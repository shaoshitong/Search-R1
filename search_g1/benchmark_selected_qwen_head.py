"""Optional CUDA benchmark for dense versus selected Qwen2.5-7B LM-head rows.

No complete model is instantiated or loaded. Random in-memory weights have the
real [152064, 3584] head shape; BF16 autocast and the repository's exact FP32
selected-token log-probability reduction are used in BOTH cases. Dense projects
every input row and subsequently selects the same noncontinuous target rows;
selected chooses those hidden rows before the otherwise identical projection.

This measures only LM-head/log-probability forward/backward. It excludes decoder,
FSDP, optimizer, rollout, retrieval, the temporary Python hook, and CPU numerical
comparison. It cannot establish whole-outer speedup or a feasible training batch.
Merely importing the module, --help, and syntax checks never initialize CUDA.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import time


HIDDEN_SIZE = 3584
VOCAB_SIZE = 152064


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=256,
                        help='Sequence/packed token rows per batch item, including non-policy rows')
    parser.add_argument('--selected-tokens', type=int, default=64,
                        help='Same noncontinuous prediction rows used by both modes')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=81723)
    parser.add_argument('--master-dtype', choices=('float32', 'bfloat16'), default='float32',
                        help='FP32 mirrors actor master/head parameters; BF16 omits FP32-master cost')
    parser.add_argument('--hf-output-fp32', action='store_true',
                        help='Emulate HF versions that cast all produced logits to FP32 before returning')
    parser.add_argument('--json-output', type=Path,
                        help='Optional small JSON receipt; tensors/weights are never saved')
    value = parser.parse_args(argv)
    if (min(value.tokens, value.selected_tokens, value.batch, value.repeats) < 1
            or value.selected_tokens > value.tokens or value.warmup < 0):
        parser.error('Require 1 <= selected-tokens <= tokens, positive batch/repeats and nonnegative warmup')
    if not value.device.startswith('cuda'):
        parser.error('This benchmark intentionally requires an explicitly selected CUDA device')
    return value


def error_statistics(reference, actual, *, chunk_elements=1048576):
    """Full CPU comparison, chunked so no extra vocabulary-size double copy exists."""
    import torch
    first = reference.detach().reshape(-1)
    second = actual.detach().reshape(-1)
    if first.shape != second.shape:
        raise ValueError('Numerical comparison tensor shapes differ')
    maximum = reference_maximum = square_error = square_reference = 0.0
    finite = True
    for start in range(0, first.numel(), chunk_elements):
        stop = start + chunk_elements
        ref = first[start:stop].to(device='cpu', dtype=torch.float64)
        got = second[start:stop].to(device='cpu', dtype=torch.float64)
        finite = finite and bool(torch.isfinite(ref).all()) and bool(torch.isfinite(got).all())
        if not finite:
            return dict(elements=first.numel(), all_finite=False,
                        max_absolute_error=None, reference_max_absolute=None,
                        root_mean_square_error=None, relative_l2_error=None,
                        compared_all_elements=False,
                        reason='A nonfinite value prevents a meaningful numerical error metric')
        difference = got - ref
        maximum = max(maximum, float(difference.abs().max()))
        reference_maximum = max(reference_maximum, float(ref.abs().max()))
        square_error += float(difference.square().sum())
        square_reference += float(ref.square().sum())
    return dict(elements=first.numel(), all_finite=finite, max_absolute_error=maximum,
                reference_max_absolute=reference_maximum,
                root_mean_square_error=math.sqrt(square_error / max(1, first.numel())),
                relative_l2_error=(math.sqrt(square_error / square_reference)
                                   if square_reference > 0 else None),
                compared_all_elements=True)


def _iteration(torch, head, hidden, positions, labels, objective_weights, mode, options):
    from search_g1.logprob_memory import selected_log_probs
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    logits = projected = log_probs = loss = None
    try:
        events[0].record()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            projected = head(hidden if mode == 'dense' else hidden.index_select(1, positions))
        # Some pinned HF implementations upcast the complete LM-head result.
        # This optional flag is applied symmetrically before either mode returns.
        if options.hf_output_fp32:
            projected = projected.float()
        logits = projected.index_select(1, positions) if mode == 'dense' else projected
        log_probs = selected_log_probs(logits.float(), labels)
        # Signed token weights exercise positive/negative policy gradients. The
        # benchmark deliberately excludes PPO/DualClip and optimizer state.
        loss = (log_probs * objective_weights).sum() / objective_weights.abs().sum()
        events[1].record()
        loss.backward()
        events[2].record()
        torch.cuda.synchronize()
        return dict(forward_ms=events[0].elapsed_time(events[1]),
                    backward_ms=events[1].elapsed_time(events[2]),
                    forward_backward_ms=events[0].elapsed_time(events[2])), log_probs.detach(), loss.detach()
    finally:
        # In particular release partial graphs after a dense allocation OOM.
        logits = projected = log_probs = loss = None


def run_case(torch, head, template, positions, labels, objective_weights, mode, options,
             reference=None):
    head.zero_grad(set_to_none=True)
    hidden = template.detach().clone().requires_grad_(True)
    last_lp = last_loss = None
    samples = []
    result = dict(mode=mode, status='starting', warmup_completed=0, measured_completed=0,
                  input_token_rows=options.batch * options.tokens,
                  projected_token_rows=options.batch * (options.tokens if mode == 'dense' else options.selected_tokens),
                  selected_policy_token_rows=options.batch * options.selected_tokens,
                  vocabulary_columns=VOCAB_SIZE, hidden_columns=HIDDEN_SIZE)
    numerical_snapshot = None
    try:
        for index in range(options.warmup + options.repeats):
            head.zero_grad(set_to_none=True)
            hidden.grad = None
            last_lp = last_loss = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
            reserved_before = torch.cuda.memory_reserved()
            started = time.perf_counter()
            sample, last_lp, last_loss = _iteration(
                torch, head, hidden, positions, labels, objective_weights, mode, options)
            sample.update(synchronized_wall_ms=(time.perf_counter() - started) * 1000,
                          allocated_before_bytes=allocated_before,
                          reserved_before_bytes=reserved_before,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            sample['peak_allocated_increment_bytes'] = sample['peak_allocated_bytes'] - allocated_before
            if index < options.warmup:
                result['warmup_completed'] += 1
            else:
                samples.append(sample)
                result['measured_completed'] += 1
        result.update(status='complete', samples=samples, summary={
            key: statistics.median(row[key] for row in samples) for key in (
                'forward_ms', 'backward_ms', 'forward_backward_ms', 'synchronized_wall_ms',
                'peak_allocated_bytes', 'peak_reserved_bytes', 'peak_allocated_increment_bytes')})
        # Numerical comparisons/copies are outside all timed events/peak samples.
        if reference is None and mode == 'dense':
            numerical_snapshot = dict(log_probs=last_lp.to(device='cpu', copy=True),
                                      loss=last_loss.to(device='cpu', copy=True),
                                      hidden_gradient=hidden.grad.detach().to(device='cpu', copy=True),
                                      weight_gradient=head.weight.grad.detach().to(device='cpu', copy=True))
            result['numerical_reference_retained_in_ram_only'] = True
        elif reference is not None:
            result['dense_comparison'] = {
                'log_probs': error_statistics(reference['log_probs'], last_lp),
                'loss': error_statistics(reference['loss'], last_loss),
                'hidden_gradient': error_statistics(reference['hidden_gradient'], hidden.grad),
                'weight_gradient': error_statistics(reference['weight_gradient'], head.weight.grad),
                'scope': 'All selected log probabilities and every hidden/head-weight gradient element; BF16 rounding may differ'}
    except torch.cuda.OutOfMemoryError as error:
        # Allocation OOM is recorded separately. Do not lower dimensions, skip
        # vocabulary columns, change labels or change math before the next mode.
        result.update(status='cuda_out_of_memory', error=str(error), samples=samples,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      numerical_comparison_available=False)
        numerical_snapshot = None
    finally:
        head.zero_grad(set_to_none=True)
        hidden.grad = None
        hidden = last_lp = last_loss = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result, numerical_snapshot


def main(argv=None):
    options = arguments(argv)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device available; no benchmark was executed')
    device = torch.device(options.device)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected CUDA device does not support the requested BF16 computation')
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)
    # Explicitly exclude TF32 from the nominally BF16-only comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    master_dtype = torch.float32 if options.master_dtype == 'float32' else torch.bfloat16
    head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False, device=device, dtype=master_dtype)
    # Same immutable random bytes, labels, selected rows and signed objective
    # weights are reused for both modes. No decoder/model/tokenizer is loaded.
    template = torch.randn(options.batch, options.tokens, HIDDEN_SIZE,
                           device=device, dtype=torch.bfloat16)
    positions = torch.randperm(options.tokens, device=device)[:options.selected_tokens].sort().values
    labels = torch.randint(VOCAB_SIZE, (options.batch, options.selected_tokens), device=device)
    weights = torch.randn(options.batch, options.selected_tokens, device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dense, reference = run_case(torch, head, template, positions, labels, weights, 'dense', options)
    selected, _ = run_case(torch, head, template, positions, labels, weights, 'selected', options,
                           reference=reference)
    if reference is None:
        selected['numerical_comparison_available'] = False
        selected['numerical_comparison_reason'] = 'Dense did not complete at these exact same dimensions'
    result = dict(format='qwen25_7b_selected_head_cuda_microbenchmark_v1',
        started_unix=started, completed_unix=time.time(),
        options={key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()},
        device=dict(index=device.index, name=torch.cuda.get_device_name(device),
                    total_memory_bytes=torch.cuda.get_device_properties(device).total_memory),
        software=dict(torch=torch.__version__, cuda=torch.version.cuda),
        geometry=dict(hidden_size=HIDDEN_SIZE, vocabulary_size=VOCAB_SIZE,
                      head_parameters=HIDDEN_SIZE * VOCAB_SIZE, compute_dtype='bfloat16',
                      head_master_dtype=str(master_dtype), log_probability_dtype='float32',
                      same_labels_positions_hidden_weights=True),
        objective='Signed weighted selected-token log probabilities with exact full-vocabulary normalization',
        modes=[dense, selected],
        limitations=['LM-head and selected-log-probability computation only; no full 7B model loaded',
                     'No decoder, FSDP, optimizer, rollout, retrieval or Python-hook throughput measured',
                     'Numerical RAM copies/comparison are outside timings and peak-memory samples',
                     'Not evidence of whole-outer speedup or micro-batch16 capacity'])
    if dense['status'] == selected['status'] == 'complete':
        result['head_only_median_time_ratio_dense_over_selected'] = (
            dense['summary']['forward_backward_ms'] / selected['summary']['forward_backward_ms'])
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if options.json_output is not None:
        options.json_output.parent.mkdir(parents=True, exist_ok=True)
        options.json_output.write_text(encoded + '\n')
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
