"""One-GPU HF reference for the saved vLLM teacher-forced continuation tokens.

Run on the remote host; only small JSON is produced. No generation, optimizer,
training update, checkpoint write, or installed package change is performed.
Attention Q/K/V and math accumulation are explicitly FP32 under SDPA-MATH;
the rest of the original HF Qwen model remains BF16. This is a full-model
reference, not a matched-attention-input kernel comparison.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def difference(a, b):
    """Same-position scalar errors; these are not a full-distribution KL."""
    if len(a) != len(b) or not a or any(len(x) != len(y) or not x for x, y in zip(a, b)):
        raise ValueError('Teacher-forced rows or token lengths differ')
    pairs = [(row, token, float(x), float(y)) for row, (xs, ys) in enumerate(zip(a, b))
             for token, (x, y) in enumerate(zip(xs, ys))]
    if not all(math.isfinite(x) and math.isfinite(y) for _, _, x, y in pairs):
        raise ValueError('Nonfinite teacher-forced log-probability')
    errors = sorted(abs(x - y) for _, _, x, y in pairs)
    return dict(tokens=len(pairs), mean_abs=statistics.mean(errors),
        median_abs=statistics.median(errors), max_abs=errors[-1],
        p99_abs=errors[min(len(errors) - 1, int(.99 * len(errors)))],
        rms=math.sqrt(statistics.mean((x-y)**2 for _, _, x, y in pairs)),
        mean_signed_a_minus_b=statistics.mean(x-y for _, _, x, y in pairs),
        worst_tokens=[dict(row=row, continuation_index=token, a=x, b=y, abs_error=abs(x-y))
                      for row, token, x, y in sorted(pairs, key=lambda r:abs(r[2]-r[3]), reverse=True)[:8]])


@contextmanager
def fp32_math_attention():
    """Process-local override, restored on exit; original HF masks/RoPE remain."""
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    original = functional.scaled_dot_product_attention
    calls = dict(count=0, input_dtypes={}, output_dtypes={}, attention_math='FP32 SDPA-MATH')

    def reference(query, key, value, *args, **kwargs):
        original_dtype = query.dtype
        mask_args = list(args)
        if mask_args and torch.is_tensor(mask_args[0]) and mask_args[0].is_floating_point():
            mask_args[0] = mask_args[0].float()
        if torch.is_tensor(kwargs.get('attn_mask')) and kwargs['attn_mask'].is_floating_point():
            kwargs['attn_mask'] = kwargs['attn_mask'].float()
        # Cast explicitly: no reliance on the installed math backend's BF16
        # reduction setting, and no Flash/efficient attention fallback.
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = original(query.float(), key.float(), value.float(), *mask_args, **kwargs)
        calls['count'] += 1
        signature = '/'.join(str(x.dtype) for x in (query, key, value))
        calls['input_dtypes'][signature] = calls['input_dtypes'].get(signature, 0) + 1
        calls['output_dtypes'][str(out.dtype)] = calls['output_dtypes'].get(str(out.dtype), 0) + 1
        return out.to(original_dtype)

    functional.scaled_dot_product_attention = reference
    try:
        yield calls
    finally:
        functional.scaled_dot_product_attention = original


def continuation_logits(logits, token_ids, original_length):
    """Token at index i is scored by logits i-1, without a new BOS/template."""
    import torch
    if not 0 < original_length < len(token_ids):
        raise ValueError('A nonempty original prefix and continuation are required')
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(token_ids):
        raise ValueError('HF returned an unexpected full-sequence logits shape')
    targets = torch.tensor(token_ids[original_length:], device=logits.device, dtype=torch.long)
    rows = logits[0, original_length - 1:len(token_ids) - 1].float()
    # vLLM prompt_logprobs score the raw vocabulary distribution. Temperature0
    # selects greedy tokens; it is not a division by zero in log-softmax.
    logprobs = torch.log_softmax(rows, dim=-1)
    return logprobs, logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def greedy_divergences(backends, teacher, original_lengths, details):
    """First differing greedy token only; no new autoregressive decode."""
    result = {}
    base = backends['xformers']['greedy_check']
    for name in ('fa2', 'fa3'):
        other = backends[name]['greedy_check']
        rows = []
        equal = 0
        for row, (x, y) in enumerate(zip(base, other)):
            a, b = x['token_ids'], y['token_ids']
            if a == b:
                equal += 1
                continue
            first = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
            item = dict(row=row, first_divergent_continuation_index=first,
                        xformers_token_id=a[first] if first < len(a) else None,
                        candidate_token_id=b[first] if first < len(b) else None)
            suffix = teacher[row][original_lengths[row]:]
            same_prefix = a[:first] == b[:first] == suffix[:first]
            item['same_prefix_as_teacher_before_divergence'] = same_prefix
            if same_prefix and first < len(details[row]) and first < min(len(a), len(b)):
                ref = details[row][first]
                item.update(reference_top1_token_id=ref['top1_id'],
                    reference_top2_token_id=ref['top2_id'],
                    reference_top1_minus_top2_logprob=ref['top1_minus_top2'],
                    reference_xformers_token_logprob=ref['target_logprob'],
                    reference_candidate_token_logprob=ref['candidate_logprobs'][name],
                    reference_xformers_minus_candidate_logprob=(ref['target_logprob'] - ref['candidate_logprobs'][name]),
                    reference_prefers=('xformers' if ref['target_logprob'] > ref['candidate_logprobs'][name]
                                       else name if ref['target_logprob'] < ref['candidate_logprobs'][name] else 'tie'))
            rows.append(item)
        result['xformers_vs_' + name] = dict(equal_sequences=equal, sequences=len(base),
            first_divergences=rows,
            interpretation='Only the first different token has a common saved prefix. Later greedy tokens have different contexts; sequence mismatch is not a same-input logprob error.')
    return result


def run(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM
    root = Path(args.root)
    payload, input_identity = read_json(root / 'inputs.json')
    teacher, teacher_identity = read_json(root / 'teacher-inputs.json')
    backends = {}; identities = {}
    for name in ('xformers', 'fa2', 'fa3'):
        backends[name], identities[name] = read_json(root / (name + '.json'))
    original = payload['short_prompts'] + payload['prompts'][:8]
    lengths = [len(row) for row in original]
    if len(teacher) != len(original) or any(ids[:n] != prefix for ids, n, prefix in zip(teacher, lengths, original)):
        raise ValueError('Saved teacher input does not extend its exact original prefix')
    for name, record in backends.items():
        if record['numerical_prompt_lengths'] != lengths or len(record['teacher_forced_logprobs']) != len(teacher):
            raise ValueError('Backend numerical rows are not bound to these saved prefixes: ' + name)
    checkpoint = args.checkpoint or payload['checkpoint']
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    print(json.dumps(dict(phase='LOADING_REFERENCE', checkpoint=checkpoint,
                         attention='explicit FP32 SDPA-MATH', batch=1)), flush=True)
    began = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', trust_remote_code=False, local_files_only=True,
        low_cpu_mem_usage=True).to(device=device).eval()
    load_seconds = time.monotonic() - began
    rotary = [dict(name=name, dtype=str(child.inv_freq.dtype)) for name, child in model.named_modules()
              if hasattr(child, 'inv_freq') and torch.is_tensor(child.inv_freq)]
    refs = []; details = []; row_times = []
    with fp32_math_attention() as audit, torch.inference_mode():
        for row, (ids, prefix_length) in enumerate(zip(teacher, lengths)):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            positions = torch.arange(len(ids), device=device, dtype=torch.long).unsqueeze(0)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                position_ids=positions, use_cache=False, return_dict=True)
            lp, chosen = continuation_logits(output.logits, ids, prefix_length)
            top_values, top_ids = lp.topk(2, dim=-1)
            candidate_values = {}
            for name, record in backends.items():
                greedy_ids = record['greedy_check'][row]['token_ids']
                if len(greedy_ids) != len(ids) - prefix_length:
                    raise ValueError('Greedy row length differs from the saved continuation')
                index = torch.tensor(greedy_ids, device=device, dtype=torch.long)
                candidate_values[name] = lp.gather(-1, index.unsqueeze(-1)).squeeze(-1).cpu().tolist()
            refs.append(chosen.cpu().tolist())
            tv, ti = top_values.cpu().tolist(), top_ids.cpu().tolist()
            details.append([dict(target_id=ids[prefix_length + j], target_logprob=value,
                top1_id=ti[j][0], top2_id=ti[j][1], top1_logprob=tv[j][0], top2_logprob=tv[j][1],
                top1_minus_top2=tv[j][0]-tv[j][1],
                candidate_logprobs={name: values[j] for name, values in candidate_values.items()})
                for j, value in enumerate(refs[-1])])
            torch.cuda.synchronize(device)
            row_times.append(time.monotonic() - started)
            print(json.dumps(dict(phase='REFERENCE_ROW_COMPLETE', row=row,
                total_rows=len(teacher), input_tokens=len(ids), continuation_tokens=len(refs[-1]),
                seconds=row_times[-1], actual_fp32_math_attention_calls=audit['count'])), flush=True)
            del output, lp, chosen, top_values, top_ids
    reference = dict(backend='HF BF16 model + explicit FP32 SDPA-MATH attention',
        teacher_forced_logprobs=refs, continuation_details=details, attention_execution=audit,
        model_dtype='bfloat16', parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        rotary_frequency_dtypes=rotary, model_class=type(model).__name__,
        attention_classes=sorted({type(m).__name__ for m in model.modules() if 'Attention' in type(m).__name__}))
    comparisons = {name + '_minus_sdpa_math': difference(record['teacher_forced_logprobs'], refs)
                   for name, record in backends.items()}
    comparisons['fa3_minus_fa2_same_teacher_tokens'] = difference(
        backends['fa3']['teacher_forced_logprobs'], backends['fa2']['teacher_forced_logprobs'])
    comparisons['fa2_minus_xformers_same_teacher_tokens'] = difference(
        backends['fa2']['teacher_forced_logprobs'], backends['xformers']['teacher_forced_logprobs'])
    result = dict(status='REFERENCE_COMPLETE', observed_unix=time.time(), checkpoint=checkpoint,
        inputs=input_identity, teacher_inputs=teacher_identity, backends=identities,
        continuation_tokens=sum(map(len, refs)), numerical_prompt_lengths=lengths,
        total_input_lengths=list(map(len, teacher)), reference=reference, comparisons=comparisons,
        greedy_divergence_analysis=greedy_divergences(backends, teacher, lengths, details),
        gpu=dict(device=str(device), name=torch.cuda.get_device_name(device),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device)),
        runtime=dict(torch=torch.__version__, transformers=__import__('transformers').__version__,
                     tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32),
        load_seconds=load_seconds, scoring_seconds=sum(row_times), per_row_seconds=row_times,
        definition='Raw vocabulary FP32 log_softmax(logits); actual suffix token at absolute index i uses logits i-1. Exact saved IDs, position0..L-1, no extra BOS/template, no padding, no generation.',
        limits=['HF batch1 dense full-sequence computation differs from vLLM packed-batch GEMM and paged decode execution.',
                'FP32 applies to SDPA attention Q/K/V and accumulation; decoder projections, MLP and LM head remain BF16.',
                'HF and vLLM may implement RoPE/normalization/projection with different kernels; this is not an isolated matched-QKV attention error measurement.',
                'Only 512 fixed continuation-token probabilities are compared, not a full-vocabulary KL or task accuracy.',
                'No automatic small-error, accuracy-preservation or production-switch acceptance is declared.'],
        training_modified=False, generation_performed=False, optimizer_loaded=False)
    output = Path(args.output) if args.output else root / 'sdpa-math-reference.json'
    write_json(output, result)
    print(json.dumps(dict(phase='REFERENCE_COMPLETE', output=str(output),
        continuation_tokens=result['continuation_tokens'], attention_calls=audit['count'],
        comparisons={name:{k:v for k,v in row.items() if k!='worst_tokens'} for name,row in comparisons.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Existing rollout-flash3-r2 directory')
    parser.add_argument('--checkpoint', help='Default: checkpoint recorded in inputs.json')
    parser.add_argument('--output', help='Default: ROOT/sdpa-math-reference.json')
    parser.add_argument('--device', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
