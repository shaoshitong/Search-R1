"""Independent one-H100 FA3 adapter versus FP32 SDPA-MATH, without a model.

Uses packed micro8 causal sequences up to4096, compact GQA and the actual
expanded28-KV head-major strides. FP32 reference leaves/gradients stay FP32;
no FA2 comparison, optimizer, sampling, production edits or automatic switch.
"""
import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True


def _need(condition, message):
    if not condition:
        raise ValueError('FA3_SDPA_' + message)


def _metrics(reference, actual):
    import torch
    a, b = reference.detach().float(), actual.detach().float()
    delta = b-a
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    norm = float(torch.linalg.vector_norm(a)) if finite else 0.
    difference = float(torch.linalg.vector_norm(delta)) if finite else 0.
    return dict(finite=finite, elements=a.numel(),
                reference_max_abs=float(a.abs().max()) if finite else None,
                max_abs=float(delta.abs().max()) if finite else None,
                relative_l2=difference/max(norm, 1e-30) if finite else None)


def _reference(values, lengths, upstream):
    """Stream each complete sequence's FP32 math graph to bound4096 memory."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v = [value.detach().float().requires_grad_(True) for value in values]
    outputs = []
    start = 0
    for length in lengths:
        end = start+length
        qs = q[start:end].transpose(0, 1).unsqueeze(0)
        group = q.shape[1]//k.shape[1]
        ks = k[start:end].repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)
        vs = v[start:end].repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(
                qs, ks, vs, dropout_p=0., is_causal=True, scale=q.shape[-1]**-.5)
        packed = out.squeeze(0).transpose(0, 1)
        packed.backward(upstream[start:end].float())
        outputs.append(packed.detach())
        del qs, ks, vs, out, packed
        start = end
    _need(start == q.shape[0], 'REFERENCE_SEQUENCE_LENGTHS_DIFFER')
    return [torch.cat(outputs)] + [value.grad.detach() for value in (q, k, v)]


def _candidate(kernel, values, cu, maximum, upstream):
    import torch
    # preserve_format keeps actual noncontiguous [128,T*128,1] K/V strides.
    tensors = [value.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
               for value in values]
    out = kernel(*tensors, cu, cu, maximum, maximum, dropout_p=0.,
                 softmax_scale=None, causal=True, window_size=(-1, -1), softcap=0., deterministic=False)
    out.backward(upstream)
    return [out.detach()] + [value.grad.detach() for value in tensors]


def run(base):
    import torch
    from flash_attn import flash_attn_varlen_func
    from search_g1 import flash3_attention as adapter
    base = Path(base).resolve(strict=True)
    run_manifest = json.loads((base/'logs/run-manifest.json').read_text())
    _need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
          'H100' in torch.cuda.get_device_name(0), 'ISOLATED_ONE_H100_REQUIRED')
    _need(run_manifest.get('training_world_size') == 4, 'NATIVE4_SOURCE_MANIFEST_REQUIRED')
    torch.cuda.set_device(0)
    backend = adapter.load_backend(run_manifest)
    audit = adapter._Audit(backend)
    kernel = adapter._Kernel(flash_attn_varlen_func, backend, audit, 'varlen')
    generator = torch.Generator(device='cuda').manual_seed(746301)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    original_reduction = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    limits = dict(output_relative_l2=.01, gradient_relative_l2=.02,
                  max_abs_relative_to_reference_peak=.02, max_abs_floor=.001)
    report = dict(format='easy7b_fa3_vs_fp32_sdpa_math_v1', base=str(base),
        diagnostic_job=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(),
        torch=torch.__version__, device=torch.cuda.get_device_name(0),
        source_commit=run_manifest['source_commit'], provenance=backend.provenance,
        independent_reference='torch SDPABackend.MATH; FP32 leaves, softmax and gradients; '
                              'autocast/TF32 disabled; complete sequence causal self-attention',
        fixed_bf16_tolerances=limits, fa2_equivalence_required=False,
        model_loaded=False, optimizer_calls=0, scheduler_calls=0, sampling_calls=0,
        production_modified=False, automatic_switch=False, cases=[], started_unix=time.time())
    # First case tests native GQA reduction. Other cases match real HF's KV4
    # repeated to28 heads, then transposed into the observed head-major strides.
    cases = [('compact_gqa', 4, 1., 1., 1.),
             ('production_expanded28', 28, 1., 1., 1.),
             ('production_expanded28_high_magnitude', 28, 3.5, 26.7, .25)]
    lengths = [31, 127, 257, 513, 1024, 1500, 2343, 4096]
    cu = torch.tensor([0]+list(itertools.accumulate(lengths)), device='cuda', dtype=torch.int32)
    total = sum(lengths)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for label, kv_heads, qscale, kscale, vscale in cases:
            values = []
            q = (torch.randn((total, 28, 128), device='cuda', generator=generator)*qscale).to(torch.bfloat16)
            values.append(q)
            for scale in (kscale, vscale):
                value = (torch.randn((total, 4, 128), device='cuda', generator=generator)*scale).to(torch.bfloat16)
                if kv_heads == 28:
                    value = value.repeat_interleave(7, dim=1)
                values.append(value.transpose(0, 1).contiguous().transpose(0, 1))
            upstream = torch.randn((total, 28, 128), device='cuda', dtype=torch.bfloat16, generator=generator)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            candidate = _candidate(kernel, values, cu, max(lengths), upstream)
            reference = _reference(values, lengths, upstream)
            comparisons = {name: _metrics(a, b) for name, a, b in
                           zip(('output', 'dq', 'dk', 'dv'), reference, candidate)}
            checks = {name: bool(value['finite'] and value['relative_l2'] <= limits[
                'output_relative_l2' if name == 'output' else 'gradient_relative_l2'] and
                value['max_abs'] <= limits['max_abs_floor'] +
                limits['max_abs_relative_to_reference_peak']*value['reference_max_abs'])
                for name, value in comparisons.items()}
            changed = [value.clone(memory_format=torch.preserve_format) for value in values]
            for value in changed:
                value[:lengths[0]].add_(3)
            with torch.no_grad():
                separate = kernel(*changed, cu, cu, max(lengths), max(lengths), causal=True)
            isolation = bool(torch.equal(separate[lengths[0]:], candidate[0][lengths[0]:]))
            boundary = total-lengths[-1]+lengths[-1]//2
            for value in changed[1:]:
                value[boundary:].add_(3)
            with torch.no_grad():
                future = kernel(*changed, cu, cu, max(lengths), max(lengths), causal=True)
            causal = bool(torch.equal(future[total-lengths[-1]:boundary], candidate[0][total-lengths[-1]:boundary]))
            torch.cuda.synchronize()
            report['cases'].append(dict(label=label, lengths=lengths, q_heads=28, kv_heads=kv_heads,
                head_dim=128, input_dtype='bfloat16', input_strides=[list(value.stride()) for value in values],
                input_scales=dict(q=qscale, k=kscale, v=vscale), reference_gradient_dtype='torch.float32',
                reference_covers_all_queries=True, comparisons=comparisons, tolerance_checks=checks,
                other_sequences_bitwise_unchanged=isolation, causal_prefix_bitwise_unchanged=causal,
                passed=all(checks.values()) and isolation and causal,
                elapsed_seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                performance_scope='diagnostic FA3 plus streaming quadratic SDPA; not a FA3 throughput measurement'))
            del values, q, value, upstream, candidate, reference, changed, separate, future
            gc.collect(); torch.cuda.empty_cache()
        report['passed'] = all(row['passed'] for row in report['cases'])
        report['execution'] = audit.report(synchronize=True)
        report['status'] = 'FA3_SDPA_MATH_FORWARD_BACKWARD_PASS' if report['passed'] else 'FA3_SDPA_MATH_TOLERANCE_REJECTED'
        report['scope'] = 'isolated packed attention implementation correctness; no FA2 bitwise or full-model training claim'
        report['finished_unix'] = time.time()
        return report
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original_reduction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='isolated manifest with explicit six-field FA3 source binding')
    args = parser.parse_args(argv)
    try:
        report = run(args.base)
        code = 0 if report['passed'] else 2
    except Exception as error:
        report = dict(status='FA3_SDPA_DIAGNOSTIC_FAILED', passed=False,
                      error_type=type(error).__name__, error=str(error), production_modified=False,
                      automatic_switch=False)
        code = 1
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
