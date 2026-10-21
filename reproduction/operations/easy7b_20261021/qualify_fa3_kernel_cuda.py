"""One-GPU FA2/FA3 kernel diagnostic; never loads or updates a policy.

Synthetic BF16 Q/K/V use the actual Qwen7B GQA geometry, eight independent
packed sequences and up to4096 context. This does NOT certify full-model PPO
equivalence, checkpoint restoration, generation, or end-to-end speedup.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True


def run(interface, output):
    import torch
    from flash_attn import flash_attn_varlen_func as fa2
    from torch.nn.attention import SDPBackend, sdpa_kernel

    interface, output = Path(interface).resolve(strict=True), Path(output).absolute()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('Require new diagnostic output in an existing private directory')
    if not os.environ.get('SLURM_JOB_ID', '').isdecimal() or torch.cuda.device_count() != 1:
        raise ValueError('Require actual one-GPU Slurm diagnostic allocation')
    if 'H100' not in torch.cuda.get_device_name(0):
        raise ValueError('Require H100 for this production-candidate comparison')
    # Resolve only the FA3 interface/extension from its pinned directory. Torch
    # and FA2 above already come from the unchanged production Python runtime.
    sys.path.insert(0, str(interface.parent))
    spec = importlib.util.spec_from_file_location('easy7b_pinned_fa3_interface', interface)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = torch.Generator(device='cuda').manual_seed(743091)
    cases = [('short_reference', [31, 64, 127, 193, 256, 384, 500, 512]),
             ('micro8_variable', [513, 719, 1024, 1500, 2048, 2777, 3072, 4096]),
             ('micro8_context4096', [4096] * 8)]
    report = dict(format='easy7b_fa3_kernel_diagnostic_v1', checked_unix=time.time(),
        job=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(), torch=torch.__version__,
        compiled_cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)), interface=str(interface),
        interface_sha256=hashlib.sha256(interface.read_bytes()).hexdigest(),
        ld_preload=os.environ.get('LD_PRELOAD'), diagnostic_only=True,
        model_loaded=False, optimizer_steps=0, scheduler_steps=0, production_modified=False,
        full_model_qualified=False, cases=[],
        thresholds=dict(output_relative_l2=.005, output_max_abs=.03125,
                        gradient_relative_l2=.01, gradient_max_abs=.125))

    def measure(function, values, cu, maximum, upstream):
        tensors = [value.detach().clone().requires_grad_(True) for value in values]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*tensors, cu, maximum)
        if not isinstance(result, torch.Tensor):
            raise TypeError('Diagnostic wrapper must retain the native autograd output tensor')
        (result.float() * upstream).sum().backward()
        end.record(); end.synchronize()
        return [result.detach()] + [value.grad.detach() for value in tensors], start.elapsed_time(end)

    def first(value):
        # This pinned older FA3 interface returns (out, softmax_lse).
        return value[0] if isinstance(value, tuple) else value

    def baseline(q, k, v, cu, maximum):
        return fa2(q, k, v, cu, cu, maximum, maximum, dropout_p=0.,
                   softmax_scale=1 / math.sqrt(128), causal=True, window_size=(-1, -1))

    def candidate(q, k, v, cu, maximum):
        return first(module.flash_attn_varlen_func(q, k, v, cu, cu, None, None,
                     maximum, maximum, softmax_scale=1 / math.sqrt(128), causal=True,
                     window_size=(-1, -1), softcap=0., num_splits=1,
                     deterministic=False))

    def difference(left, right):
        a, b = left.float(), right.float()
        delta = a - b
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        return dict(finite=finite, max_abs=float(delta.abs().max()),
                    relative_l2=float(torch.linalg.vector_norm(delta) /
                                      torch.linalg.vector_norm(a).clamp_min(1e-30)))

    for label, lengths in cases:
        cu = torch.tensor([0] + list(__import__('itertools').accumulate(lengths)),
                          device='cuda', dtype=torch.int32)
        size = sum(lengths)
        values = [torch.randn((size, heads, 128), device='cuda', dtype=torch.bfloat16,
                              generator=generator) for heads in (28, 4, 4)]
        upstream = torch.randn((size, 28, 128), device='cuda', dtype=torch.float32,
                               generator=generator)
        old, old_ms = measure(baseline, values, cu, max(lengths), upstream)
        repeated, repeated_ms = measure(baseline, values, cu, max(lengths), upstream)
        new, new_ms = measure(candidate, values, cu, max(lengths), upstream)
        comparisons = {name: difference(a, b) for name, a, b in
                       zip(('output', 'dq', 'dk', 'dv'), old, new)}
        noise = {name: difference(a, b) for name, a, b in
                 zip(('output', 'dq', 'dk', 'dv'), old, repeated)}
        accepted = all(value['finite'] and
                       value['relative_l2'] <= (.005 if name == 'output' else .01) and
                       value['max_abs'] <= (.03125 if name == 'output' else .125)
                       for name, value in comparisons.items())
        row = dict(label=label, lengths=lengths, query_heads=28, kv_heads=4,
            head_dim=128, dtype='bfloat16', causal=True, dropout=0.,
            fa3_vs_fa2=comparisons, fa2_repeat_noise=noise, accepted=accepted,
            first_call_forward_backward_ms=dict(fa2=old_ms, fa2_repeat=repeated_ms, fa3=new_ms),
            timing_is_not_warmed_production_benchmark=True)
        if label == 'short_reference':
            def reference(q, k, v, unused_cu, unused_maximum):
                result = []
                for a, b in zip(cu.tolist()[:-1], cu.tolist()[1:]):
                    qs = q[a:b].float().transpose(0, 1).unsqueeze(0)
                    ks = k[a:b].float().repeat_interleave(7, dim=1).transpose(0, 1).unsqueeze(0)
                    vs = v[a:b].float().repeat_interleave(7, dim=1).transpose(0, 1).unsqueeze(0)
                    with sdpa_kernel(SDPBackend.MATH):
                        out = torch.nn.functional.scaled_dot_product_attention(
                            qs, ks, vs, dropout_p=0., is_causal=True)
                    result.append(out.squeeze(0).transpose(0, 1))
                return torch.cat(result)
            ref, _ = measure(reference, values, cu, max(lengths), upstream)
            row['fp32_math_reference'] = {
                backend: {name: difference(r, x) for name, r, x in
                          zip(('output', 'dq', 'dk', 'dv'), ref, data)}
                for backend, data in [('fa2', old), ('fa3', new)]}
            del ref
        # Perturb K/V as well as Q: changing Q alone cannot expose a kernel
        # that incorrectly reads another sequence's keys and values.
        changed = [value.clone() for value in values]
        for value in changed:
            value[:lengths[0]].add_(3)
        with torch.no_grad():
            isolated = candidate(*changed, cu, max(lengths))
        row['other_sequences_bitwise_unchanged'] = bool(torch.equal(
            isolated[lengths[0]:], new[0][lengths[0]:]))
        # Future K/V in the final sequence must not affect its causal prefix.
        boundary = size - lengths[-1] + lengths[-1] // 2
        for value in changed[1:]:
            value[boundary:].add_(3)
        with torch.no_grad():
            causal_changed = candidate(*changed, cu, max(lengths))
        start_last = size - lengths[-1]
        row['causal_prefix_bitwise_unchanged'] = bool(torch.equal(
            causal_changed[start_last:boundary], new[0][start_last:boundary]))
        row['accepted'] = (accepted and row['other_sequences_bitwise_unchanged'] and
                           row['causal_prefix_bitwise_unchanged'])
        report['cases'].append(row)
        del old, repeated, new, values, changed, isolated, causal_changed, upstream
        torch.cuda.empty_cache()
    report['accepted'] = all(row['accepted'] for row in report['cases'])
    report['status'] = ('FA3_KERNEL_FORWARD_BACKWARD_LOW_ERROR_PASS' if report['accepted']
                        else 'FA3_KERNEL_NUMERICAL_REJECTED')
    report['finished_unix'] = time.time()
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(report), flush=True)
    if not report['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.interface, args.output)
