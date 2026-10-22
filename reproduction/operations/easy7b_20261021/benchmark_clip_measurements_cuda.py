"""Independent one-H100 clipping comparison using actual native4 shard sizes.

Only small prior rank JSON is read. Parameters are empty local geometry and
gradients are synthetic FP32: no model/checkpoint/rollout is loaded or saved.
This measures the clipping component, not FSDP communication or a full outer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def run(args):
    import torch
    from search_g1.gradient_clipping import clip_global_l2
    source = json.loads(args.geometry_report.read_text())
    geometry = source['initial_parameter_samples']
    assert geometry and all(row['dtype'] == 'torch.float32' for row in geometry)
    module = torch.nn.Module()
    module.shards = torch.nn.ParameterList([
        torch.nn.Parameter(torch.empty(row['elements'], device='cuda', dtype=torch.float32))
        for row in geometry])
    for parameter in module.parameters():
        parameter.grad = torch.empty_like(parameter)
    generator = torch.Generator(device='cuda')

    def reset():
        generator.manual_seed(917239)
        for parameter in module.parameters():
            parameter.grad.uniform_(-.1, .1, generator=generator)

    def execute(mode):
        fast = mode == 'no-extra-measurements'
        return clip_global_l2(module, 1., check_finite=False,
            diagnostic=not fast, measure_post_norm=not fast)

    # Exact full-gradient comparison is outside the measured repetitions.
    reset(); reference_norm = execute('current-lean')
    expected = [parameter.grad.clone() for parameter in module.parameters()]
    reference_post = float(module._last_clipped_global_l2)
    reset(); fast_norm = execute('no-extra-measurements')
    equality = [torch.equal(parameter.grad, value)
                for parameter, value in zip(module.parameters(), expected)]
    exact_norm = torch.equal(reference_norm, fast_norm)
    del expected
    assert all(equality) and exact_norm
    assert module._last_clipped_global_l2 is None
    assert module._last_clipped_global_l2_measured is False
    torch.cuda.synchronize()
    rows = []
    for repeat in range(args.repeats):
        order = ('current-lean', 'no-extra-measurements') if repeat % 2 == 0 else (
            'no-extra-measurements', 'current-lean')
        for mode in order:
            reset(); torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            began = time.perf_counter(); start.record()
            norm = execute(mode)
            end.record(); torch.cuda.synchronize()
            row = dict(mode=mode, repeat=repeat, wall_seconds=time.perf_counter()-began,
                cuda_seconds=start.elapsed_time(end)/1000., actual_preclip_norm=float(norm))
            rows.append(row)
            print('CLIP_MEASUREMENT_PROGRESS='+json.dumps(row), flush=True)
    summary = {mode: dict(median_wall_seconds=statistics.median(
        row['wall_seconds'] for row in rows if row['mode'] == mode),
        median_cuda_seconds=statistics.median(row['cuda_seconds'] for row in rows if row['mode'] == mode))
        for mode in ('current-lean', 'no-extra-measurements')}
    before = summary['current-lean']['median_wall_seconds']
    after = summary['no-extra-measurements']['median_wall_seconds']
    result = dict(status='GRADIENT_CLIP_LOCAL_GEOMETRY_COMPLETE',
        job_id=__import__('os').environ.get('SLURM_JOB_ID'), epoch=time.time(),
        gpu=torch.cuda.get_device_name(), source_geometry_report=str(args.geometry_report),
        source_geometry_report_sha256=hashlib.sha256(args.geometry_report.read_bytes()).hexdigest(),
        geometry_shards=len(geometry), gradient_elements=sum(row['elements'] for row in geometry),
        gradient_bytes=sum(row['elements']*4 for row in geometry), gradient_dtype='torch.float32',
        full_gradient_shards_bitwise_equal=equality, preclip_norm_bitwise_equal=exact_norm,
        reference_measured_postclip_norm=reference_post,
        fast_postclip_norm_measured=False, modes=summary, repeats=rows,
        component_elapsed_reduction_fraction=1.-after/before,
        component_seconds_saved_for_twenty_clips=20.*(before-after),
        checkpoint_loaded=False, model_loaded=False, training_adam_writes=0,
        production_modified=False, full_outer_speedup_proven=False,
        limitations=['One GPU local gradient geometry; no native4 collective timing or trained forward/Adam.',
                    'FP64 norm/coefficient and per-element clipping operations are unchanged.',
                    'Twenty-clip projection is this component only, not a full-outer speed claim.'])
    args.base.mkdir(parents=True, exist_ok=True)
    with (args.base/'result.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print('CLIP_MEASUREMENT_COMPLETE='+json.dumps({key: result[key] for key in (
        'status', 'gradient_bytes', 'component_elapsed_reduction_fraction',
        'component_seconds_saved_for_twenty_clips')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--geometry-report', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=6)
    run(parser.parse_args())
