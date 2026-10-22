"""Diagnostic clipping A/B inside the real native4 CP50/selection51 benchmark.

Only this independent benchmark process temporarily wraps gradient clipping.
Original gradients and legacy clipped gradients are cloned on their actual
devices. The new clipping result remains in place for each of twenty real Adam
writes. CPU-offloaded gradient placement is reported rather than inferred from
a synthetic CUDA benchmark. Copies, equality checks and pre-synchronization are
outside the two measured clipping bodies. No tensors are written to disk.
"""
import argparse
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from reproduction.operations.easy7b_20261021 import benchmark_native4_fastpath as native


def validate_args(args):
    if args.mode != 'leancompact' or args.steps != 20 or args.selection_step != 51:
        raise ValueError('Clipping comparison requires leancompact, twenty real writes and selection51')


class ClipComparison:
    def __init__(self, original, synchronize):
        self.original = original
        self.synchronize = synchronize
        self.rows = []
        self.geometry = None

    def timed(self, module, max_norm, kwargs, *, fast):
        before = time.perf_counter()
        self.synchronize()
        pre_sync = time.perf_counter() - before
        began = time.perf_counter()
        result = self.original(module, max_norm, **dict(kwargs, diagnostic=not fast,
            check_finite=False, measure_post_norm=not fast))
        self.synchronize()
        return result, dict(body_wall_seconds=time.perf_counter() - began,
                           pre_sync_seconds=pre_sync)

    def __call__(self, module, max_norm, **kwargs):
        import torch
        if len(self.rows) >= 20:
            raise ValueError('Clipping comparison received more than twenty calls')
        with torch.no_grad():
            named = [(name, p.grad) for name, p in module.named_parameters() if p.grad is not None]
            if not named:
                raise ValueError('Actual clipping call has no gradients')
            geometry = [dict(name=name, device=str(grad.device), dtype=str(grad.dtype),
                shape=list(grad.shape), elements=grad.numel(), bytes=grad.numel()*grad.element_size())
                for name, grad in named]
            if self.geometry is None:
                self.geometry = geometry
            placement = {}
            for row in geometry:
                device = placement.setdefault(row['device'], dict(parameters=0, elements=0, bytes=0))
                device['parameters'] += 1
                device['elements'] += row['elements']
                device['bytes'] += row['bytes']
            originals = [grad.detach().clone() for _, grad in named]
            old_norm, old_timing = self.timed(module, max_norm, kwargs, fast=False)
            old_norm = old_norm.detach().clone()
            old_clipped = [grad.detach().clone() for _, grad in named]
            for (_, grad), backup in zip(named, originals):
                grad.copy_(backup)
            new_norm, new_timing = self.timed(module, max_norm, kwargs, fast=True)
            equal = [torch.equal(grad, expected) for (_, grad), expected in zip(named, old_clipped)]
            norm_equal = torch.equal(old_norm, new_norm)
            unmeasured = (module._last_clipped_global_l2 is None and
                          module._last_clipped_global_l2_measured is False)
            row = dict(call_index=len(self.rows), context=kwargs.get('context'),
                gradient_placement=placement, gradient_parameters=len(named),
                gradient_bytes=sum(value['bytes'] for value in geometry),
                geometry_equal_to_first_call=geometry == self.geometry,
                old_lean=old_timing, new_lean=new_timing,
                all_gradient_tensors_bitwise_equal=all(equal),
                gradient_tensors_compared=len(equal), failed_gradient_names=[
                    name for (name, _), passed in zip(named, equal) if not passed],
                preclip_norm_bitwise_equal=norm_equal,
                fast_postclip_norm_explicitly_unmeasured=unmeasured,
                new_clipped_gradients_left_for_adam=True)
            self.rows.append(row)
            del originals, old_clipped, old_norm
            if len(self.rows) == 1 or len(self.rows) % 5 == 0:
                print('NATIVE4_CLIP_PROGRESS='+__import__('json').dumps(dict(
                    rank=int(os.environ.get('RANK', '0')), completed_calls=len(self.rows),
                    gradient_placement=placement, old_lean=old_timing, new_lean=new_timing,
                    all_gradient_tensors_bitwise_equal=all(equal), preclip_norm_bitwise_equal=norm_equal)), flush=True)
            if not (all(equal) and norm_equal and unmeasured):
                raise ValueError('Actual native clipping A/B differs; see per-rank comparison')
            return new_norm


def native_completion(output, rank):
    rank_report = native.read(output/('rank-%d.json' % rank))
    result = native.read(output/'result.json')
    if (rank_report.get('mode') != 'leancompact' or
            rank_report.get('actual_optimizer_writes') != 20 or
            rank_report.get('actual_optimizer_calls') != 20 or
            rank_report.get('source_binding', {}).get('selection_step') != 51 or
            result.get('status') != 'NATIVE4_REAL_ADAM_BENCHMARK_COMPLETED' or
            result.get('mode') != 'leancompact' or result.get('actual_world_size') != 4 or
            result.get('steps') != 20):
        raise ValueError('Underlying native4 benchmark did not complete its required real work')
    return dict(rank_report=str(output/('rank-%d.json' % rank)),
                rank_report_sha256=native.sha(output/('rank-%d.json' % rank)),
                result=str(output/'result.json'), result_sha256=native.sha(output/'result.json'),
                actual_optimizer_writes=20, actual_optimizer_calls=20,
                source_binding=rank_report['source_binding'])


def run(args):
    import torch
    from search_g1 import gradient_clipping
    validate_args(args)
    rank = int(os.environ['RANK'])
    output = Path(args.output).absolute()
    original = gradient_clipping.clip_global_l2
    comparison = ClipComparison(original, torch.cuda.synchronize)
    complete = None
    failure = None
    try:
        gradient_clipping.clip_global_l2 = comparison
        native.run(args)
        complete = native_completion(output, rank)
        if len(comparison.rows) != 20 or not all(
                row['all_gradient_tensors_bitwise_equal'] and row['preclip_norm_bitwise_equal']
                and row['fast_postclip_norm_explicitly_unmeasured'] for row in comparison.rows):
            raise ValueError('Require twenty complete, bitwise-equal clipping comparisons per rank')
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error)[-2000:])
        raise
    finally:
        gradient_clipping.clip_global_l2 = original
        timings = {}
        for mode in ('old_lean', 'new_lean'):
            values = [row[mode]['body_wall_seconds'] for row in comparison.rows]
            pre_sync = [row[mode]['pre_sync_seconds'] for row in comparison.rows]
            timings[mode] = dict(body_wall_sum_seconds=sum(values),
                body_wall_median_seconds=statistics.median(values) if values else None,
                pre_sync_sum_seconds=sum(pre_sync),
                pre_sync_median_seconds=statistics.median(pre_sync) if pre_sync else None)
        result = dict(status='NATIVE4_CLIP_COMPARISON_COMPLETE' if failure is None else 'FAILED',
            epoch=time.time(), rank=rank, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            mode=args.mode, steps=args.steps, selection_step=args.selection_step,
            checkpoint=str(args.checkpoint), source_base=str(args.source_base),
            actual_clipping_calls=len(comparison.rows), all_twenty_calls_equal=(
                len(comparison.rows) == 20 and all(row['all_gradient_tensors_bitwise_equal']
                and row['preclip_norm_bitwise_equal'] for row in comparison.rows)),
            initial_gradient_geometry=comparison.geometry, timings=timings, calls=comparison.rows,
            native_benchmark_completion=complete, failure=failure,
            original_clipping_function_restored=gradient_clipping.clip_global_l2 is original,
            production_modified=False, tensor_files_saved=False,
            wrapper_source_sha256=native.sha(Path(__file__)),
            clipping_source_sha256=native.sha(Path(gradient_clipping.__file__)),
            body_timing_scope='clip call and final CUDA synchronization only; pre-sync, backup/restore '
                              'copies and full gradient equality checks are excluded',
            limitations=['Diagnostic run executes both clipping paths and clones full gradients; '
                         'its full benchmark elapsed time is not a production speed estimate.',
                         'Actual gradient device placement is observed on each real native4 backward.',
                         'Uses the existing benchmark fixed EM advantages, not production grounding rewards.'])
        native.write(output/('clip-comparison-rank-%d.json' % rank), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-base', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=('leancompact',), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, choices=(20,), default=20)
    parser.add_argument('--selection-step', type=int, choices=(51,), default=51)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print('NATIVE4_CLIP_COMPARISON='+__import__('json').dumps({
            key: result[key] for key in ('status', 'rank', 'actual_clipping_calls',
                                        'all_twenty_calls_equal', 'timings')}), flush=True)
    except Exception as error:
        native.emit(args.output, 'NATIVE4_CLIP_COMPARISON_FAILED', mode=args.mode,
                    error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
