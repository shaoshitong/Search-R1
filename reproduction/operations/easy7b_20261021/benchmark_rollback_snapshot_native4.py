#!/usr/bin/env python3
"""Time CPU rollback snapshot construction in the independent legacy native4 probe.

The existing CP50/selection51 benchmark still performs its real twenty Adam
writes and unchanged forward/loss/clipping/monitor work. Only this diagnostic
process wraps LocalOptimizerTransaction.__init__. The original clone function
and constructor body execute unchanged. No production process is contacted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))


def tensor_stats(value, torch):
    """Count metadata of saved tensors, without reading/scanning tensor values."""
    if torch.is_tensor(value):
        return dict(tensors=1, bytes=value.numel()*value.element_size(),
                    cpu_tensors=int(value.device.type == 'cpu'))
    parts = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    result = dict(tensors=0, bytes=0, cpu_tensors=0)
    for item in parts:
        for name, count in tensor_stats(item, torch).items():
            result[name] += count
    return result


def timed_constructor(original, torch, rows, progress):
    def measured(transaction, *args, **kwargs):
        row = dict(index=len(rows), status='STARTED',
                   pre_constructor_cuda_sync_seconds=None,
                   constructor_body_wall_seconds=None)
        try:
            began = time.perf_counter()
            torch.cuda.synchronize()
            row['pre_constructor_cuda_sync_seconds'] = time.perf_counter()-began
            began = time.perf_counter()
            try:
                result = original(transaction, *args, **kwargs)
            finally:
                row['constructor_body_wall_seconds'] = time.perf_counter()-began
            # Original .to('cpu', copy=True) is blocking. Metadata collection,
            # printing, prior CUDA work and later Adam are outside the timer.
            row['parameter_snapshot'] = tensor_stats([saved for _, saved in transaction.parameters], torch)
            row['adam_snapshot'] = tensor_stats(transaction.state, torch)
            row['rng_snapshot'] = tensor_stats([transaction.cpu_rng, transaction.cuda_rng], torch)
            row['status'] = 'COMPLETE'
            return result
        except BaseException as error:
            row.update(status='FAILED', error_type=type(error).__name__, error=str(error)[:800])
            raise
        finally:
            rows.append(row)
            if len(rows) % 5 == 0 or row['status'] != 'COMPLETE':
                progress(dict(phase='ROLLBACK_SNAPSHOT_CONSTRUCTORS',
                    constructors_observed=len(rows), completed=sum(r['status']=='COMPLETE' for r in rows),
                    constructor_body_wall_seconds=sum(r['constructor_body_wall_seconds'] or 0. for r in rows),
                    latest_status=row['status']))
    return measured


def run(args):
    output = Path(args.output)
    rank = int(os.environ.get('RANK', '0'))
    rows = []
    state = dict(benchmark_completed=False, error=None)
    original = transaction_class = None
    implementation = {}
    try:
        import torch
        from search_g1 import optimizer_transaction
        from reproduction.operations.easy7b_20261021 import benchmark_native4_fastpath as native
        implementation = dict(path=str(Path(optimizer_transaction.__file__).resolve()),
            sha256=hashlib.sha256(Path(optimizer_transaction.__file__).read_bytes()).hexdigest())
        transaction_class = optimizer_transaction.LocalOptimizerTransaction
        original = transaction_class.__init__
        def progress(value):
            print('ROLLBACK_SNAPSHOT_TIMING='+json.dumps(dict(rank=rank, epoch=time.time(), **value),
                                                       sort_keys=True), flush=True)
        transaction_class.__init__ = timed_constructor(original, torch, rows, progress)
        # Reuse the existing native4 actor/state/data implementation verbatim.
        # CLI permits only the originally qualified legacy CP50/selection51 work.
        native.run(args)
        state['benchmark_completed'] = True
    except BaseException as error:
        state['error'] = dict(type=type(error).__name__, message=str(error)[:800])
        raise
    finally:
        if original is not None:
            transaction_class.__init__ = original
        complete = [row for row in rows if row['status'] == 'COMPLETE']
        times = [row['constructor_body_wall_seconds'] for row in complete]
        syncs = [row['pre_constructor_cuda_sync_seconds'] for row in rows
                 if row['pre_constructor_cuda_sync_seconds'] is not None]
        report = dict(format='native4_real_legacy_rollback_snapshot_timing_v1',
            rank=rank, slurm_job_id=os.environ.get('SLURM_JOB_ID'), epoch=time.time(),
            source_base=args.source_base, checkpoint=args.checkpoint,
            mode=args.mode, selection_step=args.selection_step,
            expected_actual_adam_writes=args.steps,
            benchmark_completed=state['benchmark_completed'], error=state['error'],
            constructors_observed=len(rows), completed_constructors=len(complete),
            measurement_complete=state['benchmark_completed'] and len(complete)==args.steps and len(rows)==args.steps,
            constructor_body_wall_seconds_total=sum(times),
            constructor_body_wall_seconds_median=statistics.median(times) if times else None,
            constructor_body_wall_seconds_min=min(times) if times else None,
            constructor_body_wall_seconds_max=max(times) if times else None,
            pre_constructor_cuda_sync_seconds_total=sum(syncs),
            parameter_snapshot_bytes_copied_total=sum(r['parameter_snapshot']['bytes'] for r in complete),
            adam_snapshot_bytes_copied_total=sum(r['adam_snapshot']['bytes'] for r in complete),
            parameter_snapshot_bytes_per_call=[r['parameter_snapshot']['bytes'] for r in complete],
            adam_snapshot_bytes_per_call=[r['adam_snapshot']['bytes'] for r in complete],
            snapshot_tensor_copies_all_cpu=bool(complete) and all(
                r[name]['tensors']==r[name]['cpu_tensors'] for r in complete
                for name in ('parameter_snapshot','adam_snapshot')),
            calls=rows, original_snapshot_implementation=implementation,
            wrapper_file=str(Path(__file__).resolve()),
            original_constructor_and_clone_bodies_unchanged=True,
            production_modified=False, fresh_rollout=False,
            actual_exception_rollback_timed=False, snapshot_release_timed=False,
            four_rank_sum_is_wall_time=False, full_outer_contribution_proven=False,
            scope='Original rank-local parameter/Adam/RNG snapshot constructor wall in the legacy fixed-work native4 probe.',
            limitations=['Prior CUDA synchronization is separately measured and excluded from constructor wall.',
                        'Original blocking GPU-to-CPU copies remain unchanged; metadata and progress are outside the timer.',
                        'Per-rank totals must not be added and called parallel wall time.',
                        'Does not measure an exception rollback, snapshot release, or full production outer contribution.'])
        output.mkdir(parents=True, exist_ok=True)
        path = output/('snapshot-timing-rank-%d.json' % rank)
        # The parent owns a unique diagnostic output directory and submission.
        with path.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush()
        print('ROLLBACK_SNAPSHOT_TIMING='+json.dumps(dict(rank=rank,
            phase='SNAPSHOT_TIMING_SAVED', path=str(path), measurement_complete=report['measurement_complete'],
            completed_constructors=len(complete), constructor_body_wall_seconds_total=sum(times)),
            sort_keys=True), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-base', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('legacy',), default='legacy')
    parser.add_argument('--steps', type=int, choices=(20,), default=20)
    parser.add_argument('--selection-step', type=int, choices=(51,), default=51)
    args = parser.parse_args(argv)
    report = run(args)
    return 0 if report['measurement_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
