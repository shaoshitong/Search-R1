"""One read-only JSON/log snapshot of the lean continuation, with no Slurm calls.

This observer never imports torch, reads tensors, contacts a service, starts a
watcher, or changes production. Missing producer evidence remains unknown.
The restore outer is taken from the actual target manifest, never a fixed CP34.
"""
import argparse
import json
from pathlib import Path
import re
import time

from reproduction.operations.easy7b_20261021.observe_fa3_transition_hpc import private_progress_reader

BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-lean-r1-20261022'
ERROR = re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|'
                   r'RuntimeError:|AssertionError:|ValueError:|NCCL.*(?:error|failed)', re.I)
ANSI = re.compile(r'\x1b\[[0-9;]*m')
MAX_JSON_BYTES = 32 * 1024**2


def observe(base=BASE, runtime_commit=None):
    base = Path(base)
    reader = private_progress_reader()
    errors = []

    def document(path):
        path = Path(path)
        if not path.is_file():
            return dict(path=str(path), present=False)
        try:
            if path.suffix != '.json' or path.stat().st_size > MAX_JSON_BYTES:
                return dict(path=str(path), present=True, observation_error='JSON read scope/size limit')
            value = reader.read_json(path)
            if not isinstance(value, dict):
                value = dict(observation_error='Expected a JSON object')
            if 'observation_error' in value:
                errors.append(dict(path=str(path), error=value['observation_error']))
            return dict(path=str(path), present=True, value=value)
        except OSError as error:
            errors.append(dict(path=str(path), error=str(error)))
            return dict(path=str(path), present=False, observation_error=str(error))

    def ledger(name):
        rows, issues = reader.complete_jsonl(base/'logs'/name)
        errors.extend(issues)
        return rows

    run_entry = document(base/'logs/run-manifest.json')
    run = run_entry.get('value', {})
    binding = run.get('full_state_restore') or {}
    source_step = binding.get('completed_outer')
    after_source = lambda row: (type(source_step) is int and
        type(row.get('logical_update', row.get('step'))) is int and
        row.get('logical_update', row.get('step')) > source_step)
    try:
        job = (base/'job.id').read_text().strip()
    except OSError:
        job = None
    result = dict(epoch=time.time(), base=str(base), job=job,
        requested_runtime_commit=runtime_commit, actual_runtime_commit=run.get('source_commit'),
        runtime_commit_matches=(run.get('source_commit') == runtime_commit) if runtime_commit else None,
        source_job=binding.get('source_job', '747219'), source_completed_outer=source_step,
        source_checkpoint=binding.get('source_checkpoint'),
        read_only=True, slurm_commands_called=False, network_called=False,
        tensor_files_read=False, model_loaded=False, training_modified=False,
        observer_is_training_gate=False, automatic_retry=False,
        run_identity={key: run.get(key) for key in ('run_id', 'source_commit', 'training_world_size',
            'actor_micro_batch_per_rank', 'actor_logical_micro_batch_per_rank',
            'performance_profile', 'extra_policy_monitor_forwards', 'infra_optimizations')},
        restore_binding=binding)
    docs = {}
    for name in ('logs/pipeline-status.json', 'logs/full-state-restored-workers.json',
                 'logs/full-state-restored-driver.json', 'logs/easy2560-restored-stream-before-next-outer.json',
                 'preparation/rolling-checkpoints/latest.json', 'preparation/lean-transition-retrieval-exit.json'):
        docs[name] = document(base/name)
    ranks = {str(rank): document(base/('logs/full-state-restored-rank-%d.json' % rank)) for rank in range(4)}
    result['documents'], result['actual_restore_ranks'] = docs, ranks
    aggregate = docs['logs/full-state-restored-workers.json'].get('value', {})
    driver = docs['logs/full-state-restored-driver.json'].get('value', {})
    stream = docs['logs/easy2560-restored-stream-before-next-outer.json'].get('value', {})
    actual_rows = [ranks[str(rank)].get('value', {}) for rank in range(4)]
    aggregate_ranks = {row.get('rank'): row for row in aggregate.get('ranks', []) if isinstance(row, dict)}
    expected_writes = binding.get('actual_optimizer_writes_per_rank')
    workers_equal = bool(type(source_step) is int and len(aggregate.get('ranks', [])) == 4 and
        all(row.get('rank') == rank and row.get('source_step') == source_step and
            row.get('run_id') == run.get('run_id') and row.get('slurm_job_id') == job and
            row.get('source_checkpoint') == binding.get('source_checkpoint') and
            row.get('actual_optimizer_writes') == expected_writes and
            row.get('scheduler_last_epoch') == source_step and
            all(row.get(key) is True for key in ('actual_model_collective_readback_equal',
                'actual_adam_scheduler_readback_equal', 'actual_rng_readback_equal')) and
            row == aggregate_ranks.get(rank) for rank, row in enumerate(actual_rows)) and
        len({row.get('actual_cuda_uuid') for row in actual_rows}) == 4 and
        all(row.get('actual_cuda_uuid') for row in actual_rows))
    cursor = binding.get('candidate_source_cursor')
    driver_equal = bool(type(source_step) is int and driver.get('completed_outer') == source_step and
        driver.get('source_step') == source_step and driver.get('source_job') == binding.get('source_job') and
        driver.get('actual_driver_dataequal') is True and
        driver.get('actual_driver_rng_readback_equal') is True and
        driver.get('actual_candidate_stream_readback_equal') is True)
    stream_equal = bool(type(source_step) is int and stream.get('completed_outer') == source_step and
        stream.get('actual_candidate_stream_readback_equal') is True and
        stream.get('actual_driver_rng_readback_equal') is True and
        driver.get('actual_candidate_source_cursor') == stream.get('candidate_source_cursor') and
        (cursor is None or cursor == stream.get('candidate_source_cursor')))
    result['restoration_observed'] = dict(all4_worker_model_adam_scheduler_rng_readback_equal=workers_equal,
        driver_readback_equal=driver_equal, reconstructed_stream_readback_equal=stream_equal,
        combined_actual_restore_observed=workers_equal and driver_equal and stream_equal,
        actual_worker_cuda_uuids=[row.get('actual_cuda_uuid') for row in actual_rows],
        rank0_model_tensor_readback_count=actual_rows[0].get('model_hf_exact_tensors_checked_on_rank0'),
        expected_source_writes_per_rank=expected_writes,
        actual_driver_cursor=driver.get('actual_candidate_source_cursor'),
        actual_stream_cursor=stream.get('candidate_source_cursor'))

    updates = [row for row in ledger('actual-updates.jsonl') if after_source(row)]
    attempts = [row for row in ledger('actual-attempts.jsonl') if after_source(row)]
    complete = [row for row in updates if row.get('update_status') == 'complete']
    keys = ('step', 'logical_update', 'run_id', 'slurm_job_id', 'attempt_id', 'attempt_index',
            'update_status', 'status', 'full_budget_complete', 'complete_outer', 'all_ranks_agreed',
            'real_rank_count', 'actual_optimizer_writes', 'optimizer_writes_before', 'optimizer_writes_after',
            'optimizer_writes_this_attempt', 'optimizer_calls_this_attempt', 'scheduler_advances',
            'opportunities_processed', 'opportunities_unattempted', 'no_automatic_rollback')
    project = lambda rows: [{key: row[key] for key in keys if key in row} for row in rows]
    result['actual_update_records'], result['actual_attempt_records'] = project(updates), project(attempts)
    result['latest_completed_outer'] = max((row['step'] for row in complete), default=source_step)
    result['post_restore_completed_outer_count'] = len(complete)
    result['optimizer_ranks'], result['scheduler_ranks'], result['actor_flash3_ranks'] = {}, {}, {}
    for rank in range(4):
        rows = ledger('optimizer-rank-%d.jsonl' % rank)
        writes = [row for row in rows if row.get('kind') == 'OPTIMIZER_WRITE' and
                  after_source(row.get('context', {}))]
        budgets = [row for row in rows if row.get('kind') == 'OUTER_BUDGET' and after_source(row)]
        summaries = []
        for row in budgets:
            outer = row['logical_update']
            actual_writes = [value.get('write_count') for value in writes
                             if value.get('context', {}).get('logical_update') == outer]
            value = {key: row[key] for key in ('rank', 'logical_update', 'optimizer_writes', 'cumulative_writes',
                'full_budget_complete', 'opportunities_processed', 'local_micro', 'local_logical_micro',
                'local_mini', 'gradient_accumulation', 'actual_monitor_forward_count',
                'reused_monitor_forward_count', 'extra_policy_monitor_forwards') if key in row}
            value['actual_write_counts'] = actual_writes
            value['observed_twenty_writes'] = (row.get('optimizer_writes') == 20 and len(actual_writes) == 20 and
                all(type(count) is int for count in actual_writes) and
                actual_writes == list(range(actual_writes[0], actual_writes[0] + 20)))
            value['monitor_counter_evidence_available'] = all(key in row for key in
                ('actual_monitor_forward_count', 'reused_monitor_forward_count', 'extra_policy_monitor_forwards'))
            value['observed_zero_extra_monitor_forwards'] = ((row.get('actual_monitor_forward_count') == 0 and
                row.get('reused_monitor_forward_count') == 0 and row.get('extra_policy_monitor_forwards') is False)
                if value['monitor_counter_evidence_available'] else None)
            summaries.append(value)
        result['optimizer_ranks'][str(rank)] = dict(completed_outer_budgets=summaries,
            latest_actual_write=writes[-1] if writes else None,
            zero_monitor_forward_evidence_available=bool(summaries) and all(
                row['monitor_counter_evidence_available'] for row in summaries),
            all_observed_completed_budgets_twenty_writes=all(row['observed_twenty_writes'] for row in summaries) if summaries else None,
            all_observed_completed_budgets_zero_monitor_forwards=all(row['observed_zero_extra_monitor_forwards'] for row in summaries)
                if summaries and all(row['monitor_counter_evidence_available'] for row in summaries) else None)
        schedule = [row for row in ledger('scheduler-rank-%d.jsonl' % rank) if after_source(row)]
        schedule_keys = keys + ('kind', 'writes_before', 'writes_after', 'scheduler_last_epoch_after')
        result['scheduler_ranks'][str(rank)] = [{key: row[key] for key in schedule_keys if key in row} for row in schedule]
        flash = ledger('actual-flash3-execution-rank-%d.jsonl' % rank)
        result['actor_flash3_ranks'][str(rank)] = dict(complete_records=len(flash), latest=flash[-1] if flash else None)

    # The small selection JSON already stores actual phase-close counters. Do
    # not read raw rollout .pt files or infer FA3 execution from a profile flag.
    audit_paths = sorted((base/'artifacts/mixed-selection').glob('*.json'),
                         key=lambda path: path.stat().st_mtime)[-2:]
    result['candidate_phase_observations'] = []
    for path in audit_paths:
        entry = document(path); audit = entry.get('value', {})
        phases = audit.get('candidate_generation_phases', [])
        item = dict(path=str(path), step=audit.get('step'), pool_retry=audit.get('pool_retry'),
                    phase_count=len(phases), rank_order='phase_metrics list order from native worker group',
                    ranks={})
        for rank in range(4):
            receipts = [phase.get('phase_metrics', [])[rank] for phase in phases
                        if isinstance(phase.get('phase_metrics'), list) and len(phase['phase_metrics']) == 4]
            syncs = [row.get('weight_syncs') for row in receipts]
            actual_counts = [row.get('flash3_execution', {}).get('calls') for row in receipts]
            available = [row for row in actual_counts if isinstance(row, dict)]
            item['ranks'][str(rank)] = dict(phase_weight_syncs=syncs,
                total_weight_syncs=sum(syncs) if phases and len(syncs) == len(phases) and all(type(v) is int for v in syncs) else None,
                five_phases_one_sync_observed=(len(phases) == 5 and syncs == [1, 0, 0, 0, 0]),
                latest_cumulative_rollout_fa3_calls=available[-1] if available else None,
                rollout_counter_scope='cumulative actual adapter successful returns; not summed across phases',
                rollout_counter_location='candidate_generation_phases[].phase_metrics[].flash3_execution.calls')
        item['rollout_fa3_counter_available'] = any(
            row['latest_cumulative_rollout_fa3_calls'] is not None for row in item['ranks'].values())
        if not item['rollout_fa3_counter_available']:
            item['missing_counter_note'] = ('No persisted rollout FA3 call counter observed. The runtime counter is '
                'vLLMRollout._search_g1_flash3_functions.calls; this observer does not query or modify the worker.')
        result['candidate_phase_observations'].append(item)

    log_paths = dict(training=base/'logs/search-g1-common8-training.log', trainer=base/'logs/trainer.log',
                     pipeline=base/'logs/pipeline.log')
    if job and job.isdecimal():
        log_paths['slurm'] = base/('logs/training-%s.log' % job)
    result['log_tails'] = {name: reader.tail(path, 32000) for name, path in log_paths.items()}
    result['actual_error_lines'] = {name: [ANSI.sub('', line)[-1800:] for line in value.get('tail', '').splitlines()
        if ERROR.search(line)][-10:] for name, value in result['log_tails'].items()}
    result['observation_errors'] = errors
    result['goal200_complete'] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--runtime-commit', help='Expected full target runtime SHA; difference is reported only')
    args = parser.parse_args(argv)
    print(json.dumps(observe(args.base, args.runtime_commit), sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
