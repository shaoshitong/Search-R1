"""Record one completed ordinary outer from saved local observer JSON only.

Example (add --dry-run to write nothing):
  python3 -B record_completed_combined_outer_local.py --output-dir OUTPUT_DIR \
    --evidence OUTPUT_DIR/combined-batch-search-stop-transition-live-NS.json --step 93

No host contact, tensor reads, training gates, locks, or background processes.
The final outer, which must not start outer201, uses the separate final acceptance
operator rather than this ordinary-outer recorder's next-candidate requirement.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


def local_path(value, output_dir):
    path = Path(value).expanduser()
    if path.is_absolute() or path.is_file():
        return path.resolve()
    return (output_dir / path.name).resolve()


def load_summarizer():
    path = Path(__file__).with_name('summarize_combined_production_timing.py')
    spec = importlib.util.spec_from_file_location('combined_local_timing', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_records(output_dir, evidence_path, step, health_path=None, archive_path=None):
    timing_module = load_summarizer()
    read = timing_module.read
    state_path = output_dir / 'fastpath-current-state.json'
    state = read(state_path)
    evidence = read(evidence_path)
    require(0 < step < state['total_outer'], 'Use the separate final acceptance operator for the final outer')
    require(str(evidence.get('job')) == str(state['production_job']) and
            evidence.get('base') == state['production_base'], 'Evidence is for another production run')
    require(evidence.get('actual_runtime_commit') == state['runtime_commit'], 'Runtime differs from canonical state')
    complete = next((row for row in evidence.get('actual_update_records', [])
                     if row.get('step') == step and row.get('update_status') == 'complete'
                     and row.get('full_budget_complete')), None)
    require(complete is not None, f'No actual complete update for outer{step}')
    seal = (evidence.get('documents') or {}).get(
        'preparation/rolling-checkpoints/latest.json', {}).get('value') or evidence.get('rolling_checkpoint')
    require(seal and seal.get('step') == step and
            seal.get('status') == 'EASY2560_ROLLING_CHECKPOINT_READY', f'No ready seal for outer{step}')
    next_candidate = max((row for row in evidence.get('candidate_progress', [])
                          if row.get('step') == step + 1),
                         key=lambda row: row.get('complete_questions', 0), default=None)
    require(next_candidate is not None and next_candidate.get('complete_questions', 0) >= 512,
            f'Outer{step + 1} has not saved at least512 candidate questions')

    evidence_map = deepcopy(state.get('completed_outer_evidence', {}))
    evidence_map[str(step)] = str(evidence_path)
    paths = list(dict.fromkeys(str(local_path(value, output_dir)) for key, value in
                              sorted(evidence_map.items(), key=lambda item: int(item[0])) if int(key) <= step))
    new = timing_module.summarize(paths, step)
    require(not new.get('read_errors'), 'Saved timing inputs could not all be read: ' + str(new.get('read_errors')))
    through_rows = [row for row in new['rows'] if row['step'] <= step]
    row = next((row for row in through_rows if row['step'] == step), None)
    require(row is not None, f'No locally summarized complete outer{step}')
    budgets = row['optimizer_by_rank']
    require(set(budgets) == {'0', '1', '2', '3'} and all(
        value.get('optimizer_writes') == 20 and value.get('actual_monitor_forward_count') == 0
        and value.get('full_budget_complete') for value in budgets.values()),
        f'Outer{step} lacks four actual20-write/zero-extra-monitor budgets')
    require(isinstance(row.get('full_outer_seconds'), (int, float)) and row['full_outer_seconds'] > 0,
            f'Outer{step} lacks adjacent actual before-generation timestamps')
    new['rows'] = through_rows[-5:]
    new['aggregate'] = {key: timing_module.mean(new['rows'], key) for key in
                        ('full_outer_seconds', 'candidate_fill_seconds', 'actor_update_seconds',
                         'advantage_seconds', 'checkpoint_write_seconds')}
    timing = read(local_path(state['latest_production_timing'], output_dir))
    timing['new'] = new
    baseline_mean = timing['fixed_baseline']['whole_outer_seconds_mean']
    new_mean = new['aggregate']['full_outer_seconds']['mean']
    timing['observational_comparison'].update(
        baseline_over_new_wall_ratio=baseline_mean / new_mean if new_mean else None,
        new_whole_wall_observations=new['aggregate']['full_outer_seconds']['observed_count'],
        same_input_causality_proven=False)

    post_restore = step - state['source_completed_outer']
    observation = dict(observed_unix=evidence['epoch'], evidence=str(evidence_path), latest_completed_outer=step,
        post_restore_completed_outers=post_restore, rolling_step=step, candidate_step=step + 1,
        saved_questions=next_candidate['complete_questions'], actual_errors=evidence.get('actual_error_lines'),
        observation_errors=evidence.get('observation_errors'))
    previous = read(local_path(state['latest_continuation_evidence'], output_dir))
    continuation = dict(schema=f'combined_continuation{step}_complete_next{step + 1}_v1', **observation,
        job=evidence['job'], runtime_commit=evidence['actual_runtime_commit'], checkpoint_seal=seal,
        next_candidate=next_candidate, latest_outer_timing=row, target_outer=state['total_outer'],
        update201_allowed=state['update201_allowed'],
        fixed100_pending=state.get('fixed100_pending', previous.get('fixed100_pending', True)),
        final_native51713_seven_metrics_pending=state.get('final_native51713_seven_metrics_pending',
            previous.get('final_native51713_seven_metrics_pending', True)),
        final_native4_actual_recovery_pending=state.get('final_native4_actual_recovery_pending',
            previous.get('final_native4_actual_recovery_pending', previous.get('actual_final_native4_recovery_pending', True))),
        new_jobs_submitted=0, training_modified=False, goal_complete=False)

    health = None
    if health_path:
        health_record = read(health_path)
        service = health_record['health']
        health = dict(evidence=str(health_path), observed_unix=health_record['observed_unix'],
            service_job=service['service_job_id'], completed_queries=service['statistics']['completed_queries'],
            failed_requests=service['statistics']['failed_requests'],
            gpu_forward_triggered=health_record['gpu_forward_triggered'])
        continuation['retrieval_latest_health_observation'] = health
    if archive_path:
        read(archive_path)  # Local parsing only; the original archive evidence stays intact.
        continuation['archive_consumer_latest_observation'] = str(archive_path)

    timing_path = output_dir / f'combined-production-timing-through{step}.json'
    continuation_path = output_dir / f'combined-continuation{step}-complete-next{step + 1}.json'
    old_step = state['latest_completed_outer']
    advance = step > old_step or (step == old_step and evidence['epoch'] >= state.get('observed_unix', 0))
    state['completed_outer_evidence'] = evidence_map
    if advance:
        latest_writes = {rank: value.get('latest_actual_write', {}).get('write_count',
                         budgets[rank]['cumulative_writes']) for rank, value in evidence['optimizer_ranks'].items()
                         if rank in budgets}
        state.update(observed_unix=evidence['epoch'], latest_completed_outer=step,
            new_target_completed_outer_count=post_restore, current_outer=step + 1,
            latest_candidate_progress=next_candidate, saved_current_candidate_questions=next_candidate['complete_questions'],
            production_latest_sealed_checkpoint=seal, actual_adam_writes_per_rank=latest_writes,
            latest_target_evidence=str(evidence_path), latest_continuation_observation=observation,
            latest_continuation_evidence=str(continuation_path), latest_production_timing=str(timing_path),
            combined_full_outer_measurements=sum(row.get('full_outer_seconds') is not None for row in through_rows),
            latest_training_metric=dict(outer=step, training_em_rounded=row.get('training_em_rounded'),
                reward_mean_rounded=row.get('training_reward_mean_rounded'),
                scope='selected512x5 pre-update rollouts, not fixed heldout'))
        actual_errors = {key: value for key, value in (evidence.get('actual_error_lines') or {}).items() if value}
        state['latest_error'] = actual_errors or None
        accounting = next((line.split('|') for line in evidence.get('accounting', '').splitlines()
                           if line.startswith(str(evidence['job']) + '|')), [])
        if len(accounting) > 2:
            state['production_job_state'] = accounting[2]
        if health and health['observed_unix'] >= (state.get('retrieval_latest_health_observation') or {}).get('observed_unix', 0):
            state['retrieval_latest_health_observation'] = health
        if archive_path:
            state['archive_consumer_latest_observation'] = str(archive_path)
        state['previous_goal_turn_classification'] = (
            f'PROGRESS: actual{step} full20writes/rank and native4 seal;{step + 1} '
            f'saved{next_candidate["complete_questions"]}. Actual whole{row["full_outer_seconds"]:.4f}s;'
            f'{post_restore} post-restore full outers. Training unchanged; goal200/no201/'
            'fixed validation/native51713/archive/actual final native4 recovery remain subject to final acceptance.')
    else:
        state['previous_goal_turn_classification'] = (
            f'PROGRESS: late actual outer{step} complete/seal/next-candidate evidence recorded locally; '
            f'current completed outer{old_step} and latest observations retained. Training unchanged.')
    return {timing_path: timing, continuation_path: continuation, state_path: state}, continuation, advance


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--health-evidence', type=Path)
    parser.add_argument('--archive-evidence', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Read and summarize only; write no files')
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        files, record, advance = build_records(output_dir, local_path(args.evidence, output_dir), args.step,
            local_path(args.health_evidence, output_dir) if args.health_evidence else None,
            local_path(args.archive_evidence, output_dir) if args.archive_evidence else None)
        if not args.dry_run:
            for path, value in files.items():
                atomic_json(path, value)
        print(json.dumps(dict(status='DRY_RUN_PASS' if args.dry_run else 'LOCAL_RECORD_SAVED',
            step=args.step, next_candidate_questions=record['saved_questions'], state_latest_fields_updated=advance,
            canonical_completed_outer=files[output_dir / 'fastpath-current-state.json']['latest_completed_outer'],
            full_outer_seconds=record['latest_outer_timing']['full_outer_seconds'],
            training_em_rounded=record['latest_outer_timing']['training_em_rounded'],
            reward_mean_rounded=record['latest_outer_timing']['training_reward_mean_rounded'],
            checkpoint_manifest_sha256=record['checkpoint_seal']['manifest_sha256'],
            files=[str(path) for path in files], files_written=0 if args.dry_run else len(files),
            remote_operations=False, tensor_files_read=False, training_modified=False), ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps(dict(status='LOCAL_RECORD_FAILED', error=str(error), remote_operations=False,
                              training_modified=False), ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
