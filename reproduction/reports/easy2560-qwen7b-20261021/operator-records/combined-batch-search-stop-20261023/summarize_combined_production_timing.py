"""Summarize saved observer JSON locally; never contacts a host or loads tensors.

Example: python3 -B summarize_combined_production_timing.py --new NEW.json
    --old OLD.json --baseline pre-combined-native4-baseline-latest-five.json
All output is JSON on stdout. Missing boundaries or phases remain null.
"""
import argparse
import json
from pathlib import Path
import re


def read(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value.get('stdout'), str):
        if value.get('returncode', 0):
            raise ValueError('Saved observation failed: ' + str(value.get('stderr', ''))[-500:])
        value = json.loads(value['stdout'])
    return value


def mean(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return dict(mean=sum(values) / len(values) if values else None,
                observed_count=len(values), completed_outer_count=len(rows))


def compact_actor(record):
    geometry = set()
    for shape in record.get('observed_shapes', []):
        if shape.get('q') and shape.get('k'):
            geometry.add((shape.get('dtype'), shape['q'][-2], shape['k'][-2], shape['q'][-1]))
    return dict(actual_flash3_training_executed=record.get('actual_flash3_training_executed'),
        compact_gqa_enabled=record.get('compact_gqa_enabled'),
        cumulative_forward_returns=record.get('forward_success_returns'),
        cumulative_backward_returns=record.get('kernel_backward_success_returns'),
        cumulative_qkv_gradient_returns=record.get('qkv_gradient_returns'),
        observed_qkv_geometry=[dict(dtype=d, q_heads=q, kv_heads=k, head_dim=h)
                               for d, q, k, h in sorted(geometry)])


def summarize(paths, count, baseline=None):
    snapshots, errors = [], []
    for path in paths:
        try:
            snapshots.append((read(path), str(path)))
        except (OSError, ValueError, TypeError) as error:
            errors.append(dict(file=str(path), error=str(error)))
    snapshots.sort(key=lambda item: item[0].get('epoch') or 0)
    if not snapshots:
        return dict(rows=[], read_errors=errors, status='NO_READABLE_SNAPSHOT')
    latest, latest_file = snapshots[-1]
    identity = latest.get('base')
    boundaries, complete, selections, metrics = {}, {}, {}, {}
    budgets, scheduler, actor, phase, stop, seals = {}, {}, {}, {}, {}, {}
    baseline_rows = baseline.get('per_outer', []) if baseline and baseline.get('job') == latest.get('job') else []
    for row in baseline_rows:
        step = row['step']
        boundaries[step] = (row['start_before_generation_mtime'], 'saved_baseline')
        boundaries[step + 1] = (row['next_before_generation_mtime'], 'saved_baseline')
        complete[step] = row.get('full_update_record')
        metrics[step] = dict(actor_update_seconds=row.get('actor_update_seconds'),
            advantage_seconds=row.get('advantage_seconds'), checkpoint_write_seconds=row.get('checkpoint_write_seconds'),
            producer_post_candidate_step_seconds=row.get('producer_post_candidate_step_seconds'),
            training_em_rounded=row.get('training_em_rounded'),
            training_reward_mean_rounded=row.get('training_reward_mean_rounded'), source='saved_baseline')
        selections[step] = dict(candidate_fill_seconds=row.get('candidate_fill_seconds'),
            selected_mixed=row.get('selected_mixed_questions'),
            raw_generated_policy_tokens=row.get('producer_raw_generated_policy_tokens'),
            raw_generated_trajectories=row.get('candidate_trajectories'),
            raw_actual_retrieval_calls=row.get('actual_retrieval_calls'))
        captured = row.get('captured_checkpoint_seal') or {}
        if captured.get('value'):
            seals[step] = captured['value']
    for data, filename in snapshots:
        if data.get('base') != identity:
            errors.append(dict(file=filename, error='Different run base; not merged'))
            continue
        for row in data.get('candidate_progress') or []:
            if row.get('step') is not None and isinstance(row.get('first_before_generation'), (int, float)):
                boundaries[row['step']] = (row['first_before_generation'], filename)
        for row in data.get('actual_update_records') or []:
            if row.get('update_status') == 'complete' and row.get('full_budget_complete'):
                complete[row['step']] = row
        for row in data.get('selection_summary') or []:
            selections[row['step']] = row
        for rank, value in (data.get('optimizer_ranks') or {}).items():
            for row in value.get('completed_outer_budgets', []):
                budgets.setdefault(row['logical_update'], {})[rank] = row
        for rank, records in (data.get('scheduler_ranks') or {}).items():
            for row in records:
                if row.get('kind') == 'COMPLETE':
                    scheduler.setdefault(row['logical_update'], {})[rank] = row
        for rank, value in (data.get('actor_flash3_ranks') or {}).items():
            record = value.get('latest')
            if record:
                actor.setdefault(record['logical_update'], {})[rank] = compact_actor(record)
        for row in data.get('candidate_phase_observations') or []:
            phase[row['step']] = row
        for row in data.get('search_stop_phase_observations') or []:
            stop[row['step']] = row
        receipt = (data.get('documents') or {}).get('preparation/rolling-checkpoints/latest.json', {}).get('value')
        receipt = receipt or data.get('rolling_checkpoint')
        if receipt:
            seals[receipt['step']] = receipt
        for log in (data.get('log_tails') or {}).values():
            if not isinstance(log, dict):
                continue
            for match in re.finditer(r'\bstep:(\d+)\s*-\s*([^\n]+)', log.get('tail', '')):
                values = {}
                for entry in match.group(2).split(' - '):
                    if ':' in entry:
                        key, value = entry.split(':', 1)
                        try:
                            values[key.strip()] = float(value)
                        except ValueError:
                            pass
                if 'timing_s/update_actor' in values:
                    metrics[int(match.group(1))] = dict(
                        actor_update_seconds=values.get('timing_s/update_actor'),
                        advantage_seconds=values.get('timing_s/adv'),
                        checkpoint_write_seconds=values.get('timing_s/save_checkpoint'),
                        producer_post_candidate_step_seconds=values.get('timing_s/step'),
                        training_em_rounded=values.get('train/perf/em'),
                        training_reward_mean_rounded=values.get('critic/rewards/mean'), source=filename)
    # The compact watcher may omit actual_update_records; four completed native
    # rank budgets can still establish the update without inventing a wall time.
    for step, ranks in budgets.items():
        if set(ranks) == {'0', '1', '2', '3'} and all(
                row.get('full_budget_complete') and row.get('optimizer_writes') == 20 for row in ranks.values()):
            complete.setdefault(step, dict(status='all4_actual_completed_budgets'))
    rows = []
    for step in sorted(complete)[-count:]:
        start, start_file = boundaries.get(step, (None, None))
        following, following_file = boundaries.get(step + 1, (None, None))
        selected = selections.get(step, {})
        phase_ranks = (phase.get(step) or {}).get('ranks', {})
        stop_ranks = (stop.get(step) or {}).get('ranks', {})
        rollout = {}
        for rank in sorted(set(phase_ranks) | set(stop_ranks)):
            left, right = phase_ranks.get(rank, {}), stop_ranks.get(rank, {})
            calls = left.get('latest_cumulative_rollout_fa3_calls') or (right.get('flash3_execution') or {}).get('calls')
            rollout[rank] = dict(
                phase_weight_syncs=left.get('phase_weight_syncs'), total_weight_syncs=left.get('total_weight_syncs'),
                cumulative_fa3_calls=calls, cumulative_search_action_stop=right.get('search_action_stop'),
                actual_work_latest_phase_only=right.get('actual_work'))
        row = dict(step=step, full_outer_seconds=following-start if start is not None and following is not None else None,
            before_generation_mtime=start, next_before_generation_mtime=following,
            boundary_sources=[start_file, following_file], checkpoint_seal=seals.get(step),
            candidate_fill_seconds=selected.get('candidate_fill_seconds'),
            retained_policy_tokens=selected.get('raw_generated_policy_tokens'),
            retained_token_source='selection_summary.raw_generated_policy_tokens (producer name; retained, not untruncated raw)',
            actual_raw_generated_tokens_full_outer=selected.get('actual_raw_generated_tokens'),
            raw_token_scope='Null unless an explicit full-outer actual_raw_generated_tokens counter exists. '
                'Latest-phase work/stop counters cannot reconstruct all five phases or untruncated raw output.',
            candidate_trajectories=selected.get('raw_generated_trajectories'),
            actual_retrieval_calls=selected.get('raw_actual_retrieval_calls'),
            selected_mixed_questions=selected.get('selected_mixed'), selected_questions=512,
            optimizer_by_rank={rank:{key:value.get(key) for key in ('optimizer_writes','cumulative_writes',
                'actual_monitor_forward_count','full_budget_complete')} for rank,value in budgets.get(step,{}).items()},
            scheduler_by_rank={rank:{key:value.get(key) for key in ('scheduler_advances','writes_after',
                'scheduler_last_epoch_after')} for rank,value in scheduler.get(step,{}).items()},
            actor_fa3_by_rank=actor.get(step), rollout_by_rank=rollout,
            rollout_counter_scope='Latest completed phase records contain cumulative engine counters, not per-outer deltas; '
                'phase_weight_syncs is the explicit full phase sequence. Never sum cumulative phases or rank wall times.')
        row.update(metrics.get(step, dict(actor_update_seconds=None, advantage_seconds=None,
            checkpoint_write_seconds=None, producer_post_candidate_step_seconds=None,
            training_em_rounded=None, training_reward_mean_rounded=None, source=None)))
        rows.append(row)
    return dict(job=latest.get('job'), base=identity, runtime_commit=latest.get('actual_runtime_commit'),
        latest_observed_unix=latest.get('epoch'), latest_observation=latest_file,
        latest_completed_outer=latest.get('latest_completed_outer'), rows=rows,
        aggregate={key:mean(rows,key) for key in ('full_outer_seconds','candidate_fill_seconds',
            'actor_update_seconds','advantage_seconds','checkpoint_write_seconds')},
        actual_errors=latest.get('actual_error_lines'), observation_errors=latest.get('observation_errors'),
        read_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new', nargs='+', required=True, help='Saved new-target observer JSON(s)')
    parser.add_argument('--old', nargs='+', required=True, help='Saved old-target observer JSON(s)')
    parser.add_argument('--baseline', required=True, help='Saved baseline65–69 summary JSON')
    parser.add_argument('--last', type=int, default=5)
    args = parser.parse_args()
    baseline = read(args.baseline)
    old, new = summarize(args.old, args.last, baseline), summarize(args.new, args.last)
    baseline_mean = baseline['aggregate']['full_outer_seconds']['mean']
    new_mean = new.get('aggregate', {}).get('full_outer_seconds', {}).get('mean')
    result = dict(schema='combined_production_local_timing_summary_v1', old=old, new=new,
        fixed_baseline=dict(file=args.baseline, outers=baseline['complete_measured_outers'],
            whole_outer_seconds_mean=baseline_mean, phase_aggregates=baseline['aggregate']),
        observational_comparison=dict(baseline_over_new_wall_ratio=baseline_mean/new_mean if new_mean else None,
            new_whole_wall_observations=new.get('aggregate',{}).get('full_outer_seconds',{}).get('observed_count'),
            same_input_causality_proven=False),
        wall_scope='Only adjacent actual before-generation mtimes establish whole wall; includes checkpoint/seal/inter-outer '
            'work and excludes startup. No rank wall sum or candidate+step substitution. Missing next boundary means null.',
        phase_scope='Producer rounded log phases are preserved; checkpoint write is not all sealing time; '
            'segments need not sum to whole. Training EM/reward are not fixed-validation metrics.',
        counter_scope='FA3 and stop reports are actual cumulative counters per rank. Raw and retained token counts are '
            'kept distinct; null is not zero. Different policy/data/requests prevent strict speed causality.',
        remote_operations=False, tensor_files_read=False, training_modified=False, goal_complete=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
