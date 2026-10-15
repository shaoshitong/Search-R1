"""Validate three downloaded JSON summaries and publish local nested progress only.

No network/process/training actions, tensor libraries or remote paths are read.
Missing settling metadata emits PENDING without writing status or success output.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
INVOCATION = 'a313bda919834d24b4c5d491698d39ce'
PID = '503255'
JOB = 'not-slurm;systemd:' + SOURCE
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-' + JOB
RESTORE_SHA = 'cceb1cfe9263fcf7213a375585d97a06cf54ca179f11c0f6a03985b7c2c348d6'
RESTORE, TARGET = 5, 200
BASE_REL = 'outputs/tree-nested-branch-handoff-20261013'


class Observation(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def need(condition, message, status='PENDING_EVIDENCE_NO_WRITE'):
    if not condition:
        raise Observation(status, message)


def integer(value):
    return type(value) in (int, float) and math.isfinite(value) and value == int(value)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def held(path):
    path = Path(path).resolve()
    need(path.is_file(), 'Local input unavailable: ' + str(path))
    need(stat.S_ISREG(path.lstat().st_mode), 'Ordinary local JSON required', 'REJECTED_INPUT_NO_WRITE')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size <= 12 * 1024**2, 'Input exceeds compact summary limit', 'REJECTED_INPUT_NO_WRITE')
        raw = stream.read()
        after = os.fstat(stream.fileno())
    signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    need(signature(before) == signature(after) == signature(path.stat()), 'Input changed while held')
    value = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    need(isinstance(value, dict), 'JSON object required', 'REJECTED_INPUT_NO_WRITE')
    return value, dict(path=str(path), sha256=digest(raw), bytes=len(raw))


def state(value):
    if isinstance(value, dict):
        return value
    need(isinstance(value, str), 'Actual systemd observation missing')
    return dict(line.split('=', 1) for line in value.splitlines() if '=' in line)


def counts(value, label):
    need(isinstance(value, dict) and set(value) == {str(n) for n in range(8)}, label + ' requires all eight ranks')
    need(all(integer(n) and n >= 0 for n in value.values()), label + ' has invalid actual counts', 'REJECTED_INPUT_NO_WRITE')
    return {rank: int(n) for rank, n in value.items()}


def sha64(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_unit(poll, resource, outer):
    ps = state(poll.get('systemd'))
    rs = state(resource.get('training_systemd') or resource.get('systemd')
               or (resource.get('state') or {}).get('training_unit_observation'))
    need(ps.get('InvocationID', INVOCATION) == INVOCATION and rs.get('InvocationID') == INVOCATION,
         'Exact original training Invocation differs', 'REJECTED_INPUT_NO_WRITE')
    active = all(item.get('MainPID') == PID and item.get('ActiveState') == 'active'
                 and item.get('SubState') == 'running' for item in (ps, rs))
    if not active:
        need(outer == TARGET and all(item.get('MainPID') == '0' and item.get('ActiveState') in ('inactive', 'failed')
                                    for item in (ps, rs)),
             'Unit is not the observed same active training instance', 'OBSERVED_NONACTIVE_NO_WRITE')
    return ps, rs, active


def validate_restore(poll):
    need(poll.get('native_restore_step') == RESTORE and poll.get('native_continuation_requested') is True,
         'Current native5 continuation request differs', 'REJECTED_INPUT_NO_WRITE')
    loaded = poll.get('native_load_receipt') or {}
    driver = loaded.get('driver') or {}
    workers = loaded.get('workers') or []
    need(loaded.get('status') == 'COLD_NESTED_NATIVE5_STATE_LOADED'
         and loaded.get('receipt_sha256') == RESTORE_SHA
         and driver.get('completed_outer_updates') == 5 and driver.get('training_attempts') == 5
         and driver.get('sampler_cursor') == 2560 and driver.get('sampler_epoch') == 0
         and driver.get('actual_cold_driver_state_equal') is True
         and driver.get('actual_file_sha256_verified') is True
         and counts(driver.get('completed_optimizer_calls_per_rank'), 'Actual restore calls') == {str(r):100 for r in range(8)}
         and len(workers) == 8 and {w.get('rank') for w in workers} == set(range(8))
         and all(w.get('child_code_commit') == SOURCE and w.get('world_size') == 8
                 and w.get('scheduler_step') == 5 and w.get('actual_total_optimizer_calls') == 100
                 and w.get('actual_cold_native_state_equal') is True
                 and w.get('actual_file_sha256_verified') is True for w in workers),
         'Actual native5 eight-rank/driver restore receipt is unconfirmed')
    provenance = loaded.get('provenance') or {}
    need(provenance.get('child_config_sha256') == CONFIG and provenance.get('child_code_commit') == SOURCE,
         'Actual child source/config differs', 'REJECTED_INPUT_NO_WRITE')
    return loaded, provenance


def validate_metric(poll, outer):
    candidates = [m for m in poll.get('iteration_metrics', []) if m.get('step') == outer
                  and m.get('training/actual_outer_updates') == outer]
    if not candidates and outer == TARGET:
        need(not any(m.get('step') == outer for m in poll.get('iteration_metrics', [])),
             'A present metric200 has inconsistent actual credit fields', 'REJECTED_INPUT_NO_WRITE')
        # The real final validation may run before the complete step200 line is
        # printed. Native/credit/rank/final-phase checks below remain required.
        return None
    need(candidates, 'Full iteration metric has not settled for outer ' + str(outer))
    metric = candidates[-1]
    need(all(number(value) for value in metric.values()), 'Nonfinite complete metric', 'REJECTED_INPUT_NO_WRITE')
    for name in ('timing_s/step', 'timing_s/gen', 'timing_s/update_actor'):
        need(number(metric.get(name)) and metric[name] > 0, 'Full metric lacks actual ' + name)
    processed, skipped, writes = [metric.get(name) for name in
        ('actor/optimizer_opportunities_processed', 'actor/optimizer_opportunities_skipped', 'actor/optimizer_steps')]
    need(processed == 20 and number(writes) and 1 <= writes <= 20
         and number(skipped) and 0 <= skipped <= 19 and writes + skipped == 20,
         'A full credited outer requires twenty actual opportunities and at least one actual write')
    need(integer(metric.get('training/attempt_id')) and metric['training/attempt_id'] >= outer,
         'Actual attempt ID missing')
    return metric


def validate_ranks(poll, metric, outer):
    rows = [r for r in poll.get('last_rank_summaries', []) if r.get('outer_step') == outer]
    need(len(rows) == 8 and {r.get('rank') for r in rows} == set(range(8)), 'Actual eight-rank outer update evidence pending')
    result = []
    for row in sorted(rows, key=lambda r:r['rank']):
        calls = row.get('successful_optimizer_calls')
        attempt_matches = (row.get('attempt_id') == metric['training/attempt_id'] if metric is not None
                           else integer(row.get('attempt_id')) and row['attempt_id'] >= outer)
        need(integer(calls) and 1 <= calls <= 20 and attempt_matches
             and row.get('finite_grad_norms') == calls and integer(row.get('nonzero_grad_norms'))
             and 1 <= row['nonzero_grad_norms'] <= calls, 'Actual rank write/finite/nonzero evidence incomplete')
        sample = row.get('local_shard_sample_delta') or {}
        need(integer(sample.get('count')) and sample['count'] > 0 and sample.get('finite') == sample['count']
             and integer(sample.get('nonzero')) and 0 < sample['nonzero'] <= sample['count']
             and number(sample.get('minimum')) and number(sample.get('maximum'))
             and sample['minimum'] <= sample['maximum']
             and (sample['minimum'] != 0 or sample['maximum'] != 0), 'Actual finite/nonzero fixed-coordinate sample incomplete')
        explicit_skip = row.get('opportunities_skipped', row.get('skipped_optimizer_calls'))
        explicit_processed = row.get('opportunities_processed')
        if explicit_skip is not None:
            need(integer(explicit_skip) and 0 <= explicit_skip <= 19 and calls + explicit_skip == 20,
                 'Actual rank success/skip accounting disagrees', 'REJECTED_INPUT_NO_WRITE')
        if explicit_processed is not None:
            need(explicit_processed == 20, 'Rank did not process every opportunity', 'REJECTED_INPUT_NO_WRITE')
        if row.get('scheduler_step') is not None:
            need(row['scheduler_step'] == outer, 'Actual rank scheduler differs', 'REJECTED_INPUT_NO_WRITE')
        result.append(dict(row, actual_successful_writes=int(calls), actual_opportunities_processed=explicit_processed,
                           actual_opportunities_skipped=explicit_skip,
                           missing_rank_opportunity_fields_remain_unknown=explicit_skip is None or explicit_processed is None))
    if metric is not None:
        need(sum(r['actual_successful_writes'] for r in result)/8 == metric['actor/optimizer_steps'],
             'Full metric actual writes disagree with all-rank returns', 'REJECTED_INPUT_NO_WRITE')
    else:
        need(outer == TARGET and len({r['attempt_id'] for r in result}) == 1,
             'All actual200 rank returns must identify the same credited attempt')
    return result


def validate_native(native, progress, outer):
    need(native.get('source_commit') == SOURCE and native.get('outer') == outer,
         'Selected native readback source/outer differs', 'REJECTED_INPUT_NO_WRITE')
    need(native.get('archive_matches_original_manifest') is True, 'Native JSON archival is settling; not a training fault')
    ids = native.get('identity') or {}
    expected = dict(job_id=JOB, code_commit=SOURCE, config_sha256=CONFIG, completed_step=outer, actual_update_limit=TARGET)
    need(all(ids.get(k) == v for k,v in expected.items()) and ids.get('run_id', Path(RUN).name) == Path(RUN).name,
         'Native identity differs', 'REJECTED_INPUT_NO_WRITE')
    need(native.get('manifest_status') in ('COMPLETE_NATIVE_ROLLING_STATE_SAVED', 'COMPLETE_SAME_WORLD_STATE_SAVED'),
         'A COMPLETE native writer manifest has not settled')
    record = native.get('receipt') or {}
    manifest_sha = native.get('manifest_sha256')
    need(sha64(manifest_sha) and manifest_sha == native.get('recorded_original_manifest_sha256')
         == record.get('history_manifest_sha256') == record.get('source_manifest_sha256')
         and sha64(native.get('receipt_sha256')) and record.get('completed_outer') == outer
         and record.get('source_commit') == SOURCE and record.get('status') == 'COMPLETE_NATIVE_MANIFEST_METADATA_CAPTURED',
         'Actual original/archive JSON SHA readback is incomplete')
    guard = ids.get('guard_continue_accounting') or {}
    need(guard.get('protocol') == 'tree_guard_continue_fresh_rollout_v1'
         and guard.get('completed_outer_updates') == outer
         and guard.get('training_attempts') == progress.get('training_attempts_seen'), 'Native actual credit/attempt accounting differs')
    completed = counts(guard.get('completed_optimizer_calls_per_rank'), 'Native credited calls')
    totals = counts(guard.get('total_optimizer_calls_per_rank'), 'Native actual total calls')
    need(completed == counts(progress.get('completed_actual_optimizer_calls_per_rank'), 'Persisted actual credited calls')
         and all(value >= 100 + outer - RESTORE for value in completed.values()),
         'Native versus persisted actual cumulative calls differ', 'REJECTED_INPUT_NO_WRITE')
    need(integer(guard.get('refused_attempts')) and guard['refused_attempts'] >= 0
         and guard['training_attempts'] == outer + guard['refused_attempts']
         and guard['refused_attempts'] == progress.get('all_skipped_attempts'),
         'Actual no-credit all-skip attempts must remain separately accounted')
    partial = guard.get('partial_optimizer_calls_per_rank')
    need(isinstance(partial, dict) and set(partial) <= {str(r) for r in range(8)},
         'Native partial accounting missing')
    need(all(integer(partial.get(str(r), 0)) and partial.get(str(r), 0) >= 0
             and totals[str(r)] == completed[str(r)] + partial.get(str(r), 0) for r in range(8)),
         'Actual totals omit known partial writes', 'REJECTED_INPUT_NO_WRITE')
    workers = record.get('workers') or []
    need(len(workers) == 8 and {w.get('rank') for w in workers} == set(range(8)), 'Native eight-writer metadata pending')
    for worker in workers:
        actual = totals[str(worker['rank'])]
        need(worker.get('full_worker_state_finite') is True and worker.get('actual_total_optimizer_calls') == actual
             and worker.get('observed_adam_steps') == [actual] and worker.get('scheduler_last_epoch') == outer
             and sha64(worker.get('recorded_file_sha256')) and sha64(worker.get('recorded_logical_state_sha256')),
             'Native writer finite/actual Adam/scheduler metadata incomplete')
    need(record.get('driver', {}).get('next_step') == outer+1, 'Native actual driver next_step field differs')
    return record, guard, completed, totals


def final_phase(poll, resource, active):
    stage = (poll.get('logs') or {}).get('stage.log') or ''
    if 'Final validation dataset size: 51713' in stage or 'Final validation metrics:' in stage:
        need(active, 'Final-stage log alone cannot explain an inactive unit')
        return dict(kind='actual_final_evaluation_log', normal_exit_claimed=False)
    final = resource.get('final_evaluation') or {}
    if (final.get('source_commit') == SOURCE and final.get('unit') == UNIT
            and final.get('actual_completed_outer') == TARGET and final.get('dataset_rows') == 51713
            and final.get('status') in ('FINAL_EVALUATION_STARTED', 'VALIDATION_COST_STARTED', 'VALIDATION_COST_COMPLETE')):
        return dict(kind='explicit_downloaded_final_phase', evidence=final, normal_exit_claimed=False)
    result = poll.get('run_result') or {}
    if (result.get('code_commit') == SOURCE and result.get('job_id') == JOB
            and result.get('actual_completed_outer') == TARGET
            and result.get('status') == 'ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE'):
        return dict(kind='actual_training_terminal', evidence=result, normal_exit_claimed=result.get('training_body_exit_code') == 0)
    handoff = resource.get('final_handoff') or {}
    if (handoff.get('status') == 'EXPLICIT_NESTED_TRAIN200_FINAL_EVALUATION_HANDOFF_AUTHORIZED'
            and handoff.get('training_source_commit') == SOURCE and handoff.get('training_unit') == UNIT
            and handoff.get('training_run') == RUN and handoff.get('actual_training_completed_outer') == TARGET
            and handoff.get('authority', {}).get('human_instruction')
            and handoff.get('authority', {}).get('original_instruction_time')):
        return dict(kind='explicit_authorized_final_only_handoff', evidence=handoff, normal_exit_claimed=False)
    raise Observation('PENDING_FINAL_PHASE_NO_WRITE', 'Native200 is not evidence that training201 entered; capture actual final phase/terminal/handoff')


def valid_dev(value):
    metrics = value.get('metrics') or {}
    return (value.get('status') == 'VALIDATION_COST_COMPLETE' and value.get('rows') == 512
            and integer(value.get('outer')) and 5 <= value['outer'] <= TARGET
            and metrics.get('val/qa_count/all') == 512 and number(metrics.get('val/f1/all'))
            and number(metrics.get('val/em/all')))


def dev_evidence(poll, previous, metric, outer):
    candidates = [v for v in poll.get('development', []) if valid_dev(v) and v['outer'] <= outer]
    prior = previous.get('latest_dev512')
    if isinstance(prior, dict) and valid_dev(prior) and prior['outer'] <= outer:
        candidates.append(prior)
    latest = max(candidates, key=lambda v:v['outer']) if candidates else None
    actual = next((v for v in candidates if v['outer'] == outer), None) if outer % 5 == 0 else None
    # A missing periodic report is unknown, never a recycled prior score/time.
    elapsed = metric.get('timing_s/testing') if actual is not None and metric is not None else None
    if elapsed is not None:
        need(number(elapsed) and elapsed >= 0, 'Invalid actual current dev timing', 'REJECTED_INPUT_NO_WRITE')
    updated = actual is not None and (prior is None or actual != prior)
    return actual, latest, elapsed, updated


def previous_snapshots(workspace, base):
    paths = [workspace/'outputs/STATUS.json', workspace/'outputs/tree-nested-branch/STATUS.json',
             base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json']
    result = []
    for path in paths:
        if path.is_file():
            value, pin = held(path)
            need(value.get('source_commit') == SOURCE, 'Existing status belongs to another source', 'REJECTED_INPUT_NO_WRITE')
            result.append((path, value, pin))
    return result


def monotonic(previous, outer, epoch):
    for _, value, _ in previous:
        need(value.get('goal_complete') is not True, 'Completed goal cannot be reset', 'STALE_SNAPSHOT_NO_WRITE')
        prior_outer = max([value.get(k, 0) or 0 for k in
            ('current_outer', 'last_confirmed_outer', 'last_confirmed_chain_outer', 'latest_native_checkpoint_outer')])
        prior_epoch = max([value.get(k, 0) or 0 for k in ('observed_epoch', 'last_observed_epoch')])
        need(outer >= prior_outer and epoch >= prior_epoch,
             'Selected poll is older than current progress; no status/report overwrite', 'STALE_SNAPSHOT_NO_WRITE')


def atomic_replace(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('poll', 'native-readback', 'resource'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve();base = workspace/BASE_REL;outer = args.outer
    need(RESTORE < outer <= TARGET, 'Only current child outer6..200 is permitted', 'REJECTED_INPUT_NO_WRITE')
    paths = [path if path.is_absolute() else workspace/path
             for path in (args.poll, args.native_readback, args.resource)]
    loaded = [held(path) for path in paths]
    (poll, poll_pin), (native, native_pin), (resource, resource_pin) = loaded
    need(poll.get('source_commit') == SOURCE and poll.get('unit') == UNIT and poll.get('job_id') == JOB
         and poll.get('run') == RUN and poll.get('node') == 'h100-10', 'Wrong fixed current experiment identity', 'REJECTED_INPUT_NO_WRITE')
    epoch = poll.get('epoch');need(number(epoch), 'Actual poll epoch missing')
    previous = previous_snapshots(workspace, base);monotonic(previous, outer, epoch)
    status_candidates = [(v.get('last_observed_epoch', v.get('observed_epoch', 0)), v)
                         for p,v,_ in previous if p.name == 'STATUS.json']
    need(status_candidates, 'Existing current status is required to preserve ancestry/publication/deployment fields')
    previous_status = max(status_candidates, key=lambda item:item[0])[1]
    followup = next((v for p,v,_ in previous if p.name == 'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json'), {})
    ps, rs, active = validate_unit(poll, resource, outer)
    restore_receipt, provenance = validate_restore(poll)
    progress = poll.get('progress') or {}
    need(progress.get('actual_completed_outer_from_attempt_results') == outer and not progress.get('errors'),
         'Selected outer credit is pending or the snapshot already advanced')
    metric = validate_metric(poll, outer);ranks = validate_ranks(poll, metric, outer)
    native_record, guard, completed, totals = validate_native(native, progress, outer)
    if metric is None:
        need(outer == TARGET and all(r['attempt_id'] == guard['training_attempts'] for r in ranks),
             'Actual200 rank returns and persisted native final attempt disagree')
    stage = (poll.get('logs') or {}).get('stage.log') or ''
    if outer < TARGET:
        need(active and re.search(r'epoch\s+\d+,\s*step\s+' + str(outer+1) + r'\b', stage),
             'Actual next outer stage marker is pending')
        next_outer, final = outer+1, None
    else:
        next_outer, final = None, final_phase(poll, resource, active)
    actual_dev, latest_dev, dev_seconds, score_updated = dev_evidence(poll, previous_status, metric, outer)
    metric_pending = metric is None
    total_seconds = metric['timing_s/step'] if metric is not None else None
    training_seconds = total_seconds - dev_seconds if dev_seconds is not None else total_seconds if outer % 5 else None
    latest_dev_scores = dict(outer=latest_dev.get('outer') if latest_dev else None,
        f1_percent=100*latest_dev['metrics']['val/f1/all'] if latest_dev else None,
        em_percent=100*latest_dev['metrics']['val/em/all'] if latest_dev else None,
        current_outer_dev_complete=actual_dev is not None)
    variance_names = ('tree_variance/intra_adv_population_variance',
                      'tree_variance/group_actual_adv_population_variance',
                      'tree_variance/within_root_raw_reward_population_variance',
                      'tree_variance/group_raw_reward_population_variance',
                      'tree_variance/mixed_reward_question_fraction')
    cost_names = ('rollout/candidate_attempts','rollout/candidate_accepted',
                  'rollout/candidate_fallbacks','rollout/candidate_discarded',
                  'rollout/generated_tokens','rollout/prefill_tokens','rollout/retrieval_calls',
                  'timing_s/gen','timing_s/update_actor')
    current_variances = {name:metric.get(name) if metric is not None else None for name in variance_names}
    current_candidate_costs = {name:metric.get(name) if metric is not None else None for name in cost_names}
    watcher = resource.get('state') or {}
    need(not watcher or (watcher.get('training_source_commit') == SOURCE and watcher.get('training_unit') == UNIT
         and watcher.get('run_directory') == RUN
         and watcher.get('expected_training_invocation_id') == INVOCATION),
         'Independent watcher source/run binding differs', 'REJECTED_INPUT_NO_WRITE')
    watcher_service = state(resource['watcher_systemd']) if resource.get('watcher_systemd') is not None else None
    need(not watcher.get('last_errors') and not watcher.get('audit_error_count'),
         'Independent watcher reports an audit observation; no success write')
    input_pins = dict(poll=poll_pin, native_readback=native_pin, resource=resource_pin)
    receipt_name = f'FULL_OUTER{outer}_PROGRESS_RECEIPT.json';receipt_path = base/receipt_name
    need(not receipt_path.exists(), 'Existing immutable outer receipt retained; no rewrite', 'RECEIPT_EXISTS_NO_WRITE')
    receipt_status = ('ACTUAL_TRAIN200_UPDATES_NATIVE_COMPLETE_FULL_METRIC_PENDING_FINAL_PHASE_OBSERVED'
                      if metric_pending else f'ACTUAL_FULL_OUTER{outer}_NATIVE{outer}_COMPLETE_'
                      + (f'NEXT{next_outer}_ENTERED' if next_outer else 'FINAL_PHASE_OBSERVED'))
    receipt = dict(status=receipt_status,
        observed_epoch=epoch,source_commit=SOURCE,config_sha256=CONFIG,unit=UNIT,run=RUN,restored_outer=RESTORE,
        local_control_script_sha256=digest(Path(__file__).read_bytes()),
        completed_outer=outer,actual_next_outer_entered=next_outer,final_phase_evidence=final,
        per_rank_successful_adam=completed,per_rank_actual_total_adam=totals,partial_optimizer_calls_per_rank=guard.get('partial_optimizer_calls_per_rank'),
        actual_outer_successful_writes_per_rank={str(r['rank']):r['actual_successful_writes'] for r in ranks},
        actual_outer_skipped_opportunities_per_rank={str(r['rank']):r['actual_opportunities_skipped'] for r in ranks},
        actual_outer_processed_opportunities_per_rank={str(r['rank']):r['actual_opportunities_processed'] for r in ranks},
        full_metric_processed_opportunities=metric['actor/optimizer_opportunities_processed'] if metric is not None else None,
        full_metric_skipped_opportunities=metric['actor/optimizer_opportunities_skipped'] if metric is not None else None,
        no_all20_writes_or_zero_skip_assumption=True,all_skipped_attempts=progress.get('all_skipped_attempts'),
        direct_rank_record_count=poll.get('direct_rank_records'),last_rank_summaries=ranks,
        native_manifest_sha256=native['manifest_sha256'],native_history_receipt_sha256=native['receipt_sha256'],
        native_manifest_paths=[native_record.get('source_manifest_path'),native_record.get('history_manifest_path')],
        native_driver_next_step=native_record['driver']['next_step'],native_workers=native_record['workers'],
        native_evidence_scope='Actual writer finite/file/logical metadata; immutable original/archive JSON SHA readback. PT availability is capture-time only.',
        independent_cold_reload_performed=False,tensor_contents_read=False,PT_current_availability_unknown=True,
        iteration_metrics=metric,full_iteration_metric_pending=metric_pending,full_outer_wall_seconds=total_seconds,
        latest_actual_dev512_scores=latest_dev_scores,current_training_variances=current_variances,
        current_candidate_cost_scalars=current_candidate_costs,
        unobserved_metric_fields_remain_null=metric_pending,actual_dev512=actual_dev,latest_actual_dev512=latest_dev,
        latest_dev512_outer=latest_dev.get('outer') if latest_dev else None,dev_score_updated_this_turn=score_updated,
        normal_dev512_wall_seconds=dev_seconds,training_archive_wall_seconds_excluding_current_dev=training_seconds,
        source_poll=str(args.poll),source_poll_sha256=poll_pin['sha256'],native_metadata_readback=str(args.native_readback),
        native_metadata_readback_sha256=native_pin['sha256'],input_bindings=input_pins,
        own_resources_after_save=resource.get('own_cgroup'),actual_training_pid=int(rs.get('MainPID', 0)),
        actual_training_invocation_id=INVOCATION,exact_executor_resume=False,target_outer=TARGET,goal_status='active',
        legacy_unbound_native_collector_observation=poll.get('native_manifest_history'),
        independent_source_bound_native_watcher=watcher or None,source_bound_watcher_systemd=watcher_service,
        separate_profile_observations=dict(poll=poll.get('profile'),resource=resource.get('profile')),
        watcher_metadata_lag_is_not_training_failure=True)
    updated = copy.deepcopy(previous_status)
    updated.update(status='TRAINING_RUNNING' if outer < TARGET else 'TRAIN200_COMPLETE_FINAL_WORK_PENDING',
        phase='TRAINING_RUNNING' if outer < TARGET else 'TRAIN200_COMPLETE_FINAL_WORK_PENDING',
        observed_epoch=epoch,last_observed_epoch=epoch,current_outer=outer,completed_outer=outer,new_completed_outer=outer-RESTORE,
        actual_current_restore_outer=outer,actual_current_restore_adam_per_rank=completed,
        successful_adam_per_rank=completed,actual_total_adam_per_rank=totals,
        effective_counters_known=True,actual_child_counters_known=True,
        all_skipped_attempts=progress.get('all_skipped_attempts'),direct_rank_record_count=poll.get('direct_rank_records'),
        child_successful_adam_per_rank={str(r):completed[str(r)]-100 for r in range(8)},
        last_confirmed_chain_outer=outer,last_confirmed_chain_adam_per_rank=completed,
        effective_origin_segments=[*provenance.get('effective_ancestor_segments', []),
            dict(source_run=RUN,run_id=Path(RUN).name,code_commit=SOURCE,config_sha256=CONFIG,first_outer=RESTORE+1,last_outer=outer)],
        latest_native_checkpoint_outer=outer,latest_native_manifest_sha256=native['manifest_sha256'],
        last_durable_completed_outer=outer,current_full_outer_progress_receipt=receipt_name,last_progress_receipt=receipt_name,
        actual_next_outer_entered=next_outer,active_outer_in_progress=next_outer,active_outer=next_outer,
        last_actual_poll=str(args.poll),current_systemd=ps,actual_training_progress=progress,last_rank_summaries=ranks,
        checkpoint=poll.get('native_manifests'),recent_iteration_metrics=poll.get('iteration_metrics'),
        latest_sampling_completed_outer=outer,next_outer_sampling_complete=None,
        latest_guard=poll.get('latest_guard'),latest_dev512=latest_dev,dev_score_updated=score_updated,
        dev_score_updated_this_turn=score_updated,actual_dev512=actual_dev,normal_dev512_wall_seconds=dev_seconds,
        latest_full_outer_wall_seconds=total_seconds,latest_training_archive_wall_seconds_excluding_dev=training_seconds,
        latest_ordinary_training_seconds_excluding_dev=training_seconds,
        latest_intra_adv_population_variance=metric.get('tree_variance/intra_adv_population_variance') if metric is not None else None,
        latest_actual_total_adv_population_variance=metric.get('tree_variance/group_actual_adv_population_variance') if metric is not None else None,
        latest_mixed_reward_question_fraction=metric.get('tree_variance/mixed_reward_question_fraction') if metric is not None else None,
        legacy_unbound_native_collector_observation=poll.get('native_manifest_history'),
        current_native_manifest_history=watcher or None,native_manifest_history=watcher or None,
        native_watcher_systemd=watcher_service,latest_rollout_cost=poll.get('rollout_cost', []),
        completed_outer_rollout_cost=[v for v in poll.get('rollout_cost', []) if v.get('candidate_outer') == outer],
        separate_profile_observations=receipt['separate_profile_observations'],
        normal_dev512_pending=outer % 5 == 0 and actual_dev is None,
        progress_input_bindings=input_pins,submitted_this_turn=False,new_training_submitted=False,
        running_source_hot_modified=False,extra_prelaunch_review=False,exact_executor_resume=False,
        goal_complete=False,formal_effect_complete=False,goal_status='active',full_iteration_log_pending=metric_pending,
        current_iteration_metrics=metric,current_full_outer_metric_pending=metric_pending,
        latest_actual_dev512_scores=latest_dev_scores,current_training_variances=current_variances,
        current_candidate_cost_scalars=current_candidate_costs)
    if latest_dev is not None and latest_dev in poll.get('development', []):
        updated['latest_dev512_source_commit'] = SOURCE
    if outer % 5 == 0:
        updated[f'normal_dev512_outer{outer}_pending'] = actual_dev is None
    if dev_seconds is not None:
        updated['latest_completed_dev512_wall_seconds'] = dev_seconds
    followup = copy.deepcopy(followup)
    followup.update(status=f'NORMAL_TRAINING_OUTER{outer}_FULL_' + (f'NEXT{next_outer}_ENTERED' if next_outer else 'FINAL_WORK_PENDING'),
        source_commit=SOURCE,unit=UNIT,main_pid=int(rs.get('MainPID',0)),invocation_id=INVOCATION,
        last_confirmed_outer=outer,target_outer=TARGET,restored_outer=RESTORE,actual_next_outer_entered=next_outer,
        actual_optimizer_completed_outer=outer,completed_outer=outer,observed_epoch=epoch,successful_adam_per_rank=completed,
        all_skipped_attempts=progress.get('all_skipped_attempts'),actual_total_adam_per_rank=totals,
        latest_native_checkpoint_outer=outer,last_progress_receipt=receipt_name,evidence=str(args.poll),
        latest_full_outer_wall_seconds=total_seconds,normal_dev512_wall_seconds=dev_seconds,
        latest_ordinary_training_seconds_excluding_dev=training_seconds,latest_dev512_outer=receipt['latest_dev512_outer'],
        score_updated=score_updated,normal_training_continues=outer<TARGET,goal_status='active',
        exact_executor_resume=False,full_iteration_log_pending=metric_pending,current_iteration_metrics=metric,progress_input_bindings=input_pins)
    if outer % 5 == 0:
        followup[f'normal_dev512_outer{outer}_pending'] = actual_dev is None
    followup['new_full_outers_after_this_restore'] = sorted(set([*followup.get('new_full_outers_after_this_restore', []),outer]))
    followup['full_outer_receipts'] = list(dict.fromkeys([*followup.get('full_outer_receipts', []),receipt_name]))
    if latest_dev:
        followup['latest_dev512_f1_percent'] = 100*latest_dev['metrics']['val/f1/all']
        followup['latest_dev512_em_percent'] = 100*latest_dev['metrics']['val/em/all']
    report = (f'Current nested Tree200 on original H10018.\n\nActual complete outer {outer}/200; '
        + (f'next outer {next_outer} entered in the same active instance.' if next_outer else 'actual final phase observed; training201 is not claimed.')
        + f' Source {SOURCE}; unit {UNIT}; Invocation {INVOCATION}.\n\n'
        + f'Actual cumulative credited Adam per rank: {completed}; total including known partial writes: {totals}. '
        + f'Current successful writes: {receipt["actual_outer_successful_writes_per_rank"]}; per-rank missing opportunity fields remain unknown. '
        + f'Full metric processed/skipped opportunities: {receipt["full_metric_processed_opportunities"]}/{receipt["full_metric_skipped_opportunities"]}.\n\n'
        + ('Actual200 training updates and native state are complete; the complete step200 metric has not yet been produced. All unobserved metric/timing fields remain null.\n\n' if metric_pending else '')
        + f'Current dev512: {"complete" if actual_dev else "none/not yet observed"}; latest actual dev outer {receipt["latest_dev512_outer"]}, F1_percent {json.dumps(latest_dev_scores["f1_percent"])}, EM_percent {json.dumps(latest_dev_scores["em_percent"])}. '
        + f'Current step seconds {total_seconds}; current dev seconds {dev_seconds}; training/archive excluding current dev {training_seconds}. '
        + 'Older dev time is never assigned to this outer.\n\n'
        + f'Current training variance scalars: {json.dumps(current_variances)}.\n\n'
        + f'Current candidate/rollout cost scalars: {json.dumps(current_candidate_costs)}.\n\n'
        + 'Native evidence is writer metadata and original/archive JSON SHA agreement, not independent PT cold reload; old PT availability is unknown. '
        + 'Legacy collector pending and the independent source-bound watcher are separately recorded. Normal metadata settling is pending, not a training fault.\n\n'
        + 'Origin0bef outer1, ba9 outer2–5 and current778 outer6 onward remain separate. Lost original2, failed Ray6 and unloaded eb6 startup stay cost evidence without extra credit. '
        + 'Existing source publication, cost, future HF/evaluation deployment and acceptance fields are preserved in STATUS.\n\n'
        + 'Goal remains active: real200, full51713 scoring, final HF/tokenizer proof, independent acceptance and cleanup are separate work. No exact executor resume, new launch, review or training change.\n')
    outputs = {receipt_path:json_bytes(receipt),base/f'CONTINUE_OUTER{outer}_PROGRESS_REPORT.md':report.encode(),
        workspace/'outputs/STATUS.json':json_bytes(updated),workspace/'outputs/tree-nested-branch/STATUS.json':json_bytes(updated),
        workspace/'outputs/REPORT.md':report.encode(),workspace/'outputs/tree-nested-branch/REPORT.md':report.encode(),
        base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json':json_bytes(followup)}
    # Lock only this local helper. Re-read every observed source/status immediately
    # before publication; a concurrent root update refuses the entire write batch.
    with Path(__file__).open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _, pin in loaded + [(v,pin) for _,v,pin in previous]:
            need(digest(Path(pin['path']).read_bytes()) == pin['sha256'], 'Input/status changed during preparation', 'CONCURRENT_UPDATE_NO_WRITE')
        monotonic(previous_snapshots(workspace,base), outer, epoch)
        need(not receipt_path.exists(), 'Another publisher wrote this outer receipt', 'CONCURRENT_UPDATE_NO_WRITE')
        for path, raw in outputs.items():
            if path == receipt_path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            else:
                atomic_replace(path,raw)
        mirrors=[]
        for mirror in (workspace/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
                       Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')):
            if mirror.is_dir():
                for source in (workspace/'outputs/tree-nested-branch/STATUS.json',workspace/'outputs/tree-nested-branch/REPORT.md',
                               receipt_path,base/f'CONTINUE_OUTER{outer}_PROGRESS_REPORT.md',base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json'):
                    atomic_replace(mirror/source.name, outputs[source])
                mirrors.append(str(mirror))
    print(json.dumps(dict(status='LOCAL_PROGRESS_RECEIPT_AND_STATUS_PUBLISHED',outer=outer,receipt=str(receipt_path),
                          input_bindings=input_pins,mirrors=mirrors,remote_actions_performed=False)))


if __name__ == '__main__':
    try:
        main()
    except Observation as error:
        print(json.dumps(dict(status=error.status,reason=str(error),successful_receipt_written=False,
                              status_or_report_written=False,remote_actions_performed=False)))
        sys.exit(2)
    except Exception as error:
        print(json.dumps(dict(status='LOCAL_INPUT_OR_IO_ERROR',reason=repr(error),
                              no_remote_actions=True,automatic_retry=False)))
        sys.exit(1)
