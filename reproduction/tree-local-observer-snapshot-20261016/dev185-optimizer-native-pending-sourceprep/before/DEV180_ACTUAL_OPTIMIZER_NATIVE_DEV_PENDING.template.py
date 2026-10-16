"""Frozen180 local pending producer; source preparation alone performs no publication.

Derived from the preserved35230-byte corrected175 producer (SHA69936d89...).
Scope stays actual optimizer180/native180 writer COMPLETE while dev180/FULL180
are still absent. Saved cursor181 is not actual training181 or a cold reload.
Native/guard/rank records provide real counts; no3600 total is hardcoded.
Missing FULL180 and rank opportunity fields remain unknown/null.
FULL/COST/CUM179, latest dev/curve175, HF150, actual cold load5 and source1ed
cumulative main4/remote4/PENDING0 remain protected. No SSH, collector, eval or
training operation is present. Root must skip this entry after FULL180 or a
completed dev180 is observed. This source is unexecuted/uncompiled/untested.

Future invocation, only with actual frozen root-selected values:
  python3 -B THIS_FILE --workspace WORKSPACE --pins ACTUAL_PINS_JSON \
    --pins-bytes ACTUAL_BYTES --pins-sha256 ACTUAL_SHA \
    --source-bytes ACTUAL_SOURCE_BYTES --source-sha256 ACTUAL_SOURCE_SHA

ACTUAL_PINS_JSON schema: dev180_pending_stage_execution_pins_v1
Required descriptors (file,bytes,sha256): first_poll,first_resource,
second_poll,second_resource,native180,prior_full_progress,prior_root_full.
Also require actual native_manifest_sha256/native_receipt_sha256 and the then
current canonical_before list naming exactly the14 existing paths below.
No future180 file names, digests, native counts or success are defaulted.
"""

import argparse
import copy
import datetime
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
CORE_SHA = '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
ORIGINAL_SHA = '070304de4dec953422dcddafc965fa1d1167f7cdf6d498700ff8eedc370e0dae'
BASE_REL = 'outputs/tree-nested-branch-handoff-20261013'
ORIGINAL_REL = ('work/tree-speed-repair-20261008/nested-local-observer-snapshot-source/'
                'reproduction/tree-local-observer-snapshot-20261016/dev170-optimizer-native-pending-stage/'
                'DEV170_ACTUAL_OPTIMIZER_NATIVE_DEV_PENDING_2097_2098.command.txt')
CORE_REL = ('work/tree-speed-repair-20261008/nested-native5-controls-source/reproduction/'
            'tree-eval-resume-20261013/local-control-backup/sync_nested_native_progress.py')
PHASE = 'ACTUAL_OPTIMIZER180_NATIVE180_COMPLETE_DEV180_PENDING_FULL179'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def descriptor(path, raw):
    return dict(path=str(path), bytes=len(raw), sha256=sha(raw))


def atomic(path, raw):
    temporary = path.with_name(path.name + '.180-pending.tmp')
    need(not temporary.exists(), 'Retain pre-existing temporary file: ' + str(temporary))
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--pins', required=True, type=Path)
    parser.add_argument('--pins-bytes', required=True, type=int)
    parser.add_argument('--pins-sha256', required=True)
    parser.add_argument('--source-bytes', required=True, type=int)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    w = args.workspace.resolve()
    b = w / BASE_REL
    ms = [w / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
          Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
    need(all(path.is_dir() for path in [b] + ms), 'Existing BASE and two mirrors required')
    source = Path(__file__).resolve()
    source_raw = source.read_bytes()
    need(len(source_raw) == args.source_bytes and sha(source_raw) == args.source_sha256,
         'Root-selected source byte/hash pin differs')
    original = w / ORIGINAL_REL
    original_raw = original.read_bytes()
    need(len(original_raw) == 6913 and sha(original_raw) == ORIGINAL_SHA, 'Original170 bytes changed')
    pins_path = args.pins.resolve()
    need(pins_path.is_relative_to(w), 'Execution pins must be a local workspace file')
    pins_raw = pins_path.read_bytes()
    need(len(pins_raw) == args.pins_bytes and sha(pins_raw) == args.pins_sha256,
         'Root-selected input-pin file changed')
    pins = json.loads(pins_raw)
    need(pins.get('schema') == 'dev180_pending_stage_execution_pins_v1', 'Input-pin schema differs')
    cp = w / CORE_REL
    need(sha(cp.read_bytes()) == CORE_SHA, 'Original676d validator source changed')
    spec = importlib.util.spec_from_file_location('pending180_core', cp)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)

    # Use the same cooperative publisher lock as the existing FULL alias writer.
    with cp.open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs, retained_raw = {}, {}

        def bound(path, size, digest, mirrored=False):
            path = path.resolve()
            need(type(size) is int and size > 0 and re.fullmatch(r'[a-f0-9]{64}', digest or ''),
                 'Actual bytes/SHA required: ' + str(path))
            raw = path.read_bytes()
            need(len(raw) == size and sha(raw) == digest, 'Bound input differs: ' + str(path))
            retained_raw[path] = raw
            if mirrored:
                for mirror in ms:
                    q = mirror / path.relative_to(b.resolve())
                    need(q.read_bytes() == raw, 'Published input mirror differs: ' + str(q))
                    retained_raw[q] = raw
            ref = descriptor(path, raw)
            refs[str(path)] = ref
            return json.loads(raw), ref

        def selected(key, prefix):
            pin = pins.get(key) or {}
            name = pin.get('file')
            need(isinstance(name, str) and re.fullmatch(prefix + r'\d+\.json', name),
                 'Explicit real observation filename required: ' + key)
            return bound(b / name, pin.get('bytes'), pin.get('sha256')), name

        need(not any((location / 'FULL_OUTER180_PROGRESS_RECEIPT.json').exists() for location in [b] + ms),
             'FULL180 already exists; this pending entry is obsolete')
        (p1, p1pin), p1name = selected('first_poll', 'NATIVE5_CONTINUE_POLL_')
        (r1, r1pin), r1name = selected('first_resource', 'NATIVE5_CONTINUE_RESOURCE_')
        (p2, p2pin), p2name = selected('second_poll', 'NATIVE5_CONTINUE_POLL_')
        (r2, r2pin), r2name = selected('second_resource', 'NATIVE5_CONTINUE_RESOURCE_')
        first = re.fullmatch(r'NATIVE5_CONTINUE_POLL_(\d+)\.json', p1name).group(1)
        second = re.fullmatch(r'NATIVE5_CONTINUE_POLL_(\d+)\.json', p2name).group(1)
        need(r1name == 'NATIVE5_CONTINUE_RESOURCE_' + first + '.json'
             and r2name == 'NATIVE5_CONTINUE_RESOURCE_' + second + '.json' and int(second) > int(first),
             'Two real poll/resource pairs must match and advance')
        raw_names = {p1name, r1name, p2name, r2name}
        prior_known = {
            'FULL_OUTER179_PROGRESS_RECEIPT.json': (29690, '1483192321b3c307e006c256d11a051fd316cff06df7195662df0dad15174e56'),
            'COST_OUTER179_LOCAL_RECEIPT.json': (65630, 'e0454800d8983c780dcc1a79f4a256dc09ad186a422fbd5af574c7b88b244c90'),
            'CUMULATIVE_COST_THROUGH179.json': (80323, '22e94d992099a8fc3d5f638305edfa3f0e31f72cd3b89b158d53a05ecf4bc0cb'),
        }
        values = {}
        for name, (size, digest) in prior_known.items():
            values[name], _ = bound(b / name, size, digest, mirrored=True)
        for key, pattern in [('native180', r'NATIVE_HISTORY_180_METADATA_READBACK\.json'),
                             ('prior_full_progress', r'OUTER179_FULL_NEXT180_ACTUAL_PROGRESS\.json'),
                             ('prior_root_full', r'ROOT_FULL179_PUBLICATION_READBACK_\d+_\d+\.json')]:
            pin = pins.get(key) or {};name = pin.get('file')
            need(isinstance(name, str) and re.fullmatch(pattern, name), 'Explicit actual local pin required: ' + key)
            values[name], _ = bound(b / name, pin.get('bytes'), pin.get('sha256'), mirrored=True)
        native = values['NATIVE_HISTORY_180_METADATA_READBACK.json']
        MANIFEST180 = pins.get('native_manifest_sha256');ARCHIVE180 = pins.get('native_receipt_sha256')
        need(re.fullmatch(r'[a-f0-9]{64}', MANIFEST180 or '') and re.fullmatch(r'[a-f0-9]{64}', ARCHIVE180 or ''),
             'Actual native180 manifest and receipt SHA pins required')
        need(native['manifest_sha256'] == MANIFEST180 and native['receipt_sha256'] == ARCHIVE180
             and native.get('tensor_contents_read') is False
             and native.get('independent_cold_reload_performed') is False, 'Native180 metadata/scope differs')
        record, guard, completed, totals = c.validate_native(native, p1['progress'], 180)
        need(record['driver']['next_step'] == 181, 'Actual180 saved writer cursor differs')
        # Preserve the175 repair: only the first pinned poll must retain the native save marker.
        need('TREE_ROLLING_NATIVE_STATE_SAVED=' + c.RUN
             + '/checkpoints/training_state/rolling-completed/global_step_180/manifest.json'
             in p1['logs']['stage.log'], 'Original saved180 stage log marker absent')
        observations = []
        for poll, resource in [(p1, r1), (p2, r2)]:
            ps, rs, active = c.validate_unit(poll, resource, 180)
            c.validate_restore(poll)
            need(all(poll.get(k) == v for k, v in dict(source_commit=c.SOURCE, unit=c.UNIT,
                 job_id=c.JOB, run=c.RUN, node='h100-10').items()), 'Actual training identity differs')
            need(active and poll['progress']['actual_completed_outer_from_attempt_results'] == 180
                 and not poll['progress'].get('errors') and poll.get('run_result') is None,
                 'Actual180 instance is failed, pending or already beyond180')
            need(c.counts(poll['progress']['completed_actual_optimizer_calls_per_rank'], 'actual180') == completed,
                 'Actual180 credited counters differ')
            need(poll['progress']['training_attempts_seen'] == guard['training_attempts']
                 and poll['progress']['all_skipped_attempts'] == guard['refused_attempts'], 'Guard/skip differs')
            need(max(metric['step'] for metric in poll['iteration_metrics']) == 179, 'FULL180 already exists')
            need(not any(dev.get('outer') == 180 and dev.get('status') == 'VALIDATION_COST_COMPLETE'
                         for dev in poll['development']), 'Dev180 result already exists')
            log = poll['logs']['stage.log']
            need(re.search(r'epoch\s+\d+,\s*step\s+181\b', log) is None, 'Live181 already entered')
            state = resource['state']
            need(180 in state['indexed_completed_outers'] and state['audit_error_count'] == 0
                 and not state.get('last_errors') and state.get('training_source_commit') == c.SOURCE
                 and state.get('training_unit') == c.UNIT and state.get('run_directory') == c.RUN
                 and state.get('expected_training_invocation_id') == c.INVOCATION,
                 'Actual180 watcher binding/health differs')
            events = dict(line.split() for line in resource['own_cgroup']['memory.events'].splitlines())
            need(events.get('oom') == events.get('oom_kill') == '0', 'Actual training cgroup OOM observed')
            observations.append(dict(poll_epoch=poll['epoch'], resource_epoch=resource['epoch'],
                 MainPID=ps['MainPID'], InvocationID=rs['InvocationID'],
                 service_CPU_seconds=int(rs['CPUUsageNSec']) / 1e9))
        need(observations[1]['poll_epoch'] > observations[0]['poll_epoch']
             and observations[1]['resource_epoch'] > observations[0]['resource_epoch']
             and observations[1]['service_CPU_seconds'] > observations[0]['service_CPU_seconds'],
             'Two actual observations must move forward')
        growth = {name: p2['own_output_files'][name] - p1['own_output_files'][name]
                  for name in ('stage.log', 'retriever.log')}
        need(all(value > 0 for value in growth.values()), 'Pending dev stage/retriever byte growth required')
        ranks = p2['last_rank_summaries']
        need(len(ranks) == 8 and {rank['rank'] for rank in ranks} == set(range(8)), 'Eight ranks required')
        need(all(rank['outer_step'] == 180 and rank['attempt_id'] == guard['training_attempts']
                 and type(rank['successful_optimizer_calls']) is int and 1 <= rank['successful_optimizer_calls'] <= 20
                 and rank['finite_grad_norms'] >= rank['successful_optimizer_calls']
                 and rank['nonzero_grad_norms'] > 0
                 and rank['local_shard_sample_delta']['finite'] == rank['local_shard_sample_delta']['count']
                 and rank['local_shard_sample_delta']['nonzero'] > 0 for rank in ranks),
             'Original actual finite/nonzero rank evidence incomplete')
        processed_by_rank = {};skipped_by_rank = {}
        for rank in ranks:
            key = str(rank['rank']);writes = rank['successful_optimizer_calls']
            processed = rank.get('opportunities_processed')
            skipped = rank.get('opportunities_skipped', rank.get('skipped_optimizer_calls'))
            if processed is not None:
                need(type(processed) is int and processed == 20, 'Explicit actual processed opportunities differ')
            if skipped is not None:
                need(type(skipped) is int and 0 <= skipped <= 19 and writes + skipped == 20,
                     'Explicit actual write/skip opportunities differ')
            if rank.get('scheduler_step') is not None:
                need(rank['scheduler_step'] == 180, 'Explicit actual rank scheduler differs')
            processed_by_rank[key] = processed;skipped_by_rank[key] = skipped
        latest = max((dev for dev in p2['development'] if dev['status'] == 'VALIDATION_COST_COMPLETE'),
                     key=lambda dev: dev['outer'])
        need(latest['outer'] == 175 and latest['rows'] == 512
             and latest['metrics']['val/f1/all'] == 0.5655242435515874
             and latest['metrics']['val/em/all'] == 0.4765625, 'Latest real dev175 score differs')
        full = values['FULL_OUTER179_PROGRESS_RECEIPT.json']
        cumulative = values['CUMULATIVE_COST_THROUGH179.json']
        root179 = values[pins['prior_root_full']['file']]
        need(full['completed_outer'] == 179 and full['latest_dev512_outer'] == 175
             and cumulative['through_effective_outer'] == 179
             and root179['source1ed_confirmed_actual_main_invocations_cumulative'] == 4
             and root179['source1ed_confirmed_actual_remote_incremental_invocations_cumulative'] == 4,
             'Protected FULL/CUM179/dev175/collector4 boundaries differ')
        prior_completed = c.counts(full['per_rank_successful_adam'], 'Prior179 actual credited calls')
        need(all(completed[str(rank['rank'])] - prior_completed[str(rank['rank'])]
                 == rank['successful_optimizer_calls'] for rank in ranks),
             'Actual180 native credited delta differs from original rank writes; no3600 total is inferred')
        # Retain all actual protected CUM179 parts at their existing relative paths.
        seen_parts = set()
        def retain_parts(value):
            if isinstance(value, dict):
                if value.get('storage') == 'complete_original_field_parts':
                    for part in value['parts']:
                        name = part.get('file') or part.get('path');path = Path(name)
                        path = path.resolve() if path.is_absolute() else (b / path).resolve()
                        need(path.is_relative_to(b.resolve()), 'Cost part outside existing BASE')
                        if path not in seen_parts:
                            seen_parts.add(path);part_value, _ = bound(path, part['bytes'], part['sha256'], mirrored=True)
                            retain_parts(part_value)
                else:
                    for item in value.values():retain_parts(item)
            elif isinstance(value, list):
                for item in value:retain_parts(item)
        retain_parts(cumulative)

        paths = [w / 'outputs/STATUS.json', w / 'outputs/tree-nested-branch/STATUS.json',
                 b / 'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json', b / 'CURRENT_OBSERVER_CONNECTION_STATE.json']
        reports = [w / 'outputs/REPORT.md', w / 'outputs/tree-nested-branch/REPORT.md',
                   *[mirror / 'REPORT.md' for mirror in ms]]
        allp = paths + [mirror / path.name for mirror in ms for path in paths[1:]] + reports
        need(len(allp) == len(set(allp)) == 14, 'Closed14 canonical path set differs')
        actual_pins = pins.get('canonical_before')
        need(isinstance(actual_pins, list) and len(actual_pins) == 14, 'Root actual closed14 pins required')
        by_path = {str(Path(pin['path']).resolve()): pin for pin in actual_pins}
        need(len(by_path) == 14 and set(by_path) == {str(path.resolve()) for path in allp},
             'Actual canonical pins must name exactly the closed14 paths')
        prior = {}
        for path in allp:
            pin = by_path[str(path.resolve())]
            raw = path.read_bytes()
            need(type(pin.get('bytes')) is int and len(raw) == pin['bytes'] and sha(raw) == pin.get('sha256'),
                 'Root-selected canonical pin changed: ' + str(path))
            prior[path] = raw
        need(prior[paths[0]] == prior[paths[1]] and len({prior[path] for path in reports}) == 1,
             'Prior canonical/status/report mirrors differ')
        for mirror in ms:
            for path in paths[1:]:
                need(prior[mirror / path.name] == prior[path], 'Prior canonical mirror differs')
        old_values = [json.loads(prior[path]) for path in paths]
        for old in old_values:
            need(old['completed_outer'] == old['current_outer'] == old['last_complete_full_metric_outer'] == 179
                 and old['active_outer'] == 180 and old['actual_optimizer_completed_outer'] == 179
                 and old['latest_native_checkpoint_outer'] == 179 and old['goal_status'] == 'active'
                 and old.get('goal_complete') is not True, 'Prior canonical stage already changed')
            need(old['latest_dev512_outer'] == old['learning_curve_complete_through_outer'] == 175
                 and old['dev_native_cost_actual_last_collected_outer'] == 175
                 and old['actual_current_restore_outer'] == 5
                 and old['actual_current_restore_adam_per_rank'] == {str(rank): 100 for rank in range(8)}
                 and old['dev_native_cost_collector_new_entry_actual_main_invocations'] == 4
                 and old['dev_native_cost_collector_actual_remote_incremental_invocations'] == 4
                 and old['dev_native_cost_collector_preceding_local_pending_invocations'] == 0,
                 'Protected dev/curve/cold-load/collector boundary differs')
            need(old['current_HF_permanent_milestone150_complete'] is True
                 and old['current_dev512'] is None and old['actual_dev512'] is None,
                 'HF150 or existing null current dev differs')
            epoch_fields = ('observed_epoch', 'last_observed_epoch', 'last_confirmed_epoch',
                            'last_actual_poll_epoch', 'last_current_progress_poll_epoch')
            need(all((old.get(key) or 0) <= p2['epoch'] for key in epoch_fields), 'Newer canonical epoch exists')

        suffix = first + '_' + second
        pending_path = b / ('OUTER180_UPDATES_NATIVE_DEV_PENDING_ACTUAL_' + suffix + '.json')
        closure_path = b / ('OUTER180_PENDING_CANONICAL_LOCAL_PUBLICATION_' + suffix + '.json')
        need(all(not (root / path.name).exists() for root in [b] + ms
                 for path in [pending_path, closure_path]), 'Retain existing pending/publication receipts')
        now = datetime.datetime.now(datetime.timezone.utc)
        snapshot = b / ('outer180_optimizer_native_dev_pending_prior_' + now.strftime('%Y%m%dT%H%M%S%fZ'))
        pending = dict(schema='nested_dev180_verified_wait_v1', status='ACTUAL_NATIVE180_DEV180_RESULT_PENDING',
             created_utc=now.isoformat(), goal_status='active', source_commit=c.SOURCE, config_sha256=c.CONFIG,
             unit=c.UNIT, run=c.RUN, original_PID=int(c.PID), invocation_id=c.INVOCATION,
             actual_observations=observations, actual_optimizer_completed_outer=180,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             child_successful_adam_per_rank={key: value - 100 for key, value in completed.items()},
             actual_outer_successful_writes_per_rank={str(rank['rank']): rank['successful_optimizer_calls'] for rank in ranks},
             original_rank_records=ranks, actual_outer_processed_opportunities_per_rank=processed_by_rank,
             actual_outer_skipped_opportunities_per_rank=skipped_by_rank,
             full_metric_processed_opportunities=None, full_metric_skipped_opportunities=None, full_metric_successful_writes=None,
             rank_opportunity_and_guard_skip_fields='Only explicit original per-rank opportunity fields are retained; absent fields stay null. FULL180 processed/skipped/write metrics stay null while FULL180 is absent. skipped_empty_minibatches is separate; no missing zero or20-opportunity count inferred.',
             latest_saved_native_outer=180, native_manifest_sha256=MANIFEST180,
             native_archive_receipt_sha256=ARCHIVE180, actual_cold_loaded_native_outer=5,
             independent_cold_reload_performed=False, tensor_contents_read=False, native_saved_driver_next_step=181,
             saved_cursor181_is_not_live_outer181=True, current_outer180_full_metric=None,
             current_outer180_wall_seconds=None, current_outer180_training_variances=None, current_outer180_reward=None,
             dev180_result=None, latest_completed_dev512=latest, score_updated=False,
             stage_and_retriever_log_growth=growth, input_refs=copy.deepcopy(refs),
             raw_observation_sources_scope='Original selected poll/resource files are BASE-only actual bytes/SHA-bound sources. Native180 and prior FULL/progress/COST/CUM/root179 products retain three-location two-pass byte readbacks.',
             raw_observation_files=sorted(raw_names), canonical_FULL_boundary=179, canonical_publication_updated=False,
             outer181_entered_claimed=False, whole_goal_complete=False, training_or_remote_mutation=False,
             scope='Actual180 optimizer/native saved; dev180 and FULL180 pending. FULL/COST/CUM179 and latest dev/curve175 remain previous complete boundaries. This evidence receipt itself does not mutate canonical; the separate local publication receipt records that patch.')
        pending_raw = encoded(pending)
        pending_pin = descriptor(pending_path, pending_raw)
        common = dict(status=PHASE, phase=PHASE, current_phase=PHASE, training_phase=PHASE,
             current_observation_status=PHASE, observed_current_training_phase=PHASE,
             actual_optimizer_completed_outer=180, last_confirmed_optimizer_outer=180,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             successful_adam_per_rank=completed, per_rank_successful_adam=completed,
             per_rank_actual_total_adam=totals,
             child_successful_adam_per_rank={key: value - 100 for key, value in completed.items()},
             actual_physical_rank_optimizer_calls=sum(totals.values()),
             actual_outer_successful_writes_per_rank=pending['actual_outer_successful_writes_per_rank'],
             actual_outer_processed_opportunities_per_rank=processed_by_rank,
             actual_outer_skipped_opportunities_per_rank=skipped_by_rank,
             actual_training_progress=p2['progress'], all_skipped_attempts=p2['progress']['all_skipped_attempts'],
             latest_native_checkpoint_outer=180, last_confirmed_native_saved_outer=180, latest_saved_native_outer=180,
             native_saved_driver_next_step=181, native_saved_cursor_is_not_live_next_outer_entry=True,
             native_manifest_sha256=MANIFEST180, latest_native_manifest_sha256=MANIFEST180,
             native_manifest_history_receipt_sha256=ARCHIVE180,
             latest_native_metadata_scope='Actual180 writer/archive JSON metadata only; actual cold load stays5; no tensor hash/read/reload.',
             actual_current_restore_scope='Actual cold-loaded5/100 each remains unchanged; writer-complete native180 is a distinct saved boundary.',
             successful_adam_per_rank_scope='Actual credited native5 restore plus direct child6..180; failed ancestor calls excluded; FULL accounting remains179.',
             child_successful_adam_per_rank_scope='Actual cumulative credited180 minus actual restored100 each; no cold restore180 or completed181 claim.',
             current_actual_optimizer_progress=pending_pin, current_live_validation_progress=pending_pin,
             optimizer180_completion_before_periodic_dev=pending_pin, dev180_result=None,
             dev180_result_pending=True, normal_dev512_outer180_pending=True, normal_dev512_pending=True,
             current_dev512_outer=180, current_dev_boundary_scope='Current180 evaluation pending/null; latest complete dev175 F1/EM and curve175 remain unchanged.',
             current_active_outer_full_metric_pending=True, active_outer_optimizer_completion_confirmed=True,
             active_outer_full_update_unconfirmed=False, current_outer180_full_metric=None,
             current_outer180_wall_seconds=None, current_outer180_training_variances=None, current_outer180_reward=None,
             current_outer180_full_metric_pending=True, current_outer180_full_metric_processed_opportunities=None,
             current_outer180_full_metric_skipped_opportunities=None, current_outer180_full_metric_successful_writes=None,
             current_full_field_boundary_scope='Existing completed_outer/current_outer/current metric, timing, reward and variance retain FULL179 definitions. Explicit actual_optimizer_completed_outer/native saved fields are180. Active evaluation180; no live181 entry claim.',
             current_pending_stage_rank_opportunity_scope=pending['rank_opportunity_and_guard_skip_fields'],
             actual_observations=observations, current_pending_stage_log_growth=growth,
             current_pending_stage_source=descriptor(source, source_raw), current_pending_stage_input_pins=descriptor(pins_path, pins_raw),
             current_pending_stage_evidence=pending_pin,
             current_native_manifest_history=r2['state'], current_resource_observation=r2pin,
             current_systemd=p2['systemd'], last_rank_observation=p2pin['path'],
             observed_epoch=p2['epoch'], last_observed_epoch=p2['epoch'], last_confirmed_epoch=p2['epoch'],
             last_actual_poll_epoch=p2['epoch'], last_actual_poll=p2pin['path'], last_actual_poll_sha256=p2pin['sha256'],
             last_current_progress_poll=p2pin['path'], last_current_progress_poll_epoch=p2['epoch'],
             last_confirmed_resource_epoch=r2['epoch'], same_instance_confirmed=True,
             current_observation_unknown=False, current_remote_health_known=True,
             current_remote_state='KNOWN_AT_ACTUAL_BOUND_POLL', current_observation_training_action='none',
             current_observation_error=None, current_observation_failure_evidence=None,
             last_nonterminal_service_CPU_seconds=observations[-1]['service_CPU_seconds'],
             last_nonterminal_service_CPUUsageNSec=int(c.state(r2['training_systemd'])['CPUUsageNSec']),
             current_pending_dev180_collector_main_invocations=0, current_pending_dev180_collector_remote_incremental_invocations=0,
             current_pending_dev180_collector_scope='No180 collector invocation. Existing source1ed executor-attested cumulative main4/remote4/PENDING0 is unchanged.',
             updated_at=now.isoformat(), recorded_at_utc=now.isoformat(),
             health='Healthy original PID503255/Invocationa313bda919834d24b4c5d491698d39ce; optimizer/native180 saved, dev180 pending, FULL179.')
        output = {}
        changed = []
        for path, old in zip(paths, old_values):
            new = copy.deepcopy(old)
            archive_key = 'historical_current_fields_before_optimizer180_native180_dev_pending'
            need(archive_key not in new, 'Prior180 pending archive already exists')
            replaced = {key: copy.deepcopy(old[key]) for key, value in common.items()
                        if key in old and old[key] != value}
            new[archive_key] = replaced
            new.update(copy.deepcopy(common))
            need({key: value for key, value in old.items() if key not in common}
                 == {key: value for key, value in new.items() if key not in common and key != archive_key},
                 'Protected canonical field changed')
            need(new['completed_outer'] == new['current_outer'] == new['last_complete_full_metric_outer'] == 179
                 and new['active_outer'] == 180 and new['actual_optimizer_completed_outer'] == 180,
                 'FULL179 versus optimizer180 field definition differs')
            raw = encoded(new)
            output[path] = raw
            if path != paths[0]:
                for mirror in ms:
                    output[mirror / path.name] = raw
            changed.append(dict(path=str(path), changed_current_keys=sorted(replaced), added_keys=sorted(set(common) - set(old))))
        need(output[paths[0]] == output[paths[1]], 'Updated status aliases differ')
        report_note = ('\nActual optimizer/native180 complete; periodic dev180 remains pending. Actual native credited Adam by rank=' + json.dumps(completed, sort_keys=True) + '; actual totals=' + json.dumps(totals, sort_keys=True) + '; child credited calls=' + json.dumps({key: value -100 for key,value in completed.items()}, sort_keys=True) + ' after actual cold-loaded native5/100 each. Native saved driver cursor181 does not claim live181 entry. '
             'Completed/current/FULL metric fields retain FULL179, with COST/CUM179 immutable. Current180 metric/wall/reward/variance and dev180 result remain null. Latest complete dev175: F1=56.552424355158735%, EM=47.65625% (512=NQ256+Hot256); curve175 and HF150 remain unchanged. '
             'Source1ed confirmed cumulative main4/remote4/PENDING0 remains unchanged;180 collector calls0. Original PID503255/Invocationa313bda919834d24b4c5d491698d39ce remains active with two bound advancing polls and zero OOM. Goal active; no new training, restore, evaluation, collector or remote operation. '
             'Pending evidence: ' + pending_path.name + '; original14 documents retained in ' + str(snapshot / 'SNAPSHOT_MANIFEST.json') + '.\n')
        report_raw = prior[reports[0]] + report_note.encode()
        for path in reports:
            output[path] = report_raw
        need(set(output) == set(allp), 'Only the closed14 canonical/report paths may be patched')

        def inputs_unchanged():
            need(not any((location / 'FULL_OUTER180_PROGRESS_RECEIPT.json').exists() for location in [b] + ms),
                 'FULL180 already exists; refuse obsolete pending publication')
            need(source.read_bytes() == source_raw and pins_path.read_bytes() == pins_raw
                 and original.read_bytes() == original_raw and sha(cp.read_bytes()) == CORE_SHA,
                 'Source/pins/dependency changed')
            need(all(path.read_bytes() == raw for path, raw in retained_raw.items()), 'Bound evidence changed')

        inputs_unchanged()
        need(all(path.read_bytes() == raw for path, raw in prior.items()), 'Concurrent canonical writer detected')
        snapshot.mkdir()
        members = []
        for index, (path, raw) in enumerate(prior.items()):
            stored_path = snapshot / (f'{index:02d}-' + path.name + '.gz')
            stored = gzip.compress(raw, mtime=0)
            stored_path.write_bytes(stored)
            need(stored_path.read_bytes() == stored and gzip.decompress(stored) == raw,
                 'Exact prior canonical gzip readback differs')
            members.append(dict(original_path=str(path), original_bytes=len(raw), original_sha256=sha(raw),
                 stored_gzip_path=str(stored_path), stored_bytes=len(stored), stored_sha256=sha(stored),
                 gzip_mtime=0, exact_original_bytes_readback=True))
        manifest_path = snapshot / 'SNAPSHOT_MANIFEST.json'
        manifest_raw = encoded(dict(files=members, created_at_utc=now.isoformat(), closed_original_documents=14))
        manifest_path.write_bytes(manifest_raw)
        need(manifest_path.read_bytes() == manifest_raw, 'Snapshot manifest readback differs')
        inputs_unchanged()
        need(all(path.read_bytes() == raw for path, raw in prior.items()), 'Concurrent writer before canonical replace')
        for path, raw in output.items():
            atomic(path, raw)
        for root in [b] + ms:
            atomic(root / pending_path.name, pending_raw)
        readbacks = []
        for index in range(2):
            inputs_unchanged()
            need(all(path.read_bytes() == raw for path, raw in output.items()), 'Canonical/report output readback differs')
            need(all((root / pending_path.name).read_bytes() == pending_raw for root in [b] + ms),
                 'Pending receipt mirror readback differs')
            for member in members:
                stored = Path(member['stored_gzip_path']).read_bytes()
                raw = gzip.decompress(stored)
                need(len(stored) == member['stored_bytes'] and sha(stored) == member['stored_sha256']
                     and len(raw) == member['original_bytes'] and sha(raw) == member['original_sha256'],
                     'Retained closed14 gzip original changed')
            readbacks.append(dict(pass_number=index + 1, observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 canonical_documents=14, input_bytes_unchanged=True, mirrors_byte_equal=True,
                 closed14_original_gzip_bytes_SHA_verified=True))
            if index == 0:
                time.sleep(1)
        closure = dict(schema='outer180_pending_canonical_local_publication_v1', status=PHASE,
             created_at_utc=now.isoformat(), source=descriptor(source, source_raw), execution_input_pins=descriptor(pins_path, pins_raw),
             historical_successful170_source=descriptor(original, original_raw), pinned_core_sha256=CORE_SHA,
             pending_evidence=pending_pin, actual_observations=observations,
             canonical_FULL_boundary=179, actual_optimizer_completed_outer=180, latest_saved_native_outer=180,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             active_outer=180, native_saved_driver_next_step=181, live181_entry_claimed=False,
             actual_cold_loaded_native_outer=5, current_dev180_result=None, latest_completed_dev_outer=175,
             curve_complete_through=175, immutable_cost_and_cumulative_through=179, HF_permanent_milestone=150,
             source1ed_actual_cumulative_main_invocations=4, source1ed_actual_cumulative_remote_invocations=4,
             dev180_new_collector_main_invocations=0, dev180_new_remote_collector_invocations=0,
             prior_closed14_gzip_manifest=descriptor(manifest_path, manifest_raw),
             before=[descriptor(path, raw) for path, raw in prior.items()],
             after=[descriptor(path, raw) for path, raw in output.items()],
             current_changes=changed, actual_local_readback_observations=readbacks,
             protected_canonical_fields_equal=True, protected_immutable_evidence_unchanged=True,
             raw_observation_scope='BASE-only real bytes/SHA; no raw mirror publication claim',
             global_atomic_transaction=False, training_or_remote_mutation=False,
             independent_cold_reload_performed=False, tensor_contents_read=False,
             score_updated=False, FULL180_or_dev180_score_claimed=False, goal_status='active', goal_complete=False)
        closure_raw = encoded(closure)
        for root in [b] + ms:
            atomic(root / closure_path.name, closure_raw)
        for index in range(2):
            inputs_unchanged()
            need(all(path.read_bytes() == raw for path, raw in output.items())
                 and all((root / closure_path.name).read_bytes() == closure_raw
                         and (root / pending_path.name).read_bytes() == pending_raw for root in [b] + ms),
                 'Final actual receipt/canonical two-pass readback differs')
            if index == 0:
                time.sleep(1)
        print(json.dumps(dict(pending180=pending_pin, publication=descriptor(closure_path, closure_raw),
             closed14_snapshot=descriptor(manifest_path, manifest_raw), canonical_FULL=179,
             actual_optimizer_native=180, dev180_result=None, live181_claimed=False,
             two_actual_readbacks=True, training_or_remote_mutation=False, goal_status='active')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='DEV180_PENDING_STAGE_NO_SUCCESS_RECEIPT', error=str(error),
                             remote_actions=False)), file=sys.stderr)
        raise SystemExit(2)
