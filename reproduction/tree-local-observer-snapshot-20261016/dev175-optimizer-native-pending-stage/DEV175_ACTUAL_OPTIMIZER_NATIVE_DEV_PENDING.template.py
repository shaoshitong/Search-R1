"""Frozen local source template; saving this file does not execute a publication.

Root executes only after selecting a real later poll/resource and capturing the
14 then-current canonical/report pins. No unknown poll or digest is defaulted.

Invocation (all size/hash arguments are root's actual frozen input values):
  python3 -B THIS_FILE --workspace WORKSPACE --pins ACTUAL_PINS_JSON \
    --pins-bytes ACTUAL_BYTES --pins-sha256 ACTUAL_SHA \
    --source-bytes ACTUAL_SOURCE_BYTES --source-sha256 ACTUAL_SOURCE_SHA

ACTUAL_PINS_JSON schema: dev175_pending_stage_execution_pins_v1
  second_poll: {file: NATIVE5_CONTINUE_POLL_<actual>.json, bytes: int, sha256: str}
  second_resource: {file: NATIVE5_CONTINUE_RESOURCE_<same actual>.json,
                    bytes: int, sha256: str}
  canonical_before: [{path: <absolute real path>, bytes: int, sha256: str}, ...]
canonical_before must contain exactly the existing 14 paths constructed below.

Derivation: the immutable successful 170 pending-stage command (6913 bytes,
070304de4dec953422dcddafc965fa1d1167f7cdf6d498700ff8eedc370e0dae).
Its validate_native/unit, finite/nonzero rank, skip, no FULL/dev/live-next and
two-observation checks are retained. Raw poll/resource bytes remain BASE-only.
The added canonical patch preserves the existing FULL174 meaning of completed/
current fields and explicitly advances only actual optimizer/native fields175.
FULL/COST/CUM174, latest dev/curve170, HF150, actual cold load5 and collector
main3/remote3/PENDING0 remain protected; current175 metric/score stays unknown.
There is no remote call, tensor read, collector, evaluation or training action.
This source has not been executed, compiled, tested or reviewed by its preparer.
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
KNOWN = {
    'NATIVE5_CONTINUE_POLL_2219.json': (649026, '1ea6e680bb8a1785d2a314149727bb8259b2d1c1a8e5b430c83e7de5efe5828f'),
    'NATIVE5_CONTINUE_RESOURCE_2219.json': (19270, '419e7a6b3d8c0555ad20f485a346987cabdab23ed01f9b34af5d95c3bdccf7da'),
    'NATIVE_HISTORY_175_METADATA_READBACK.json': (8355, 'a9695084fd4edccefcca87c74e7c84696f98f10326f715d68e0b5fd1645ec099'),
    'FULL_OUTER174_PROGRESS_RECEIPT.json': (29678, '67a083bab1e057b93186192d32b8c3a281500eef944bd1c85b34165c4ed80cce'),
    'OUTER174_FULL_NEXT175_ACTUAL_PROGRESS.json': (10849, '9b1ec63af70bc40fd658332b0d493be8cff3498eba7ef8b79bb37c97467113ca'),
    'COST_OUTER174_LOCAL_RECEIPT.json': (65357, 'a9d0846517ff4b3c00b243389c6a0a6fefcc1120750f30c50a35276b2fd38cdb'),
    'CUMULATIVE_COST_THROUGH174.json': (78546, '367d527b02acc5df61fbb80d16d438dafa86c3b66c883eda112690fa5b4f6faa'),
    'ROOT_FULL174_PUBLICATION_READBACK_2193_2195.json': (18057, '53cb6eacab21a4a7393c875542b88e66e9549da2f831f8181baac3699ff35c9a'),
}
MANIFEST175 = '9a2ed6963513cee2c0032540360802d40dd06cc3e8300708b3fcb74898a0bd92'
ARCHIVE175 = '88947503c014c11842c84857659ec6e5c51cd1ebcd1f0352588d93f1bc1a5a7c'
EXPECTED_COUNTS = {str(rank): 3500 for rank in range(8)}
PHASE = 'ACTUAL_OPTIMIZER175_NATIVE175_COMPLETE_DEV175_PENDING_FULL174'


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
    temporary = path.with_name(path.name + '.175-pending.tmp')
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
    need(pins.get('schema') == 'dev175_pending_stage_execution_pins_v1', 'Input-pin schema differs')
    cp = w / CORE_REL
    need(sha(cp.read_bytes()) == CORE_SHA, 'Original676d validator source changed')
    spec = importlib.util.spec_from_file_location('pending175_core', cp)
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
                    q = mirror / path.name
                    need(q.read_bytes() == raw, 'Published input mirror differs: ' + str(q))
                    retained_raw[q] = raw
            ref = descriptor(path, raw)
            refs[str(path)] = ref
            return json.loads(raw), ref

        def selected(key, prefix):
            pin = pins.get(key) or {}
            name = pin.get('file')
            need(isinstance(name, str) and re.fullmatch(prefix + r'\d+\.json', name),
                 'Explicit real second observation filename required: ' + key)
            return bound(b / name, pin.get('bytes'), pin.get('sha256')), name

        (p2, p2pin), p2name = selected('second_poll', 'NATIVE5_CONTINUE_POLL_')
        (r2, r2pin), r2name = selected('second_resource', 'NATIVE5_CONTINUE_RESOURCE_')
        second = re.fullmatch(r'NATIVE5_CONTINUE_POLL_(\d+)\.json', p2name).group(1)
        need(r2name == 'NATIVE5_CONTINUE_RESOURCE_' + second + '.json' and int(second) > 2219,
             'The real second poll/resource must be a matching later pair')
        raw_names = {'NATIVE5_CONTINUE_POLL_2219.json', 'NATIVE5_CONTINUE_RESOURCE_2219.json', p2name, r2name}
        values = {}
        for name, (size, digest) in KNOWN.items():
            values[name], _ = bound(b / name, size, digest, mirrored=name not in raw_names)
        p1 = values['NATIVE5_CONTINUE_POLL_2219.json']
        r1 = values['NATIVE5_CONTINUE_RESOURCE_2219.json']
        native = values['NATIVE_HISTORY_175_METADATA_READBACK.json']
        need(native['manifest_sha256'] == MANIFEST175 and native['receipt_sha256'] == ARCHIVE175
             and native.get('tensor_contents_read') is False
             and native.get('independent_cold_reload_performed') is False, 'Native175 metadata/scope differs')
        record, guard, completed, totals = c.validate_native(native, p1['progress'], 175)
        need(completed == totals == EXPECTED_COUNTS and record['driver']['next_step'] == 176,
             'Actual175 writer totals/cursor differ')
        observations = []
        for poll, resource in [(p1, r1), (p2, r2)]:
            ps, rs, active = c.validate_unit(poll, resource, 175)
            c.validate_restore(poll)
            need(all(poll.get(k) == v for k, v in dict(source_commit=c.SOURCE, unit=c.UNIT,
                 job_id=c.JOB, run=c.RUN, node='h100-10').items()), 'Actual training identity differs')
            need(active and poll['progress']['actual_completed_outer_from_attempt_results'] == 175
                 and not poll['progress'].get('errors') and poll.get('run_result') is None,
                 'Actual175 instance is failed, pending or already beyond175')
            need(c.counts(poll['progress']['completed_actual_optimizer_calls_per_rank'], 'actual175') == completed,
                 'Actual175 credited counters differ')
            need(poll['progress']['training_attempts_seen'] == guard['training_attempts']
                 and poll['progress']['all_skipped_attempts'] == guard['refused_attempts'], 'Guard/skip differs')
            need(max(metric['step'] for metric in poll['iteration_metrics']) == 174, 'FULL175 already exists')
            need(not any(dev.get('outer') == 175 and dev.get('status') == 'VALIDATION_COST_COMPLETE'
                         for dev in poll['development']), 'Dev175 result already exists')
            log = poll['logs']['stage.log']
            need(re.search(r'epoch\s+\d+,\s*step\s+176\b', log) is None, 'Live176 already entered')
            need('TREE_ROLLING_NATIVE_STATE_SAVED=' + c.RUN
                 + '/checkpoints/training_state/rolling-completed/global_step_175/manifest.json' in log,
                 'Original saved175 stage log marker absent')
            state = resource['state']
            need(175 in state['indexed_completed_outers'] and state['audit_error_count'] == 0
                 and not state.get('last_errors') and state.get('training_source_commit') == c.SOURCE
                 and state.get('training_unit') == c.UNIT and state.get('run_directory') == c.RUN
                 and state.get('expected_training_invocation_id') == c.INVOCATION,
                 'Actual175 watcher binding/health differs')
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
        need(all(rank['outer_step'] == rank['attempt_id'] == 175
                 and rank['successful_optimizer_calls'] > 0
                 and rank['finite_grad_norms'] >= rank['successful_optimizer_calls']
                 and rank['nonzero_grad_norms'] > 0
                 and rank['local_shard_sample_delta']['finite'] == rank['local_shard_sample_delta']['count']
                 and rank['local_shard_sample_delta']['nonzero'] > 0 for rank in ranks),
             'Original actual finite/nonzero rank evidence incomplete')
        latest = max((dev for dev in p2['development'] if dev['status'] == 'VALIDATION_COST_COMPLETE'),
                     key=lambda dev: dev['outer'])
        need(latest['outer'] == 170 and latest['rows'] == 512
             and latest['metrics']['val/f1/all'] == 0.5742947048611112
             and latest['metrics']['val/em/all'] == 0.490234375, 'Latest real dev170 score differs')
        full = values['FULL_OUTER174_PROGRESS_RECEIPT.json']
        cumulative = values['CUMULATIVE_COST_THROUGH174.json']
        root174 = values['ROOT_FULL174_PUBLICATION_READBACK_2193_2195.json']
        need(full['completed_outer'] == 174 and full['latest_dev512_outer'] == 170
             and cumulative['through_effective_outer'] == 174
             and root174['source1ed_confirmed_actual_main_invocations_cumulative'] == 3
             and root174['source1ed_confirmed_actual_remote_incremental_invocations_cumulative'] == 3,
             'Protected FULL/CUM174/dev170/collector3 boundaries differ')
        # Retain and reread original cost-field parts; no rewrite or new accounting.
        for value in cumulative.values():
            if isinstance(value, dict) and value.get('storage') == 'complete_original_field_parts':
                for part in value['parts']:
                    name = part.get('file') or part.get('path')
                    path = Path(name)
                    path = path if path.is_absolute() else b / path
                    need(path.parent.resolve() == b.resolve(), 'Cost part outside existing BASE')
                    bound(path, part['bytes'], part['sha256'], mirrored=True)

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
            need(old['completed_outer'] == old['current_outer'] == old['last_complete_full_metric_outer'] == 174
                 and old['active_outer'] == 175 and old['actual_optimizer_completed_outer'] == 174
                 and old['latest_native_checkpoint_outer'] == 174 and old['goal_status'] == 'active'
                 and old.get('goal_complete') is not True, 'Prior canonical stage already changed')
            need(old['latest_dev512_outer'] == old['learning_curve_complete_through_outer'] == 170
                 and old['dev_native_cost_actual_last_collected_outer'] == 170
                 and old['actual_current_restore_outer'] == 5
                 and old['actual_current_restore_adam_per_rank'] == {str(rank): 100 for rank in range(8)}
                 and old['dev_native_cost_collector_new_entry_actual_main_invocations'] == 3
                 and old['dev_native_cost_collector_actual_remote_incremental_invocations'] == 3
                 and old['dev_native_cost_collector_preceding_local_pending_invocations'] == 0,
                 'Protected dev/curve/cold-load/collector boundary differs')
            need(old['current_HF_permanent_milestone150_complete'] is True
                 and old['current_dev512'] is None and old['actual_dev512'] is None,
                 'HF150 or existing null current dev differs')
            epoch_fields = ('observed_epoch', 'last_observed_epoch', 'last_confirmed_epoch',
                            'last_actual_poll_epoch', 'last_current_progress_poll_epoch')
            need(all((old.get(key) or 0) <= p2['epoch'] for key in epoch_fields), 'Newer canonical epoch exists')

        suffix = '2219_' + second
        pending_path = b / ('OUTER175_UPDATES_NATIVE_DEV_PENDING_ACTUAL_' + suffix + '.json')
        closure_path = b / ('OUTER175_PENDING_CANONICAL_LOCAL_PUBLICATION_' + suffix + '.json')
        need(all(not (root / path.name).exists() for root in [b] + ms
                 for path in [pending_path, closure_path]), 'Retain existing pending/publication receipts')
        now = datetime.datetime.now(datetime.timezone.utc)
        snapshot = b / ('outer175_optimizer_native_dev_pending_prior_' + now.strftime('%Y%m%dT%H%M%S%fZ'))
        pending = dict(schema='nested_dev175_verified_wait_v1', status='ACTUAL_NATIVE175_DEV175_RESULT_PENDING',
             created_utc=now.isoformat(), goal_status='active', source_commit=c.SOURCE, config_sha256=c.CONFIG,
             unit=c.UNIT, run=c.RUN, original_PID=int(c.PID), invocation_id=c.INVOCATION,
             actual_observations=observations, actual_optimizer_completed_outer=175,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             child_successful_adam_per_rank={key: value - 100 for key, value in completed.items()},
             actual_outer_successful_writes_per_rank={str(rank['rank']): rank['successful_optimizer_calls'] for rank in ranks},
             original_rank_records=ranks,
             rank_opportunity_and_guard_skip_fields='Original processed/guard-skip fields absent and unknown; skipped_empty_minibatches is separate. No missing zero inferred.',
             latest_saved_native_outer=175, native_manifest_sha256=MANIFEST175,
             native_archive_receipt_sha256=ARCHIVE175, actual_cold_loaded_native_outer=5,
             independent_cold_reload_performed=False, tensor_contents_read=False, native_saved_driver_next_step=176,
             saved_cursor176_is_not_live_outer176=True, current_outer175_full_metric=None,
             current_outer175_wall_seconds=None, current_outer175_training_variances=None, current_outer175_reward=None,
             dev175_result=None, latest_completed_dev512=latest, score_updated=False,
             stage_and_retriever_log_growth=growth, input_refs=copy.deepcopy(refs),
             raw_observation_sources_scope='Original selected poll/resource files are BASE-only actual bytes/SHA-bound sources. Native175 and prior FULL/progress/COST/CUM/root174 products retain three-location two-pass byte readbacks.',
             raw_observation_files=sorted(raw_names), canonical_FULL_boundary=174, canonical_publication_updated=False,
             outer176_entered_claimed=False, whole_goal_complete=False, training_or_remote_mutation=False,
             scope='Actual175 optimizer/native saved; dev175 and FULL175 pending. FULL/COST/CUM174 and latest dev/curve170 remain previous complete boundaries. This evidence receipt itself does not mutate canonical; the separate local publication receipt records that patch.')
        pending_raw = encoded(pending)
        pending_pin = descriptor(pending_path, pending_raw)
        common = dict(status=PHASE, phase=PHASE, current_phase=PHASE, training_phase=PHASE,
             current_observation_status=PHASE, observed_current_training_phase=PHASE,
             actual_optimizer_completed_outer=175, last_confirmed_optimizer_outer=175,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             successful_adam_per_rank=completed, per_rank_successful_adam=completed,
             per_rank_actual_total_adam=totals,
             child_successful_adam_per_rank={key: value - 100 for key, value in completed.items()},
             actual_physical_rank_optimizer_calls=sum(totals.values()),
             actual_outer_successful_writes_per_rank=pending['actual_outer_successful_writes_per_rank'],
             actual_outer_processed_opportunities_per_rank={str(rank): None for rank in range(8)},
             actual_outer_skipped_opportunities_per_rank={str(rank): None for rank in range(8)},
             actual_training_progress=p2['progress'], all_skipped_attempts=p2['progress']['all_skipped_attempts'],
             latest_native_checkpoint_outer=175, last_confirmed_native_saved_outer=175, latest_saved_native_outer=175,
             native_saved_driver_next_step=176, native_saved_cursor_is_not_live_next_outer_entry=True,
             native_manifest_sha256=MANIFEST175, latest_native_manifest_sha256=MANIFEST175,
             native_manifest_history_receipt_sha256=ARCHIVE175,
             latest_native_metadata_scope='Actual175 writer/archive JSON metadata only; actual cold load stays5; no tensor hash/read/reload.',
             actual_current_restore_scope='Actual cold-loaded5/100 each remains unchanged; writer-complete native175 is a distinct saved boundary.',
             successful_adam_per_rank_scope='Actual credited native5 restore plus direct child6..175; failed ancestor calls excluded; FULL accounting remains174.',
             child_successful_adam_per_rank_scope='Actual cumulative credited175 minus actual restored100 each; no cold restore175 or completed176 claim.',
             current_actual_optimizer_progress=pending_pin, current_live_validation_progress=pending_pin,
             optimizer175_completion_before_periodic_dev=pending_pin, dev175_result=None,
             dev175_result_pending=True, normal_dev512_outer175_pending=True, normal_dev512_pending=True,
             current_dev512_outer=175, current_dev_boundary_scope='Current175 evaluation pending/null; latest complete dev170 F1/EM and curve170 remain unchanged.',
             current_active_outer_full_metric_pending=True, active_outer_optimizer_completion_confirmed=True,
             active_outer_full_update_unconfirmed=False, current_outer175_full_metric=None,
             current_outer175_wall_seconds=None, current_outer175_training_variances=None, current_outer175_reward=None,
             current_outer175_full_metric_pending=True,
             current_full_field_boundary_scope='Existing completed_outer/current_outer/current metric, timing, reward and variance retain FULL174 definitions. Explicit actual_optimizer_completed_outer/native saved fields are175. Active evaluation175; no live176 entry claim.',
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
             current_pending_dev175_collector_main_invocations=0, current_pending_dev175_collector_remote_incremental_invocations=0,
             current_pending_dev175_collector_scope='No175 collector invocation. Existing source1ed executor-attested cumulative main3/remote3/PENDING0 is unchanged.',
             updated_at=now.isoformat(), recorded_at_utc=now.isoformat(),
             health='Healthy original PID503255/Invocationa313bda919834d24b4c5d491698d39ce; optimizer/native175 saved, dev175 pending, FULL174.')
        output = {}
        changed = []
        for path, old in zip(paths, old_values):
            new = copy.deepcopy(old)
            archive_key = 'historical_current_fields_before_optimizer175_native175_dev_pending'
            need(archive_key not in new, 'Prior175 pending archive already exists')
            replaced = {key: copy.deepcopy(old[key]) for key, value in common.items()
                        if key in old and old[key] != value}
            new[archive_key] = replaced
            new.update(copy.deepcopy(common))
            need({key: value for key, value in old.items() if key not in common}
                 == {key: value for key, value in new.items() if key not in common and key != archive_key},
                 'Protected canonical field changed')
            need(new['completed_outer'] == new['current_outer'] == new['last_complete_full_metric_outer'] == 174
                 and new['active_outer'] == 175 and new['actual_optimizer_completed_outer'] == 175,
                 'FULL174 versus optimizer175 field definition differs')
            raw = encoded(new)
            output[path] = raw
            if path != paths[0]:
                for mirror in ms:
                    output[mirror / path.name] = raw
            changed.append(dict(path=str(path), changed_current_keys=sorted(replaced), added_keys=sorted(set(common) - set(old))))
        need(output[paths[0]] == output[paths[1]], 'Updated status aliases differ')
        report_note = ('\nActual optimizer/native175 complete; periodic dev175 remains pending. Eight ranks each have3500 actual credited Adam calls; child calls each3400 after actual cold-loaded native5/100 each. Native saved driver cursor176 does not claim live176 entry. '
             'Completed/current/FULL metric fields retain FULL174, with COST/CUM174 immutable. Current175 metric/wall/reward/variance and dev175 result remain null. Latest complete dev170: F1=57.42947048611112%, EM=49.0234375% (512=NQ256+Hot256); curve170 and HF150 remain unchanged. '
             'Source1ed confirmed cumulative main3/remote3/PENDING0 remains unchanged;175 collector calls0. Original PID503255/Invocationa313bda919834d24b4c5d491698d39ce remains active with two bound advancing polls and zero OOM. Goal active; no new training, restore, evaluation, collector or remote operation. '
             'Pending evidence: ' + pending_path.name + '; original14 documents retained in ' + str(snapshot / 'SNAPSHOT_MANIFEST.json') + '.\n')
        report_raw = prior[reports[0]] + report_note.encode()
        for path in reports:
            output[path] = report_raw
        need(set(output) == set(allp), 'Only the closed14 canonical/report paths may be patched')

        def inputs_unchanged():
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
        closure = dict(schema='outer175_pending_canonical_local_publication_v1', status=PHASE,
             created_at_utc=now.isoformat(), source=descriptor(source, source_raw), execution_input_pins=descriptor(pins_path, pins_raw),
             historical_successful170_source=descriptor(original, original_raw), pinned_core_sha256=CORE_SHA,
             pending_evidence=pending_pin, actual_observations=observations,
             canonical_FULL_boundary=174, actual_optimizer_completed_outer=175, latest_saved_native_outer=175,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             active_outer=175, native_saved_driver_next_step=176, live176_entry_claimed=False,
             actual_cold_loaded_native_outer=5, current_dev175_result=None, latest_completed_dev_outer=170,
             curve_complete_through=170, immutable_cost_and_cumulative_through=174, HF_permanent_milestone=150,
             source1ed_actual_cumulative_main_invocations=3, source1ed_actual_cumulative_remote_invocations=3,
             dev175_new_collector_main_invocations=0, dev175_new_remote_collector_invocations=0,
             prior_closed14_gzip_manifest=descriptor(manifest_path, manifest_raw),
             before=[descriptor(path, raw) for path, raw in prior.items()],
             after=[descriptor(path, raw) for path, raw in output.items()],
             current_changes=changed, actual_local_readback_observations=readbacks,
             protected_canonical_fields_equal=True, protected_immutable_evidence_unchanged=True,
             raw_observation_scope='BASE-only real bytes/SHA; no raw mirror publication claim',
             global_atomic_transaction=False, training_or_remote_mutation=False,
             independent_cold_reload_performed=False, tensor_contents_read=False,
             score_updated=False, FULL175_or_dev175_score_claimed=False, goal_status='active', goal_complete=False)
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
        print(json.dumps(dict(pending175=pending_pin, publication=descriptor(closure_path, closure_raw),
             closed14_snapshot=descriptor(manifest_path, manifest_raw), canonical_FULL=174,
             actual_optimizer_native=175, dev175_result=None, live176_claimed=False,
             two_actual_readbacks=True, training_or_remote_mutation=False, goal_status='active')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='DEV175_PENDING_STAGE_NO_SUCCESS_RECEIPT', error=str(error),
                             remote_actions=False)), file=sys.stderr)
        raise SystemExit(2)
