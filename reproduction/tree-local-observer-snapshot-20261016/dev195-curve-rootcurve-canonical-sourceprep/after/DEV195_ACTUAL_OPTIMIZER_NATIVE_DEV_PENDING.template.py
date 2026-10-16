"""Frozen195 local pending producer; source preparation alone performs no publication.

Derived from published190 producer44127B SHA2abdb4ab6a7b8fbf6fd1d9dc76bcbcb63265d592a94960ff8476895ba20a6737; commitdeecf5becd3f8e9045b6d3fa0a497b3095048353; future195 invocation only.
Scope stays actual optimizer195/native195 writer COMPLETE while dev195/FULL195
are still absent. Saved cursor196 is not actual training196 or a cold reload.
Native/guard/rank records provide real counts; no3900 total is hardcoded.
Missing FULL195 and rank opportunity fields remain unknown/null.
FULL/COST/CUM194, latest dev/curve190, HF150, actual cold load5 and source1ed
cumulative main7/remote7/PENDING0 remain protected. No SSH, collector, eval or
training operation is present. Root must skip this entry after FULL195 or a
completed dev195 is observed. This source is unexecuted/uncompiled/untested.

Future invocation, only with actual frozen root-selected values:
  python3 -B THIS_FILE --workspace WORKSPACE --pins ACTUAL_PINS_JSON \
    --pins-bytes ACTUAL_BYTES --pins-sha256 ACTUAL_SHA \
    --source-bytes ACTUAL_SOURCE_BYTES --source-sha256 ACTUAL_SOURCE_SHA

ACTUAL_PINS_JSON schema: dev195_pending_stage_execution_pins_v1
Required descriptors (file,bytes,sha256): first_poll,first_resource,
second_poll,second_resource,native195,prior_full,prior_root_full.
prior_full must be the real FULL_OUTER194_PROGRESS_RECEIPT.json. Prior COST/CUM194
bytes/SHA are derived only from the real bound ROOT_FULL194 receipt; none are defaulted.
Also require actual native_manifest_sha256/native_receipt_sha256 and the then
current canonical_before list naming exactly the14 existing paths below.
No future195 file names, digests, native counts or success are defaulted.
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
PHASE = 'ACTUAL_OPTIMIZER195_NATIVE195_COMPLETE_DEV195_PENDING_FULL194'


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
    temporary = path.with_name(path.name + '.195-pending.tmp')
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
    need(pins.get('schema') == 'dev195_pending_stage_execution_pins_v1', 'Input-pin schema differs')
    cp = w / CORE_REL
    need(sha(cp.read_bytes()) == CORE_SHA, 'Original676d validator source changed')
    spec = importlib.util.spec_from_file_location('pending195_core', cp)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)

    # Use the same cooperative publisher lock as the existing FULL alias writer.
    with cp.open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs, retained_raw = {}, {}

        def bound(path, size, digest, mirrored=False):
            path = path.resolve()
            need(not mirrored or path.is_relative_to(b.resolve()), 'Mirrored evidence outside actual BASE')
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

        def reject_obsolete_local():
            need(not any((location / 'FULL_OUTER195_PROGRESS_RECEIPT.json').exists() for location in [b] + ms),
                 'FULL195 already exists; this pending entry is obsolete')
            for location in [b] + ms:
                control_path = location / 'dev_native_cost_summaries/DEV_OUTER0185_SUMMARY.json'
                if control_path.exists():
                    control = json.loads(control_path.read_bytes())
                    need(control.get('status') != 'VALIDATION_COST_COMPLETE',
                         'Actual complete dev195 control exists; refuse obsolete pending publication')
        reject_obsolete_local()
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
        values = {}
        for key, pattern in [('native195', r'NATIVE_HISTORY_195_METADATA_READBACK\.json'),
                             ('prior_full', r'FULL_OUTER194_PROGRESS_RECEIPT\.json'),
                             ('prior_root_full', r'ROOT_FULL194_PUBLICATION_READBACK_\d+_\d+\.json')]:
            pin = pins.get(key) or {};name = pin.get('file')
            need(isinstance(name, str) and re.fullmatch(pattern, name), 'Explicit actual local pin required: ' + key)
            values[name], _ = bound(b / name, pin.get('bytes'), pin.get('sha256'), mirrored=True)

        def locate_existing_base(name):
            need(isinstance(name, str) and name, 'Actual existing bound path required')
            path = Path(name)
            need('..' not in path.parts, 'Parent path traversal refused')
            if path.is_absolute():
                path = path.resolve()
            elif path.parts[0] in ('outputs', 'work'):
                path = (w / path).resolve()
            else:
                path = (b / path).resolve()
            need(path.is_relative_to(b.resolve()), 'Protected input must stay within actual BASE')
            return path

        root194 = values[pins['prior_root_full']['file']]
        need(root194.get('completed_outer') == 194 and root194.get('active_outer') == 195
             and root194.get('latest_saved_native_outer') == 194
             and root194.get('source_commit') == c.SOURCE and root194.get('main_pid') == int(c.PID)
             and root194.get('invocation_id') == c.INVOCATION
             and isinstance(root194.get('status'), str) and root194['status'].startswith('ACTUAL_')
             and root194['status'].endswith('_VERIFIED'), 'Actual ROOT194 identity/boundary differs')
        prior_full_ref = root194.get('full_receipt') or {}
        prior_full_path = locate_existing_base(prior_full_ref.get('path') or prior_full_ref.get('file'))
        need(prior_full_path == (b / pins['prior_full']['file']).resolve()
             and prior_full_ref.get('bytes') == pins['prior_full']['bytes']
             and prior_full_ref.get('sha256') == pins['prior_full']['sha256'],
             'Actual ROOT194 FULL descriptor differs from the selected FULL194')
        cost_ref = root194.get('cost_receipt') or {}
        cost_path = locate_existing_base(cost_ref.get('path') or cost_ref.get('file'))
        need(cost_path.name == 'COST_OUTER194_LOCAL_RECEIPT.json', 'ROOT194 actual COST boundary differs')
        values[cost_path.name], _ = bound(cost_path, cost_ref.get('bytes'), cost_ref.get('sha256'), mirrored=True)
        cost_files = root194.get('cost_files')
        need(isinstance(cost_files, list), 'Actual ROOT194 dynamic cost descriptors required')
        cumulative_refs = [ref for ref in cost_files if isinstance(ref, dict)
                           and Path(ref.get('path') or ref.get('file') or '').name == 'CUMULATIVE_COST_THROUGH194.json']
        need(len(cumulative_refs) == 1, 'ROOT194 must bind exactly one actual CUM194 descriptor')
        cumulative_ref = cumulative_refs[0]
        cumulative_path = locate_existing_base(cumulative_ref.get('path') or cumulative_ref.get('file'))
        values[cumulative_path.name], _ = bound(cumulative_path, cumulative_ref.get('bytes'),
                                               cumulative_ref.get('sha256'), mirrored=True)
        native = values['NATIVE_HISTORY_195_METADATA_READBACK.json']
        MANIFEST195 = pins.get('native_manifest_sha256');ARCHIVE195 = pins.get('native_receipt_sha256')
        need(re.fullmatch(r'[a-f0-9]{64}', MANIFEST195 or '') and re.fullmatch(r'[a-f0-9]{64}', ARCHIVE195 or ''),
             'Actual native195 manifest and receipt SHA pins required')
        need(native['manifest_sha256'] == MANIFEST195 and native['receipt_sha256'] == ARCHIVE195
             and native.get('tensor_contents_read') is False
             and native.get('independent_cold_reload_performed') is False, 'Native195 metadata/scope differs')
        record, guard, completed, totals = c.validate_native(native, p1['progress'], 195)
        need(record['driver']['next_step'] == 196, 'Actual195 saved writer cursor differs')
        # Preserve the190 repair: only the first pinned poll must retain the native save marker.
        need('TREE_ROLLING_NATIVE_STATE_SAVED=' + c.RUN
             + '/checkpoints/training_state/rolling-completed/global_step_195/manifest.json'
             in p1['logs']['stage.log'], 'Original saved195 stage log marker absent')
        observations = []
        for poll, resource in [(p1, r1), (p2, r2)]:
            ps, rs, active = c.validate_unit(poll, resource, 195)
            c.validate_restore(poll)
            need(all(poll.get(k) == v for k, v in dict(source_commit=c.SOURCE, unit=c.UNIT,
                 job_id=c.JOB, run=c.RUN, node='h100-10').items()), 'Actual training identity differs')
            need(active and poll['progress']['actual_completed_outer_from_attempt_results'] == 195
                 and not poll['progress'].get('errors') and poll.get('run_result') is None,
                 'Actual195 instance is failed, pending or already beyond195')
            need(c.counts(poll['progress']['completed_actual_optimizer_calls_per_rank'], 'actual195') == completed,
                 'Actual195 credited counters differ')
            need(poll['progress']['training_attempts_seen'] == guard['training_attempts']
                 and poll['progress']['all_skipped_attempts'] == guard['refused_attempts'], 'Guard/skip differs')
            need(max(metric['step'] for metric in poll['iteration_metrics']) == 194, 'FULL195 already exists')
            need(not any(dev.get('outer') == 195 and dev.get('status') == 'VALIDATION_COST_COMPLETE'
                         for dev in poll['development']), 'Dev195 result already exists')
            log = poll['logs']['stage.log']
            need(re.search(r'epoch\s+\d+,\s*step\s+196\b', log) is None, 'Live196 already entered')
            state = resource['state']
            need(195 in state['indexed_completed_outers'] and state['audit_error_count'] == 0
                 and not state.get('last_errors') and state.get('training_source_commit') == c.SOURCE
                 and state.get('training_unit') == c.UNIT and state.get('run_directory') == c.RUN
                 and state.get('expected_training_invocation_id') == c.INVOCATION,
                 'Actual195 watcher binding/health differs')
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
        need(all(rank['outer_step'] == 195 and rank['attempt_id'] == guard['training_attempts']
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
                need(rank['scheduler_step'] == 195, 'Explicit actual rank scheduler differs')
            processed_by_rank[key] = processed;skipped_by_rank[key] = skipped
        latest = max((dev for dev in p2['development'] if dev['status'] == 'VALIDATION_COST_COMPLETE'),
                     key=lambda dev: dev['outer'])
        need(latest['outer'] == 190 and latest['rows'] == 512, 'Actual complete prior dev190 required; score is not defaulted')
        full = values['FULL_OUTER194_PROGRESS_RECEIPT.json']
        cumulative = values['CUMULATIVE_COST_THROUGH194.json']
        root194 = values[pins['prior_root_full']['file']]
        need(full['status'] == 'ACTUAL_FULL_OUTER194_NATIVE194_COMPLETE_NEXT195_ENTERED'
             and full['source_commit'] == c.SOURCE and full['config_sha256'] == c.CONFIG
             and values['COST_OUTER194_LOCAL_RECEIPT.json']['outer'] == 194
             and values['COST_OUTER194_LOCAL_RECEIPT.json']['status'] == 'ACTUAL_OUTER194_COST_ARCHIVED'
             and values['COST_OUTER194_LOCAL_RECEIPT.json']['source_commit'] == c.SOURCE
             and values['COST_OUTER194_LOCAL_RECEIPT.json']['config_sha256'] == c.CONFIG
             and full['completed_outer'] == 194 and full['latest_dev512_outer'] == 190
             and cumulative['through_effective_outer'] == 194
             and root194['source1ed_confirmed_actual_main_invocations_cumulative'] == 7
             and root194['source1ed_confirmed_actual_remote_incremental_invocations_cumulative'] == 7,
             'Protected FULL/CUM194/dev190/collector7 boundaries differ')
        need(full['latest_actual_dev512']['outer'] == 190 and full['latest_actual_dev512']['status'] == 'VALIDATION_COST_COMPLETE'
             and latest['metrics'] == full['latest_actual_dev512']['metrics']
             and latest['ledger_sha256'] == full['latest_actual_dev512']['ledger_sha256'],
             'Actual dev190 score/ledger must match bound FULL194; no future score is defaulted')
        def descriptor_nodes(value):
            if isinstance(value, dict):
                if (value.get('path') or value.get('file')) and type(value.get('bytes')) is int and value.get('sha256'):
                    yield value
                else:
                    for child in value.values():yield from descriptor_nodes(child)
            elif isinstance(value, list):
                for child in value:yield from descriptor_nodes(child)
        prior_owner_refs = {}
        for ref in descriptor_nodes(root194):
            name = ref.get('path') or ref.get('file')
            if Path(name).name != 'DEV5_190_NATIVE_COST_LOCAL_RECEIPT.json':continue
            path = locate_existing_base(name)
            need(path not in prior_owner_refs or prior_owner_refs[path] == (ref['bytes'], ref['sha256']), 'Conflicting actual prior190 owner pins')
            prior_owner_refs[path] = (ref['bytes'], ref['sha256'])
        need(len(prior_owner_refs) == 1, 'ROOT194 must bind the actual unique prior190 owner receipt')
        owner_path, (owner_size, owner_sha) = next(iter(prior_owner_refs.items()))
        owner190, _ = bound(owner_path, owner_size, owner_sha, mirrored=True)
        need(owner190['same_entry_executor_attested_cumulative_main_invocations'] == owner190['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == 7,
             'Actual prior190 cumulative source1ed7 chain required')
        historical_owner = owner190
        for expected_outer in (190, 185, 180, 175, 170, 165, 160):
            need(historical_owner['outer'] == expected_outer
                 and historical_owner['collector_source_commit'] == '1edaf50bb8e14de51db936e0859e369792011233'
                 and historical_owner['collector_source_sha256'] == '545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0'
                 and historical_owner['new_entry_main_invocations_this_task'] == historical_owner['actual_remote_incremental_collection_invocations'] == 1
                 and historical_owner['preceding_local_pending_invocations'] == historical_owner['remote_collector_exit_code'] == 0,
                 'Actual prior source1ed owner chain differs')
            if expected_outer != 160:
                ref = historical_owner['same_entry_prior_actual_executor_attestation']
                historical_owner, _ = bound(locate_existing_base(ref.get('path') or ref.get('file')),
                                           ref['bytes'], ref['sha256'], mirrored=True)
        prior_completed = c.counts(full['per_rank_successful_adam'], 'Prior194 actual credited calls')
        need(all(completed[str(rank['rank'])] - prior_completed[str(rank['rank'])]
                 == rank['successful_optimizer_calls'] for rank in ranks),
             'Actual195 native credited delta differs from original rank writes; no3900 total is inferred')
        # Retain all actual protected CUM194 parts at their existing relative paths.
        seen_parts = set()
        def retain_parts(value):
            if isinstance(value, dict):
                if value.get('storage') == 'complete_original_field_parts':
                    for part in value['parts']:
                        name = part.get('file') or part.get('path')
                        path = locate_existing_base(name)
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
            need(old['completed_outer'] == old['current_outer'] == old['last_complete_full_metric_outer'] == 194
                 and old['active_outer'] == 195 and old['actual_optimizer_completed_outer'] == 194
                 and old['latest_native_checkpoint_outer'] == 194 and old['goal_status'] == 'active'
                 and old.get('goal_complete') is not True, 'Prior canonical stage already changed')
            need(old['latest_dev512_outer'] == old['learning_curve_complete_through_outer'] == 190
                 and old['dev_native_cost_actual_last_collected_outer'] == 190
                 and old['actual_current_restore_outer'] == 5
                 and old['actual_current_restore_adam_per_rank'] == {str(rank): 100 for rank in range(8)}
                 and old['dev_native_cost_collector_new_entry_actual_main_invocations'] == 7
                 and old['dev_native_cost_collector_actual_remote_incremental_invocations'] == 7
                 and old['dev_native_cost_collector_preceding_local_pending_invocations'] == 0,
                 'Protected dev/curve/cold-load/collector boundary differs')
            need(old['current_HF_permanent_milestone150_complete'] is True
                 and old['current_dev512'] is None and old['actual_dev512'] is None,
                 'HF150 or existing null current dev differs')
            epoch_fields = ('observed_epoch', 'last_observed_epoch', 'last_confirmed_epoch',
                            'last_actual_poll_epoch', 'last_current_progress_poll_epoch')
            need(all((old.get(key) or 0) <= p2['epoch'] for key in epoch_fields), 'Newer canonical epoch exists')

        suffix = first + '_' + second
        pending_path = b / ('OUTER195_UPDATES_NATIVE_DEV_PENDING_ACTUAL_' + suffix + '.json')
        closure_path = b / ('OUTER195_PENDING_CANONICAL_LOCAL_PUBLICATION_' + suffix + '.json')
        need(all(not (root / path.name).exists() for root in [b] + ms
                 for path in [pending_path, closure_path]), 'Retain existing pending/publication receipts')
        now = datetime.datetime.now(datetime.timezone.utc)
        snapshot = b / ('outer195_optimizer_native_dev_pending_prior_' + now.strftime('%Y%m%dT%H%M%S%fZ'))
        pending = dict(schema='nested_dev195_verified_wait_v1', status='ACTUAL_NATIVE195_DEV195_RESULT_PENDING',
             created_utc=now.isoformat(), goal_status='active', source_commit=c.SOURCE, config_sha256=c.CONFIG,
             unit=c.UNIT, run=c.RUN, original_PID=int(c.PID), invocation_id=c.INVOCATION,
             actual_observations=observations, actual_optimizer_completed_outer=195,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             child_successful_adam_per_rank={key: value - 100 for key, value in completed.items()},
             actual_outer_successful_writes_per_rank={str(rank['rank']): rank['successful_optimizer_calls'] for rank in ranks},
             original_rank_records=ranks, actual_outer_processed_opportunities_per_rank=processed_by_rank,
             actual_outer_skipped_opportunities_per_rank=skipped_by_rank,
             full_metric_processed_opportunities=None, full_metric_skipped_opportunities=None, full_metric_successful_writes=None,
             rank_opportunity_and_guard_skip_fields='Only explicit original per-rank opportunity fields are retained; absent fields stay null. FULL195 processed/skipped/write metrics stay null while FULL195 is absent. skipped_empty_minibatches is separate; no missing zero or20-opportunity count inferred.',
             latest_saved_native_outer=195, native_manifest_sha256=MANIFEST195,
             native_archive_receipt_sha256=ARCHIVE195, actual_cold_loaded_native_outer=5,
             independent_cold_reload_performed=False, tensor_contents_read=False, native_saved_driver_next_step=196,
             saved_cursor196_is_not_live_outer196=True, current_outer195_full_metric=None,
             current_outer195_wall_seconds=None, current_outer195_training_variances=None, current_outer195_reward=None,
             dev195_result=None, latest_completed_dev512=latest, score_updated=False,
             stage_and_retriever_log_growth=growth, input_refs=copy.deepcopy(refs),
             raw_observation_sources_scope='Original selected poll/resource files are BASE-only actual bytes/SHA-bound sources. Native195 and prior FULL/COST/CUM/root194 products retain three-location two-pass byte readbacks.',
             raw_observation_files=sorted(raw_names), canonical_FULL_boundary=194, canonical_publication_updated=False,
             outer196_entered_claimed=False, whole_goal_complete=False, training_or_remote_mutation=False,
             scope='Actual195 optimizer/native saved; dev195 and FULL195 pending. FULL/COST/CUM194 and latest dev/curve190 remain previous complete boundaries. This evidence receipt itself does not mutate canonical; the separate local publication receipt records that patch.')
        pending_raw = encoded(pending)
        pending_pin = descriptor(pending_path, pending_raw)
        common = dict(status=PHASE, phase=PHASE, current_phase=PHASE, training_phase=PHASE,
             current_observation_status=PHASE, observed_current_training_phase=PHASE,
             actual_optimizer_completed_outer=195, last_confirmed_optimizer_outer=195,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             successful_adam_per_rank=completed, per_rank_successful_adam=completed,
             per_rank_actual_total_adam=totals,
             child_successful_adam_per_rank={key: value - 100 for key, value in completed.items()},
             actual_physical_rank_optimizer_calls=sum(totals.values()),
             actual_outer_successful_writes_per_rank=pending['actual_outer_successful_writes_per_rank'],
             actual_outer_processed_opportunities_per_rank=processed_by_rank,
             actual_outer_skipped_opportunities_per_rank=skipped_by_rank,
             actual_training_progress=p2['progress'], all_skipped_attempts=p2['progress']['all_skipped_attempts'],
             latest_native_checkpoint_outer=195, last_confirmed_native_saved_outer=195, latest_saved_native_outer=195,
             native_saved_driver_next_step=196, native_saved_cursor_is_not_live_next_outer_entry=True,
             native_manifest_sha256=MANIFEST195, latest_native_manifest_sha256=MANIFEST195,
             native_manifest_history_receipt_sha256=ARCHIVE195,
             latest_native_metadata_scope='Actual195 writer/archive JSON metadata only; actual cold load stays5; no tensor hash/read/reload.',
             actual_current_restore_scope='Actual cold-loaded5/100 each remains unchanged; writer-complete native195 is a distinct saved boundary.',
             successful_adam_per_rank_scope='Actual credited native5 restore plus direct child6..195; failed ancestor calls excluded; FULL accounting remains194.',
             child_successful_adam_per_rank_scope='Actual cumulative credited195 minus actual restored100 each; no cold restore195 or completed196 claim.',
             current_actual_optimizer_progress=pending_pin, current_live_validation_progress=pending_pin,
             optimizer195_completion_before_periodic_dev=pending_pin, dev195_result=None,
             dev195_result_pending=True, normal_dev512_outer195_pending=True, normal_dev512_pending=True,
             current_dev512_outer=195, current_dev_boundary_scope='Current195 evaluation pending/null; latest complete dev190 F1/EM and curve190 remain unchanged.',
             current_active_outer_full_metric_pending=True, active_outer_optimizer_completion_confirmed=True,
             active_outer_full_update_unconfirmed=False, current_outer195_full_metric=None,
             current_outer195_wall_seconds=None, current_outer195_training_variances=None, current_outer195_reward=None,
             current_outer195_full_metric_pending=True, current_outer195_full_metric_processed_opportunities=None,
             current_outer195_full_metric_skipped_opportunities=None, current_outer195_full_metric_successful_writes=None,
             current_full_field_boundary_scope='Existing completed_outer/current_outer/current metric, timing, reward and variance retain FULL194 definitions. Explicit actual_optimizer_completed_outer/native saved fields are195. Active evaluation195; no live196 entry claim.',
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
             current_pending_dev195_collector_main_invocations=0, current_pending_dev195_collector_remote_incremental_invocations=0,
             current_pending_dev195_collector_scope='No195 collector invocation. Existing source1ed executor-attested cumulative main7/remote7/PENDING0 is unchanged.',
             updated_at=now.isoformat(), recorded_at_utc=now.isoformat(),
             health='Healthy original PID503255/Invocationa313bda919834d24b4c5d491698d39ce; optimizer/native195 saved, dev195 pending, FULL194.')
        output = {}
        changed = []
        for path, old in zip(paths, old_values):
            new = copy.deepcopy(old)
            archive_key = 'historical_current_fields_before_optimizer195_native195_dev_pending'
            need(archive_key not in new, 'Prior195 pending archive already exists')
            replaced = {key: copy.deepcopy(old[key]) for key, value in common.items()
                        if key in old and old[key] != value}
            new[archive_key] = replaced
            new.update(copy.deepcopy(common))
            need({key: value for key, value in old.items() if key not in common}
                 == {key: value for key, value in new.items() if key not in common and key != archive_key},
                 'Protected canonical field changed')
            need(new['completed_outer'] == new['current_outer'] == new['last_complete_full_metric_outer'] == 194
                 and new['active_outer'] == 195 and new['actual_optimizer_completed_outer'] == 195,
                 'FULL194 versus optimizer195 field definition differs')
            raw = encoded(new)
            output[path] = raw
            if path != paths[0]:
                for mirror in ms:
                    output[mirror / path.name] = raw
            changed.append(dict(path=str(path), changed_current_keys=sorted(replaced), added_keys=sorted(set(common) - set(old))))
        need(output[paths[0]] == output[paths[1]], 'Updated status aliases differ')
        report_note = ('\nActual optimizer/native195 complete; periodic dev195 remains pending. Actual native credited Adam by rank=' + json.dumps(completed, sort_keys=True) + '; actual totals=' + json.dumps(totals, sort_keys=True) + '; child credited calls=' + json.dumps({key: value -100 for key,value in completed.items()}, sort_keys=True) + ' after actual cold-loaded native5/100 each. Native saved driver cursor196 does not claim live196 entry. '
             'Completed/current/FULL metric fields retain FULL194, with COST/CUM194 immutable. Current195 metric/wall/reward/variance and dev195 result remain null. Latest complete dev190: F1=' + str(latest['metrics']['val/f1/all'] * 100) + '%, EM=' + str(latest['metrics']['val/em/all'] * 100) + '% (512=NQ256+Hot256); curve190 and HF150 remain unchanged. '
             'Source1ed confirmed cumulative main7/remote7/PENDING0 remains unchanged;195 collector calls0. Original PID503255/Invocationa313bda919834d24b4c5d491698d39ce remains active with two bound advancing polls and zero OOM. Goal active; no new training, restore, evaluation, collector or remote operation. '
             'Pending evidence: ' + pending_path.name + '; original14 documents retained in ' + str(snapshot / 'SNAPSHOT_MANIFEST.json') + '.\n')
        report_raw = prior[reports[0]] + report_note.encode()
        for path in reports:
            output[path] = report_raw
        need(set(output) == set(allp), 'Only the closed14 canonical/report paths may be patched')

        def inputs_unchanged():
            reject_obsolete_local()
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
        closure = dict(schema='outer195_pending_canonical_local_publication_v1', status=PHASE,
             created_at_utc=now.isoformat(), source=descriptor(source, source_raw), execution_input_pins=descriptor(pins_path, pins_raw),
             historical_successful170_source=descriptor(original, original_raw), pinned_core_sha256=CORE_SHA,
             pending_evidence=pending_pin, actual_observations=observations,
             canonical_FULL_boundary=194, actual_optimizer_completed_outer=195, latest_saved_native_outer=195,
             actual_successful_adam_per_rank=completed, actual_total_adam_per_rank=totals,
             active_outer=195, native_saved_driver_next_step=196, live196_entry_claimed=False,
             actual_cold_loaded_native_outer=5, current_dev195_result=None, latest_completed_dev_outer=190,
             curve_complete_through=190, immutable_cost_and_cumulative_through=194, HF_permanent_milestone=150,
             source1ed_actual_cumulative_main_invocations=7, source1ed_actual_cumulative_remote_invocations=7,
             dev195_new_collector_main_invocations=0, dev195_new_remote_collector_invocations=0,
             prior_closed14_gzip_manifest=descriptor(manifest_path, manifest_raw),
             before=[descriptor(path, raw) for path, raw in prior.items()],
             after=[descriptor(path, raw) for path, raw in output.items()],
             current_changes=changed, actual_local_readback_observations=readbacks,
             protected_canonical_fields_equal=True, protected_immutable_evidence_unchanged=True,
             raw_observation_scope='BASE-only real bytes/SHA; no raw mirror publication claim',
             global_atomic_transaction=False, training_or_remote_mutation=False,
             independent_cold_reload_performed=False, tensor_contents_read=False,
             score_updated=False, FULL195_or_dev195_score_claimed=False, goal_status='active', goal_complete=False)
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
        print(json.dumps(dict(pending195=pending_pin, publication=descriptor(closure_path, closure_raw),
             closed14_snapshot=descriptor(manifest_path, manifest_raw), canonical_FULL=194,
             actual_optimizer_native=195, dev195_result=None, live196_claimed=False,
             two_actual_readbacks=True, training_or_remote_mutation=False, goal_status='active')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='DEV195_PENDING_STAGE_NO_SUCCESS_RECEIPT', error=str(error),
                             remote_actions=False)), file=sys.stderr)
        raise SystemExit(2)
