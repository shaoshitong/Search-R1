#!/usr/bin/env python3
"""Future dev175 local preflight/owner receipt; never invokes SSH or collector.

Adapted from dev170_owner_local_receipt.py, 18260 bytes, SHA256
a3afdda89718b06c8d144e55a06701c214af158a0af3ef731137ed6d3d11e27e.
This source preparation is not evidence that outer175 or its dev512 has finished.

Only after root has observed actual FULL/native/CUM/COST175 and the original
COMPLETE dev175 control, supply a SHA/byte-bound root gate JSON with:
  schema: dev175_root_owner_gate_pins_v1
  status: ACTUAL_DEV175_FULL_NATIVE_CUM_COST_CONTROL_READY
  outer: 175
  collector_source_commit / collector_source_sha256: the fixed source below
  collector_executor_agent: the one root-authorized actual executor
  authorized_main_invocations / authorized_remote_incremental_invocations: 1
  source_pins: mapping of the four exact names in FUTURE_NAMES to
               {path, bytes, sha256} from actual local bytes
  pending_evidence: {path, bytes, sha256} of an actual source-bound local JSON
  control_ready: {outer: 175, raw_summary_utf8: original remote JSON bytes
                 decoded as UTF-8, source: original stable-read descriptor}
The control source descriptor has path/resolved_path/bytes/sha256/stable_read;
it is checked against the original raw bytes, not followed remotely. No local
DEV_OUTER0175_SUMMARY copy is needed before the unique collector invocation.

The pending evidence JSON must carry schema dev175_source_bound_pending_readback_v1,
outer/source_commit/source_sha256/executor_agent/observed_utc, actual nonempty
observations and evidence_scope, preceding_local_pending_invocations=0 and
preceding_pending_remote_operations=false. These remain executor attestations;
they do not constitute an independent global process audit.

Future invocation (all variables must come from actual future readback):
  python3 -B work/tree-speed-repair-20261008/dev175_owner_local_receipt.py preflight \
    --root-pins "$ACTUAL_ROOT_GATE" --root-pins-bytes "$ACTUAL_ROOT_GATE_BYTES" \
    --root-pins-sha256 "$ACTUAL_ROOT_GATE_SHA256"
After root authorizes the one collector main1/remote1 and it actually exits0:
  python3 -B work/tree-speed-repair-20261008/dev175_owner_local_receipt.py receipt \
    --root-pins "$ACTUAL_ROOT_GATE" --root-pins-bytes "$ACTUAL_ROOT_GATE_BYTES" \
    --root-pins-sha256 "$ACTUAL_ROOT_GATE_SHA256" \
    --execution-bytes "$ACTUAL_EXECUTION_BYTES" \
    --execution-sha256 "$ACTUAL_EXECUTION_SHA256"
The execution JSON retains the dev170 execution fields and additionally binds
root_gate_input_pins, pre_execution_input_pins and preceding_pending_evidence.
This script never starts, retries or recursively calls that collector command.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

W = Path(__file__).resolve().parents[2]
B = W / 'outputs/tree-nested-branch-handoff-20261013'
M = [W / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
     Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
S = W / 'work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py'
H = W / 'work/tree-speed-repair-20261008/ssh_retained18.py'
ENTRY = '1edaf50bb8e14de51db936e0859e369792011233'
SOURCE_SHA = '545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0'
HELPER_SHA = '8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'
CMD = 'env TREE_MGMT_CONTROL_PATH=/tmp/tree-mgmt-3fw9a1ab/control python3 -B work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py --base outputs/tree-nested-branch-handoff-20261013 --ssh-helper work/tree-speed-repair-20261008/ssh_retained18.py --through-outer 175'
FUTURE_NAMES = ('FULL_OUTER175_PROGRESS_RECEIPT.json', 'NATIVE_HISTORY_175_METADATA_READBACK.json',
                'CUMULATIVE_COST_THROUGH175.json', 'COST_OUTER175_LOCAL_RECEIPT.json')
PRIOR_NAMES = {
    'DEV5_170_NATIVE_COST_SUPPLEMENT.json': (7058, '4b34f52a1a3a8447f23c4ebdb8c55a83c9254c8fc72cf8ba3ad367d69b2e6722'),
    'DEV5_170_NATIVE_COST_LOCAL_RECEIPT.json': (59548, '29c0ea107ed07fe91b2d76742246c3909b0a32380927015022153c79cc364415'),
    'ROOT_DEV170_NATIVE_COST_ACTUAL_READBACK.json': (80637, 'f1afb5d6c6c686fe771acb9f04f065bfd10d5f46199024f1fe6711a780007d13'),
    'dev_native_cost_summaries/DEV_OUTER0170_SUMMARY.json': (3606, '17998fd4d10e221ddc095f2d8251c6890c5cb810dad7a19fef388301954cd94a'),
}
F, R, PARTS, FIELDS, REMOTE_REFS = {}, {}, [], {}, []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def locate(name):
    p = Path(name)
    assert p.parts and '..' not in p.parts, name
    return p if p.is_absolute() else W / p if p.parts[0] in ('outputs', 'work') else B / p


def read(name, digest=None, size=None):
    p = locate(name).resolve(strict=True)
    assert p.is_relative_to(W), str(p)
    assert p.suffix in ('.json', '.md', '.py', '.patch', '.csv', '.yaml', '.yml', '.sh'), str(p)
    raw = p.read_bytes()
    assert 0 < len(raw) < 4000000, str(p)
    assert digest is None or digest_ok(digest) and sha(raw) == digest, 'SHA ' + str(p)
    assert size is None or type(size) is int and len(raw) == size, 'bytes ' + str(p)
    mirrored = p.is_relative_to(B) and not p.name.startswith(('NATIVE5_CONTINUE_POLL_', 'NATIVE5_CONTINUE_RESOURCE_'))
    ref = dict(path=str(p.relative_to(W)), bytes=len(raw), sha256=sha(raw), mirror_checked=mirrored)
    assert str(p) not in F or F[str(p)] == ref, 'changed ' + str(p)
    if mirrored:
        for mirror in M:
            assert (mirror / p.relative_to(B)).read_bytes() == raw, 'mirror ' + str(p)
    F[str(p)], R[str(p)] = ref, raw
    return json.loads(raw) if p.suffix == '.json' else raw


def reference(name):
    p = locate(name).resolve(strict=True)
    if str(p) not in F:
        read(str(p))
    return {k: v for k, v in F[str(p)].items() if k != 'mirror_checked'}


def bound(ref, hint=None):
    assert isinstance(ref, dict) and digest_ok(ref.get('sha256'))
    assert type(ref.get('bytes')) is int and 0 < ref['bytes'] < 4000000
    name = ref.get('path', ref.get('file', hint))
    assert isinstance(name, str), (ref, hint)
    if name.startswith(('/dataset/', '/mnt/')):
        REMOTE_REFS.append(ref)
        return None
    return read(name, ref['sha256'], ref['bytes'])


def exact_ref(ref, name):
    assert locate(ref.get('path', ref.get('file', ''))).resolve(strict=True) == locate(name).resolve(strict=True), name
    value = bound(ref)
    assert value is not None, name
    return value


def refs(value):
    if isinstance(value, list):
        for item in value:
            refs(item)
    elif isinstance(value, dict):
        if digest_ok(value.get('sha256')) and type(value.get('bytes')) is int and (value.get('path') or value.get('file')):
            bound(value)
            return
        for key, item in value.items():
            if isinstance(item, dict) and digest_ok(item.get('sha256')) and type(item.get('bytes')) is int and Path(key).suffix in ('.json', '.md', '.py', '.patch', '.csv', '.yaml', '.yml', '.sh'):
                bound(item, key)
            else:
                refs(item)


def expand(value, field):
    if isinstance(value, dict) and value.get('storage') == 'complete_original_field_parts':
        kind = value['container_type']
        assert kind in ('list', 'mapping') and isinstance(value['parts'], list) and value['parts']
        assert type(value['original_entries']) is int and value['original_entries'] >= 0
        result, seen = ([], set()) if kind == 'list' else ({}, set())
        for ref in value['parts']:
            assert ref.get('json_selector') == 'value' and ref['file'] not in seen
            seen.add(ref['file'])
            part = bound(ref)
            assert part['schema'] == 'cumulative_scalar_field_part_v1' and part['original_field'] == field
            chunk = part['value']
            assert isinstance(chunk, list if kind == 'list' else dict)
            if kind == 'list':
                result.extend(chunk)
            else:
                assert not set(result).intersection(chunk)
                result.update(chunk)
            PARTS.append(dict(original_field=field, **ref))
        assert len(result) == value['original_entries']
        FIELDS[field] = dict(entries=len(result), parts=len(value['parts']), container_type=kind)
        return expand(result, field)
    if isinstance(value, dict):
        return {key: expand(item, field + '.' + key) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, field + '[' + str(i) + ']') for i, item in enumerate(value)]
    return value


def row_parts(summary):
    value = summary['rows']
    if isinstance(value, list):
        return []
    assert value['storage'] == 'complete_original_field_parts' and value['container_type'] == 'list'
    assert type(value['original_entries']) is int and isinstance(value['parts'], list) and value['parts']
    seen, count, result = set(), 0, []
    for ref in value['parts']:
        assert ref.get('json_selector') == 'value' and ref['file'] not in seen
        seen.add(ref['file'])
        part = bound(ref)
        assert isinstance(part, dict) and isinstance(part['value'], list)
        count += len(part['value'])
        result.append(reference(ref['file']))
    assert count == value['original_entries']
    return result


def two_pass():
    observations = []
    for number in range(2):
        for name, ref in F.items():
            p, raw = Path(name), Path(name).read_bytes()
            assert raw == R[name] and len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], name
            if ref['mirror_checked']:
                for mirror in M:
                    assert (mirror / p.relative_to(B)).read_bytes() == raw, name
        observations.append(dict(pass_number=number + 1, observed_utc=datetime.now(timezone.utc).isoformat(), files=len(F), mirrored_files=sum(x['mirror_checked'] for x in F.values())))
        if number == 0:
            time.sleep(1)
    return observations


def publish(name, value):
    assert name in ('DEV5_175_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json', 'DEV5_175_NATIVE_COST_LOCAL_RECEIPT.json')
    raw = encode(value)
    assert len(raw) <= 100000
    for folder in [B] + M:
        p = folder / name
        assert not p.exists() or p.read_bytes() == raw, str(p)
    for folder in [B] + M:
        p = folder / name
        if not p.exists():
            p.write_bytes(raw)
    for number in range(2):
        for folder in [B] + M:
            assert (folder / name).read_bytes() == raw
        if number == 0:
            time.sleep(1)
    return dict(path=str(B / name), bytes=len(raw), sha256=sha(raw), copies=3, byte_readback_passes=2)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate(rows, values):
    known = [v for v in values if numeric(v)]
    total = math.fsum(known) if known else None
    return dict(total=total if len(known) == len(values) else None, known_sum=total,
                missing_outers=[row['outer'] for row, value in zip(rows, values) if not numeric(value)])


def dev_pin(mod, dev):
    n = dev['outer']
    sc = mod.PARENT if n == 5 else mod.CHILD
    run = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + sc
    assert isinstance(dev['ledger'], str) and Path(dev['ledger']).name == dev['ledger']
    assert dev['rows'] == 512 and dev['phase'] == 'periodic'
    assert dev.get('source_commit', sc) == sc
    assert dev['ledger_sha256'] is None or digest_ok(dev['ledger_sha256'])
    assert numeric(dev['F1']) and numeric(dev['EM']) and 0 <= dev['F1'] <= 1 and 0 <= dev['EM'] <= 1
    return dict(outer=n, code_commit=sc, config_sha256=mod.CONFIGS[sc], model_identity=mod.MODEL,
                data_identity=mod.DATA, job_id='not-slurm;systemd:' + sc, run=run,
                path=run + '/output/validation-cost/' + dev['ledger'], phase=dev['phase'],
                ledger_sha256=dev['ledger_sha256'], F1=dev['F1'], EM=dev['EM'])


def verify_gate_inputs(args, mod):
    gate = read(str(args.root_pins), args.root_pins_sha256, args.root_pins_bytes)
    assert gate['schema'] == 'dev175_root_owner_gate_pins_v1' and gate['outer'] == 175
    assert gate['status'] == 'ACTUAL_DEV175_FULL_NATIVE_CUM_COST_CONTROL_READY'
    assert gate['collector_source_commit'] == ENTRY and gate['collector_source_sha256'] == SOURCE_SHA
    assert isinstance(gate['collector_executor_agent'], str) and gate['collector_executor_agent'].startswith('/root/')
    assert type(gate['authorized_main_invocations']) is int and gate['authorized_main_invocations'] == 1
    assert type(gate['authorized_remote_incremental_invocations']) is int and gate['authorized_remote_incremental_invocations'] == 1
    assert set(gate['source_pins']) == set(FUTURE_NAMES)
    full, native, rawcum, cost = [exact_ref(gate['source_pins'][name], name) for name in FUTURE_NAMES]
    cumulative = {key: expand(value, key) for key, value in rawcum.items()}
    assert full['status'] == 'ACTUAL_FULL_OUTER175_NATIVE175_COMPLETE_NEXT176_ENTERED'
    assert full['completed_outer'] == cumulative['through_effective_outer'] == native['outer'] == cost['outer'] == 175
    assert cost['status'] == 'ACTUAL_OUTER175_COST_ARCHIVED'
    assert full['source_commit'] == native['source_commit'] == cost['source_commit'] == mod.CHILD
    assert full['config_sha256'] == cost['config_sha256'] == mod.CONFIGS[mod.CHILD]
    assert full['actual_training_pid'] == cost['actual_training_pid'] == 503255
    assert full['actual_training_invocation_id'] == cost['actual_training_invocation_id'] == 'a313bda919834d24b4c5d491698d39ce'
    identity, receipt = native['identity'], native['receipt']
    assert identity['code_commit'] == receipt['source_commit'] == mod.CHILD
    assert identity['config_sha256'] == mod.CONFIGS[mod.CHILD]
    assert identity['data_identity'] == mod.DATA and identity['model_identity'] == mod.MODEL
    assert identity['actual_update_limit'] == 200 and identity['completed_step'] == receipt['completed_outer'] == 175
    assert native['archive_matches_original_manifest'] is True
    assert native['manifest_status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED'
    assert receipt['status'] == 'COMPLETE_NATIVE_MANIFEST_METADATA_CAPTURED'
    assert native['tensor_contents_read'] is False and native['independent_cold_reload_performed'] is False
    assert receipt['tensor_contents_read'] is False and receipt['independent_cold_reload_performed'] is False
    assert full['native_manifest_sha256'] == native['manifest_sha256'] == native['recorded_original_manifest_sha256'] == receipt['history_manifest_sha256'] == receipt['source_manifest_sha256'] == cost['native_manifest_sha256']
    assert full['native_history_receipt_sha256'] == native['receipt_sha256'] == cost['native_history_receipt_sha256']
    assert receipt['driver']['next_step'] == full['native_driver_next_step'] == 176
    assert {v['rank'] for v in receipt['workers']} == set(range(8)) and len(receipt['workers']) == 8
    assert all(v['scheduler_last_epoch'] == 175 and v['full_worker_state_finite'] is True for v in receipt['workers'])
    actual = full['actual_dev512']
    assert actual['outer'] == 175 and actual['rows'] == 512 and actual['status'] == 'VALIDATION_COST_COMPLETE'
    assert actual['phase'] == 'periodic' and digest_ok(actual['ledger_sha256'])
    metrics = actual['metrics']
    assert metrics['val/qa_count/all'] == 512 and metrics['val/qa_count/nq'] == metrics['val/qa_count/hotpotqa'] == 256
    assert numeric(metrics['val/f1/all']) and numeric(metrics['val/em/all'])
    assert 0 <= metrics['val/f1/all'] <= 1 and 0 <= metrics['val/em/all'] <= 1
    assert cost['actual_dev512'] == actual
    devs = cumulative['completed_dev_evaluations']
    assert [row['outer'] for row in devs] == list(range(5, 176, 5))
    assert devs[-1]['ledger'] == actual['path'] and devs[-1]['ledger_sha256'] == actual['ledger_sha256']
    assert devs[-1]['F1'] == metrics['val/f1/all'] and devs[-1]['EM'] == metrics['val/em/all']
    assert devs[-1]['testing_phase_wall_seconds'] == full['normal_dev512_wall_seconds']
    assert devs[-1]['rows'] == 512 and devs[-1]['phase'] == actual['phase']
    assert cumulative['pending_full_metric_outers'] == []
    counted = cumulative['rows_counted_once']
    assert [row['outer'] for row in counted] == list(range(1, 176))
    assert len({row['row_id'] for row in counted}) == 175 and counted[-1]['source_commit'] == mod.CHILD
    assert cumulative['dev_phase_already_in_outer_wall'] is True
    assert cost['aggregate_candidate_supplement_added_again'] is False
    assert cost['dev5_120_supplement_independent_not_added_to_outer_or_service_cpu'] is True
    assert cost['dev_native_supplement_not_added_again'] is True
    assert cost['prefix_candidate_and_CPU_subsets_not_added_again'] is True
    assert cost['dev_collector_called'] is False and cost['final200_entry_called'] is False
    assert cost['sampling_shape']['question_count'] == 512 and cost['sampling_shape']['training_leaf_count'] == 2560
    exact_ref(cost['source_inputs']['FULL'], FUTURE_NAMES[0])
    exact_ref(cost['source_inputs']['native_metadata'], FUTURE_NAMES[1])
    exact_ref(cost['cumulative_snapshot'], FUTURE_NAMES[2])
    refs(full['input_bindings'])
    refs(cost['source_inputs'])
    artifacts = cost['mirrored_artifacts']
    fixed = {'BRANCH_FALLBACK_RATES_OUTER0175.json', 'CANDIDATE_COST_EXTENSION_INPUTS.json',
             'CANDIDATE_COST_EXTENSION_OUTER0175.json', 'CANDIDATE_COST_EXTENSION_OUTER0175.md',
             'CUMULATIVE_COST_THROUGH175.json', 'CUMULATIVE_COST_THROUGH175.md',
             'candidate_cost_backfill/CANDIDATE_OUTER0175.json'}
    assert set(artifacts) == fixed | {part['file'] for part in PARTS}
    assert cost['mirrored_artifact_count'] == len(artifacts)
    for name, ref in artifacts.items():
        exact_ref(ref, name)
    for part in PARTS:
        assert artifacts[part['file']]['bytes'] == part['bytes'] and artifacts[part['file']]['sha256'] == part['sha256']
    pending = bound(gate['pending_evidence'])
    assert pending['schema'] == 'dev175_source_bound_pending_readback_v1' and pending['outer'] == 175
    assert pending['source_commit'] == ENTRY and pending['source_sha256'] == SOURCE_SHA
    assert pending['executor_agent'] == gate['collector_executor_agent']
    assert type(pending['preceding_local_pending_invocations']) is int and pending['preceding_local_pending_invocations'] == 0
    assert pending['preceding_pending_remote_operations'] is False
    assert isinstance(pending['observed_utc'], str) and pending['observed_utc']
    assert isinstance(pending['observations'], list) and pending['observations']
    assert isinstance(pending['evidence_scope'], str) and pending['evidence_scope']
    ready = gate['control_ready']
    assert ready['outer'] == 175 and isinstance(ready['raw_summary_utf8'], str)
    raw = ready['raw_summary_utf8'].encode('utf8')
    assert 0 < len(raw) <= 100000
    meta = ready['source']
    assert digest_ok(meta['sha256']) and meta['bytes'] == len(raw) and meta['sha256'] == sha(raw)
    assert meta['stable_read'] is True and meta['path'] == dev_pin(mod, devs[-1])['path']
    assert isinstance(meta['resolved_path'], str) and meta['resolved_path'].startswith('/dataset/')
    REMOTE_REFS.append(meta)
    control = json.loads(raw)
    assert control['training_protocol'] == 'tree_fresh200_five_leaf_v1' and control['physical_gpus'] == 8
    assert control['metrics'] == metrics
    saved = dict(path='dev_native_cost_summaries/DEV_OUTER0175_SUMMARY.json', bytes=len(raw), sha256=sha(raw), both_mirrors_byte_equal=True)
    mod.control_row(dev_pin(mod, devs[-1]), raw, meta, saved)
    return gate, full, native, rawcum, cumulative, cost, raw


def main():
    if not __debug__:
        raise RuntimeError('Required input gates cannot be disabled with python -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'receipt'])
    parser.add_argument('--root-pins', required=True, type=Path)
    parser.add_argument('--root-pins-bytes', required=True, type=int)
    parser.add_argument('--root-pins-sha256', required=True)
    parser.add_argument('--execution-bytes', type=int)
    parser.add_argument('--execution-sha256')
    args = parser.parse_args()
    assert args.root_pins_bytes > 0 and digest_ok(args.root_pins_sha256)
    if args.phase == 'receipt':
        assert args.execution_bytes is not None and args.execution_bytes > 0 and digest_ok(args.execution_sha256)
    read(str(S), SOURCE_SHA, 27911)
    read(str(H), HELPER_SHA, 4411)
    read('work/tree-speed-repair-20261008/dev170_owner_local_receipt.py', 'a3afdda89718b06c8d144e55a06701c214af158a0af3ef731137ed6d3d11e27e', 18260)
    read(str(Path(__file__).resolve()))
    # Source SHA is fixed first; import exposes pure readers/row validation only.
    # Its __main__ collector entry is never called here; the SSH helper is not imported.
    spec = importlib.util.spec_from_file_location('dev175_local_owner_collector_functions', S)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = {name: read(name, digest, size) for name, (size, digest) in PRIOR_NAMES.items()}
    oldowner = old['DEV5_170_NATIVE_COST_LOCAL_RECEIPT.json']
    priorroot = old['ROOT_DEV170_NATIVE_COST_ACTUAL_READBACK.json']
    assert priorroot['status'] == 'PASS_ACTUAL_DEV170_LOCAL_ARTIFACT_READBACK' and priorroot['outer'] == 170
    assert priorroot['inconsistencies'] == []
    for name, ref in priorroot['source_pins'].items():
        exact_ref(dict(path=name, **ref), name)
    olderowner = bound(oldowner['same_entry_prior_actual_executor_attestation'])
    oldestowner = bound(olderowner['same_entry_prior_actual_executor_attestation'])
    for owner in (oldestowner, olderowner, oldowner):
        assert owner['collector_source_commit'] == ENTRY and owner['collector_source_sha256'] == SOURCE_SHA
        assert owner['new_entry_main_invocations_this_task'] == owner['actual_remote_incremental_collection_invocations'] == 1
        assert owner['preceding_local_pending_invocations'] == 0 and owner['preceding_pending_remote_operations'] is False
        assert owner['remote_collector_exit_code'] == 0
    assert olderowner['same_entry_executor_attested_cumulative_main_invocations'] == olderowner['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == 2
    assert oldowner['same_entry_executor_attested_cumulative_main_invocations'] == oldowner['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == 3
    prior_summary = old['DEV5_170_NATIVE_COST_SUPPLEMENT.json']
    refs(prior_summary)
    prior_parts = row_parts(prior_summary)
    assert len(prior_parts) == 2
    prior = mod.rows_from_summary(B, M, prior_summary)
    assert len(prior) == 34 and [row['outer'] for row in prior] == list(range(5, 171, 5))
    assert len(oldowner['local_original_control_refs']) == 34
    for row, oldref in zip(prior, oldowner['local_original_control_refs']):
        assert oldref['outer'] == row['outer']
        bound(row['local_original_summary'])
        assert {k: oldref[k] for k in ('path', 'bytes', 'sha256')} == reference(row['local_original_summary']['path'])
    gate, full, native, rawcum, cumulative, cost, ready_control_raw = verify_gate_inputs(args, mod)
    actual, devs = full['actual_dev512'], cumulative['completed_dev_evaluations']
    for row, dev in zip(prior, devs[:-1]):
        raw = R[str(locate(row['local_original_summary']['path']).resolve())]
        assert mod.control_row(dev_pin(mod, dev), raw, row['source_control_summary'], row['local_original_summary']) == row
    if args.phase == 'preflight':
        for name in ('DEV5_175_NATIVE_COST_SUPPLEMENT.json', 'DEV5_175_NATIVE_COST_COLLECTION_ERROR.json',
                     'DEV5_175_NATIVE_COST_COLLECTOR_EXECUTION.json', 'DEV5_175_NATIVE_COST_LOCAL_RECEIPT.json'):
            assert not (B / name).exists(), name
        result = dict(schema='dev175_local_preflight_input_pins_v1', outer=175,
            observed_utc=datetime.now(timezone.utc).isoformat(), status='LOCAL_INPUTS_READY_NO_COLLECTOR_INVOKED',
            command=CMD, source_commit=ENTRY, collector_source_sha256=SOURCE_SHA, helper_sha256=HELPER_SHA,
            root_gate_input_pins=reference(str(args.root_pins)), preceding_pending_evidence=gate['pending_evidence'],
            original_remote_control_ready=gate['control_ready']['source'], actual_dev175_ledger_sha256=actual['ledger_sha256'],
            original_remote_control_byte_evidence_in_root_gate=True, local_dev175_control_copy_required=False,
            collector_executor_agent=gate['collector_executor_agent'], authorized_main_invocations=1,
            authorized_remote_incremental_invocations=1, prior_control_outers=list(range(5, 171, 5)),
            same_entry_prior_confirmed_main_invocations=3, same_entry_prior_confirmed_remote_incremental_invocations=3,
            source_CUM_dynamic_fields=FIELDS, source_CUM_complete_parts=len(PARTS), cost_mirrored_artifact_refs=len(cost['mirrored_artifacts']),
            preserved_input_refs=list(F.values()), local_readback_observations=two_pass(),
            invocation_count_scope='Prior source1ed SHA-bound executor attestations160+165+170 total main3/remote3, corroborated by products; this preflight does not count outer175 or claim a global process audit.',
            remote_operations=False, collector_invoked=False)
        print(json.dumps(publish('DEV5_175_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json', result)))
        return
    assert not (B / 'DEV5_175_NATIVE_COST_COLLECTION_ERROR.json').exists()
    execution = read('DEV5_175_NATIVE_COST_COLLECTOR_EXECUTION.json', args.execution_sha256, args.execution_bytes)
    assert execution['source_commit'] == ENTRY and execution['source_sha256'] == SOURCE_SHA and execution['helper_sha256'] == HELPER_SHA
    assert execution['actual_command'] == CMD and type(execution['exit_code']) is int and execution['exit_code'] == 0
    assert execution['executor_agent'] == gate['collector_executor_agent']
    assert type(execution['new_entry_main_invocations_this_task']) is int and execution['new_entry_main_invocations_this_task'] == 1
    assert type(execution['actual_remote_incremental_collection_invocations']) is int and execution['actual_remote_incremental_collection_invocations'] == 1
    assert type(execution['preceding_local_pending_invocations']) is int and execution['preceding_local_pending_invocations'] == 0
    assert execution['preceding_pending_remote_operations'] is False
    exact_ref(execution['root_gate_input_pins'], str(args.root_pins))
    assert execution['root_gate_input_pins']['sha256'] == args.root_pins_sha256 and execution['root_gate_input_pins']['bytes'] == args.root_pins_bytes
    assert execution['preceding_pending_evidence'] == gate['pending_evidence']
    pre = exact_ref(execution['pre_execution_input_pins'], 'DEV5_175_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json')
    assert pre['schema'] == 'dev175_local_preflight_input_pins_v1' and pre['outer'] == 175
    assert pre['status'] == 'LOCAL_INPUTS_READY_NO_COLLECTOR_INVOKED' and pre['collector_invoked'] is False
    assert pre['source_commit'] == ENTRY and pre['collector_source_sha256'] == SOURCE_SHA and pre['command'] == CMD
    assert pre['root_gate_input_pins'] == reference(str(args.root_pins))
    assert pre['preceding_pending_evidence'] == gate['pending_evidence']
    assert pre['collector_executor_agent'] == execution['executor_agent']
    for ref in pre['preserved_input_refs']:
        bound(ref)
    emitted = json.loads(execution['stdout'])
    assert emitted['controls'] == 35 and emitted['newly_read_remote_control_outers'] == [175]
    assert emitted['reused_original_control_outers'] == list(range(5, 171, 5))
    assert emitted['original_cumulative_unchanged'] is True and emitted['remote_files_written'] is False
    summ = exact_ref(emitted['supplement'], 'DEV5_175_NATIVE_COST_SUPPLEMENT.json')
    exact_ref(emitted['markdown'], 'DEV5_175_NATIVE_COST_SUPPLEMENT.md')
    assert summ['status'] == 'ACTUAL_COMPLETED_DEV_CONTROL_SUMMARY_COST_READBACK' and summ['through_outer'] == 175
    exact_ref(summ['source_cumulative'], FUTURE_NAMES[2])
    exact_ref(summ['source_full_receipt'], FUTURE_NAMES[0])
    exact_ref(summ['prior_immutable_supplement'], 'DEV5_170_NATIVE_COST_SUPPLEMENT.json')
    assert summ['model_identity'] == mod.MODEL and summ['data_identity'] == mod.DATA
    assert summ['split_identity']['evaluated_rows'] == 512 and summ['split_identity']['nq_rows'] == summ['split_identity']['hotpotqa_rows'] == 256
    assert summ['split_identity']['independent_split_file_sha256'] is None
    assert summ['read_only'] is True and summ['remote_files_written'] is False
    assert summ['jsonl_rollout_tensor_or_model_read'] is False and summ['immutable_cumulative_rewritten'] is False
    assert summ['STATUS_REPORT_FUP_or_training_modified'] is False and summ['tests_or_review_performed'] is False
    refs(summ)
    new_parts = row_parts(summ)
    rows = mod.rows_from_summary(B, M, summ)
    assert len(rows) == 35 and rows[:-1] == prior and [row['outer'] for row in rows] == list(range(5, 176, 5))
    assert summ['newly_read_remote_control_outers'] == [175] and summ['reused_original_control_outers'] == list(range(5, 171, 5))
    control_refs, pins = [], []
    for row, dev in zip(rows, devs):
        pin = dev_pin(mod, dev)
        pins.append(pin)
        bound(row['local_original_summary'])
        raw = R[str(locate(row['local_original_summary']['path']).resolve())]
        assert mod.control_row(pin, raw, row['source_control_summary'], row['local_original_summary']) == row
        control_refs.append(dict(outer=row['outer'], **reference(row['local_original_summary']['path'])))
    last = rows[-1]
    assert locate(last['local_original_summary']['path']).resolve() == (B / 'dev_native_cost_summaries/DEV_OUTER0175_SUMMARY.json').resolve()
    assert R[str(locate(last['local_original_summary']['path']).resolve())] == ready_control_raw
    assert last['source_control_summary'] == gate['control_ready']['source']
    assert last['ledger_sha256'] == actual['ledger_sha256']
    assert last['precise_engine_emitted_tokens'] is None and last['scoring_call_event_total'] is None
    program = sha(mod.REMOTE.replace('PIN_DATA', repr(pins[-1:])).encode())
    assert summ['remote_program_sha256'] == program and summ['collector_source_sha256'] == SOURCE_SHA and summ['ssh_helper_sha256'] == HELPER_SHA
    assert summ['precise_engine_emitted_tokens'] is None and summ['scoring_call_event_total'] is None
    assert numeric(summ['collector_cpu_seconds']) and summ['collector_cpu_seconds'] >= 0
    assert summ['collector_cpu_scope'] == 'This small new readback process only; not added to validation/service CPU.'
    assert set(summ['native_counter_totals']) == {key for row in rows for key in (row['counters'] or {})}
    for key, record in summ['native_counter_totals'].items():
        assert record == aggregate(rows, [(row['counters'] or {}).get(key) for row in rows])
    assert summ['original_validation_observer_wall_seconds'] == aggregate(rows, [row['wall_seconds'] for row in rows])
    assert summ['original_validation_trainer_process_cpu_seconds'] == aggregate(rows, [row['trainer_process_cpu_seconds'] for row in rows])
    assert summ['prior_fullmetric_testing_phase_wall_seconds'] == cumulative['completed_dev_testing_phase_wall_seconds']
    assert emitted['native_counter_totals'] == summ['native_counter_totals']
    observations = two_pass()
    value = dict(schema='dev_native_cost_actual_local_execution_receipt_v2', outer=175,
        observed_utc=datetime.now(timezone.utc).isoformat(), status='ACTUAL_DEV175_COST_COLLECTED_AND_LOCAL_MIRRORS_READ_BACK',
        collector_executor_agent=execution['executor_agent'], local_readback_executor_agent=execution['executor_agent'],
        root_remote_collector_execution_claimed=False, new_entry_main_invocations_this_task=1,
        actual_remote_incremental_collection_invocations=1, preceding_local_pending_invocations=0,
        preceding_pending_remote_operations=False, preceding_pending_source_bound_evidence=gate['pending_evidence'],
        remote_collector_exit_code=execution['exit_code'], same_entry_prior_actual_executor_attestation=reference('DEV5_170_NATIVE_COST_LOCAL_RECEIPT.json'),
        same_entry_prior_confirmed_main_invocations=3, same_entry_prior_confirmed_remote_incremental_invocations=3,
        same_entry_executor_attested_cumulative_main_invocations=4, same_entry_executor_attested_cumulative_remote_incremental_invocations=4,
        invocation_count_scope='This actual outer175 delta main1/remote1/PENDING0; same source1ed SHA-bound executor attestations160+165+170+175 total main4/remote4, corroborated by products. No independent complete system-wide process-log audit is claimed.',
        actual_dev175_metrics=actual['metrics'], full_outer175_wall_seconds=full['full_outer_wall_seconds'],
        actual_command_template=CMD, collector_source_commit=ENTRY, collector_source=reference(str(S)),
        collector_source_sha256=SOURCE_SHA, remote_program_sha256=program, ssh_helper_sha256=HELPER_SHA,
        source_FULL=reference(FUTURE_NAMES[0]), source_CUM=reference(FUTURE_NAMES[2]),
        source_native_metadata=reference(FUTURE_NAMES[1]), cost_outer_receipt=reference(FUTURE_NAMES[3]),
        root_gate_input_pins=reference(str(args.root_pins)), original_remote_control_ready=gate['control_ready']['source'],
        source_CUM_completed_dev_storage=rawcum['completed_dev_evaluations'].get('storage') if isinstance(rawcum['completed_dev_evaluations'], dict) else 'inline_list',
        source_CUM_completed_dev_entries=len(devs),
        source_CUM_all_complete_field_parts=[dict(path=str(B.relative_to(W) / part['file']), bytes=part['bytes'], sha256=part['sha256'], original_field=part['original_field']) for part in PARTS],
        source_CUM_dynamic_fields=FIELDS, source_CUM_original_bytes_sha_retained=True,
        source_CUM_parsed_dev_view_does_not_rewrite_original=True, cost_mirrored_artifact_refs=len(cost['mirrored_artifacts']),
        cost_mirrored_artifact_refs_bound_to_actual_bytes=True,
        supplement=reference('DEV5_175_NATIVE_COST_SUPPLEMENT.json'), markdown=reference('DEV5_175_NATIVE_COST_SUPPLEMENT.md'),
        prior_immutable_supplement=reference('DEV5_170_NATIVE_COST_SUPPLEMENT.json'),
        prior_root_actual_readback=reference('ROOT_DEV170_NATIVE_COST_ACTUAL_READBACK.json'),
        controls=len(rows), reused_original_control_outers=list(range(5, 171, 5)), newly_read_remote_control_outers=[175],
        local_original_control_refs=control_refs, rows_storage=summ['rows'].get('storage') if isinstance(summ['rows'], dict) else 'inline_list',
        rows_part_refs=new_parts, prior_rows_part_refs=prior_parts,
        rows_readback_function='SHA-bound collect_nested_dev_costs.rows_from_summary; no collector main invoked by this builder',
        prior34_rows_exact_object_equal=True, prior34_original_controls_SHA_and_bytes_preserved=True,
        prior_DEV170_two_original_rows_parts_SHA_and_bytes_preserved=True,
        new_dev175_original_control=last['source_control_summary'], new_dev175_counters=last['counters'],
        new_dev175_observer_wall_seconds=last['wall_seconds'], new_dev175_trainer_process_cpu_seconds=last['trainer_process_cpu_seconds'],
        new_dev175_ledger_sha256=last['ledger_sha256'], new_dev175_control_summary_sha256=last['local_original_summary']['sha256'],
        control_summary_SHA_and_value_ledger_SHA_have_distinct_roles=True,
        fullmetric_dev175_wall_seconds=full['normal_dev512_wall_seconds'], native_counter_totals=summ['native_counter_totals'],
        original_validation_observer_wall_seconds=summ['original_validation_observer_wall_seconds'],
        original_validation_trainer_process_cpu_seconds=summ['original_validation_trainer_process_cpu_seconds'],
        precise_engine_emitted_tokens=None, scoring_call_event_total=None,
        new_remote_metadata_collector_cpu_seconds=summ['collector_cpu_seconds'], metadata_collector_cpu_scope=summ['collector_cpu_scope'],
        accounting_scope='Dev observer wall and trainer-process CPU are contained in FULL testing/service scopes; not added again to CUM or service CPU. Metadata collector CPU is a separate small readback process. Submitted prefill repeats events; returned nonpad is not precise engine emitted; HTTP calls and query events remain distinct.',
        execution_record=reference('DEV5_175_NATIVE_COST_COLLECTOR_EXECUTION.json'),
        pre_execution_input_pins=reference('DEV5_175_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json'),
        owner_receipt_builder=reference(str(Path(__file__).resolve())), adapted_from_reader_source=reference('work/tree-speed-repair-20261008/dev170_owner_local_receipt.py'),
        mirrors_actual_byte_readback_passes=2, local_readback_observations=observations,
        actual_mirror_roots=[str(p) for p in M], audited_local_files=sorted(F.values(), key=lambda x: x['path']),
        source_raw_poll_resource_scope='Existing raw poll/resource are local BASE-only SHA-bound observations; product artifacts are read from BASE and both mirrors twice. Root gate remote descriptors are not followed by this script. No new raw remote read is claimed.',
        old_DEV170_outputs_rewritten=False, immutable_CUM_rewritten=False, training_or_evaluation_modified=False,
        tensor_model_or_JSONL_read=False, canonical_STATUS_REPORT_FUP_observer_modified=False, tests_or_review_performed=False,
        ROOT_FULL175_or_final200_or51713_completion_claimed=False)
    print(json.dumps(publish('DEV5_175_NATIVE_COST_LOCAL_RECEIPT.json', value)))


if __name__ == '__main__':
    main()
