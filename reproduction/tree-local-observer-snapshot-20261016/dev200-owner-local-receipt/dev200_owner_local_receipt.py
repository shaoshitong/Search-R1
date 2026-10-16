#!/usr/bin/env python3
"""Future periodic dev200 local preflight/owner receipt; no SSH or collector main.

Prepared from frozen dev180 owner44417B/16c02d34...; source preparation is not
actual200 acceptance. Original source1ed collector is reused unchanged with
--through-outer200. Future actual FULL200 must have complete original step200
scalars and FINAL_PHASE_OBSERVED (never metric-pending or NEXT201). Native cursor
201 is only a saved boundary; no training201, ROOT_FULL200, or slow final51713
completion is required. Dev200 comes only from its original periodic512 ledger.

Root gate: schema dev200_root_owner_gate_pins_v1, outer200,
status ACTUAL_DEV200_FULL_NATIVE_CUM_COST_CONTROL_READY; fixed collector source,
sole executor and main1/remote1 authorization; source_pins exact FUTURE_NAMES;
actual pending_evidence, original stable control_ready rawbytes; prior_inputs
outer195 plus supplement, owner_receipt, root_actual_readback, original_control
and curve {json,csv,publication,binding,root_readback} actual {path,bytes,sha256}.
No future195/200 SHA, metric, registry, candidate, Adam total or success count is
pre-filled. The verified prior receipt chain supplies actual counts, then +1 only
after successful200 execution. All prior controls, every chain DEVpart, and five
original195 curve files are protected by exact bytes; historical curve registry
references remain as-of descriptors, not a claim their old registry bytes exist.

The pending JSON uses dev200_source_bound_pending_readback_v1 and the existing
source/executor/actual observations/PENDING0 fields. It is a sole-executor
attestation plus artifact evidence, not a global process audit. This builder has
only preflight and receipt phases, the same --root-pins/--root-pins-bytes/
--root-pins-sha256 and receipt --execution-bytes/--execution-sha256 CLI as180.
See DEV200_OWNER_SOURCE_PREPARATION.json for the exact future interface. Never
count this source preparation, a phase start, or a PENDING response as collection.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import re
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
CMD = 'env TREE_MGMT_CONTROL_PATH=/tmp/tree-mgmt-3fw9a1ab/control python3 -B work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py --base outputs/tree-nested-branch-handoff-20261013 --ssh-helper work/tree-speed-repair-20261008/ssh_retained18.py --through-outer 200'
FUTURE_NAMES = ('FULL_OUTER200_PROGRESS_RECEIPT.json', 'NATIVE_HISTORY_200_METADATA_READBACK.json',
                'CUMULATIVE_COST_THROUGH200.json', 'COST_OUTER200_LOCAL_RECEIPT.json')
PRIOR_OUTER = 195
PRIOR_NAMES = {
    'supplement': 'DEV5_195_NATIVE_COST_SUPPLEMENT.json',
    'owner_receipt': 'DEV5_195_NATIVE_COST_LOCAL_RECEIPT.json',
    'root_actual_readback': 'ROOT_DEV195_NATIVE_COST_ACTUAL_READBACK.json',
    'original_control': 'dev_native_cost_summaries/DEV_OUTER0195_SUMMARY.json',
}
PRIOR_CURVE_NAMES = {
    'json': 'LEARNING_CURVE_COMPLETE_THROUGH195.json',
    'csv': 'LEARNING_CURVE_COMPLETE_THROUGH195.csv',
    'publication': 'LEARNING_CURVE_THROUGH195_LOCAL_PUBLICATION.json',
    'binding': 'LEARNING_CURVE_THROUGH195_COST_SUPPLEMENT_BINDING.json',
    'root_readback': 'ROOT_CURVE_DEV_COST_THROUGH195_ACTUAL_READBACK.json',
}
KNOWN_SOURCE1ED_ROOT = dict(path='DEV5_160_NATIVE_COST_LOCAL_RECEIPT.json', bytes=19440,
    sha256='a169dcb969c0d9228847b47b577e7058fea888ac3fbc52e542c232594a1eee0d')
PRIOR_CHAIN, PRIOR_CURVE = [], {}

F, R, PARTS, FIELDS, REMOTE_REFS, TRAINING = {}, {}, [], {}, [], {}
CORE = W / 'work/tree-speed-repair-20261008/nested-native5-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/sync_nested_native_progress.py'
CORE_SHA = '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
# Exact FIELDS values from frozen variance200_record_coverage.py, read as source
# text during preparation; that source is byte-bound but never imported here.
VARIANCE_SOURCE = W / 'work/tree-speed-repair-20261008/variance200_record_coverage.py'
VARIANCE_SOURCE_SHA = 'bf44ce6316fa976fffba96db218d2f193f82be70877bc8729e1b7cef51ac0885'
REQUIRED_200_SCALARS = (
    'critic/rewards/mean', 'critic/score/mean',
    'tree_variance/within_root_raw_reward_population_variance',
    'tree_variance/intra_adv_population_variance', 'tree_variance/group_actual_adv_population_variance',
    'tree_variance/same_reward_question_fraction', 'tree_variance/mixed_reward_question_fraction',
    'tree_variance/exact_reward_one_question_count', 'tree_variance/exact_reward_zero_question_count',
    'tree_variance/root0_same_reward_fraction', 'tree_variance/root0_mixed_reward_fraction',
    'tree_variance/root1_same_reward_fraction', 'tree_variance/root1_mixed_reward_fraction',
    'tree_variance/same_reward_tree_fraction', 'tree_variance/mixed_reward_tree_fraction',
)



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
    assert name in ('DEV5_200_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json', 'DEV5_200_NATIVE_COST_LOCAL_RECEIPT.json')
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
    assert dev['rows'] == 512 and dev['phase'] == ('acceptance' if n == 5 else 'periodic')
    assert dev.get('source_commit', sc) == sc
    assert dev['ledger_sha256'] is None or digest_ok(dev['ledger_sha256'])
    assert numeric(dev['F1']) and numeric(dev['EM']) and 0 <= dev['F1'] <= 1 and 0 <= dev['EM'] <= 1
    return dict(outer=n, code_commit=sc, config_sha256=mod.CONFIGS[sc], model_identity=mod.MODEL,
                data_identity=mod.DATA, job_id='not-slurm;systemd:' + sc, run=run,
                path=run + '/output/validation-cost/' + dev['ledger'], phase=dev['phase'],
                ledger_sha256=dev['ledger_sha256'], F1=dev['F1'], EM=dev['EM'])


def load_prior_inputs(gate, mod):
    # Every195 hash/size comes from the future actual manifest, never sourceprep.
    assert gate['schema'] == 'dev200_root_owner_gate_pins_v1' and gate['outer'] == 200
    prior_inputs = gate['prior_inputs']
    assert prior_inputs['outer'] == PRIOR_OUTER
    assert set(prior_inputs) == {'outer', *PRIOR_NAMES, 'curve'}
    old = {key: exact_ref(prior_inputs[key], name) for key, name in PRIOR_NAMES.items()}
    oldowner, priorroot, summary = old['owner_receipt'], old['root_actual_readback'], old['supplement']
    assert oldowner['outer'] == priorroot['outer'] == summary['through_outer'] == PRIOR_OUTER
    assert priorroot['status'] == 'PASS_ACTUAL_DEV195_LOCAL_ARTIFACT_READBACK' and priorroot['inconsistencies'] == []
    assert summary['status'] == 'ACTUAL_COMPLETED_DEV_CONTROL_SUMMARY_COST_READBACK'
    for name, ref in priorroot['source_pins'].items():
        exact_ref(ref if ref.get('path') or ref.get('file') else dict(path=name, **ref), name)
    assert reference(PRIOR_NAMES['owner_receipt']) == reference(priorroot['actual_owner_receipt']['path'])
    exact_ref(priorroot['actual_owner_receipt'], PRIOR_NAMES['owner_receipt'])
    # Reverse linked actual receipt chain, with a pinned already-observed160 root.
    owners, owner_refs, seen = [oldowner], [reference(PRIOR_NAMES['owner_receipt'])], set()
    while True:
        owner = owners[-1]
        outer = owner['outer']
        assert type(outer) is int and outer % 5 == 0 and 160 <= outer <= PRIOR_OUTER and outer not in seen
        seen.add(outer)
        assert owner['schema'] == 'dev_native_cost_actual_local_execution_receipt_v2'
        assert owner['status'] == 'ACTUAL_DEV%d_COST_COLLECTED_AND_LOCAL_MIRRORS_READ_BACK' % outer
        assert owner['collector_source_commit'] == ENTRY and owner['collector_source_sha256'] == SOURCE_SHA
        assert type(owner['new_entry_main_invocations_this_task']) is int and owner['new_entry_main_invocations_this_task'] == 1
        assert type(owner['actual_remote_incremental_collection_invocations']) is int and owner['actual_remote_incremental_collection_invocations'] == 1
        assert type(owner['preceding_local_pending_invocations']) is int and owner['preceding_local_pending_invocations'] == 0
        assert owner['preceding_pending_remote_operations'] is False
        assert type(owner['remote_collector_exit_code']) is int and owner['remote_collector_exit_code'] == 0
        assert owner['controls'] == outer // 5
        exact_ref(owner['collector_source'], str(S))
        for key in ('source_FULL', 'source_CUM', 'source_native_metadata', 'cost_outer_receipt', 'supplement', 'markdown'):
            bound(owner[key])
        bound(owner['prior_immutable_supplement'])
        for ref in owner['source_CUM_all_complete_field_parts'] + owner['rows_part_refs'] + owner.get('prior_rows_part_refs', []):
            bound(ref)
        controls = owner['local_original_control_refs']
        assert len(controls) == outer // 5 and [ref['outer'] for ref in controls] == list(range(5, outer + 1, 5))
        for ref in controls:
            bound(ref)
        if owner.get('execution_record') is not None:
            execution = bound(owner['execution_record'])
            assert execution['source_commit'] == ENTRY and execution['source_sha256'] == SOURCE_SHA
            assert execution['exit_code'] == 0 and execution['new_entry_main_invocations_this_task'] == 1
            assert execution['actual_remote_incremental_collection_invocations'] == 1
            assert execution['preceding_local_pending_invocations'] == 0 and execution['preceding_pending_remote_operations'] is False
        previous = owner.get('same_entry_prior_actual_executor_attestation')
        if previous is None:
            assert outer == 160 and owner_refs[-1] == reference(KNOWN_SOURCE1ED_ROOT['path'])
            exact_ref(KNOWN_SOURCE1ED_ROOT, KNOWN_SOURCE1ED_ROOT['path'])
            break
        preceding = bound(previous)
        assert preceding['outer'] < outer
        owners.append(preceding)
        owner_refs.append(reference(previous['path']))
    ordered = list(reversed(owners))
    assert [owner['outer'] for owner in ordered[:4]] == [160, 165, 170, 175]
    for count, (owner, ref) in enumerate(zip(ordered, reversed(owner_refs)), 1):
        for key in ('same_entry_executor_attested_cumulative_main_invocations', 'same_entry_executor_attested_cumulative_remote_incremental_invocations'):
            if count > 1:
                assert type(owner[key]) is int and owner[key] == count
            elif key in owner:
                assert owner[key] == count
        for key in ('same_entry_prior_confirmed_main_invocations', 'same_entry_prior_confirmed_remote_incremental_invocations'):
            if key in owner:
                assert owner[key] == count - 1
        PRIOR_CHAIN.append(dict(outer=owner['outer'], cumulative_actual_executor_attested_main_invocations=count,
            cumulative_actual_executor_attested_remote_incremental_invocations=count, receipt=ref))
    refs(summary)
    prior_parts = row_parts(summary)
    assert prior_parts == [reference(ref['path']) for ref in oldowner['rows_part_refs']]
    prior = mod.rows_from_summary(B, M, summary)
    assert len(prior) == 39 and [row['outer'] for row in prior] == list(range(5, 196, 5))
    assert len(oldowner['local_original_control_refs']) == 39
    for row, oldref in zip(prior, oldowner['local_original_control_refs']):
        assert oldref['outer'] == row['outer']
        bound(row['local_original_summary'])
        assert {key: oldref[key] for key in ('path', 'bytes', 'sha256')} == reference(row['local_original_summary']['path'])
    exact_ref(prior_inputs['original_control'], prior[-1]['local_original_summary']['path'])
    assert prior_inputs['original_control']['bytes'] == prior[-1]['local_original_summary']['bytes']
    assert prior_inputs['original_control']['sha256'] == prior[-1]['local_original_summary']['sha256']
    # Protect old curve objects as complete SHA-bound files. Registry references
    # within these historical documents are descriptors as-of their publications;
    # no old mutable registry bytes are followed or claimed to remain readable.
    assert set(prior_inputs['curve']) == set(PRIOR_CURVE_NAMES)
    curve_docs = {key: exact_ref(prior_inputs['curve'][key], name) for key, name in PRIOR_CURVE_NAMES.items()}
    curve = curve_docs['json']
    assert [point['outer'] for point in curve['training_points']] == list(range(1, 196))
    assert [point['outer'] for point in curve['completed_dev512_points']] == list(range(5, 196, 5))
    rootcurve = curve_docs['root_readback']
    assert rootcurve['status'] == 'PASS_ACTUAL_CURVE195_LOCAL_ARTIFACT_READBACK'
    assert rootcurve['outer'] == 195 and rootcurve.get('inconsistencies', []) == []
    exact_ref(rootcurve['references']['curve'], PRIOR_CURVE_NAMES['json'])
    exact_ref(rootcurve['references']['csv'], PRIOR_CURVE_NAMES['csv'])
    exact_ref(rootcurve['references']['curve_publication'], PRIOR_CURVE_NAMES['publication'])
    exact_ref(rootcurve['cost_supplement_binding'], PRIOR_CURVE_NAMES['binding'])
    exact_ref(rootcurve['root_dev195_actual_readback'], PRIOR_NAMES['root_actual_readback'])
    assert curve_docs['binding']['through_outer'] == 195
    for key, name in PRIOR_CURVE_NAMES.items():
        PRIOR_CURVE[key] = reference(name)
    return old, oldowner, priorroot, summary, prior_parts, prior, len(owners)


def verify_gate_inputs(args, mod, core):
    gate = read(str(args.root_pins), args.root_pins_sha256, args.root_pins_bytes)
    assert gate['schema'] == 'dev200_root_owner_gate_pins_v1' and gate['outer'] == 200
    assert gate['status'] == 'ACTUAL_DEV200_FULL_NATIVE_CUM_COST_CONTROL_READY'
    assert gate['collector_source_commit'] == ENTRY and gate['collector_source_sha256'] == SOURCE_SHA
    assert isinstance(gate['collector_executor_agent'], str) and gate['collector_executor_agent'].startswith('/root/')
    assert type(gate['authorized_main_invocations']) is int and gate['authorized_main_invocations'] == 1
    assert type(gate['authorized_remote_incremental_invocations']) is int and gate['authorized_remote_incremental_invocations'] == 1
    assert set(gate['source_pins']) == set(FUTURE_NAMES)
    full, native, rawcum, cost = [exact_ref(gate['source_pins'][name], name) for name in FUTURE_NAMES]
    cumulative = {key: expand(value, key) for key, value in rawcum.items()}
    assert full['status'] == 'ACTUAL_FULL_OUTER200_NATIVE200_COMPLETE_FINAL_PHASE_OBSERVED'
    assert full['full_iteration_metric_pending'] is False and full['unobserved_metric_fields_remain_null'] is False
    assert full['completed_outer'] == cumulative['through_effective_outer'] == native['outer'] == cost['outer'] == 200
    assert cost['status'] == 'ACTUAL_OUTER200_COST_ARCHIVED'
    assert full['source_commit'] == native['source_commit'] == cost['source_commit'] == mod.CHILD
    assert full['config_sha256'] == cost['config_sha256'] == mod.CONFIGS[mod.CHILD]
    assert full['actual_training_pid'] == cost['actual_training_pid']
    assert full['actual_training_invocation_id'] == cost['actual_training_invocation_id'] == 'a313bda919834d24b4c5d491698d39ce'
    identity, receipt = native['identity'], native['receipt']
    assert identity['code_commit'] == receipt['source_commit'] == mod.CHILD
    assert identity['config_sha256'] == mod.CONFIGS[mod.CHILD]
    assert identity['data_identity'] == mod.DATA and identity['model_identity'] == mod.MODEL
    assert identity['actual_update_limit'] == 200 and identity['completed_step'] == receipt['completed_outer'] == 200
    assert native['archive_matches_original_manifest'] is True
    assert native['manifest_status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED'
    assert receipt['status'] == 'COMPLETE_NATIVE_MANIFEST_METADATA_CAPTURED'
    assert native['tensor_contents_read'] is False and native['independent_cold_reload_performed'] is False
    assert receipt['tensor_contents_read'] is False and receipt['independent_cold_reload_performed'] is False
    assert full['native_manifest_sha256'] == native['manifest_sha256'] == native['recorded_original_manifest_sha256'] == receipt['history_manifest_sha256'] == receipt['source_manifest_sha256'] == cost['native_manifest_sha256']
    assert full['native_history_receipt_sha256'] == native['receipt_sha256'] == cost['native_history_receipt_sha256']
    assert receipt['driver']['next_step'] == full['native_driver_next_step'] == 201
    assert {v['rank'] for v in receipt['workers']} == set(range(8)) and len(receipt['workers']) == 8
    assert all(v['scheduler_last_epoch'] == 200 and v['full_worker_state_finite'] is True for v in receipt['workers'])
    actual = full['actual_dev512']
    assert actual['outer'] == 200 and actual['rows'] == 512 and actual['status'] == 'VALIDATION_COST_COMPLETE'
    assert actual['phase'] == 'periodic' and digest_ok(actual['ledger_sha256'])
    metrics = actual['metrics']
    assert metrics['val/qa_count/all'] == 512 and metrics['val/qa_count/nq'] == metrics['val/qa_count/hotpotqa'] == 256
    assert numeric(metrics['val/f1/all']) and numeric(metrics['val/em/all'])
    assert 0 <= metrics['val/f1/all'] <= 1 and 0 <= metrics['val/em/all'] <= 1
    assert cost['actual_dev512'] == actual
    devs = cumulative['completed_dev_evaluations']
    assert [row['outer'] for row in devs] == list(range(5, 201, 5))
    assert devs[-1]['ledger'] == actual['path'] and devs[-1]['ledger_sha256'] == actual['ledger_sha256']
    assert devs[-1]['F1'] == metrics['val/f1/all'] and devs[-1]['EM'] == metrics['val/em/all']
    assert devs[-1]['testing_phase_wall_seconds'] == full['normal_dev512_wall_seconds']
    assert devs[-1]['rows'] == 512 and devs[-1]['phase'] == actual['phase']
    assert cumulative['pending_full_metric_outers'] == []
    counted = cumulative['rows_counted_once']
    assert [row['outer'] for row in counted] == list(range(1, 201))
    assert len({row['row_id'] for row in counted}) == 200 and counted[-1]['source_commit'] == mod.CHILD
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
    assert full['local_control_script_sha256'] == CORE_SHA
    poll = exact_ref(full['input_bindings']['poll'], full['source_poll'])
    resource = bound(full['input_bindings']['resource'])
    assert poll['source_commit'] == mod.CHILD and poll['run'] == full['run']
    assert poll['progress']['actual_completed_outer_from_attempt_results'] == 200
    assert not poll['progress'].get('errors')
    ps, rs, active = core.validate_unit(poll, resource, 200)
    assert full['actual_training_pid'] == int(rs['MainPID'])
    assert full['actual_next_outer_entered'] is None
    final = core.final_phase(poll, resource, active)
    assert full['final_phase_evidence'] == final
    metric = core.validate_metric(poll, 200)
    assert metric is not None, 'Original complete step200 metric is mandatory for this periodic-dev owner'
    assert len([value for value in poll['iteration_metrics'] if value.get('step') == 200]) == 1
    assert all(numeric(metric.get(name)) for name in REQUIRED_200_SCALARS)
    ranks = core.validate_ranks(poll, metric, 200)
    record, guard, completed, totals = core.validate_native(native, poll['progress'], 200)
    assert record == receipt and full['iteration_metrics'] == metric
    core.validate_restore(poll)
    assert full['last_rank_summaries'] == ranks
    assert full['actual_dev512'] == next(value for value in poll['development'] if value['outer'] == 200)
    assert full['per_rank_successful_adam'] == completed
    assert full['per_rank_actual_total_adam'] == totals == cost['actual_rank_successful_total_adam']
    assert full['partial_optimizer_calls_per_rank'] == guard['partial_optimizer_calls_per_rank']
    writes = {str(rank['rank']): rank['actual_successful_writes'] for rank in ranks}
    processed = {str(rank['rank']): rank['actual_opportunities_processed'] for rank in ranks}
    skipped = {str(rank['rank']): rank['actual_opportunities_skipped'] for rank in ranks}
    assert full['actual_outer_successful_writes_per_rank'] == writes == cost['actual_outer_successful_writes_per_rank']
    assert full['actual_outer_processed_opportunities_per_rank'] == processed == cost['independent_rank_processed_opportunities']
    assert full['actual_outer_skipped_opportunities_per_rank'] == skipped == cost['independent_rank_skipped_opportunities']
    assert full['full_metric_processed_opportunities'] == metric['actor/optimizer_opportunities_processed'] == cost['full_metric_processed_opportunities']
    assert full['full_metric_skipped_opportunities'] == metric['actor/optimizer_opportunities_skipped'] == cost['full_metric_skipped_opportunities']
    accounting = cumulative['successful_optimizer_accounting']
    assert accounting['direct_observed_calls_per_rank'] == completed
    assert accounting['direct_observed_rank_call_events'] == sum(completed.values())
    assert accounting['successful_calls_per_rank_through200'] == totals
    assert accounting['successful_rank_call_events_through200'] == sum(totals.values())
    latest = accounting['latest_outer_original_receipt_opportunity_fields']
    for key in ('per_rank_actual_total_adam', 'partial_optimizer_calls_per_rank', 'actual_outer_successful_writes_per_rank',
                'actual_outer_processed_opportunities_per_rank', 'actual_outer_skipped_opportunities_per_rank',
                'full_metric_processed_opportunities', 'full_metric_skipped_opportunities'):
        assert latest[key] == full[key]
    TRAINING.update(actual_total_adam_per_rank=totals, completed_successful_adam_per_rank=completed,
        partial_optimizer_calls_per_rank=guard['partial_optimizer_calls_per_rank'], actual_successful_writes_per_rank=writes,
        independent_rank_processed_opportunities=processed, independent_rank_skipped_opportunities=skipped,
        full_metric_processed_opportunities=metric['actor/optimizer_opportunities_processed'],
        full_metric_skipped_opportunities=metric['actor/optimizer_opportunities_skipped'],
        full_metric_actual_optimizer_writes=metric['actor/optimizer_steps'], native_driver_next_step=receipt['driver']['next_step'],
        native_cursor_is_saved_boundary_only=True, actual_next_outer_entered=None,
        final_phase_evidence=final, original_complete_step200_scalar_source=reference(full['source_poll']),
        original_step200_required_scalars={name: metric[name] for name in REQUIRED_200_SCALARS},
        actual_training201_entry_claimed=False, slow_final51713_completion_required=False,
        same_original_training_instance_active=active, exact_fullmetric_core_source=reference(str(CORE)),
        original15_scalar_field_definition_source=reference(str(VARIANCE_SOURCE)),
        count_scope='Existing core676d validates complete20 opportunities, per-rank1..20 writes, actual guard skips, finite/nonzero samples and cumulative/partial accounting. No fixed4000 Adam total or missing rank opportunity value is inferred.')
    artifacts = cost['mirrored_artifacts']
    fixed = {'BRANCH_FALLBACK_RATES_OUTER0200.json', 'CANDIDATE_COST_EXTENSION_INPUTS.json',
             'CANDIDATE_COST_EXTENSION_OUTER0200.json', 'CANDIDATE_COST_EXTENSION_OUTER0200.md',
             'CUMULATIVE_COST_THROUGH200.json', 'CUMULATIVE_COST_THROUGH200.md',
             'candidate_cost_backfill/CANDIDATE_OUTER0200.json'}
    assert set(artifacts) == fixed | {part['file'] for part in PARTS}
    assert cost['mirrored_artifact_count'] == len(artifacts)
    for name, ref in artifacts.items():
        exact_ref(ref, name)
    for part in PARTS:
        assert artifacts[part['file']]['bytes'] == part['bytes'] and artifacts[part['file']]['sha256'] == part['sha256']
    pending = bound(gate['pending_evidence'])
    assert pending['schema'] == 'dev200_source_bound_pending_readback_v1' and pending['outer'] == 200
    assert pending['source_commit'] == ENTRY and pending['source_sha256'] == SOURCE_SHA
    assert pending['executor_agent'] == gate['collector_executor_agent']
    assert type(pending['preceding_local_pending_invocations']) is int and pending['preceding_local_pending_invocations'] == 0
    assert pending['preceding_pending_remote_operations'] is False
    assert isinstance(pending['observed_utc'], str) and pending['observed_utc']
    assert isinstance(pending['observations'], list) and pending['observations']
    assert isinstance(pending['evidence_scope'], str) and pending['evidence_scope']
    ready = gate['control_ready']
    assert ready['outer'] == 200 and isinstance(ready['raw_summary_utf8'], str)
    raw = ready['raw_summary_utf8'].encode('utf8')
    assert 0 < len(raw) <= 100000
    meta = ready['source']
    assert digest_ok(meta['sha256']) and meta['bytes'] == len(raw) and meta['sha256'] == sha(raw)
    assert meta['stable_read'] is True and meta['path'] == dev_pin(mod, devs[-1])['path']
    alias = '/dataset/shaoshitong/search-r1-rl/tree-grpo/'
    retained = '/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/'
    assert meta['path'].startswith(alias)
    assert meta['resolved_path'] == retained + meta['path'][len(alias):]
    REMOTE_REFS.append(meta)
    control = json.loads(raw)
    assert control['training_protocol'] == 'tree_fresh200_five_leaf_v1' and control['physical_gpus'] == 8
    assert control['metrics'] == metrics
    saved = dict(path='dev_native_cost_summaries/DEV_OUTER0200_SUMMARY.json', bytes=len(raw), sha256=sha(raw), both_mirrors_byte_equal=True)
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
    read(str(CORE), CORE_SHA, 39098)
    read(str(VARIANCE_SOURCE), VARIANCE_SOURCE_SHA, 29073)
    read('work/tree-speed-repair-20261008/dev180_owner_local_receipt.py', '16c02d34b66a54b259f221552cd8be2a48b73255f4c876fddd86572e0e5832e0', 44417)
    read(str(Path(__file__).resolve()))
    # Both source SHAs are fixed first; import exposes pure metadata readers/gates.
    # Neither __main__ entry is called here; the SSH helper is not imported.
    spec = importlib.util.spec_from_file_location('dev200_local_owner_collector_functions', S)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location('dev200_published_core_metadata_gates', CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    prior_gate = read(str(args.root_pins), args.root_pins_sha256, args.root_pins_bytes)
    old, oldowner, priorroot, prior_summary, prior_parts, prior, prior_count = load_prior_inputs(prior_gate, mod)
    gate, full, native, rawcum, cumulative, cost, ready_control_raw = verify_gate_inputs(args, mod, core)
    actual, devs = full['actual_dev512'], cumulative['completed_dev_evaluations']
    for row, dev in zip(prior, devs[:-1]):
        raw = R[str(locate(row['local_original_summary']['path']).resolve())]
        assert mod.control_row(dev_pin(mod, dev), raw, row['source_control_summary'], row['local_original_summary']) == row
    if args.phase == 'preflight':
        for name in ('DEV5_200_NATIVE_COST_SUPPLEMENT.json', 'DEV5_200_NATIVE_COST_COLLECTION_ERROR.json',
                     'DEV5_200_NATIVE_COST_COLLECTOR_EXECUTION.json', 'DEV5_200_NATIVE_COST_LOCAL_RECEIPT.json'):
            assert not (B / name).exists(), name
        result = dict(schema='dev200_local_preflight_input_pins_v1', outer=200,
            observed_utc=datetime.now(timezone.utc).isoformat(), status='LOCAL_INPUTS_READY_NO_COLLECTOR_INVOKED',
            command=CMD, source_commit=ENTRY, collector_source_sha256=SOURCE_SHA, helper_sha256=HELPER_SHA,
            root_gate_input_pins=reference(str(args.root_pins)), preceding_pending_evidence=gate['pending_evidence'],
            original_remote_control_ready=gate['control_ready']['source'], actual_dev200_ledger_sha256=actual['ledger_sha256'],
            original_remote_control_byte_evidence_in_root_gate=True, local_dev200_control_copy_required=False,
            collector_executor_agent=gate['collector_executor_agent'], authorized_main_invocations=1,
            authorized_remote_incremental_invocations=1, prior_control_outers=list(range(5, 196, 5)),
            same_entry_prior_confirmed_main_invocations=prior_count, same_entry_prior_confirmed_remote_incremental_invocations=prior_count,
            actual200_optimizer_and_final_phase=TRAINING,
            source_CUM_dynamic_fields=FIELDS, source_CUM_complete_parts=len(PARTS), cost_mirrored_artifact_refs=len(cost['mirrored_artifacts']),
            preserved_input_refs=list(F.values()), local_readback_observations=two_pass(),
            invocation_count_scope='Actual prior source1ed SHA-bound executor receipt chain counted dynamically, corroborated by original immutable products; this preflight does not count outer200 or claim a global process audit.',
            same_entry_actual_prior_owner_chain=PRIOR_CHAIN, prior_inputs=prior_gate['prior_inputs'], prior_curve_byte_refs=PRIOR_CURVE,
            remote_operations=False, collector_invoked=False)
        print(json.dumps(publish('DEV5_200_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json', result)))
        return
    assert not (B / 'DEV5_200_NATIVE_COST_COLLECTION_ERROR.json').exists()
    execution = read('DEV5_200_NATIVE_COST_COLLECTOR_EXECUTION.json', args.execution_sha256, args.execution_bytes)
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
    pre = exact_ref(execution['pre_execution_input_pins'], 'DEV5_200_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json')
    assert pre['schema'] == 'dev200_local_preflight_input_pins_v1' and pre['outer'] == 200
    assert pre['status'] == 'LOCAL_INPUTS_READY_NO_COLLECTOR_INVOKED' and pre['collector_invoked'] is False
    assert pre['source_commit'] == ENTRY and pre['collector_source_sha256'] == SOURCE_SHA and pre['command'] == CMD
    assert pre['root_gate_input_pins'] == reference(str(args.root_pins))
    assert pre['preceding_pending_evidence'] == gate['pending_evidence']
    assert pre['collector_executor_agent'] == execution['executor_agent']
    for ref in pre['preserved_input_refs']:
        bound(ref)
    emitted = json.loads(execution['stdout'])
    assert emitted['controls'] == 40 and emitted['newly_read_remote_control_outers'] == [200]
    assert emitted['reused_original_control_outers'] == list(range(5, 196, 5))
    assert emitted['original_cumulative_unchanged'] is True and emitted['remote_files_written'] is False
    summ = exact_ref(emitted['supplement'], 'DEV5_200_NATIVE_COST_SUPPLEMENT.json')
    exact_ref(emitted['markdown'], 'DEV5_200_NATIVE_COST_SUPPLEMENT.md')
    assert summ['status'] == 'ACTUAL_COMPLETED_DEV_CONTROL_SUMMARY_COST_READBACK' and summ['through_outer'] == 200
    exact_ref(summ['source_cumulative'], FUTURE_NAMES[2])
    exact_ref(summ['source_full_receipt'], FUTURE_NAMES[0])
    exact_ref(summ['prior_immutable_supplement'], 'DEV5_195_NATIVE_COST_SUPPLEMENT.json')
    assert summ['model_identity'] == mod.MODEL and summ['data_identity'] == mod.DATA
    assert summ['split_identity']['evaluated_rows'] == 512 and summ['split_identity']['nq_rows'] == summ['split_identity']['hotpotqa_rows'] == 256
    assert summ['split_identity']['independent_split_file_sha256'] is None
    assert summ['read_only'] is True and summ['remote_files_written'] is False
    assert summ['jsonl_rollout_tensor_or_model_read'] is False and summ['immutable_cumulative_rewritten'] is False
    assert summ['STATUS_REPORT_FUP_or_training_modified'] is False and summ['tests_or_review_performed'] is False
    refs(summ)
    new_parts = row_parts(summ)
    rows = mod.rows_from_summary(B, M, summ)
    assert len(rows) == 40 and rows[:-1] == prior and [row['outer'] for row in rows] == list(range(5, 201, 5))
    assert summ['newly_read_remote_control_outers'] == [200] and summ['reused_original_control_outers'] == list(range(5, 196, 5))
    control_refs, pins = [], []
    for row, dev in zip(rows, devs):
        pin = dev_pin(mod, dev)
        pins.append(pin)
        bound(row['local_original_summary'])
        raw = R[str(locate(row['local_original_summary']['path']).resolve())]
        assert mod.control_row(pin, raw, row['source_control_summary'], row['local_original_summary']) == row
        control_refs.append(dict(outer=row['outer'], **reference(row['local_original_summary']['path'])))
    last = rows[-1]
    assert locate(last['local_original_summary']['path']).resolve() == (B / 'dev_native_cost_summaries/DEV_OUTER0200_SUMMARY.json').resolve()
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
    value = dict(schema='dev_native_cost_actual_local_execution_receipt_v2', outer=200,
        observed_utc=datetime.now(timezone.utc).isoformat(), status='ACTUAL_DEV200_COST_COLLECTED_AND_LOCAL_MIRRORS_READ_BACK',
        collector_executor_agent=execution['executor_agent'], local_readback_executor_agent=execution['executor_agent'],
        root_remote_collector_execution_claimed=False, new_entry_main_invocations_this_task=1,
        actual_remote_incremental_collection_invocations=1, preceding_local_pending_invocations=0,
        preceding_pending_remote_operations=False, preceding_pending_source_bound_evidence=gate['pending_evidence'],
        remote_collector_exit_code=execution['exit_code'], same_entry_prior_actual_executor_attestation=reference('DEV5_195_NATIVE_COST_LOCAL_RECEIPT.json'),
        same_entry_prior_confirmed_main_invocations=prior_count, same_entry_prior_confirmed_remote_incremental_invocations=prior_count,
        same_entry_executor_attested_cumulative_main_invocations=prior_count + 1, same_entry_executor_attested_cumulative_remote_incremental_invocations=prior_count + 1,
        invocation_count_scope='This actual outer200 delta main1/remote1/PENDING0 plus the dynamically counted prior source1ed SHA-bound executor receipt chain; products corroborate those attestations. No independent complete system-wide process-log audit is claimed.',
        same_entry_actual_prior_owner_chain=PRIOR_CHAIN, prior_inputs=prior_gate['prior_inputs'], prior_curve_byte_refs=PRIOR_CURVE,
        actual_dev200_metrics=actual['metrics'], full_outer200_wall_seconds=full['full_outer_wall_seconds'],
        actual200_optimizer_and_final_phase=TRAINING, full_metric_core_source=reference(str(CORE)),
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
        supplement=reference('DEV5_200_NATIVE_COST_SUPPLEMENT.json'), markdown=reference('DEV5_200_NATIVE_COST_SUPPLEMENT.md'),
        prior_immutable_supplement=reference('DEV5_195_NATIVE_COST_SUPPLEMENT.json'),
        prior_root_actual_readback=reference('ROOT_DEV195_NATIVE_COST_ACTUAL_READBACK.json'),
        controls=len(rows), reused_original_control_outers=list(range(5, 196, 5)), newly_read_remote_control_outers=[200],
        local_original_control_refs=control_refs, rows_storage=summ['rows'].get('storage') if isinstance(summ['rows'], dict) else 'inline_list',
        rows_part_refs=new_parts, prior_rows_part_refs=prior_parts,
        rows_readback_function='SHA-bound collect_nested_dev_costs.rows_from_summary; no collector main invoked by this builder',
        prior39_rows_exact_object_equal=True, prior39_original_controls_SHA_and_bytes_preserved=True,
        prior_DEV195_and_all_preceding_owner_rows_parts_SHA_and_bytes_preserved=True,
        prior_curve_objects_complete_file_bytes_preserved=True, prior_curve_nested_registry_refs_are_historical_descriptors=True,
        historical_registry_original_bytes_currently_verified=False,
        new_dev200_original_control=last['source_control_summary'], new_dev200_counters=last['counters'],
        new_dev200_observer_wall_seconds=last['wall_seconds'], new_dev200_trainer_process_cpu_seconds=last['trainer_process_cpu_seconds'],
        new_dev200_ledger_sha256=last['ledger_sha256'], new_dev200_control_summary_sha256=last['local_original_summary']['sha256'],
        control_summary_SHA_and_value_ledger_SHA_have_distinct_roles=True,
        fullmetric_dev200_wall_seconds=full['normal_dev512_wall_seconds'], native_counter_totals=summ['native_counter_totals'],
        original_validation_observer_wall_seconds=summ['original_validation_observer_wall_seconds'],
        original_validation_trainer_process_cpu_seconds=summ['original_validation_trainer_process_cpu_seconds'],
        precise_engine_emitted_tokens=None, scoring_call_event_total=None,
        new_remote_metadata_collector_cpu_seconds=summ['collector_cpu_seconds'], metadata_collector_cpu_scope=summ['collector_cpu_scope'],
        accounting_scope='Dev observer wall and trainer-process CPU are contained in FULL testing/service scopes; not added again to CUM or service CPU. Metadata collector CPU is a separate small readback process. Submitted prefill repeats events; returned nonpad is not precise engine emitted; HTTP calls and query events remain distinct.',
        execution_record=reference('DEV5_200_NATIVE_COST_COLLECTOR_EXECUTION.json'),
        pre_execution_input_pins=reference('DEV5_200_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json'),
        owner_receipt_builder=reference(str(Path(__file__).resolve())), adapted_from_reader_source=reference('work/tree-speed-repair-20261008/dev180_owner_local_receipt.py'),
        mirrors_actual_byte_readback_passes=2, local_readback_observations=observations,
        actual_mirror_roots=[str(p) for p in M], audited_local_files=sorted(F.values(), key=lambda x: x['path']),
        source_raw_poll_resource_scope='Existing raw poll/resource are local BASE-only SHA-bound observations; product artifacts are read from BASE and both mirrors twice. Root gate remote descriptors are not followed by this script. No new raw remote read is claimed.',
        old_DEV195_and_prior_control_parts_curve_outputs_rewritten=False, immutable_CUM_rewritten=False, training_or_evaluation_modified=False,
        tensor_model_or_JSONL_read=False, canonical_STATUS_REPORT_FUP_observer_modified=False, tests_or_review_performed=False,
        ROOT_FULL200_or_slow_final51713_completion_claimed=False, actual_training201_entry_claimed=False)
    print(json.dumps(publish('DEV5_200_NATIVE_COST_LOCAL_RECEIPT.json', value)))


if __name__ == '__main__':
    main()
