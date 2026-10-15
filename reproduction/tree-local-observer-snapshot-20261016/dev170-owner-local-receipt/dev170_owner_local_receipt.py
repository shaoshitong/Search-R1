#!/usr/bin/env python3
"""Local metadata preflight and owner receipt only; never invokes SSH or collector."""
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
INNER = '846e3e75def4d13d2d130b90dff648cbd8cef3cf7ced987ffc606d335661867b'
CMD = 'env TREE_MGMT_CONTROL_PATH=/tmp/tree-mgmt-3fw9a1ab/control python3 -B work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py --base outputs/tree-nested-branch-handoff-20261013 --ssh-helper work/tree-speed-repair-20261008/ssh_retained18.py --through-outer 170'
F, R, PARTS, FIELDS, REMOTE_REFS = {}, {}, [], {}, []

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()

def locate(name):
    p = Path(name)
    return p if p.is_absolute() else W / p if p.parts[0] in ('outputs', 'work') else B / p

def read(name, digest=None, size=None):
    p = locate(name).resolve(strict=True)
    assert p.is_relative_to(W), str(p)
    assert p.suffix in ('.json', '.md', '.py', '.patch', '.csv', '.yaml', '.yml', '.sh'), str(p)
    raw = p.read_bytes()
    assert len(raw) < 4000000, str(p)
    assert digest is None or sha(raw) == digest, 'SHA ' + str(p)
    assert size is None or len(raw) == size, 'bytes ' + str(p)
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
    name = ref.get('path', ref.get('file', hint))
    assert isinstance(name, str), (ref, hint)
    if name.startswith(('/dataset/', '/mnt/')):
        REMOTE_REFS.append(ref)
        return None
    return read(name, ref['sha256'], ref['bytes'])

def refs(value):
    if isinstance(value, list):
        for item in value:
            refs(item)
    elif isinstance(value, dict):
        if isinstance(value.get('sha256'), str) and len(value['sha256']) == 64 and type(value.get('bytes')) is int and (value.get('path') or value.get('file')):
            bound(value)
            return
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(item.get('sha256'), str) and len(item['sha256']) == 64 and type(item.get('bytes')) is int and Path(key).suffix in ('.json', '.md', '.py', '.patch', '.csv', '.yaml', '.yml', '.sh'):
                bound(item, key)
            else:
                refs(item)

def expand(value, field):
    if isinstance(value, dict) and value.get('storage') == 'complete_original_field_parts':
        kind = value['container_type']
        assert kind in ('list', 'mapping')
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'receipt'])
    args = parser.parse_args()
    read(str(S), SOURCE_SHA, 27911)
    read(str(H), HELPER_SHA, 4411)
    spec = importlib.util.spec_from_file_location('dev170_local_owner_collector_functions', S)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    oldowner = read('DEV5_165_NATIVE_COST_LOCAL_RECEIPT.json')
    olderowner = read('DEV5_160_NATIVE_COST_LOCAL_RECEIPT.json')
    assert oldowner['collector_source_commit'] == olderowner['collector_source_commit'] == ENTRY
    assert oldowner['collector_source_sha256'] == olderowner['collector_source_sha256'] == SOURCE_SHA
    assert oldowner['same_entry_executor_attested_cumulative_main_invocations'] == oldowner['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == 2
    old = read('DEV5_165_NATIVE_COST_SUPPLEMENT.json')
    prior = mod.rows_from_summary(B, M, old)
    assert len(prior) == 33 and [row['outer'] for row in prior] == list(range(5, 166, 5))
    refs(old)
    for row in prior:
        bound(row['local_original_summary'])
    full = read('FULL_OUTER170_PROGRESS_RECEIPT.json', 'a1d5f8b8a8af07324594657e5b26b16a670dba5d3ecd27fb7d27dac8040390be', 31896)
    native = read('NATIVE_HISTORY_170_METADATA_READBACK.json', '923b57c2d99790babfc843ea66ce497dd6b94d7fa616dcaed94e432bf8e0b635', 8355)
    rawcum = read('CUMULATIVE_COST_THROUGH170.json')
    cumulative = {key: expand(value, key) for key, value in rawcum.items()}
    assert full['completed_outer'] == cumulative['through_effective_outer'] == native['outer'] == 170
    assert full['source_commit'] == mod.CHILD and full['config_sha256'] == mod.CONFIGS[mod.CHILD]
    assert full['actual_training_pid'] == 503255 and full['actual_training_invocation_id'] == 'a313bda919834d24b4c5d491698d39ce'
    actual = full['actual_dev512']
    assert actual['outer'] == 170 and actual['rows'] == 512 and actual['ledger_sha256'] == INNER
    assert actual['metrics']['val/qa_count/all'] == 512 and actual['metrics']['val/qa_count/nq'] == actual['metrics']['val/qa_count/hotpotqa'] == 256
    assert actual['metrics']['val/f1/all'] == 0.5742947048611112 and actual['metrics']['val/em/all'] == 0.490234375
    assert [row['outer'] for row in cumulative['completed_dev_evaluations']] == list(range(5, 171, 5))
    cost = read('COST_OUTER170_LOCAL_RECEIPT.json')
    assert cost['outer'] == 170 and cost['status'] == 'ACTUAL_OUTER170_COST_ARCHIVED'
    refs(cost)
    if args.phase == 'preflight':
        assert not (B / 'DEV5_170_NATIVE_COST_SUPPLEMENT.json').exists()
        assert not (B / 'DEV5_170_NATIVE_COST_COLLECTION_ERROR.json').exists()
        result = dict(schema='dev170_local_preflight_input_pins_v1', observed_utc=datetime.now(timezone.utc).isoformat(), status='LOCAL_INPUTS_READY_NO_COLLECTOR_INVOKED', command=CMD, source_commit=ENTRY, collector_source_sha256=SOURCE_SHA, helper_sha256=HELPER_SHA, preserved_input_refs=list(F.values()), prior_control_outers=list(range(5, 166, 5)), local_readback_observations=two_pass(), remote_operations=False, collector_invoked=False)
        print(json.dumps(publish('DEV5_170_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json', result)))
        return
    pre = read('DEV5_170_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json')
    for ref in pre['preserved_input_refs']:
        bound(ref)
    execution = read('DEV5_170_NATIVE_COST_COLLECTOR_EXECUTION.json')
    assert execution['actual_command'] == CMD and execution['exit_code'] == 0
    assert execution['new_entry_main_invocations_this_task'] == execution['actual_remote_incremental_collection_invocations'] == 1
    assert execution['preceding_local_pending_invocations'] == 0 and execution['preceding_pending_remote_operations'] is False
    emitted = json.loads(execution['stdout'])
    assert emitted['controls'] == 34 and emitted['newly_read_remote_control_outers'] == [170] and emitted['reused_original_control_outers'] == list(range(5, 166, 5))
    summ = read('DEV5_170_NATIVE_COST_SUPPLEMENT.json')
    read('DEV5_170_NATIVE_COST_SUPPLEMENT.md')
    refs(summ)
    rows = mod.rows_from_summary(B, M, summ)
    assert len(rows) == 34 and rows[:-1] == prior and [row['outer'] for row in rows] == list(range(5, 171, 5))
    assert summ['newly_read_remote_control_outers'] == [170] and summ['reused_original_control_outers'] == list(range(5, 166, 5))
    control_refs, pins = [], []
    for row, dev in zip(rows, cumulative['completed_dev_evaluations']):
        n = row['outer']
        sc = mod.PARENT if n == 5 else mod.CHILD
        run = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + sc
        pin = dict(outer=n, code_commit=sc, config_sha256=mod.CONFIGS[sc], model_identity=mod.MODEL, data_identity=mod.DATA, job_id='not-slurm;systemd:' + sc, run=run, path=run + '/output/validation-cost/' + dev['ledger'], phase=dev['phase'], ledger_sha256=dev['ledger_sha256'], F1=dev['F1'], EM=dev['EM'])
        pins.append(pin)
        bound(row['local_original_summary'])
        raw = R[str(locate(row['local_original_summary']['path']).resolve())]
        assert mod.control_row(pin, raw, row['source_control_summary'], row['local_original_summary']) == row
        control_refs.append(dict(outer=n, **reference(row['local_original_summary']['path'])))
    last = rows[-1]
    assert last['ledger_sha256'] == INNER and last['precise_engine_emitted_tokens'] is None and last['scoring_call_event_total'] is None
    program = sha(mod.REMOTE.replace('PIN_DATA', repr(pins[-1:])).encode())
    assert summ['remote_program_sha256'] == program and summ['collector_source_sha256'] == SOURCE_SHA and summ['ssh_helper_sha256'] == HELPER_SHA
    assert summ['precise_engine_emitted_tokens'] is None and summ['scoring_call_event_total'] is None
    for key, record in summ['native_counter_totals'].items():
        values = [(row['counters'] or {}).get(key) for row in rows]
        known = [v for v in values if mod.numeric(v)]
        total = math.fsum(known) if known else None
        assert record == dict(total=total if len(known) == len(values) else None, known_sum=total, missing_outers=[row['outer'] for row, value in zip(rows, values) if not mod.numeric(value)])
    observations = two_pass()
    value = dict(schema='dev_native_cost_actual_local_execution_receipt_v2', outer=170, observed_utc=datetime.now(timezone.utc).isoformat(), status='ACTUAL_DEV170_COST_COLLECTED_AND_LOCAL_MIRRORS_READ_BACK', collector_executor_agent='/root/dev170_cost_once', local_readback_executor_agent='/root/dev170_cost_once', root_remote_collector_execution_claimed=False, new_entry_main_invocations_this_task=1, actual_remote_incremental_collection_invocations=1, preceding_local_pending_invocations=0, preceding_pending_remote_operations=False, remote_collector_exit_code=0, same_entry_prior_actual_executor_attestation=reference('DEV5_165_NATIVE_COST_LOCAL_RECEIPT.json'), same_entry_prior_confirmed_main_invocations=2, same_entry_prior_confirmed_remote_incremental_invocations=2, same_entry_executor_attested_cumulative_main_invocations=3, same_entry_executor_attested_cumulative_remote_incremental_invocations=3, invocation_count_scope='This outer170 delta main1/remote1/PENDING0; same source1ed SHA-bound executor attestations160+165+170 total main3/remote3, corroborated by products. No independent complete system-wide process-log audit is claimed.', actual_dev170_metrics=actual['metrics'], full_outer170_wall_seconds=full['full_outer_wall_seconds'], actual_command_template=CMD, collector_source_commit=ENTRY, collector_source=reference(str(S)), collector_source_sha256=SOURCE_SHA, remote_program_sha256=program, ssh_helper_sha256=HELPER_SHA, source_FULL=reference('FULL_OUTER170_PROGRESS_RECEIPT.json'), source_CUM=reference('CUMULATIVE_COST_THROUGH170.json'), source_native_metadata=reference('NATIVE_HISTORY_170_METADATA_READBACK.json'), cost_outer_receipt=reference('COST_OUTER170_LOCAL_RECEIPT.json'), source_CUM_completed_dev_storage=rawcum['completed_dev_evaluations'].get('storage') if isinstance(rawcum['completed_dev_evaluations'], dict) else 'inline_list', source_CUM_completed_dev_entries=34, source_CUM_all_complete_field_parts=[dict(path=str(B.relative_to(W) / part['file']), bytes=part['bytes'], sha256=part['sha256'], original_field=part['original_field']) for part in PARTS], source_CUM_dynamic_fields=FIELDS, source_CUM_original_bytes_sha_retained=True, source_CUM_parsed_dev_view_does_not_rewrite_original=True, supplement=reference('DEV5_170_NATIVE_COST_SUPPLEMENT.json'), markdown=reference('DEV5_170_NATIVE_COST_SUPPLEMENT.md'), prior_immutable_supplement=reference('DEV5_165_NATIVE_COST_SUPPLEMENT.json'), controls=34, reused_original_control_outers=list(range(5, 166, 5)), newly_read_remote_control_outers=[170], local_original_control_refs=control_refs, rows_storage=summ['rows'].get('storage') if isinstance(summ['rows'], dict) else 'inline_list', rows_part_refs=[dict(path=str(B.relative_to(W) / p['file']), bytes=p['bytes'], sha256=p['sha256']) for p in summ['rows'].get('parts', [])] if isinstance(summ['rows'], dict) else [], prior_rows_part_refs=[dict(path=str(B.relative_to(W) / p['file']), bytes=p['bytes'], sha256=p['sha256']) for p in old['rows'].get('parts', [])] if isinstance(old['rows'], dict) else [], rows_readback_function='reviewed collect_nested_dev_costs.rows_from_summary', prior33_rows_exact_object_equal=True, prior33_original_controls_SHA_and_bytes_preserved=True, new_dev170_original_control=last['source_control_summary'], new_dev170_counters=last['counters'], new_dev170_observer_wall_seconds=last['wall_seconds'], new_dev170_trainer_process_cpu_seconds=last['trainer_process_cpu_seconds'], new_dev170_ledger_sha256=last['ledger_sha256'], new_dev170_control_summary_sha256=last['local_original_summary']['sha256'], control_summary_SHA_and_value_ledger_SHA_have_distinct_roles=True, fullmetric_dev170_wall_seconds=full['normal_dev512_wall_seconds'], native_counter_totals=summ['native_counter_totals'], original_validation_observer_wall_seconds=summ['original_validation_observer_wall_seconds'], original_validation_trainer_process_cpu_seconds=summ['original_validation_trainer_process_cpu_seconds'], precise_engine_emitted_tokens=None, scoring_call_event_total=None, new_remote_metadata_collector_cpu_seconds=summ['collector_cpu_seconds'], metadata_collector_cpu_scope=summ['collector_cpu_scope'], accounting_scope='Dev observer wall/CPU contained in FULL testing/service scopes; not added again to CUM or service CPU. Submitted prefill repeats events; returned nonpad is not precise engine emitted; HTTP calls and query events remain distinct.', execution_record=reference('DEV5_170_NATIVE_COST_COLLECTOR_EXECUTION.json'), pre_execution_input_pins=reference('DEV5_170_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json'), owner_receipt_builder=reference(str(Path(__file__).resolve())), mirrors_actual_byte_readback_passes=2, local_readback_observations=observations, actual_mirror_roots=[str(p) for p in M], audited_local_files=sorted(F.values(), key=lambda x: x['path']), source_raw_poll_resource_scope='Existing raw poll/resource are local BASE-only SHA-bound observations; product artifacts are read from BASE and both mirrors twice. No new raw remote read is claimed.', old_DEV165_outputs_rewritten=False, immutable_CUM_rewritten=False, training_or_evaluation_modified=False, tensor_model_or_JSONL_read=False, canonical_STATUS_REPORT_FUP_observer_modified=False, tests_or_review_performed=False, ROOT_FULL170_or_final200_or51713_completion_claimed=False)
    print(json.dumps(publish('DEV5_170_NATIVE_COST_LOCAL_RECEIPT.json', value)))

if __name__ == '__main__':
    main()
