#!/usr/bin/env python3
"""Future one-time independent actual DEV190 local bytes readback.

Prepared from frozen dev185_independent_actual_readback.py (52848 bytes, SHA256
74c2cdabc404182ed3b09cb1f9239f4f0d87e3daf18df93e11e939ce209352f5).
Current real successes end at180 (5); future185/190 are never presumed successful.
This source is not a completion artifact, a training gate, or a new review.
Run only after the actual unique source1ed main1/remote1 collector exits0 and
the frozen owner publishes its actual successful DEV190 local receipt.
All unknown future pins are required CLI inputs or read from SHA-bound actual
artifacts. The reader parses original local bytes itself; owner PASS fields
cannot substitute for the input/control/part/execution checks below.
No SSH, collector import/main, evaluation, tensor, JSONL, or training operation.

Future invocation, with values captured from actual future local bytes:
  python3 -B work/tree-speed-repair-20261008/dev190_independent_actual_readback.py \
    --root-pins "$ACTUAL_ROOT_GATE" --root-pins-bytes "$ACTUAL_ROOT_GATE_BYTES" \
    --root-pins-sha256 "$ACTUAL_ROOT_GATE_SHA256" \
    --owner-bytes "$ACTUAL_OWNER_BYTES" --owner-sha256 "$ACTUAL_OWNER_SHA256" \
    --execution-bytes "$ACTUAL_EXECUTION_BYTES" \
    --execution-sha256 "$ACTUAL_EXECUTION_SHA256"
It publishes only ROOT_DEV190_NATIVE_COST_ACTUAL_READBACK.json, after actual
input checks and two real byte-readback passes, to BASE and both local mirrors.
An existing output is never overwritten. This file must not be run during
source preparation; no future190 success, pins, counters, or metrics exist here.
"""
import argparse
import ast
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
SCRIPT_ROOT = W / 'work/tree-speed-repair-20261008'
CORE = SCRIPT_ROOT / 'nested-native5-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/sync_nested_native_progress.py'
CORE_SHA = '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
S = SCRIPT_ROOT / 'nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py'
ORIGINAL = SCRIPT_ROOT / 'dev185_independent_actual_readback.py'
OWNER_SOURCE = SCRIPT_ROOT / 'dev190_owner_local_receipt.py'
# Final owner source is frozen independently; this is a source pin, not future success.
OWNER_SOURCE_PIN = (54157, 'c425e121f0ce978976d773193abebea17ed5c399f523132d1e2020082ad1da5a')
ENTRY = '1edaf50bb8e14de51db936e0859e369792011233'
SOURCE_SHA = '545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0'
HELPER_SHA = '8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'
C = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
P = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
CONFIGS = {C: '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3',
           P: '22beaec8aa2b449b496754aacfd9b1e7c89009c9c27a8eff3aef92b768b28b59'}
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + C
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
INV = 'a313bda919834d24b4c5d491698d39ce'
CMD = 'env TREE_MGMT_CONTROL_PATH=/tmp/tree-mgmt-3fw9a1ab/control python3 -B work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py --base outputs/tree-nested-branch-handoff-20261013 --ssh-helper work/tree-speed-repair-20261008/ssh_retained18.py --through-outer 190'
FUTURE_NAMES = ('FULL_OUTER190_PROGRESS_RECEIPT.json', 'NATIVE_HISTORY_190_METADATA_READBACK.json',
                'CUMULATIVE_COST_THROUGH190.json', 'COST_OUTER190_LOCAL_RECEIPT.json')
OWNER_NAME = 'DEV5_190_NATIVE_COST_LOCAL_RECEIPT.json'
EXECUTION_NAME = 'DEV5_190_NATIVE_COST_COLLECTOR_EXECUTION.json'
PRE_NAME = 'DEV5_190_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json'
SUPPLEMENT_NAME = 'DEV5_190_NATIVE_COST_SUPPLEMENT.json'
OUTPUT_NAME = 'ROOT_DEV190_NATIVE_COST_ACTUAL_READBACK.json'

F, R, REMOTE_REFS = {}, {}, []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def unique(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, 'duplicate JSON key: ' + key
        result[key] = value
    return result


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), 'nonfinite JSON scalar'
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def load(raw):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    finite(value)
    return value


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
    assert str(p) not in F or F[str(p)] == ref and R[str(p)] == raw, 'changed ' + str(p)
    if mirrored:
        for mirror in M:
            assert (mirror / p.relative_to(B)).read_bytes() == raw, 'mirror ' + str(p)
    F[str(p)], R[str(p)] = ref, raw
    return load(raw) if p.suffix == '.json' else raw


def reference(name):
    p = locate(name).resolve(strict=True)
    assert str(p) in F, 'reference before actual read: ' + str(p)
    return {key: value for key, value in F[str(p)].items() if key != 'mirror_checked'}


def bound(ref, hint=None):
    assert isinstance(ref, dict) and digest_ok(ref.get('sha256')), ref
    assert type(ref.get('bytes')) is int and ref['bytes'] > 0, ref
    name = ref.get('path') or ref.get('file') or hint
    assert isinstance(name, str), (ref, hint)
    if name.startswith(('/dataset/', '/mnt/')):
        if ref not in REMOTE_REFS:
            REMOTE_REFS.append(ref)
        return None
    return read(name, ref['sha256'], ref['bytes'])


def exact_ref(ref, name):
    assert locate(ref.get('path') or ref.get('file') or '').resolve(strict=True) == locate(name).resolve(strict=True), name
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


def part_file(ref):
    name = ref['file']
    assert isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts
    assert ref.get('json_selector') == 'value'
    return str(locate(name).resolve(strict=True))


def expand(value, field, parts, fields):
    if isinstance(value, dict) and value.get('storage') == 'complete_original_field_parts':
        kind = value['container_type']
        assert kind in ('list', 'mapping') and isinstance(value['parts'], list) and value['parts']
        assert type(value['original_entries']) is int and value['original_entries'] >= 0
        result = [] if kind == 'list' else {}
        seen = {part_file(ref) for ref in parts}
        for ref in value['parts']:
            name = part_file(ref)
            assert name not in seen, 'duplicate cumulative part: ' + name
            seen.add(name)
            part = bound(ref)
            assert part['schema'] == 'cumulative_scalar_field_part_v1' and part['original_field'] == field
            chunk = part['value']
            assert isinstance(chunk, list if kind == 'list' else dict)
            if kind == 'list':
                result.extend(chunk)
            else:
                assert not set(result).intersection(chunk), 'duplicate cumulative mapping entries'
                result.update(chunk)
            parts.append(dict(original_field=field, **ref))
        assert len(result) == value['original_entries'], 'missing cumulative entries: ' + field
        assert field not in fields
        fields[field] = dict(entries=len(result), parts=len(value['parts']), container_type=kind)
        return expand(result, field, parts, fields)
    if isinstance(value, dict):
        return {key: expand(item, field + '.' + key, parts, fields) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, field + '[' + str(i) + ']', parts, fields) for i, item in enumerate(value)]
    return value


def rows_from_summary(summary):
    value, parts = summary['rows'], []
    if isinstance(value, list):
        rows = value
    else:
        assert value['storage'] == 'complete_original_field_parts' and value['container_type'] == 'list'
        assert type(value['original_entries']) is int and value['original_entries'] >= 0
        assert isinstance(value['parts'], list) and value['parts']
        rows, seen = [], set()
        for ref in value['parts']:
            name = part_file(ref)
            assert name not in seen, 'duplicate DEV row part: ' + name
            seen.add(name)
            part = bound(ref)
            assert isinstance(part, dict) and isinstance(part['value'], list)
            rows.extend(part['value'])
            parts.append(dict(ref))
        assert len(rows) == value['original_entries'], 'missing DEV rows'
    assert all(isinstance(row, dict) for row in rows)
    assert all(type(row['outer']) is int for row in rows)
    assert len({row['outer'] for row in rows}) == len(rows), 'duplicate DEV outer'
    return rows, parts


def dev_pin(dev):
    n = dev['outer']
    sc = P if n == 5 else C
    run = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + sc
    assert isinstance(dev['ledger'], str) and Path(dev['ledger']).name == dev['ledger']
    assert dev['rows'] == 512 and dev['phase'] == ('acceptance' if n == 5 else 'periodic') and dev.get('source_commit', sc) == sc
    assert dev['ledger_sha256'] is None or digest_ok(dev['ledger_sha256'])
    assert numeric(dev['F1']) and numeric(dev['EM']) and 0 <= dev['F1'] <= 1 and 0 <= dev['EM'] <= 1
    return dict(outer=n, code_commit=sc, config_sha256=CONFIGS[sc], model_identity=MODEL,
                data_identity=DATA, job_id='not-slurm;systemd:' + sc, run=run,
                path=run + '/output/validation-cost/' + dev['ledger'], phase=dev['phase'],
                ledger_sha256=dev['ledger_sha256'], F1=dev['F1'], EM=dev['EM'])


def control_row(pin, raw, meta, saved):
    # Same row contract as pinned source1ed, independently parsed here.
    value = load(raw)
    for key in ('code_commit', 'config_sha256', 'data_identity', 'model_identity', 'job_id'):
        assert value.get(key) == pin[key], 'control identity: ' + key
    assert sha(raw) == meta['sha256'] and len(raw) == meta['bytes'] and meta['path'] == pin['path']
    assert meta.get('stable_read') is True and isinstance(meta['resolved_path'], str)
    assert meta['resolved_path'].startswith(('/dataset/', '/mnt/'))
    assert value.get('run') in (pin['run'] + '/output', str(Path(meta['resolved_path']).parent.parent))
    assert value['actual_completed_outer'] == pin['outer'] and value['status'] == 'VALIDATION_COST_COMPLETE'
    assert value['evaluated_rows'] == value['dataset_rows'] == 512 and value['phase'] == pin['phase']
    metrics = value['metrics']
    assert metrics['val/qa_count/all'] == 512 and metrics['val/qa_count/nq'] == metrics['val/qa_count/hotpotqa'] == 256
    assert metrics['val/f1/all'] == pin['F1'] and metrics['val/em/all'] == pin['EM']
    assert digest_ok(value['ledger_sha256']) and (pin['ledger_sha256'] is None or value['ledger_sha256'] == pin['ledger_sha256'])
    assert isinstance(value.get('counters'), (dict, type(None)))
    for counter in (value.get('counters') or {}).values():
        assert counter is None or numeric(counter) and counter >= 0
    for key in ('wall_seconds', 'trainer_process_cpu_seconds'):
        assert value.get(key) is None or numeric(value[key]) and value[key] >= 0
    return dict(outer=pin['outer'], code_commit=pin['code_commit'], config_sha256=pin['config_sha256'],
        run=pin['run'], phase=value['phase'], actual_completed_outer=value['actual_completed_outer'],
        evaluated_rows=value['evaluated_rows'], counters=value.get('counters'), source_control_summary=meta,
        local_original_summary=saved, ledger_sha256=value.get('ledger_sha256'),
        start_epoch=value.get('start_epoch'), end_epoch=value.get('end_epoch'),
        wall_seconds=value.get('wall_seconds'), trainer_process_cpu_seconds=value.get('trainer_process_cpu_seconds'),
        pad_token_id=value.get('pad_token_id'), eos_token_id=value.get('eos_token_id'),
        observer_source_sha256=value.get('observer_source_sha256'), cost_count_scope=value.get('cost_count_scope'),
        limitations=value.get('limitations'), precise_engine_emitted_tokens=None, scoring_call_event_total=None)


def aggregate(rows, values):
    known = [value for value in values if numeric(value)]
    total = math.fsum(known) if known else None
    return dict(total=total if len(known) == len(values) else None, known_sum=total,
                missing_outers=[row['outer'] for row, value in zip(rows, values) if not numeric(value)])


def executor_counts(owner, outer, cumulative=None):
    assert owner['outer'] == outer and owner['collector_source_commit'] == ENTRY
    assert owner['collector_source_sha256'] == SOURCE_SHA
    for key in ('new_entry_main_invocations_this_task', 'actual_remote_incremental_collection_invocations'):
        assert type(owner[key]) is int and owner[key] == 1
    assert type(owner['preceding_local_pending_invocations']) is int and owner['preceding_local_pending_invocations'] == 0
    assert owner['preceding_pending_remote_operations'] is False
    assert type(owner['remote_collector_exit_code']) is int and owner['remote_collector_exit_code'] == 0
    if cumulative is not None:
        for key in ('same_entry_executor_attested_cumulative_main_invocations', 'same_entry_executor_attested_cumulative_remote_incremental_invocations'):
            assert type(owner[key]) is int and owner[key] == cumulative


def two_pass():
    observations = []
    for number in range(2):
        for name, ref in F.items():
            p, raw = Path(name), Path(name).read_bytes()
            assert raw == R[name] and len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], name
            if ref['mirror_checked']:
                for mirror in M:
                    assert (mirror / p.relative_to(B)).read_bytes() == raw, name
        observations.append(dict(input_readback_pass=number + 1, observed_utc=datetime.now(timezone.utc).isoformat(),
            unique_files=len(F), mirrored_files=sum(ref['mirror_checked'] for ref in F.values()), mirror_roots=[str(m) for m in M]))
        print(json.dumps(observations[-1]))
        if number == 0:
            time.sleep(1)
    return observations


def publish(value):
    raw = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
    assert 0 < len(raw) <= 100000, 'small independent receipt budget'
    targets = [folder / OUTPUT_NAME for folder in [B] + M]
    # A second invocation does not replace or regenerate a successful receipt.
    assert not any(target.exists() for target in targets), 'independent actual receipt already exists'
    for target in targets:
        assert target.parent.is_dir()
    for target in targets:
        with target.open('xb') as stream:
            stream.write(raw)
    publication_observations = []
    for number in range(2):
        for target in targets:
            assert target.read_bytes() == raw
        for name, ref in F.items():
            p = Path(name)
            assert p.read_bytes() == R[name], name
            if ref['mirror_checked']:
                for mirror in M:
                    assert (mirror / p.relative_to(B)).read_bytes() == R[name], name
        observation = dict(receipt_readback_pass=number + 1, observed_utc=datetime.now(timezone.utc).isoformat(),
                           copies=len(targets), source_inputs_and_mirrors_preserved=True)
        publication_observations.append(observation)
        print(json.dumps(observation))
        if number == 0:
            time.sleep(1)
    print(json.dumps(dict(status=value['status'], path=str(targets[0]), bytes=len(raw), sha256=sha(raw),
        publication_readback_observations=publication_observations, actual_ref_counts=value['actual_ref_counts'])))



# Future prior185 bytes are supplied by the actual gate/request, never fixed180 pins.
PRIOR_KEYS = {'outer','supplement','owner_receipt','root_actual_readback','execution','original_control','full','native','cumulative','cost','curve'}
PRIOR_CURVE_KEYS = {'json','csv','publication','binding','root_readback'}
PRIOR_ASOF_EXCLUSIONS = []
PRIOR_NAMES = dict(supplement='DEV5_185_NATIVE_COST_SUPPLEMENT.json',owner_receipt='DEV5_185_NATIVE_COST_LOCAL_RECEIPT.json',root_actual_readback='ROOT_DEV185_NATIVE_COST_ACTUAL_READBACK.json',execution='DEV5_185_NATIVE_COST_COLLECTOR_EXECUTION.json',original_control='dev_native_cost_summaries/DEV_OUTER0185_SUMMARY.json',full='FULL_OUTER185_PROGRESS_RECEIPT.json',native='NATIVE_HISTORY_185_METADATA_READBACK.json',cumulative='CUMULATIVE_COST_THROUGH185.json',cost='COST_OUTER185_LOCAL_RECEIPT.json')
PRIOR_CURVE_NAMES = dict(json='LEARNING_CURVE_COMPLETE_THROUGH185.json',csv='LEARNING_CURVE_COMPLETE_THROUGH185.csv',publication='LEARNING_CURVE_THROUGH185_LOCAL_PUBLICATION.json',binding='LEARNING_CURVE_THROUGH185_COST_SUPPLEMENT_BINDING.json',root_readback='ROOT_CURVE_DEV_COST_THROUGH185_ACTUAL_READBACK.json')


def verify_prior_inputs(prior_inputs, mod, core):
    """Verify only actual185 SHA-bound evidence; no prior success is defaulted."""
    import csv, io
    assert isinstance(prior_inputs,dict) and set(prior_inputs)==PRIOR_KEYS and prior_inputs['outer']==185
    assert set(prior_inputs['curve'])==PRIOR_CURVE_KEYS
    old={name:exact_ref(prior_inputs[key],name) for key,name in PRIOR_NAMES.items()}
    curve={key:exact_ref(prior_inputs['curve'][key],name) for key,name in PRIOR_CURVE_NAMES.items()}
    # Only actual descriptor occurrences in these three exact old185 documents
    # are asof exceptions. The original registry raw bytes are not read/verified.
    def provenance_refs(value):
        if isinstance(value,list):
            for item in value:yield from provenance_refs(item)
        elif isinstance(value,dict):
            if digest_ok(value.get('sha256')) and type(value.get('bytes')) is int and (value.get('path') or value.get('file')):
                yield value
            else:
                for key,item in value.items():
                    if isinstance(item,dict) and digest_ok(item.get('sha256')) and type(item.get('bytes')) is int and not (item.get('path') or item.get('file')) and Path(key).suffix in ('.json','.csv','.md','.py','.txt','.sh','.patch'):
                        yield dict(path=key,**item)
                    else:yield from provenance_refs(item)
    registry_path=(B/'CANDIDATE_COST_EXTENSION_INPUTS.json').resolve(strict=True)
    actual_asof_identity=None
    for key in ('publication','binding','root_readback'):
        source_document=reference(PRIOR_CURVE_NAMES[key]);occurrences=[]
        for ref in provenance_refs(curve[key]):
            name=ref.get('path',ref.get('file'))
            if not name.startswith(('/dataset/','/mnt/')) and locate(name).resolve(strict=True)==registry_path:
                assert type(ref['bytes']) is int and ref['bytes']>0 and digest_ok(ref['sha256'])
                identity=(str(registry_path),ref['bytes'],ref['sha256'])
                assert actual_asof_identity is None or actual_asof_identity==identity
                actual_asof_identity=identity;occurrences.append(dict(ref))
            else:bound(ref)
        assert occurrences, 'Each exact185 curve provenance document must contain its real asof registry descriptor'
        record=dict(source_document=source_document,actual_historical_asof_registry_descriptors=occurrences,
            actual_occurrences=len(occurrences),source_document_original_bytes_verified=True,
            historical_registry_original_raw_bytes_currently_verified=False,
            scope='Only these exact actual185 original-document descriptor occurrences are asof; every other enumerated reference remains strict. Current190 registry is separately bound only from actual COST190.')
        if record not in PRIOR_ASOF_EXCLUSIONS:PRIOR_ASOF_EXCLUSIONS.append(record)
    owner=old[PRIOR_NAMES['owner_receipt']];root=old[PRIOR_NAMES['root_actual_readback']]
    execution=old[PRIOR_NAMES['execution']];full=old[PRIOR_NAMES['full']];native=old[PRIOR_NAMES['native']]
    assert owner['outer']==185 and owner['status']=='ACTUAL_DEV185_COST_COLLECTED_AND_LOCAL_MIRRORS_READ_BACK'
    assert root['outer']==185 and root['status']=='PASS_ACTUAL_DEV185_LOCAL_ARTIFACT_READBACK' and root['inconsistencies']==[]
    assert root['actual_ref_counts']['new_dev_rows']==root['actual_ref_counts']['all_original_control_summaries']==37
    for name,ref in root['source_pins'].items():exact_ref(ref if ref.get('path') or ref.get('file') else dict(path=name,**ref),name)
    for key,name in [('source_FULL',PRIOR_NAMES['full']),('source_native_metadata',PRIOR_NAMES['native']),('source_CUM',PRIOR_NAMES['cumulative']),('cost_outer_receipt',PRIOR_NAMES['cost']),('supplement',PRIOR_NAMES['supplement']),('execution_record',PRIOR_NAMES['execution'])]:exact_ref(owner[key],name)
    assert execution['exit_code']==0 and execution['source_commit']==ENTRY and execution['source_sha256']==SOURCE_SHA
    assert execution['new_entry_main_invocations_this_task']==execution['actual_remote_incremental_collection_invocations']==1
    assert execution['preceding_local_pending_invocations']==0 and execution['preceding_pending_remote_operations'] is False
    owners=[owner]
    for outer in (180,175,170,165,160):
        preceding=bound(owners[-1]['same_entry_prior_actual_executor_attestation'])
        assert preceding['outer']==outer;owners.append(preceding)
    assert [x['outer'] for x in reversed(owners)]==[160,165,170,175,180,185]
    for count,item in enumerate(reversed(owners),1):
        assert item['collector_source_commit']==ENTRY and item['collector_source_sha256']==SOURCE_SHA
        assert item['new_entry_main_invocations_this_task']==item['actual_remote_incremental_collection_invocations']==1
        assert item['preceding_local_pending_invocations']==item['remote_collector_exit_code']==0 and item['preceding_pending_remote_operations'] is False
        if count>1:assert item['same_entry_executor_attested_cumulative_main_invocations']==item['same_entry_executor_attested_cumulative_remote_incremental_invocations']==count
    assert owner['same_entry_executor_attested_cumulative_main_invocations']==owner['same_entry_executor_attested_cumulative_remote_incremental_invocations']==6
    assert full['completed_outer']==native['outer']==old[PRIOR_NAMES['cost']]['outer']==185
    assert full['status']=='ACTUAL_FULL_OUTER185_NATIVE185_COMPLETE_NEXT186_ENTERED' and full['full_iteration_metric_pending'] is False
    assert full['source_commit']==core.SOURCE and full['config_sha256']==core.CONFIG
    assert full['actual_dev512']['outer']==185 and full['actual_dev512']['rows']==512
    poll=bound(full['input_bindings']['poll']);resource=bound(full['input_bindings']['resource'])
    _,_,active=core.validate_unit(poll,resource,185);assert active
    core.validate_restore(poll)
    metric=core.validate_metric(poll,185);ranks=core.validate_ranks(poll,metric,185)
    _,_,completed,totals=core.validate_native(native,poll['progress'],185)
    assert full['iteration_metrics']==metric and full['last_rank_summaries']==ranks
    assert full['per_rank_successful_adam']==completed and full['per_rank_actual_total_adam']==totals
    assert re.search(r'epoch\s+\d+,\s*step\s+186\b',(poll.get('logs') or {}).get('stage.log',''))
    prior_parts=[]
    def restore(value,field=None):
        if isinstance(value,dict) and value.get('storage')=='complete_original_field_parts':
            kind=value['container_type'];assert kind in ('list','mapping')
            result=[] if kind=='list' else {};seen=set()
            for ref in value['parts']:
                assert ref.get('json_selector')=='value'
                name=ref.get('path',ref.get('file'));path=locate(name).resolve(strict=True);assert path not in seen;seen.add(path)
                part=bound(ref);chunk=part['value'];prior_parts.append(reference(name))
                if field is not None:assert part['original_field']==field
                assert isinstance(chunk,list if kind=='list' else dict)
                if kind=='list':result.extend(chunk)
                else:assert not set(result).intersection(chunk);result.update(chunk)
            assert len(result)==value['original_entries'];return result
        return value
    cumulative={key:restore(value,key) for key,value in old[PRIOR_NAMES['cumulative']].items()}
    assert cumulative['through_effective_outer']==185 and [x['outer'] for x in cumulative['rows_counted_once']]==list(range(1,186))
    assert [x['outer'] for x in cumulative['completed_dev_evaluations']]==list(range(5,186,5))
    rows=restore(old[PRIOR_NAMES['supplement']]['rows'])
    assert len(rows)==37 and [row['outer'] for row in rows]==list(range(5,186,5))
    assert len(owner['local_original_control_refs'])==37
    for row,ref in zip(rows,owner['local_original_control_refs']):
        assert ref['outer']==row['outer'];raw=bound(row['local_original_summary'])
        exact_ref(ref,row['local_original_summary']['path'])
        assert raw['actual_completed_outer']==row['outer'] and raw['status']=='VALIDATION_COST_COMPLETE'
        assert raw['phase']==('acceptance' if row['outer']==5 else 'periodic') and raw['dataset_rows']==raw['evaluated_rows']==512
        assert raw['ledger_sha256']==row['ledger_sha256'] and row['source_control_summary']['sha256']==row['local_original_summary']['sha256']
    for item in owners:
        for key in ('rows_part_refs','prior_rows_part_refs','source_CUM_all_complete_field_parts'):
            for ref in item.get(key,[]):bound(ref)
    last=old[PRIOR_NAMES['original_control']]
    assert last['ledger_sha256']==rows[-1]['ledger_sha256']==full['actual_dev512']['ledger_sha256']
    assert last['metrics']==full['actual_dev512']['metrics'] and root['actual185_identity']['control_file_sha256']==prior_inputs['original_control']['sha256']
    assert curve['json']['training_point_count']==len(curve['json']['training_points'])==185
    assert curve['json']['completed_dev512_point_count']==len(curve['json']['completed_dev512_points'])==37
    assert [x['outer'] for x in curve['json']['training_points']]==list(range(1,186))
    assert [x['outer'] for x in curve['json']['completed_dev512_points']]==list(range(5,186,5))
    assert len(list(csv.DictReader(io.StringIO(curve['csv'].decode('utf8')))))==222
    for key in ('publication','binding','root_readback'):assert curve[key]['counts']==dict(training=185,actual_completed_dev512=37,csv_data_rows=222)
    assert curve['root_readback']['outer']==185 and curve['root_readback']['status']=='PASS_ACTUAL_CURVE185_LOCAL_ARTIFACT_READBACK'
    # Old185 curve files/CSV prefix are held raw by their exact manifest pins.
    # Their old asof registry descriptors are provenance, never the current190 registry.
    return old


def main():
    if not __debug__:
        raise RuntimeError('Required actual-byte gates cannot be disabled with python -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-pins', required=True, type=Path)
    parser.add_argument('--root-pins-bytes', required=True, type=int)
    parser.add_argument('--root-pins-sha256', required=True)
    parser.add_argument('--owner-bytes', required=True, type=int)
    parser.add_argument('--owner-sha256', required=True)
    parser.add_argument('--execution-bytes', required=True, type=int)
    parser.add_argument('--execution-sha256', required=True)
    args = parser.parse_args()
    for prefix in ('root_pins', 'owner', 'execution'):
        assert getattr(args, prefix + '_bytes') > 0 and digest_ok(getattr(args, prefix + '_sha256'))
    assert not any((folder / OUTPUT_NAME).exists() for folder in [B] + M), 'one-time actual reader already published'
    read(str(ORIGINAL), '74c2cdabc404182ed3b09cb1f9239f4f0d87e3daf18df93e11e939ce209352f5', 52848)
    assert OWNER_SOURCE_PIN is not None, 'final owner source must be frozen and SHA-bound before actual use'
    read(str(OWNER_SOURCE), OWNER_SOURCE_PIN[1], OWNER_SOURCE_PIN[0])
    sr = read(str(S), SOURCE_SHA, 27911)
    read(str(SCRIPT_ROOT / 'ssh_retained18.py'), HELPER_SHA, 4411)
    read(str(CORE), CORE_SHA, 39098)
    spec = importlib.util.spec_from_file_location('dev190_independent_published_core_metadata_gates', CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    read(str(Path(__file__).resolve()))
    # Extract a literal for hash corroboration only; no collector module is imported.
    remote_assignments = [node.value for node in ast.parse(sr).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'REMOTE' for target in node.targets)]
    assert len(remote_assignments) == 1
    remote_program_template = ast.literal_eval(remote_assignments[0])
    assert isinstance(remote_program_template, str) and remote_program_template.count('PIN_DATA') == 1
    prior_gate = read(str(args.root_pins), args.root_pins_sha256, args.root_pins_bytes)
    old = verify_prior_inputs(prior_gate['prior_inputs'], None, core)
    oldsummary = old['DEV5_185_NATIVE_COST_SUPPLEMENT.json']
    oldowner = old['DEV5_185_NATIVE_COST_LOCAL_RECEIPT.json']
    oldroot = old['ROOT_DEV185_NATIVE_COST_ACTUAL_READBACK.json']
    prior, prior_parts = rows_from_summary(oldsummary)
    assert [row['outer'] for row in prior] == list(range(5, 186, 5))
    prior_parts_snapshot = [dict(reference(part['file'])) for part in prior_parts]
    prior_controls = {item['outer']: item for item in oldroot['control_file_and_inner_ledger_bindings']}
    assert len(prior_controls) == len(prior) and set(prior_controls) == set(range(5, 186, 5))
    gate = read(str(args.root_pins), args.root_pins_sha256, args.root_pins_bytes)
    assert gate['schema'] == 'dev190_root_owner_gate_pins_v1' and gate['outer'] == 190
    assert gate['status'] == 'ACTUAL_DEV190_FULL_NATIVE_CUM_COST_CONTROL_READY'
    assert gate['collector_source_commit'] == ENTRY and gate['collector_source_sha256'] == SOURCE_SHA
    assert isinstance(gate['collector_executor_agent'], str) and gate['collector_executor_agent'].startswith('/root/')
    for key in ('authorized_main_invocations', 'authorized_remote_incremental_invocations'):
        assert type(gate[key]) is int and gate[key] == 1
    assert set(gate['source_pins']) == set(FUTURE_NAMES)
    owner = read(OWNER_NAME, args.owner_sha256, args.owner_bytes)
    assert owner['status'] == 'ACTUAL_DEV190_COST_COLLECTED_AND_LOCAL_MIRRORS_READ_BACK'
    executor_counts(owner, 190, 7)
    assert owner['prior_inputs'] == gate['prior_inputs']
    assert owner['same_entry_prior_confirmed_main_invocations'] == owner['same_entry_prior_confirmed_remote_incremental_invocations'] == 6
    exact_ref(owner['same_entry_prior_actual_executor_attestation'], 'DEV5_185_NATIVE_COST_LOCAL_RECEIPT.json')
    exact_ref(owner['prior_root_actual_readback'], 'ROOT_DEV185_NATIVE_COST_ACTUAL_READBACK.json')
    exact_ref(owner['owner_receipt_builder'], str(OWNER_SOURCE))
    exact_ref(owner['collector_source'], str(S))
    exact_ref(owner['root_gate_input_pins'], str(args.root_pins))
    assert owner['collector_executor_agent'] == owner['local_readback_executor_agent'] == gate['collector_executor_agent']
    full, native, rawcum, cost = [exact_ref(gate['source_pins'][name], name) for name in FUTURE_NAMES]
    for field, name in zip(('source_FULL', 'source_native_metadata', 'source_CUM', 'cost_outer_receipt'), FUTURE_NAMES):
        exact_ref(owner[field], name)
    parts, fields = [], {}
    cumulative = {key: expand(value, key, parts, fields) for key, value in rawcum.items()}
    assert owner['source_CUM_dynamic_fields'] == fields
    expected_owner_parts = [dict(path=str(B.relative_to(W) / part['file']), bytes=part['bytes'], sha256=part['sha256'], original_field=part['original_field']) for part in parts]
    assert owner['source_CUM_all_complete_field_parts'] == expected_owner_parts
    assert full['status'] == 'ACTUAL_FULL_OUTER190_NATIVE190_COMPLETE_NEXT191_ENTERED'
    assert cost['status'] == 'ACTUAL_OUTER190_COST_ARCHIVED'
    assert full['completed_outer'] == native['outer'] == cost['outer'] == cumulative['through_effective_outer'] == 190
    for key, value in dict(source_commit=C, config_sha256=CONFIGS[C], unit=UNIT, run=RUN,
        actual_training_pid=503255, actual_training_invocation_id=INV, goal_status='active').items():
        assert full[key] == value, key
    assert cost['source_commit'] == C and cost['config_sha256'] == CONFIGS[C]
    assert cost['actual_training_pid'] == 503255 and cost['actual_training_invocation_id'] == INV
    identity, native_receipt = native['identity'], native['receipt']
    for key, value in dict(code_commit=C, config_sha256=CONFIGS[C], model_identity=MODEL,
        data_identity=DATA, completed_step=190, actual_update_limit=200,
        run_id=Path(RUN).name, job_id='not-slurm;systemd:' + C).items():
        assert identity[key] == value, key
    assert native['source_commit'] == native_receipt['source_commit'] == C
    assert native_receipt['completed_outer'] == 190 and native_receipt['status'] == 'COMPLETE_NATIVE_MANIFEST_METADATA_CAPTURED'
    assert native['archive_matches_original_manifest'] is True and native['manifest_status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED'
    assert full['native_manifest_sha256'] == native['manifest_sha256'] == native['recorded_original_manifest_sha256'] == native_receipt['history_manifest_sha256'] == native_receipt['source_manifest_sha256'] == cost['native_manifest_sha256']
    assert full['native_history_receipt_sha256'] == native['receipt_sha256'] == cost['native_history_receipt_sha256']
    assert digest_ok(native['manifest_sha256']) and digest_ok(native['receipt_sha256'])
    # Successful Adam calls come only from actual future FULL/native evidence;
    # guard-skipped opportunities do not imply an expected 190*20 count.
    totals = full['per_rank_actual_total_adam']
    assert isinstance(totals, dict) and set(totals) == {str(rank) for rank in range(8)}
    assert all(type(value) is int and value >= 0 for value in totals.values())
    optimizer = cumulative['successful_optimizer_accounting']
    completed = full['per_rank_successful_adam'];partial = full['partial_optimizer_calls_per_rank']
    assert isinstance(completed, dict) and set(completed) == set(totals)
    assert all(type(value) is int and value >= 0 for value in completed.values())
    assert isinstance(partial, dict) and set(partial) <= set(totals)
    assert all(type(value) is int and value >= 0 for value in partial.values())
    guard = identity['guard_continue_accounting']
    assert guard['completed_optimizer_calls_per_rank'] == completed
    assert guard['total_optimizer_calls_per_rank'] == totals and guard['partial_optimizer_calls_per_rank'] == partial
    assert all(totals[key] == completed[key] + partial.get(key, 0) for key in totals)
    assert optimizer['successful_calls_per_rank_through190'] == totals
    assert optimizer['direct_observed_calls_per_rank'] == completed
    assert optimizer['successful_rank_call_events_through190'] == sum(totals.values())
    assert optimizer['direct_observed_rank_call_events'] == sum(completed.values())
    workers = native_receipt['workers']
    assert len(workers) == 8 and {worker['rank'] for worker in workers} == set(range(8))
    assert full['native_workers'] == workers
    for worker in workers:
        rank_total = totals[str(worker['rank'])]
        assert type(worker['actual_total_optimizer_calls']) is int and worker['actual_total_optimizer_calls'] == rank_total
        assert worker['observed_adam_steps'] == [rank_total]
        assert worker['scheduler_last_epoch'] == 190 and worker['full_worker_state_finite'] is True
    # Saved cursor191 is a native boundary; actual191 entry is separately read
    # from the FULL-bound original poll stage marker below.
    assert native_receipt['driver']['next_step'] == full['native_driver_next_step'] == 191
    for document in (native, native_receipt):
        assert document['tensor_contents_read'] is False and document['independent_cold_reload_performed'] is False
    counted, devs = cumulative['rows_counted_once'], cumulative['completed_dev_evaluations']
    assert [row['outer'] for row in counted] == list(range(1, 191))
    assert len({row['row_id'] for row in counted}) == len(counted) and counted[-1]['source_commit'] == C
    assert [dev['outer'] for dev in devs] == list(range(5, 191, 5)) and cumulative['pending_full_metric_outers'] == []
    assert owner['source_CUM_completed_dev_entries'] == len(devs)
    actual, metrics = full['actual_dev512'], full['actual_dev512']['metrics']
    assert actual['outer'] == 190 and actual['rows'] == 512 and actual['status'] == 'VALIDATION_COST_COMPLETE'
    assert actual['phase'] == 'periodic' and digest_ok(actual['ledger_sha256'])
    assert metrics['val/qa_count/all'] == 512 and metrics['val/qa_count/nq'] == metrics['val/qa_count/hotpotqa'] == 256
    assert numeric(metrics['val/f1/all']) and numeric(metrics['val/em/all'])
    assert 0 <= metrics['val/f1/all'] <= 1 and 0 <= metrics['val/em/all'] <= 1
    assert cost['actual_dev512'] == actual and owner['actual_dev190_metrics'] == metrics
    assert devs[-1]['ledger'] == actual['path'] and devs[-1]['ledger_sha256'] == actual['ledger_sha256']
    assert devs[-1]['F1'] == metrics['val/f1/all'] and devs[-1]['EM'] == metrics['val/em/all']
    assert devs[-1]['testing_phase_wall_seconds'] == full['normal_dev512_wall_seconds']
    poll = bound(full['input_bindings']['poll'])
    resource = bound(full['input_bindings']['resource'])
    assert poll['source_commit'] == C and poll['unit'] == UNIT and poll['run'] == RUN
    assert poll['progress']['actual_completed_outer_from_attempt_results'] == 190
    assert not poll['progress']['errors']
    assert full['actual_next_outer_entered'] == 191
    assert full['local_control_script_sha256'] == '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
    ps, rs, active = core.validate_unit(poll, resource, 190)
    core.validate_restore(poll)
    metric = core.validate_metric(poll, 190)
    ranks = core.validate_ranks(poll, metric, 190)
    native_record, native_guard, actual_completed, actual_totals = core.validate_native(native, poll['progress'], 190)
    assert active and native_record == native_receipt and native_guard == guard
    assert actual_completed == completed and actual_totals == totals
    assert full['iteration_metrics'] == metric and full['last_rank_summaries'] == ranks
    assert cost['actual_rank_successful_total_adam'] == totals
    writes = {str(row['rank']): row['actual_successful_writes'] for row in ranks}
    processed = {str(row['rank']): row['actual_opportunities_processed'] for row in ranks}
    skipped = {str(row['rank']): row['actual_opportunities_skipped'] for row in ranks}
    assert full['actual_outer_successful_writes_per_rank'] == cost['actual_outer_successful_writes_per_rank'] == writes
    assert full['actual_outer_processed_opportunities_per_rank'] == cost['independent_rank_processed_opportunities'] == processed
    assert full['actual_outer_skipped_opportunities_per_rank'] == cost['independent_rank_skipped_opportunities'] == skipped
    assert full['full_metric_processed_opportunities'] == cost['full_metric_processed_opportunities'] == metric['actor/optimizer_opportunities_processed']
    assert full['full_metric_skipped_opportunities'] == cost['full_metric_skipped_opportunities'] == metric['actor/optimizer_opportunities_skipped']
    latest_accounting = optimizer['latest_outer_original_receipt_opportunity_fields']
    for key in ('per_rank_actual_total_adam','partial_optimizer_calls_per_rank','actual_outer_successful_writes_per_rank','actual_outer_processed_opportunities_per_rank','actual_outer_skipped_opportunities_per_rank','full_metric_processed_opportunities','full_metric_skipped_opportunities'):
        assert latest_accounting[key] == full[key]
    stage = poll['logs']['stage.log']
    assert isinstance(stage, str)
    actual191_marker = re.search(r'epoch\s+\d+,\s*step\s+191\b', stage)
    assert actual191_marker is not None, 'actual191 requires original FULL-bound poll stage evidence'
    for state_value in (poll['systemd'], resource.get('training_systemd') or resource.get('systemd')
                        or (resource.get('state') or {}).get('training_unit_observation')):
        state = state_value if isinstance(state_value, dict) else dict(line.split('=', 1) for line in state_value.splitlines() if '=' in line)
        assert state['MainPID'] == '503255' and state['ActiveState'] == 'active' and state['SubState'] == 'running'
        assert state.get('InvocationID', INV) == INV
    assert any(item.get('step') == 190 and item == full['iteration_metrics'] for item in poll['iteration_metrics'])
    assert any(item.get('ledger_sha256') == actual['ledger_sha256'] and item.get('metrics') == metrics for item in poll['development'])
    exact_ref(cost['source_inputs']['FULL'], FUTURE_NAMES[0])
    exact_ref(cost['source_inputs']['native_metadata'], FUTURE_NAMES[1])
    exact_ref(cost['cumulative_snapshot'], FUTURE_NAMES[2])
    refs(full['input_bindings'])
    refs(cost['source_inputs'])
    fixed_artifacts = {'BRANCH_FALLBACK_RATES_OUTER0190.json', 'CANDIDATE_COST_EXTENSION_INPUTS.json',
        'CANDIDATE_COST_EXTENSION_OUTER0190.json', 'CANDIDATE_COST_EXTENSION_OUTER0190.md',
        'CUMULATIVE_COST_THROUGH190.json', 'CUMULATIVE_COST_THROUGH190.md', 'candidate_cost_backfill/CANDIDATE_OUTER0190.json'}
    artifacts = cost['mirrored_artifacts']
    assert set(artifacts) == fixed_artifacts | {part['file'] for part in parts}
    assert cost['mirrored_artifact_count'] == owner['cost_mirrored_artifact_refs'] == len(artifacts)
    for name, ref in artifacts.items():
        exact_ref(ref, name)
    for part in parts:
        assert artifacts[part['file']]['sha256'] == part['sha256'] and artifacts[part['file']]['bytes'] == part['bytes']
    # Accounting stays in its original scope: CPU subsets are never added again.
    assert cumulative['dev_phase_already_in_outer_wall'] is True
    for key in ('dev5_120_supplement_independent_not_added_to_outer_or_service_cpu',
        'dev_native_supplement_not_added_again', 'prefix_candidate_and_CPU_subsets_not_added_again'):
        assert cost[key] is True, key
    assert cost['aggregate_candidate_supplement_added_again'] is False
    assert cost['dev_collector_called'] is False and cost['final200_entry_called'] is False
    assert cost['sampling_shape']['question_count'] == 512 and cost['sampling_shape']['training_leaf_count'] == 2560
    pending = bound(gate['pending_evidence'])
    assert pending['schema'] == 'dev190_source_bound_pending_readback_v1' and pending['outer'] == 190
    assert pending['source_commit'] == ENTRY and pending['source_sha256'] == SOURCE_SHA
    assert pending['executor_agent'] == gate['collector_executor_agent']
    assert type(pending['preceding_local_pending_invocations']) is int and pending['preceding_local_pending_invocations'] == 0
    assert pending['preceding_pending_remote_operations'] is False
    assert isinstance(pending['observed_utc'], str) and pending['observed_utc']
    assert isinstance(pending['observations'], list) and pending['observations']
    assert isinstance(pending['evidence_scope'], str) and pending['evidence_scope']
    assert owner['preceding_pending_source_bound_evidence'] == gate['pending_evidence']
    ready = gate['control_ready']
    assert ready['outer'] == 190 and isinstance(ready['raw_summary_utf8'], str)
    ready_raw = ready['raw_summary_utf8'].encode('utf8')
    assert 0 < len(ready_raw) <= 100000
    assert sha(ready_raw) == ready['source']['sha256'] and len(ready_raw) == ready['source']['bytes']
    assert load(ready_raw)['metrics'] == metrics
    bound(ready['source'])
    assert owner['original_remote_control_ready'] == ready['source']
    assert not (B / 'DEV5_190_NATIVE_COST_COLLECTION_ERROR.json').exists()
    execution = read(EXECUTION_NAME, args.execution_sha256, args.execution_bytes)
    exact_ref(owner['execution_record'], EXECUTION_NAME)
    assert execution['source_commit'] == ENTRY and execution['source_sha256'] == SOURCE_SHA and execution['helper_sha256'] == HELPER_SHA
    assert execution['actual_command'] == owner['actual_command_template'] == CMD
    assert execution['executor_agent'] == gate['collector_executor_agent']
    assert type(execution['exit_code']) is int and execution['exit_code'] == 0
    for key in ('new_entry_main_invocations_this_task', 'actual_remote_incremental_collection_invocations'):
        assert type(execution[key]) is int and execution[key] == 1
    assert type(execution['preceding_local_pending_invocations']) is int and execution['preceding_local_pending_invocations'] == 0
    assert execution['preceding_pending_remote_operations'] is False
    exact_ref(execution['root_gate_input_pins'], str(args.root_pins))
    assert execution['root_gate_input_pins']['sha256'] == args.root_pins_sha256 and execution['root_gate_input_pins']['bytes'] == args.root_pins_bytes
    assert execution['preceding_pending_evidence'] == gate['pending_evidence']
    pre = exact_ref(execution['pre_execution_input_pins'], PRE_NAME)
    exact_ref(owner['pre_execution_input_pins'], PRE_NAME)
    assert pre['schema'] == 'dev190_local_preflight_input_pins_v1' and pre['outer'] == 190
    assert pre['status'] == 'LOCAL_INPUTS_READY_NO_COLLECTOR_INVOKED' and pre['collector_invoked'] is False
    assert pre['source_commit'] == ENTRY and pre['collector_source_sha256'] == SOURCE_SHA and pre['command'] == CMD
    assert pre['prior_inputs'] == gate['prior_inputs']
    assert pre['root_gate_input_pins'] == reference(str(args.root_pins))
    assert pre['preceding_pending_evidence'] == gate['pending_evidence']
    assert pre['collector_executor_agent'] == execution['executor_agent']
    assert pre['authorized_main_invocations'] == pre['authorized_remote_incremental_invocations'] == 1
    assert pre['original_remote_control_ready'] == ready['source']
    assert pre['actual_dev190_ledger_sha256'] == actual['ledger_sha256']
    for ref in pre['preserved_input_refs']:
        bound(ref)
    emitted = load(execution['stdout'])
    assert emitted['controls'] == 38 and emitted['newly_read_remote_control_outers'] == [190]
    assert emitted['reused_original_control_outers'] == list(range(5, 186, 5))
    assert emitted['original_cumulative_unchanged'] is True and emitted['remote_files_written'] is False
    summary = exact_ref(emitted['supplement'], SUPPLEMENT_NAME)
    exact_ref(owner['supplement'], SUPPLEMENT_NAME)
    exact_ref(emitted['markdown'], 'DEV5_190_NATIVE_COST_SUPPLEMENT.md')
    exact_ref(owner['markdown'], 'DEV5_190_NATIVE_COST_SUPPLEMENT.md')
    assert summary['summary_schema'] == 'tree_dev_native_cost_supplement_v2'
    assert summary['status'] == 'ACTUAL_COMPLETED_DEV_CONTROL_SUMMARY_COST_READBACK' and summary['through_outer'] == 190
    exact_ref(summary['source_cumulative'], FUTURE_NAMES[2])
    exact_ref(summary['source_full_receipt'], FUTURE_NAMES[0])
    exact_ref(summary['prior_immutable_supplement'], 'DEV5_185_NATIVE_COST_SUPPLEMENT.json')
    exact_ref(owner['prior_immutable_supplement'], 'DEV5_185_NATIVE_COST_SUPPLEMENT.json')
    refs(summary['source_completed_evidence_bindings'])
    assert summary['model_identity'] == MODEL and summary['data_identity'] == DATA
    assert summary['split_identity']['evaluated_rows'] == 512
    assert summary['split_identity']['nq_rows'] == summary['split_identity']['hotpotqa_rows'] == 256
    assert summary['split_identity']['independent_split_file_sha256'] is None
    for key in ('remote_files_written', 'jsonl_rollout_tensor_or_model_read', 'immutable_cumulative_rewritten',
        'STATUS_REPORT_FUP_or_training_modified', 'tests_or_review_performed'):
        assert summary[key] is False, key
    assert summary['read_only'] is True
    rows, row_parts = rows_from_summary(summary)
    assert [row['outer'] for row in rows] == summary['completed_dev_outers'] == list(range(5, 191, 5))
    assert len(rows) == len(prior) + 1 == 38
    assert rows[:-1] == prior
    assert summary['newly_read_remote_control_outers'] == [190]
    assert summary['reused_original_control_outers'] == list(range(5, 186, 5))
    assert owner['controls'] == len(rows) and owner['newly_read_remote_control_outers'] == [190]
    assert owner['reused_original_control_outers'] == summary['reused_original_control_outers']
    owner_controls = owner['local_original_control_refs']
    assert len(owner_controls) == len(rows) and len({ref['outer'] for ref in owner_controls}) == len(rows)
    assert [ref['outer'] for ref in owner_controls] == [row['outer'] for row in rows]
    controls, pins = [], []
    for row, dev, owner_ref in zip(rows, devs, owner_controls):
        n, pin = row['outer'], dev_pin(dev)
        pins.append(pin)
        name = 'dev_native_cost_summaries/DEV_OUTER%04d_SUMMARY.json' % n
        control = exact_ref(row['local_original_summary'], name)
        raw = R[str(locate(name).resolve(strict=True))]
        assert control_row(pin, raw, row['source_control_summary'], row['local_original_summary']) == row
        assert {key: owner_ref[key] for key in ('path', 'bytes', 'sha256')} == reference(name)
        if n == 190:
            assert control['training_protocol'] == 'tree_fresh200_five_leaf_v1' and control['physical_gpus'] == 8
        if n < 190:
            previous = prior_controls[n]
            assert len(raw) == previous['bytes'] and sha(raw) == previous['sha256']
            assert row['ledger_sha256'] == previous['inner_ledger_sha256'] and row == prior[n // 5 - 1]
        controls.append(dict(outer=n, **reference(name), inner_ledger_sha256=row['ledger_sha256'],
            source_commit=pin['code_commit'], original_bytes_preserved_from185=n < 190))
    last = rows[-1]
    current_raw = R[str(locate('dev_native_cost_summaries/DEV_OUTER0190_SUMMARY.json').resolve(strict=True))]
    assert current_raw == ready_raw and last['source_control_summary'] == ready['source']
    assert last['ledger_sha256'] == actual['ledger_sha256'] and load(current_raw)['metrics'] == metrics
    assert last['counters'] and numeric(last['wall_seconds']) and numeric(last['trainer_process_cpu_seconds'])
    assert isinstance(last['cost_count_scope'], str) and 'excludes training generations' in last['cost_count_scope']
    assert last['precise_engine_emitted_tokens'] is None and last['scoring_call_event_total'] is None
    for part, pinned in zip(prior_parts, prior_parts_snapshot):
        bound(part)
        assert reference(part['file']) == pinned
    for ref in owner['rows_part_refs']:
        bound(ref)
    assert [reference(ref.get('path') or ref.get('file')) for ref in owner['rows_part_refs']] == [reference(part['file']) for part in row_parts]
    for ref in owner['prior_rows_part_refs']:
        bound(ref)
    assert owner['prior_rows_part_refs'] == prior_parts_snapshot
    program = sha(remote_program_template.replace('PIN_DATA', repr(pins[-1:])).encode())
    assert program == summary['remote_program_sha256'] == owner['remote_program_sha256']
    assert summary['collector_source_sha256'] == owner['collector_source_sha256'] == sha(sr)
    assert summary['ssh_helper_sha256'] == owner['ssh_helper_sha256'] == HELPER_SHA
    assert numeric(summary['collector_cpu_seconds']) and summary['collector_cpu_seconds'] >= 0
    assert summary['collector_cpu_scope'] == 'This small new readback process only; not added to validation/service CPU.'
    assert summary['precise_engine_emitted_tokens'] is None and summary['scoring_call_event_total'] is None
    assert set(summary['native_counter_totals']) == {key for row in rows for key in (row['counters'] or {})}
    for key, record in summary['native_counter_totals'].items():
        assert record == aggregate(rows, [(row['counters'] or {}).get(key) for row in rows]), key
    assert summary['original_validation_observer_wall_seconds'] == aggregate(rows, [row['wall_seconds'] for row in rows])
    assert summary['original_validation_trainer_process_cpu_seconds'] == aggregate(rows, [row['trainer_process_cpu_seconds'] for row in rows])
    assert summary['prior_fullmetric_testing_phase_wall_seconds'] == cumulative['completed_dev_testing_phase_wall_seconds']
    assert emitted['native_counter_totals'] == owner['native_counter_totals'] == summary['native_counter_totals']
    assert owner['original_validation_observer_wall_seconds'] == summary['original_validation_observer_wall_seconds']
    assert owner['original_validation_trainer_process_cpu_seconds'] == summary['original_validation_trainer_process_cpu_seconds']
    observations = two_pass()
    value = dict(schema='root_dev190_native_cost_actual_local_readback_v1',
        status='PASS_ACTUAL_DEV190_LOCAL_ARTIFACT_READBACK', observed_utc=datetime.now(timezone.utc).isoformat(),
        executor='/root/dev190_actual_readback', outer=190, goal_status='active',
        scope='Independent actual local artifact/reference/original-byte parsing only after successful actual190 owner; no extra training gate/review, collector, SSH, or training test.',
        candidate_entry_commit=ENTRY, prior_inputs=gate['prior_inputs'], prior185_curve_registry_asof_exclusions=PRIOR_ASOF_EXCLUSIONS, prior185_registry_original_raw_bytes_currently_verified=False, adapted_from_reader_source=reference(str(ORIGINAL)),
        frozen_owner_source=reference(str(OWNER_SOURCE)), independent_reader_source=reference(str(Path(__file__).resolve())),
        collector_source=reference(str(S)), root_gate_input_pins=reference(str(args.root_pins)),
        actual_owner_receipt=reference(OWNER_NAME), execution_record=reference(EXECUTION_NAME),
        source_pins={name: reference(name) for name in FUTURE_NAMES + (SUPPLEMENT_NAME, OWNER_NAME, EXECUTION_NAME, PRE_NAME)},
        actual_ref_counts=dict(cost_mirrored_artifacts=len(artifacts), cum_complete_fields=len(fields),
            cum_parts=len(parts), cum_completed_dev_entries=len(devs), new_dev_row_parts=len(row_parts),
            new_dev_rows=len(rows), old_dev_row_parts=len(prior_parts), old_dev_rows=len(prior),
            all_original_control_summaries=len(controls), preserved_old_control_summaries=len(prior),
            new_remote_control_summaries=len(rows) - len(prior), unique_audited_files=len(F),
            unique_mirrored_files=sum(ref['mirror_checked'] for ref in F.values()), input_byte_readback_passes=len(observations), mirror_roots=len(M)),
        cumulative_fields_dynamic=fields, cumulative_part_refs=parts, new_row_part_refs=row_parts,
        old_row_part_refs=prior_parts, control_file_and_inner_ledger_bindings=controls,
        prior37_rows_exact_object_equal=True, prior37_control_original_bytes_SHA_preserved=True,
        all_original_DEV185_row_parts_exact_bytes_SHA_preserved=True,
        original_CUM190_raw_bytes_SHA_preserved=True,
        actual190_identity=dict(source_commit=C, config_sha256=CONFIGS[C], model_identity=MODEL, data_identity=DATA,
            rows=512, nq_rows=256, hotpotqa_rows=256, F1=metrics['val/f1/all'], EM=metrics['val/em/all'],
            phase=actual['phase'], ledger=actual['path'], inner_ledger_sha256=actual['ledger_sha256'],
            control_file_bytes=len(current_raw), control_file_sha256=sha(current_raw), actual_total_adam_per_rank=totals,
            completed_successful_adam_per_rank=completed, partial_optimizer_calls_per_rank=partial,
            actual_successful_writes_per_rank=writes, independent_rank_processed_opportunities=processed, independent_rank_skipped_opportunities=skipped,
            full_metric_processed_opportunities=metric['actor/optimizer_opportunities_processed'], full_metric_skipped_opportunities=metric['actor/optimizer_opportunities_skipped'],
            full_metric_core_source=reference(str(CORE)),
            scheduler_last_epoch=190, native_driver_next_step=191,
            native_cursor_is_saved_boundary_only=True, actual_next_outer_entered=full['actual_next_outer_entered'],
            actual_next_entry_stage_marker=actual191_marker.group(), actual_next_entry_bound_poll=full['input_bindings']['poll'],
            actual_adam_count_scope='Actual FULL/native rank counts, allowing guarded skips; no190*20 assumption.', training_pid=503255, training_invocation=INV,
            first_FULL_poll=full['input_bindings']['poll'], first_FULL_resource=full['input_bindings']['resource'],
            native_metadata_only=True, tensor_contents_read=False, independent_cold_reload_performed=False,
            PT_file_sha256_recomputed=False, PT_current_availability_after_capture_is_unknown=True),
        actual_new_collection_binding=dict(executor_receipt_attested_new_entry_main_invocations=1,
            executor_receipt_attested_remote_incremental_invocations=1,
            executor_receipt_attested_preceding_pending_invocations=0, executor_exit_code=0,
            source_bound_pending_evidence=reference(gate['pending_evidence']['path']),
            same_entry_executor_attested_cumulative_main_invocations_160_plus165_plus170_plus175_plus180_plus185_plus190=owner['same_entry_executor_attested_cumulative_main_invocations'],
            same_entry_executor_attested_cumulative_remote_incremental_invocations_160_plus165_plus170_plus175_plus180_plus185_plus190=owner['same_entry_executor_attested_cumulative_remote_incremental_invocations'],
            artifact_corroboration_new_remote_outers=[190], artifact_corroboration_reused_outers=list(range(5, 186, 5)),
            independently_reconstructed_remote_program_sha256=program, remote_program_pin_count=len(pins[-1:]),
            evidence_limit='Counts are source1ed SHA-bound executor attestations, corroborated by original190 control/row, prior160/165/170/175/180/185 receipts, pre-execution and source-bound PENDING0 evidence, exit0 execution, and a one-pin program hash. No independent global process audit or new remote execution observation is claimed.',
            this_readback_collector_or_SSH_invocations=0),
        actual190_counters=last['counters'], validated_native_dev_counter_totals=summary['native_counter_totals'],
        original_validation_observer_wall_seconds=summary['original_validation_observer_wall_seconds'],
        original_validation_trainer_process_cpu_seconds=summary['original_validation_trainer_process_cpu_seconds'],
        accounting=dict(validation_wall_CPU_already_in_outer_service_not_added_again=True,
            prefix_candidate_CPU_subsets_not_added_again=True, metadata_collector_CPU_not_added=True,
            precise_engine_emitted_tokens=None, scoring_call_event_total=None, independent_split_file_sha256=None,
            final_service_CPU_seconds=None, submitted_prefill_is_repeated_events=True,
            returned_nonpad_is_not_precise_engine_emitted=True, HTTP_calls_and_query_events_distinct=True,
            physical8gpu_wall_capacity_not_exclusive_billable_usage=True),
        two_pass_local_input_and_mirror_bytes_verified=True, input_readback_observations=observations,
        mirrors=[str(mirror) for mirror in M], files=sorted(F.values(), key=lambda ref: ref['path']),
        remote_descriptors_not_followed=REMOTE_REFS, inconsistencies=[],
        owner_completion_gate='Actual root gate, actual successful owner, and actual execution are independently SHA/byte-bound required inputs; all control and dynamic part bytes are independently parsed here.',
        raw_input_mirror_policy='Raw poll/resource retain BASE-only policy; every product artifact uses BASE and both mirrors in each actual readback pass.',
        reviewer_actions=dict(remote_operations=False, collector_or_main_invoked=False,
            training_or_evaluation_modified=False, code_review_or_training_tests=False,
            tensor_model_rollout_JSONL_read=False, canonical_STATUS_REPORT_FUP_observer_modified=False,
            old_DEV185_outputs_rewritten=False, immutable_CUM_rewritten=False),
        receipt_publication=dict(new_file_only=True, copies=len(M) + 1, required_actual_byte_readback_passes=2),
        cold_restore_scope='Existing native5/100each scope is unchanged; no native190 cold reload is performed or claimed.',
        raw_observation_sources_scope='Only SHA-bound actual local FULL190/native/poll/resource/control/receipt/execution bytes. No new remote observation, final200/51713/HF/cleanup or goal completion is claimed.')
    publish(value)


if __name__ == '__main__':
    main()
