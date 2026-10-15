#!/usr/bin/env python3
"""Pure local candidate200 supplement; never runs SSH or rewrites CUM200.

Requires actual FULL/native/CUM200 and the existing collected candidate200.
This closes training candidate metadata only, not the complete experiment goal.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
PARENT = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
ORIGIN = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
RUN_ID = 'tree-nested-branch-fresh200-not-slurm;systemd:' + SOURCE
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/' + RUN_ID
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
STEM = 'FINAL_CANDIDATE_COST_THROUGH200'
CUM = 'CUMULATIVE_COST_THROUGH200.json'
FULL = 'FULL_OUTER200_PROGRESS_RECEIPT.json'
NATIVE = 'NATIVE_HISTORY_200_METADATA_READBACK.json'
REGISTRY = 'CANDIDATE_COST_EXTENSION_INPUTS.json'
INDEX = 'CANDIDATE_COST_EXTENSION_OUTER0200.json'
PART = 'candidate_cost_backfill/CANDIDATE_OUTER0200.json'
MAX_BYTES = 100000
DEPENDENCIES = {'append_nested_local_costs': '6c27a48be242ad8288f285f332e0b141cdad94aa1302205423b7c5aa68d3245d',
                'backfill_nested_candidate_costs': 'f64872751d87d497f46b860d27be5c64147d37051df4b605f04f5b40ab77c4d4'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class LocalInputs:
    def __init__(self, base, mirrors):
        self.base, self.mirrors, self.refs = base.resolve(strict=True), mirrors, {}

    def read(self, name, expected_sha=None, expected_bytes=None, limit=4000000):
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts and path.suffix == '.json', 'Only explicit relative local JSON metadata is allowed')
        resolved = (self.base / path).resolve(strict=True)
        need(resolved.is_relative_to(self.base) and resolved.is_file(), 'Metadata path escaped local audit base')
        need(resolved.stat().st_size <= limit, 'Local scalar metadata size budget exceeded')
        raw = resolved.read_bytes()
        need(expected_sha is None or sha(raw) == expected_sha, 'Input SHA differs: ' + name)
        need(expected_bytes is None or len(raw) == expected_bytes, 'Input bytes differ: ' + name)
        need(all((m / path).read_bytes() == raw for m in self.mirrors), 'Input mirror bytes differ: ' + name)
        self.refs[name] = dict(file=name, bytes=len(raw), sha256=sha(raw))
        return raw

    def reference(self, ref, limit=MAX_BYTES):
        need(isinstance(ref, dict) and isinstance(ref.get('file'), str) and digest(ref.get('sha256')) and type(ref.get('bytes')) is int and ref['bytes'] > 0, 'Invalid complete-field reference')
        return self.read(ref['file'], ref['sha256'], ref['bytes'], limit)

    def expand(self, value, field, schema='cumulative_scalar_field_part_v1'):
        if not isinstance(value, dict) or value.get('storage') != 'complete_original_field_parts':
            if isinstance(value, dict):
                return {key:self.expand(item, field + '.' + key, schema) for key, item in value.items()}
            if isinstance(value, list):
                return [self.expand(item, field + '[' + str(key) + ']', schema) for key, item in enumerate(value)]
            return value
        kind = value.get('container_type');count = value.get('original_entries');parts = value.get('parts')
        need(kind in ('mapping', 'list') and type(count) is int and count >= 0 and isinstance(parts, list), 'Invalid CUM complete-field descriptor: ' + field)
        result = {} if kind == 'mapping' else [];seen = set()
        for ref in parts:
            need(isinstance(ref, dict) and ref.get('json_selector') == 'value', 'CUM part selector differs: ' + field)
            need(ref.get('file') not in seen, 'Duplicate CUM part: ' + field);seen.add(ref.get('file'))
            part = json.loads(self.reference(ref))
            need(isinstance(part, dict) and part.get('schema') == schema and part.get('original_field') == field, 'CUM part original field/schema differs: ' + field)
            chunk = part.get('value')
            need(isinstance(chunk, dict if kind == 'mapping' else list), 'CUM part container type differs: ' + field)
            if kind == 'mapping':
                need(not set(result).intersection(chunk), 'Duplicate CUM mapping entries: ' + field);result.update(chunk)
            else:
                result.extend(chunk)
        need(len(result) == count, 'CUM original entry count differs: ' + field)
        return self.expand(result, field, schema)


def module(name):
    path = Path(__file__).with_name(name + '.py')
    need(sha(path.read_bytes()) == DEPENDENCIES[name], 'Published local dependency SHA differs: ' + name)
    spec = importlib.util.spec_from_file_location('final_candidate_' + name, path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def native_counts(evidence, cumulative):
    full, native = evidence['full'], evidence['native'];identity = native.get('identity', {})
    expected = dict(code_commit=SOURCE, config_sha256=CONFIG, model_identity=MODEL, data_identity=DATA,
                    completed_step=200, actual_update_limit=200, run_id=RUN_ID, job_id='not-slurm;systemd:' + SOURCE)
    need(all(identity.get(k) == v for k, v in expected.items()), 'Native200 identity differs')
    guard = identity.get('guard_continue_accounting', {})
    totals = guard.get('completed_optimizer_calls_per_rank')
    need(guard.get('completed_outer_updates') == 200 and isinstance(totals, dict) and set(totals) == {str(r) for r in range(8)}, 'Native200 completed guard/ranks differ')
    need(all(type(v) is int and v >= 0 for v in totals.values()) and len(set(totals.values())) == 1, 'Native200 actual rank Adam counts are inconsistent')
    accounting = cumulative['successful_optimizer_accounting']
    need(full.get('per_rank_actual_total_adam') == totals and accounting.get('successful_calls_per_rank_through200') == totals, 'FULL/CUM/native actual Adam counts differ')
    need(accounting.get('missing_direct_record_outers') == [] and accounting.get('direct_observed_calls_per_rank') == totals, 'CUM directly recorded rank writes differ from native200')
    workers = native['receipt']['workers']
    for worker in workers:
        count = totals[str(worker['rank'])]
        need(worker.get('actual_total_optimizer_calls') == count and worker.get('observed_adam_steps') == [count] and worker.get('scheduler_last_epoch') == 200, 'Native200 worker actual Adam/scheduler differs')
        need(worker.get('full_worker_state_finite') is True and digest(worker.get('recorded_file_sha256')) and digest(worker.get('recorded_logical_state_sha256')) and type(worker.get('file_bytes_at_capture')) is int and worker['file_bytes_at_capture'] > 0, 'Native200 worker writer metadata is incomplete')
    driver = native['receipt']['driver']
    need(driver.get('next_step') == 201 and digest(driver.get('recorded_file_sha256')) and digest(driver.get('recorded_logical_state_sha256')) and type(driver.get('file_bytes_at_capture')) is int and driver['file_bytes_at_capture'] > 0, 'Native200 driver metadata is incomplete')
    return dict(actual_adam_per_rank=totals, actual_rank_call_events=sum(totals.values()),
                workers=copy.deepcopy(workers), driver=copy.deepcopy(driver),
                driver_cursor_scope='Saved next_step201 only; no actual training201 entry or cold reload is claimed.')


def add_terminal_outcomes(original, terminal):
    result = copy.deepcopy(original)
    need(set(result) == {'all', 'accepted', 'discarded', 'fallback'}, 'CUM outcome groups differ')
    for outcome, record in result.items():
        need(record.get('known_outers') == list(range(1, 200)) and record.get('missing_outers') == [200], 'Original outcome coverage must be exactly1..199, missing only200')
        fields = record['field_sums'];new = terminal[outcome]['field_sums']
        need(set(fields) == set(new), 'Terminal candidate scalar field set differs')
        record['known_outers'].append(200);record['missing_outers'] = []
        for key, value in fields.items():
            need(value.get('known_outers') == list(range(1, 200)) and value.get('missing_outers') == [200] and value.get('total') is None and numeric(value.get('known_sum')), 'Original field coverage must lack only200: ' + key)
            current = new[key]
            if numeric(current):
                value['known_sum'] = math.fsum([value['known_sum'], current]);value['total'] = value['known_sum']
                value['known_outers'].append(200);value['missing_outers'] = []
            else:
                need(current is None, 'Terminal scalar must be finite numeric or null')
                # Preserve the producer meaning: known_sum sums complete outer fields.
                # Within-outer partial known sums stay in the pinned original candidate200.
                value['total'] = None
    return result


def prepare(base, cumulative_sha, registry_sha):
    need(digest(cumulative_sha) and digest(registry_sha), 'Explicit original CUM/registry SHA256 required')
    initial = (base / CUM).read_bytes();need(len(initial) <= MAX_BYTES and sha(initial) == cumulative_sha, 'Original CUM200 SHA/size differs')
    raw_cum = json.loads(initial);mirrors = [Path(x) for x in raw_cum['mirrors']]
    need(len(mirrors) == 2 and len(set(mirrors)) == 2 and all(p.is_absolute() for p in mirrors), 'Two actual inline mirror paths required')
    inputs = LocalInputs(base, mirrors);inputs.read(CUM, cumulative_sha, len(initial), MAX_BYTES)
    cumulative = {k:inputs.expand(v, k) for k, v in raw_cum.items()}
    need(cumulative.get('through_effective_outer') == 200 and cumulative.get('target_outer') == 200 and cumulative.get('origin') == 'fresh' and cumulative.get('pending_full_metric_outers') == [], 'Actual complete fresh-origin CUM200 required')
    rows = cumulative['rows_counted_once']
    need([r.get('outer') for r in rows] == list(range(1, 201)), 'CUM effective coverage must be exactly1..200')
    need(all(r.get('source_commit') == (ORIGIN if r['outer'] == 1 else PARENT if r['outer'] <= 5 else SOURCE) for r in rows), 'Effective source ancestry differs')
    local = module('append_nested_local_costs');collector = module('backfill_nested_candidate_costs')
    evidence = local.actual_inputs(base, base / FULL)
    full = evidence['full']
    need(evidence['step'] == evidence['target'] == 200 and full.get('source_commit') == SOURCE and full.get('config_sha256') == CONFIG and full.get('unit') == UNIT and full.get('run') == RUN and evidence['native_name'] == NATIVE and evidence['poll'].get('experiment_origin') == 'fresh', 'FULL200 original task identity differs')
    need(evidence['metric'].get('step') == 200 and numeric(evidence['metric'].get('timing_s/step')), 'Actual full metric200 is required')
    original_bindings = cumulative['immutable_metadata_input_hashes']
    for name, ref in evidence['bindings'].items():
        need(original_bindings.get(name) == ref, 'CUM does not bind original FULL/poll/native/resource: ' + name)
        inputs.read(name, ref['sha256'], ref['bytes'])
    counts = native_counts(evidence, cumulative)
    book_ref = cumulative['derived_input_snapshots']['index']
    book = json.loads(inputs.read(book_ref['file'], book_ref['sha256_at_collection'], book_ref['bytes']))
    need(book['effective_history']['latest_effective_outer'] == 200 and [r['outer'] for r in book['effective_rows']] == list(range(1, 201)), 'CUM-bound credited book coverage differs')
    row_ref = next(r for r in book['effective_rows'] if r['outer'] == 200)
    block_ref = next(r for r in book['scalar_blocks'] if r['file'] == row_ref['block_file'])
    block = json.loads(inputs.read(block_ref['file'], block_ref['sha256'], block_ref['bytes']))
    row = copy.deepcopy(block['rows'][row_ref['block_row_index']])
    registry = json.loads(inputs.read(REGISTRY, registry_sha, limit=MAX_BYTES))
    refs = [r for r in registry['indexes'] if r.get('outer') == 200]
    need(len(refs) == 1 and refs[0].get('file') == INDEX, 'Exactly one candidate200 registry entry required')
    extension = json.loads(inputs.read(INDEX, refs[0]['sha256'], refs[0]['bytes'], MAX_BYTES))
    need(len(extension.get('parts', [])) == 1 and extension['parts'][0].get('file') == PART, 'Candidate200 index part differs')
    part_ref = extension['parts'][0]
    part = json.loads(inputs.read(PART, part_ref['sha256'], part_ref['bytes'], MAX_BYTES))
    need(part.get('unit') == UNIT and part.get('run_directory') == RUN and part.get('completed_rollout_steps') == [200], 'Candidate200 unit/run/completion differs')
    supplement = part['source_pin'].get('supplemental_controller_poll')
    if supplement:
        inputs.read(supplement['file'], supplement['sha256'], supplement['bytes'])
    expected_pin = collector.completed_pin(base, book, 200, local, Path(supplement['file']) if supplement else None)
    need(part['source_pin'] == expected_pin, 'Candidate200 original collector source pin differs')
    prior_join = cumulative['candidate_cost_backfill_join']
    need(prior_join.get('applied_outers') == list(range(3, 200)), 'Original candidate joins must contain3..199 once and exclude200')
    need([item.get('outer') for item in prior_join.get('inputs', [])] == list(range(3, 200)), 'Original candidate join input coverage differs')
    inputs.read(prior_join['index_file'], prior_join['index_sha256'], limit=MAX_BYTES)
    join_args = SimpleNamespace(base=base, candidate_backfill_index=Path(prior_join['index_file']), candidate_backfill_sha256=prior_join['index_sha256'], candidate_extension_inputs=Path(REGISTRY))
    joined = local.join_candidate_backfill(join_args, book, [row])
    need(joined['applied_outers'] == [200] and len(joined['inputs']) == 1, 'Exactly one validated terminal candidate join required')
    terminal = part['rollouts'][0];outcomes = terminal['candidate_outcomes']
    controller = terminal['rollout_controller_cost']
    need(controller.get('model_identity') == MODEL and controller.get('data_identity') == DATA, 'Candidate200 model/data identity differs')
    counts200 = {key:outcomes[key]['candidate_count'] for key in ('all', 'accepted', 'discarded', 'fallback')}
    need(all(type(v) is int and v >= 0 for v in counts200.values()) and counts200['all'] == counts200['accepted'] + counts200['discarded'] and counts200['fallback'] == outcomes['accepted_fallback']['candidate_count'] <= counts200['accepted'] and outcomes['unknown_outcome']['candidate_count'] == 0, 'Candidate200 outcome partition/fallback overlap differs')
    need(counts200 == row['candidate_cost_detail']['candidate_counts'], 'Candidate200 counts differ from bound FULL scalar row')
    result = copy.deepcopy(cumulative)
    result['candidate_outcome_costs_partial'] = add_terminal_outcomes(cumulative['candidate_outcome_costs_partial'], outcomes)
    result['candidate_cost_backfill_join']['applied_outers'].append(200)
    result['candidate_cost_backfill_join']['inputs'].extend(joined['inputs'])
    result['candidate_cost_backfill_join']['terminal200_join'] = joined
    result['summary_schema'] = 'tree_nested_final_training_candidate_cost_supplement_v1'
    result['created_utc'] = datetime.now(timezone.utc).isoformat()
    result['final_candidate_supplement'] = dict(scope='Training candidate outcomes only; no whole-goal/final51713/HF-cold-reload/cleanup/terminal-service acceptance.',
        original_CUM200=inputs.refs[CUM], registry_snapshot=inputs.refs[REGISTRY], terminal_candidate_index=inputs.refs[INDEX], terminal_candidate_part=inputs.refs[PART],
        terminal_native_writer=counts, terminal_candidate_original_scalars=terminal, terminal200_counted_once=True,
        sampling_dev_driver_CPU_fallback_not_added=True, training201_entry_claimed=False, independent_cold_reload_performed=False,
        final51713_completion_claimed=False, whole_goal_completion_claimed=False, resource_cleanup_claimed=False,
        unchanged_CUM_fields=[k for k in cumulative if k not in ('candidate_outcome_costs_partial', 'candidate_cost_backfill_join', 'summary_schema', 'created_utc')])
    need(all(result[k] == cumulative[k] for k in result['final_candidate_supplement']['unchanged_CUM_fields']), 'Unrelated cumulative cost field changed')
    missing = {outcome:{key:value['missing_outers'] for key, value in record['field_sums'].items() if value['missing_outers']} for outcome, record in result['candidate_outcome_costs_partial'].items()}
    result['final_candidate_supplement']['remaining_unknown_terminal_scalar_fields'] = missing
    result['final_candidate_supplement']['candidate_outcome_scalar_coverage_complete'] = not any(missing.values())
    return result, inputs, mirrors


def bounded_fields(result, max_bytes=MAX_BYTES):
    result = copy.deepcopy(result);artifacts = {}
    def split(value, field):
        kind = 'mapping' if isinstance(value, dict) else 'list'
        need(isinstance(value, (dict, list)), 'Oversized final scalar cannot be split')
        chunks = [];chunk = {} if kind == 'mapping' else []
        for key, item in (value.items() if kind == 'mapping' else enumerate(value)):
            trial = {key:item} if kind == 'mapping' else [item]
            if len(encoded(dict(schema='final_candidate_scalar_field_part_v1', original_field=field, value=trial))) > max_bytes:
                item = split(item, field + ('.' + str(key) if kind == 'mapping' else '[' + str(key) + ']'))
            trial = copy.deepcopy(chunk)
            if kind == 'mapping':trial[key] = item
            else:trial.append(item)
            if len(encoded(dict(schema='final_candidate_scalar_field_part_v1', original_field=field, value=trial))) > max_bytes and chunk:
                chunks.append(chunk);chunk = {key:item} if kind == 'mapping' else [item]
            else:chunk = trial
            need(len(encoded(dict(schema='final_candidate_scalar_field_part_v1', original_field=field, value=chunk))) <= max_bytes, 'One original final scalar item exceeds part budget')
        if chunk:chunks.append(chunk)
        refs = [];safe = re.sub(r'[^A-Za-z0-9_]', '_', field)[:80] + '_' + sha(field.encode())[:12]
        for number, chunk in enumerate(chunks):
            name = f'final_candidate_cost_parts/{STEM}_{safe}_PART{number:03d}.json'
            raw = encoded(dict(schema='final_candidate_scalar_field_part_v1', original_field=field, value=chunk));artifacts[name] = raw
            refs.append(dict(file=name, bytes=len(raw), sha256=sha(raw), json_selector='value'))
        return dict(storage='complete_original_field_parts', container_type=kind, original_entries=len(value), parts=refs)
    for field in sorted(result, key=lambda key:len(encoded(result[key])), reverse=True):
        if len(encoded(result)) <= max_bytes:
            break
        value = result[field]
        if not isinstance(value, (dict, list)):
            continue
        result[field] = split(value, field)
    need(len(encoded(result)) <= max_bytes, 'Final candidate descriptor exceeds complete-field budget')
    artifacts[STEM + '.json'] = encoded(result)
    return artifacts


def publish(base, mirrors, artifacts):
    for name, raw in artifacts.items():
        need(len(raw) <= MAX_BYTES, 'Small final metadata output required')
        for folder in [base] + mirrors:
            path = folder / name
            need(not path.exists() or path.read_bytes() == raw, 'Existing immutable final output differs: ' + name)
    for name, raw in artifacts.items():
        for folder in [base] + mirrors:
            path = folder / name;path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                tmp = path.with_name(path.name + '.tmp');tmp.write_bytes(raw);os.replace(tmp, path)
    for _ in range(2):
        for name, raw in artifacts.items():
            need(all((folder / name).read_bytes() == raw for folder in [base] + mirrors), 'Final output byte readback differs')


def run(args):
    base = args.base.resolve(strict=True)
    required = [CUM, FULL, NATIVE, REGISTRY, INDEX, PART]
    absent = [name for name in required if not (base / name).exists()]
    if absent:
        return dict(status='PENDING_ACTUAL200_CANDIDATE_EVIDENCE', missing=absent, outputs_written=False, remote_operations=False)
    receipt_name = STEM + '_LOCAL_RECEIPT.json'
    if (base / receipt_name).exists():
        prior = json.loads((base / receipt_name).read_bytes());mirrors = [Path(p) for p in prior['mirrors']]
        need(prior['source_CUM200']['sha256'] == args.cumulative_sha256 and prior['source_registry']['sha256'] == args.registry_sha256 and prior['source_script_sha256'] == sha(Path(__file__).read_bytes()), 'Existing final receipt input/source binding differs')
        source_raw = (base / CUM).read_bytes()
        need(sha(source_raw) == args.cumulative_sha256 and json.loads(source_raw)['mirrors'] == prior['mirrors'], 'Existing final receipt original CUM/mirrors differ')
        need(prior['input_refs'].get(CUM) == prior['source_CUM200'] and prior['input_refs'].get(REGISTRY) == prior['source_registry'], 'Existing final receipt omitted original input binding')
        inputs = LocalInputs(base, mirrors)
        for ref in prior['input_refs'].values():inputs.reference(ref, 4000000)
        for ref in prior['output_refs'].values():
            if ref['file'].endswith('.json'):
                inputs.reference(ref)
            else:
                need(ref['file'] == STEM + '.md', 'Unexpected final non-JSON output reference')
                raw = (base / ref['file']).read_bytes()
                need(len(raw) == ref['bytes'] and sha(raw) == ref['sha256'] and all((m/ref['file']).read_bytes() == raw for m in mirrors), 'Existing final Markdown differs')
        need(all((m/receipt_name).read_bytes() == (base/receipt_name).read_bytes() for m in mirrors), 'Existing final receipt mirror differs')
        return dict(status='ALREADY_EXISTS_BYTES_VERIFIED', receipt=receipt_name, outputs_written=False, remote_operations=False)
    result, inputs, mirrors = prepare(base, args.cumulative_sha256, args.registry_sha256)
    artifacts = bounded_fields(result)
    artifacts[STEM + '.md'] = ('# 最后训练候选成本补充\n\n实际FULL/native/CUM200与candidate200全部绑定；仅补末轮一次，原CUM200及所有输入不改。\n\n'
        'sampling、outer wall、dev与driver/service CPU沿用原累计范围；fallback与accepted重叠，候选子集不再加到总采样。DP padding分摊、独立结果wall、FLOPs/实际GPU计费/能耗未知。\n\n'
        '此文档仅闭合训练candidate结局元数据；未声称51713完成、HF冷重载、资源清理、服务终态或整个goal完成。保存cursor201不代表实际训练201。\n').encode()
    refs = {name:dict(file=name, bytes=len(raw), sha256=sha(raw)) for name, raw in artifacts.items()}
    receipt = dict(schema='final_candidate_cost_local_receipt_v1', created_utc=result['created_utc'], scope='Training candidate metadata only',
        source_CUM200=inputs.refs[CUM], source_registry=inputs.refs[REGISTRY], source_script_sha256=sha(Path(__file__).read_bytes()),
        local_dependency_source_sha256=DEPENDENCIES,
        input_refs=inputs.refs, output_refs=refs, mirrors=[str(m) for m in mirrors], terminal200_counted_once=True,
        candidate_outcome_scalar_coverage_complete=result['final_candidate_supplement']['candidate_outcome_scalar_coverage_complete'],
        original_sampling_totals_unchanged=True, original_CUM200_rewritten=False, canonical_or_training_modified=False,
        remote_operations=False, tensor_model_or_raw_rollout_read=False, independent_cold_reload_performed=False,
        final51713_or_HF_or_cleanup_or_terminal_CPU_acceptance_claimed=False, whole_goal_completion_claimed=False,
        both_mirrors_byte_readback_passes=2)
    artifacts[receipt_name] = encoded(receipt)
    # Recheck all SHA-bound inputs before any output and after publication.
    for ref in inputs.refs.values():inputs.read(ref['file'], ref['sha256'], ref['bytes'])
    publish(base, mirrors, artifacts)
    for ref in inputs.refs.values():inputs.read(ref['file'], ref['sha256'], ref['bytes'])
    return dict(status='ACTUAL_TRAINING_CANDIDATE200_SUPPLEMENT_PUBLISHED', artifacts={name:dict(bytes=len(raw), sha256=sha(raw)) for name, raw in artifacts.items()},
                candidate_outcome_scalar_coverage_complete=receipt['candidate_outcome_scalar_coverage_complete'], remote_operations=False, whole_goal_completion_claimed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--cumulative-sha256', required=True, help='Original immutable CUM200 raw-file SHA256')
    parser.add_argument('--registry-sha256', required=True, help='Actual registry snapshot SHA256 including candidate200')
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
