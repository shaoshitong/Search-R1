#!/usr/bin/env python3
"""Prepared final200 local curve readback; source preparation is not acceptance.

From frozen root_curve180_actual_readback.py22278B/d8dadba4... . Future CLI:
python3 -B work/tree-speed-repair-20261008/root_curve200_actual_readback.py \
  --pins "$ACTUAL_ROOT_CURVE200_PINS" --pins-bytes "$ACTUAL_ROOT_CURVE200_PINS_BYTES" \
  --pins-sha256 "$ACTUAL_ROOT_CURVE200_PINS_SHA256"

Input schema dev200_root_curve_actual_readback_input_pins_v1/through_outer200,
source_pins exactly four curve200 products, ROOT_DEV200 and the actual source3e05
FINAL_CANDIDATE_COST_THROUGH200_LOCAL_RECEIPT.json. The bound build manifest
supplies every actual prior195 curve descriptor and actual firstFULL/native/CUM
196..200. Original195 train/39dev object text and actual CSV prefix preserved.
Only writes ROOT_CURVE_DEV_COST_THROUGH200_ACTUAL_READBACK.json. No actual201,
ROOT_FULL200, finalalias, slowfinal51713 completion, remote access, collector,
training, canonical mutation, compile, test or review is required/performed.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import math
import re
import time

W = Path(__file__).resolve().parents[2]
B = W / 'outputs/tree-nested-branch-handoff-20261013'
M = [W / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
     Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
NAMES = ('LEARNING_CURVE_COMPLETE_THROUGH200.json', 'LEARNING_CURVE_COMPLETE_THROUGH200.csv',
         'LEARNING_CURVE_THROUGH200_LOCAL_PUBLICATION.json', 'LEARNING_CURVE_THROUGH200_COST_SUPPLEMENT_BINDING.json')
ROOT_DEV = 'ROOT_DEV200_NATIVE_COST_ACTUAL_READBACK.json'
OUTPUT = 'ROOT_CURVE_DEV_COST_THROUGH200_ACTUAL_READBACK.json'
SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
ENTRY = '1edaf50bb8e14de51db936e0859e369792011233'
COLLECTOR_SHA = '545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0'
TERMINAL_RECEIPT = 'FINAL_CANDIDATE_COST_THROUGH200_LOCAL_RECEIPT.json'
INPUTS = {}


def h(raw):
    return hashlib.sha256(raw).hexdigest()


def locate(name):
    p = Path(name)
    assert p.parts and '..' not in p.parts
    p = p if p.is_absolute() else W / p if p.parts[0] in ('outputs', 'work') else B / p
    p = p.resolve(strict=True)
    assert p.is_relative_to(W) and p.suffix in ('.json', '.csv', '.py', '.txt', '.md', '.sh', '.patch')
    return p


def pin(name, digest=None, size=None):
    p = locate(name)
    raw = p.read_bytes()
    assert 0 < len(raw) < 4000000
    assert digest is None or isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) and h(raw) == digest
    assert size is None or type(size) is int and len(raw) == size
    mirrored = p.is_relative_to(B) and not p.name.startswith(('NATIVE5_CONTINUE_POLL_', 'NATIVE5_CONTINUE_RESOURCE_'))
    if mirrored:
        for mirror in M:
            assert (mirror / p.relative_to(B)).read_bytes() == raw
    ref = dict(path=str(p), bytes=len(raw), sha256=h(raw))
    assert str(p) not in INPUTS or INPUTS[str(p)] == (raw, ref, mirrored)
    INPUTS[str(p)] = (raw, ref, mirrored)
    return ref


def bound(ref, expected_name=None):
    assert isinstance(ref, dict) and isinstance(ref.get('path', ref.get('file')), str)
    name = ref.get('path', ref.get('file'))
    if expected_name is not None:
        assert locate(name) == locate(expected_name)
    return pin(name, ref['sha256'], ref['bytes'])


def read(name):
    p = locate(name)
    pin(p)
    return json.loads(INPUTS[str(p)][0])


def segments(name, key):
    text = INPUTS[str(locate(name))][0].decode('utf8')
    match = re.search('"' + key + '"\\s*:\\s*\\[', text)
    assert match
    i, result, dec = match.end(), [], json.JSONDecoder()
    while True:
        while text[i].isspace() or text[i] == ',':
            i += 1
        if text[i] == ']':
            return result
        _, end = dec.raw_decode(text, i)
        result.append(text[i:end])
        i = end


def two_pass():
    observations = []
    for number in range(2):
        for name, (raw, ref, mirrored) in INPUTS.items():
            assert Path(name).read_bytes() == raw and len(raw) == ref['bytes'] and h(raw) == ref['sha256']
            if mirrored:
                for mirror in M:
                    assert (mirror / Path(name).relative_to(B)).read_bytes() == raw
        observations.append(dict(pass_number=number + 1, observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 unique_files=len(INPUTS), mirrored_files=sum(v[2] for v in INPUTS.values())))
        if number == 0:
            time.sleep(1)
    return observations


def terminal_candidate_readback(receipt_pin, full, rawcum, cost):
    receipt_ref = bound(receipt_pin, TERMINAL_RECEIPT)
    receipt = read(TERMINAL_RECEIPT)
    source_ref = pin(W / 'work/tree-speed-repair-20261008/nested-final-candidate-cost-fix-source/reproduction/tree-eval-resume-20261013/local-control-backup/finalize_nested_candidate_costs.py',
        '8ff7b52e88e2cce19dad244914c32ebec9fd25f18489b4856cc3213dd654eb96', 28237)
    assert receipt['schema'] == 'final_candidate_cost_local_receipt_v1'
    assert receipt['source_script_sha256'] == source_ref['sha256'] and receipt['terminal200_counted_once'] is True
    assert receipt['original_CUM200_rewritten'] is False and receipt['original_sampling_totals_unchanged'] is True
    assert receipt['remote_operations'] is False and receipt['whole_goal_completion_claimed'] is False
    bound(receipt['source_CUM200'], 'CUMULATIVE_COST_THROUGH200.json')
    registry_ref = bound(receipt['source_registry'], 'CANDIDATE_COST_EXTENSION_INPUTS.json')
    assert registry_ref == bound(cost['registry'], 'CANDIDATE_COST_EXTENSION_INPUTS.json')
    input_refs = {name:bound(ref, name) for name,ref in receipt['input_refs'].items()}
    output_refs = {name:bound(ref, name) for name,ref in receipt['output_refs'].items()}
    root_name, markdown_name = 'FINAL_CANDIDATE_COST_THROUGH200.json', 'FINAL_CANDIDATE_COST_THROUGH200.md'
    assert root_name in output_refs and markdown_name in output_refs
    final_parts = []
    def expand(value, field, schema, part_refs):
        if isinstance(value, dict) and value.get('storage') == 'complete_original_field_parts':
            kind = value['container_type']
            assert kind in ('list','mapping') and value['parts']
            result, seen = ([] if kind == 'list' else {}), set()
            for ref in value['parts']:
                assert ref['file'] not in seen and ref['json_selector'] == 'value'
                seen.add(ref['file'])
                part_ref = bound(ref)
                part = read(ref['file'])
                assert part['schema'] == schema and part['original_field'] == field
                assert isinstance(part['value'], list if kind == 'list' else dict)
                part_refs.append(part_ref)
                if kind == 'list':
                    result.extend(part['value'])
                else:
                    assert not set(result).intersection(part['value'])
                    result.update(part['value'])
            assert len(result) == value['original_entries']
            return expand(result, field, schema, part_refs)
        if isinstance(value, dict):
            return {key:expand(item,field+'.'+key,schema,part_refs) for key,item in value.items()}
        if isinstance(value, list):
            return [expand(item,field+'['+str(i)+']',schema,part_refs) for i,item in enumerate(value)]
        return value
    original = {field:expand(value,field,'cumulative_scalar_field_part_v1',[]) for field,value in rawcum.items()}
    finalraw = read(root_name)
    final = {field:expand(value,field,'final_candidate_scalar_field_part_v1',final_parts) for field,value in finalraw.items()}
    assert set(output_refs) == {root_name,markdown_name} | {Path(ref['path']).relative_to(B).as_posix() for ref in final_parts}
    meta = final['final_candidate_supplement']
    assert final['summary_schema'] == 'tree_nested_final_training_candidate_cost_supplement_v1' and final['through_effective_outer'] == 200
    assert meta['terminal200_counted_once'] is True and meta['sampling_dev_driver_CPU_fallback_not_added'] is True
    assert meta['training201_entry_claimed'] is meta['final51713_completion_claimed'] is meta['whole_goal_completion_claimed'] is False
    bound(meta['original_CUM200'], 'CUMULATIVE_COST_THROUGH200.json')
    bound(meta['registry_snapshot'], 'CANDIDATE_COST_EXTENSION_INPUTS.json')
    assert set(meta['unchanged_CUM_fields']) == set(original) - {'candidate_outcome_costs_partial','candidate_cost_backfill_join','summary_schema','created_utc'}
    assert all(final[key] == original[key] for key in meta['unchanged_CUM_fields'])
    assert original['candidate_cost_backfill_join']['applied_outers'] == list(range(3,200))
    assert final['candidate_cost_backfill_join']['applied_outers'] == list(range(3,201))
    assert [item['outer'] for item in final['candidate_cost_backfill_join']['inputs']] == list(range(3,201))
    terminal_part = bound(meta['terminal_candidate_part'], 'candidate_cost_backfill/CANDIDATE_OUTER0200.json')
    assert read(terminal_part['path'])['rollouts'][0] == meta['terminal_candidate_original_scalars']
    assert meta['terminal_native_writer']['actual_adam_per_rank'] == full['per_rank_actual_total_adam']
    return dict(receipt=receipt_ref,summary=output_refs[root_name],markdown=output_refs[markdown_name],source_entry=source_ref,
        source_entry_commit='3e05dfd9bf2fd296e32ea9d8d7c072d2669aaacb',source_original_CUM200=pin('CUMULATIVE_COST_THROUGH200.json'),
        source_registry=registry_ref,all_input_refs=list(input_refs.values()),all_output_refs=list(output_refs.values()),complete_field_parts=final_parts,
        terminal200_counted_once=True,original_CUM200_raw_bytes_unchanged=True,candidate200_applied_to_terminal_supplement_only=True,
        candidate_outcome_scalar_coverage_complete=receipt['candidate_outcome_scalar_coverage_complete'],actual_training201_required=False,
        slow_final51713_completion_required=False,added_again_to_curve_or_service_cost=False)


def main():
    if not __debug__:
        raise RuntimeError('Required actual input gates cannot be disabled with -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pins', required=True, type=Path)
    parser.add_argument('--pins-bytes', required=True, type=int)
    parser.add_argument('--pins-sha256', required=True)
    args = parser.parse_args()
    manifest_ref = pin(args.pins, args.pins_sha256, args.pins_bytes)
    manifest = read(args.pins)
    assert manifest['schema'] == 'dev200_root_curve_actual_readback_input_pins_v1' and manifest['through_outer'] == 200
    assert set(manifest['source_pins']) == set(NAMES) | {ROOT_DEV, TERMINAL_RECEIPT}
    for name, ref in manifest['source_pins'].items():
        bound(ref, name)
    rootdev = read(ROOT_DEV)
    assert rootdev['status'] == 'PASS_ACTUAL_DEV200_LOCAL_ARTIFACT_READBACK' and rootdev['outer'] == 200
    assert rootdev['inconsistencies'] == [] and rootdev['two_pass_local_input_and_mirror_bytes_verified'] is True
    assert rootdev['prior39_rows_exact_object_equal'] is True and rootdev['prior39_control_original_bytes_SHA_preserved'] is True
    assert rootdev['all_original_DEV195_row_parts_exact_bytes_SHA_preserved'] is True
    for name, ref in rootdev['source_pins'].items():
        bound(ref, name)
    curve, publication, binding = read(NAMES[0]), read(NAMES[2]), read(NAMES[3])
    curve_ref, csv_ref = pin(NAMES[0]), pin(NAMES[1])
    build_manifest_ref = bound(binding['build_input_manifest'])
    assert publication['build_input_manifest'] == build_manifest_ref
    build_manifest = read(build_manifest_ref['path'])
    prior_inputs = build_manifest['prior_inputs']
    assert prior_inputs['outer'] == 195
    owner = read('DEV5_200_NATIVE_COST_LOCAL_RECEIPT.json')
    assert owner['prior_inputs'] == prior_inputs
    curve_names = dict(json='LEARNING_CURVE_COMPLETE_THROUGH195.json', csv='LEARNING_CURVE_COMPLETE_THROUGH195.csv',
        publication='LEARNING_CURVE_THROUGH195_LOCAL_PUBLICATION.json', binding='LEARNING_CURVE_THROUGH195_COST_SUPPLEMENT_BINDING.json',
        root_readback='ROOT_CURVE_DEV_COST_THROUGH195_ACTUAL_READBACK.json')
    assert set(prior_inputs['curve']) == set(curve_names)
    prior_curve_refs = {key:bound(prior_inputs['curve'][key], name) for key,name in curve_names.items()}
    old_ref, old_csv_ref = prior_curve_refs['json'], prior_curve_refs['csv']
    oldroot = read(prior_curve_refs['root_readback']['path'])
    assert oldroot['status'] == 'PASS_ACTUAL_CURVE195_LOCAL_ARTIFACT_READBACK' and oldroot['outer'] == 195
    old = read(old_ref['path'])
    assert curve['training_points'][:195] == old['training_points'] and curve['completed_dev512_points'][:39] == old['completed_dev512_points']
    assert len(curve['training_points']) == curve['training_point_count'] == 200
    assert len(curve['completed_dev512_points']) == curve['completed_dev512_point_count'] == 40
    assert segments(NAMES[0], 'training_points')[:195] == segments(old_ref['path'], 'training_points')
    assert segments(NAMES[0], 'completed_dev512_points')[:39] == segments(old_ref['path'], 'completed_dev512_points')
    oldcsv, newcsv = INPUTS[str(locate(old_csv_ref['path']))][0], INPUTS[str(locate(NAMES[1]))][0]
    assert newcsv.startswith(oldcsv) and len(list(csv.DictReader(io.StringIO(newcsv.decode('utf8'))))) == 240
    assert binding['through_outer'] == 200 and binding['counts'] == publication['counts'] == dict(training=200, actual_completed_dev512=40, csv_data_rows=240)
    assert binding['actual_remote_incremental_collection_invocations'] == binding['new_entry_main_invocations_this_task_by_collector_executor'] == 1
    assert binding['preceding_local_pending_invocations'] == 0 and binding['preceding_pending_remote_operations'] is False
    assert binding['ROOT_FULL200_used_as_input'] is False and publication['ROOT_FULL200_used_as_input'] is False
    assert binding['original_wall_CPU_double_added'] is False and binding['supplemental_totals_added_to_curve_or_CUM'] is False
    assert binding['HF150_boundary_advanced'] is False and publication['HF150_boundary_advanced'] is False
    refs = {key: bound(ref) for key, ref in binding['references'].items()}
    assert refs['curve'] == curve_ref and refs['csv'] == csv_ref and refs['curve_publication'] == pin(NAMES[2])
    assert refs['root_dev_native_cost_readback'] == pin(ROOT_DEV)
    assert refs['full200'] == pin('FULL_OUTER200_PROGRESS_RECEIPT.json')
    build_manifest_ref = bound(binding['build_input_manifest'])
    assert publication['build_input_manifest'] == build_manifest_ref
    build_manifest = read(build_manifest_ref['path'])
    assert build_manifest['schema'] == 'dev200_actual_curve_build_input_pins_v1' and build_manifest['through_outer'] == 200
    required = {ROOT_DEV} | {f'FULL_OUTER{o}_PROGRESS_RECEIPT.json' for o in range(196, 201)} | {f'NATIVE_HISTORY_{o}_METADATA_READBACK.json' for o in range(196, 201)} | {f'CUMULATIVE_COST_THROUGH{o}.json' for o in range(196, 201)}
    assert set(build_manifest['source_pins']) == required
    for name, ref in build_manifest['source_pins'].items():
        bound(ref, name)
    core_path = W / 'work/tree-speed-repair-20261008/nested-native5-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/sync_nested_native_progress.py'
    pin(core_path, '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1', 39098)
    spec = importlib.util.spec_from_file_location('root_curve200_published_core_metadata_only', core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    pin(W / 'work/tree-speed-repair-20261008/dev200_owner_local_receipt.py', '6db63ce679c3a088607d267e3bfee9cb87147447cd93eec4592426dda95358cf', 51265)
    pin(W / 'work/tree-speed-repair-20261008/dev200_independent_actual_readback.py', '5777a2a365213c004ee4ab3983db66ecefc02621dc86ac531336e61b9fba2425', 60678)
    pin(W / 'work/tree-speed-repair-20261008/variance200_record_coverage.py', 'bf44ce6316fa976fffba96db218d2f193f82be70877bc8729e1b7cef51ac0885', 29073)
    original15_fields = ('critic/rewards/mean','critic/score/mean','tree_variance/within_root_raw_reward_population_variance','tree_variance/intra_adv_population_variance','tree_variance/group_actual_adv_population_variance','tree_variance/same_reward_question_fraction','tree_variance/mixed_reward_question_fraction','tree_variance/exact_reward_one_question_count','tree_variance/exact_reward_zero_question_count','tree_variance/root0_same_reward_fraction','tree_variance/root0_mixed_reward_fraction','tree_variance/root1_same_reward_fraction','tree_variance/root1_mixed_reward_fraction','tree_variance/same_reward_tree_fraction','tree_variance/mixed_reward_tree_fraction')
    def own_cum_parts(value, field, result):
        if isinstance(value, dict) and value.get('storage') == 'complete_original_field_parts':
            assert value['container_type'] in ('list','mapping') and value['parts']
            seen, assembled = set(), ([] if value['container_type'] == 'list' else {})
            for ref in value['parts']:
                assert ref['file'] not in seen and ref['json_selector'] == 'value'
                seen.add(ref['file'])
                part_ref = bound(ref)
                part = read(ref['file'])
                assert part['schema'] == 'cumulative_scalar_field_part_v1' and part['original_field'] == field
                assert isinstance(part['value'], list if value['container_type'] == 'list' else dict)
                result.append(part_ref)
                if isinstance(assembled, list):
                    assembled.extend(part['value'])
                else:
                    assert not set(assembled).intersection(part['value'])
                    assembled.update(part['value'])
            assert len(assembled) == value['original_entries']
            own_cum_parts(assembled, field, result)
        elif isinstance(value, dict):
            for key,item in value.items():
                own_cum_parts(item, field+'.'+key, result)
        elif isinstance(value, list):
            for index,item in enumerate(value):
                own_cum_parts(item, field+'['+str(index)+']', result)
    new = []
    scalar_fields = dict(raw_within_tree_reward_population_variance='tree_variance/within_root_raw_reward_population_variance',
        raw_group_reward_population_variance='tree_variance/group_raw_reward_population_variance',
        pure_intra_adv_population_variance='tree_variance/intra_adv_population_variance',
        total_actual_adv_population_variance='tree_variance/group_actual_adv_population_variance',
        training_reward_mean='critic/rewards/mean', training_score_mean='critic/score/mean',
        mixed_reward_question_fraction='tree_variance/mixed_reward_question_fraction',
        same_reward_question_fraction='tree_variance/same_reward_question_fraction',
        mixed_reward_tree_fraction='tree_variance/mixed_reward_tree_fraction',
        same_reward_tree_fraction='tree_variance/same_reward_tree_fraction', generation_wall_seconds='timing_s/gen')
    for outer in range(196, 201):
        point = next(x for x in curve['training_points'] if x['outer'] == outer)
        full_name = f'FULL_OUTER{outer}_PROGRESS_RECEIPT.json'
        full = read(full_name)
        full_ref = pin(full_name, point['source_receipt_sha256'])
        assert full['completed_outer'] == point['outer'] == outer and full['source_commit'] == point['source_commit'] == SOURCE
        assert full['config_sha256'] == point['config_sha256'] == CONFIG
        assert full['actual_training_invocation_id'] == 'a313bda919834d24b4c5d491698d39ce'
        poll_ref = bound(full['input_bindings']['poll'])
        poll_path = locate(poll_ref['path'])
        assert poll_path.parent == B and poll_path.name.startswith('NATIVE5_CONTINUE_POLL_')
        assert locate(point['source_poll']) == locate(full['source_poll']) == poll_path
        assert point['source_poll_sha256'] == full['source_poll_sha256'] == poll_ref['sha256']
        poll = read(poll_path)
        metrics = [x for x in poll['iteration_metrics'] if int(x['step']) == outer]
        assert len(metrics) == 1 and metrics[0] == full['iteration_metrics'] and not poll['progress'].get('errors')
        resource_ref = bound(full['input_bindings']['resource'])
        resource = read(resource_ref['path'])
        ps, rs, active = core.validate_unit(poll, resource, outer)
        assert full['actual_training_pid'] == int(rs['MainPID'])
        if outer < 200:
            assert active and full['actual_next_outer_entered'] == outer + 1
            assert full['status'] == f'ACTUAL_FULL_OUTER{outer}_NATIVE{outer}_COMPLETE_NEXT{outer+1}_ENTERED'
            assert re.search(r'epoch\s+\d+,\s*step\s+'+str(outer+1)+r'\b', (poll.get('logs') or {}).get('stage.log') or '') is not None
        else:
            assert full['status'] == 'ACTUAL_FULL_OUTER200_NATIVE200_COMPLETE_FINAL_PHASE_OBSERVED'
            assert full['actual_next_outer_entered'] is None and full['full_iteration_metric_pending'] is False and full['unobserved_metric_fields_remain_null'] is False
            assert full['final_phase_evidence'] == core.final_phase(poll, resource, active)
        assert metrics[0] == core.validate_metric(poll, outer)
        assert all(isinstance(metrics[0].get(field), (int,float)) and not isinstance(metrics[0][field],bool) and math.isfinite(metrics[0][field]) for field in original15_fields)
        assert point['original15_scalar_fields'] == {field:metrics[0][field] for field in original15_fields}
        core.validate_restore(poll)
        assert full['last_rank_summaries'] == core.validate_ranks(poll, metrics[0], outer)
        for key, selector in scalar_fields.items():
            assert point[key] == metrics[0].get(selector), (outer, key)
        assert point['actual_cumulative_adam_calls_per_rank'] == full['per_rank_actual_total_adam']
        assert point['actual_outer_adam_calls_per_rank'] == full['actual_outer_successful_writes_per_rank']
        assert point['actual_outer_skipped_opportunities_per_rank'] == full['actual_outer_skipped_opportunities_per_rank']
        assert point['actual_outer_processed_opportunities_per_rank'] == full['actual_outer_processed_opportunities_per_rank']
        assert point['step_wall_seconds'] == full['full_outer_wall_seconds']
        assert point['normal_dev_testing_seconds'] == full['normal_dev512_wall_seconds']
        assert point['training_archive_wall_seconds_excluding_current_dev'] == full['training_archive_wall_seconds_excluding_current_dev']
        bound(point['cumulative_cost_source'], f'CUMULATIVE_COST_THROUGH{outer}.json')
        own_cum = read(f'CUMULATIVE_COST_THROUGH{outer}.json')
        assert own_cum['through_effective_outer'] == outer and own_cum['pending_full_metric_outers'] == []
        own_parts = []
        for field,value in own_cum.items():
            own_cum_parts(value, field, own_parts)
        assert point['cumulative_cost_complete_parts'] == own_parts
        native_ref = bound(full['input_bindings']['native_readback'], f'NATIVE_HISTORY_{outer}_METADATA_READBACK.json')
        native = read(native_ref['path'])
        assert native['outer'] == outer and native['archive_matches_original_manifest'] is True
        record, guard, completed, totals = core.validate_native(native, poll['progress'], outer)
        assert full['per_rank_successful_adam'] == completed and full['per_rank_actual_total_adam'] == totals
        assert record['driver']['next_step'] == outer + 1
        if outer < 200:
            assert full['actual_dev512'] is None and point['current_outer_completed_dev512'] is None
        new.append(dict(outer=outer, full=full_ref, poll=poll_ref, native_metadata=native_ref,
                        cumulative_cost=point['cumulative_cost_source'], cumulative_complete_parts=own_parts, original_first_FULL_input_binding_used=True))
    full = read('FULL_OUTER200_PROGRESS_RECEIPT.json')
    actual = full['actual_dev512']
    dev = next(x for x in curve['completed_dev512_points'] if x['outer'] == 200)
    assert dev['rows'] == actual['rows'] == 512 and actual['metrics']['val/qa_count/all'] == 512
    assert actual['metrics']['val/qa_count/nq'] == actual['metrics']['val/qa_count/hotpotqa'] == 256
    assert actual['status'] == dev['status'] == 'VALIDATION_COST_COMPLETE'
    assert dev['ledger_sha256'] == actual['ledger_sha256'] and dev['ledger_path'] == actual['path']
    assert dev['f1'] == actual['metrics']['val/f1/all'] and dev['em'] == actual['metrics']['val/em/all']
    assert dev['nq_f1'] == actual['metrics']['val/f1/nq'] and dev['hotpotqa_f1'] == actual['metrics']['val/f1/hotpotqa']
    sp = W / 'work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py'
    pin(sp, COLLECTOR_SHA, 27911)
    spec = importlib.util.spec_from_file_location('root_curve200_published_decoder_only', sp)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    summary = read('DEV5_200_NATIVE_COST_SUPPLEMENT.json')
    prior_ref = bound(prior_inputs['supplement'], 'DEV5_195_NATIVE_COST_SUPPLEMENT.json')
    prior_summary = read(prior_ref['path'])
    rows, oldrows = decoder.rows_from_summary(B, M, summary), decoder.rows_from_summary(B, M, prior_summary)
    assert len(rows) == 40 and len(oldrows) == 39 and rows[:-1] == oldrows
    parts = [bound(ref) for ref in binding['dev_native_cost_rows_part_references']]
    prior_parts = [pin(ref['file'], ref['sha256'], ref['bytes']) for ref in prior_summary['rows'].get('parts', [])] if isinstance(prior_summary['rows'], dict) else []
    prior_owner_ref = bound(prior_inputs['owner_receipt'], 'DEV5_195_NATIVE_COST_LOCAL_RECEIPT.json')
    prior_owner = read(prior_owner_ref['path'])
    assert prior_parts == [bound(ref) for ref in prior_owner['rows_part_refs']]
    source_parts = summary['rows'].get('parts', []) if isinstance(summary['rows'], dict) else []
    assert parts == [pin(ref['file'], ref['sha256'], ref['bytes']) for ref in source_parts]
    controls = []
    for row in rows:
        saved = bound(row['local_original_summary'])
        value = read(saved['path'])
        assert saved['sha256'] == row['source_control_summary']['sha256'] and saved['bytes'] == row['source_control_summary']['bytes']
        assert value['ledger_sha256'] == row['ledger_sha256'] and value['actual_completed_outer'] == row['outer']
        controls.append(saved)
    assert controls == binding['all_original_control_summary_references']
    last = rows[-1]
    control = read(refs['dev200_original_control_summary']['path'])
    assert last['outer'] == 200 and last['ledger_sha256'] == dev['ledger_sha256'] == control['ledger_sha256']
    assert control['metrics'] == actual['metrics']
    assert refs['dev200_original_control_summary'] == controls[-1]
    costparts = [bound(ref) for ref in binding['cumulative_cost_part_references']]
    cum = read('CUMULATIVE_COST_THROUGH200.json')
    original_parts = []
    def collect_parts(value, field):
        if isinstance(value, dict) and value.get('storage') == 'complete_original_field_parts':
            for ref in value['parts']:
                bound(ref)
                item = read(ref['file'])
                assert ref['json_selector'] == 'value' and item['original_field'] == field
                assert item['schema'] == 'cumulative_scalar_field_part_v1'
                original_parts.append(pin(ref['file'], ref['sha256'], ref['bytes']))
                collect_parts(item['value'], field)
        elif isinstance(value, dict):
            for key, item in value.items():
                collect_parts(item, field + '.' + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                collect_parts(item, field + '[' + str(i) + ']')
    for field, value in cum.items():
        collect_parts(value, field)
    assert sorted(costparts, key=lambda ref: ref['path']) == sorted(original_parts, key=lambda ref: ref['path'])
    assert len(costparts) == rootdev['actual_ref_counts']['cum_parts']
    costrefs = [bound(ref) for ref in binding['cost_outer_actual_artifact_references']]
    cost = read('COST_OUTER200_LOCAL_RECEIPT.json')
    assert costrefs == [bound(ref) for ref in cost['mirrored_artifacts'].values()]
    assert len(costrefs) == cost['mirrored_artifact_count'] == rootdev['actual_ref_counts']['cost_mirrored_artifacts']
    owner = read('DEV5_200_NATIVE_COST_LOCAL_RECEIPT.json')
    assert owner['collector_source_commit'] == ENTRY and owner['collector_source_sha256'] == COLLECTOR_SHA
    actual_chain = owner['same_entry_actual_prior_owner_chain']
    prior_count = len(actual_chain)
    assert prior_count > 0 and [entry['outer'] for entry in actual_chain[:4]] == [160,165,170,175]
    assert actual_chain[-1]['outer'] == 195
    for count, entry in enumerate(actual_chain, 1):
        historical_ref = bound(entry['receipt'])
        historical = read(historical_ref['path'])
        assert historical['outer'] == entry['outer'] and historical['collector_source_commit'] == ENTRY and historical['collector_source_sha256'] == COLLECTOR_SHA
        assert historical['new_entry_main_invocations_this_task'] == historical['actual_remote_incremental_collection_invocations'] == 1
        assert historical['remote_collector_exit_code'] == historical['preceding_local_pending_invocations'] == 0 and historical['preceding_pending_remote_operations'] is False
        if count > 1:
            assert bound(historical['same_entry_prior_actual_executor_attestation']) == bound(actual_chain[count-2]['receipt'])
            assert historical['same_entry_executor_attested_cumulative_main_invocations'] == historical['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == count
        for ref in historical['rows_part_refs'] + historical.get('prior_rows_part_refs',[]) + historical['source_CUM_all_complete_field_parts'] + historical['local_original_control_refs']:
            bound(ref)
    cumulative_count = prior_count + 1
    assert owner['same_entry_prior_confirmed_main_invocations'] == owner['same_entry_prior_confirmed_remote_incremental_invocations'] == prior_count
    assert owner['same_entry_executor_attested_cumulative_main_invocations'] == owner['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == cumulative_count
    assert binding['actual_source1ed_prior_invocations'] == publication['actual_source1ed_prior_invocations'] == prior_count
    assert binding['actual_source1ed_cumulative_invocations'] == publication['actual_source1ed_cumulative_invocations'] == cumulative_count
    actual_collection = rootdev['actual_new_collection_binding']
    assert actual_collection['same_entry_executor_attested_cumulative_main_invocations'] == actual_collection['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == cumulative_count
    assert owner['new_entry_main_invocations_this_task'] == owner['actual_remote_incremental_collection_invocations'] == 1
    assert owner['preceding_local_pending_invocations'] == owner['remote_collector_exit_code'] == 0 and owner['preceding_pending_remote_operations'] is False
    for key, value in binding['collector_invocation_attestations'].items():
        assert owner[key] == value
    assert bound(build_manifest['terminal_candidate_receipt'], TERMINAL_RECEIPT) == bound(manifest['source_pins'][TERMINAL_RECEIPT], TERMINAL_RECEIPT)
    terminal_binding = terminal_candidate_readback(manifest['source_pins'][TERMINAL_RECEIPT], full, cum, cost)
    assert binding['terminal_candidate200_supplement_binding'] == publication['terminal_candidate200_supplement_binding'] == curve['terminal_candidate200_supplement_binding'] == terminal_binding
    assert refs['terminal_candidate200_receipt'] == terminal_binding['receipt'] and refs['terminal_candidate200_summary'] == terminal_binding['summary']
    source_ref = pin(Path(__file__).resolve())
    observations = two_pass()
    result = dict(schema='root_curve_and_dev_native_cost_through200_actual_paged_readback_v2',
        status='PASS_ACTUAL_CURVE200_LOCAL_ARTIFACT_READBACK', created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        executor='/root', outer=200, scope='Actual local immutable-source and own first-FULL original-poll reconciliation after successful actual200 owner and independent ROOT_DEV200. No new remote read/evaluation/test/code review/tensor access.',
        actual_input_manifest=manifest_ref, reader_source=source_ref, references=refs, curve200=curve_ref, csv200=csv_ref,
        old_curve195=old_ref, old_csv195=old_csv_ref, old195_training_dev_objects_raw_fragments_and_actual_CSV_prefix_preserved=True, old_CSV_prefix_bytes=len(oldcsv), old_CSV_prefix_sha256=h(oldcsv),
        new196_200_bindings=new, actual_completed_dev200=dev, new_dev5_200=refs['dev_native_cost'],
        rows_storage=summary['rows'].get('storage', 'inline_list') if isinstance(summary['rows'], dict) else 'inline_list',
        actual_decoded_rows=len(rows), all_complete_rows_parts=parts, cumulative_cost_complete_field_parts=costparts,
        cost_outer_actual_artifact_references=costrefs, terminal_candidate200_supplement_binding=terminal_binding, old39_dev_control_rows_unchanged=True,
        old39_control_original_bytes_SHA_preserved=True, old195_all_row_parts_preserved=prior_parts, prior195_curve_original_file_refs=prior_curve_refs,
        actual_dev200_control=refs['dev200_original_control_summary'], root_dev200_actual_readback=pin(ROOT_DEV),
        cost_supplement_binding=pin(NAMES[3]), native=refs['native200_metadata'], actual_native_counters=last['counters'],
        observer_wall_seconds=last['wall_seconds'], trainer_process_cpu_seconds=last['trainer_process_cpu_seconds'],
        fullmetric_dev_seconds=full['normal_dev512_wall_seconds'], wall_CPU_added_again=False,
        metadata_collector_CPU_is_separate=True, precise_engine_emitted_tokens=None, scoring_call_event_total=None,
        counts=dict(training=200, actual_completed_dev512=40, csv_data_rows=240),
        same_entry_executor_attested_cumulative_main_invocations=cumulative_count, same_entry_executor_attested_cumulative_remote_incremental_invocations=cumulative_count,
        invocation_count_scope='Actual source1ed receipt chain dynamically byte-bound and counted, followed by successful actual200 main1/remote1; no future count assumed and no independent global process audit.',
        both_mirrors_actual_byte_readback=True, actual_input_readback_observations=observations,
        cold_load_actual_native_outer=5, cold_load_actual_adam_each=100, HF_permanent_milestone_outer=150,
        HF200_reload_claimed=False, ROOT_FULL200_used_as_input=False, remote_operations=False,
        actual_training201_entry_claimed=False, finalalias_required_as_input=False, slow_final51713_completion_required=False, canonical_modified=False, goal_status='active', goal_complete=False)
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) <= 100000
    destinations = [root / OUTPUT for root in [B] + M]
    assert all(not p.exists() for p in destinations)
    for p in destinations:
        with p.open('xb') as stream:
            stream.write(raw)
    publication_observations = []
    for number in range(2):
        for p in destinations:
            assert p.read_bytes() == raw
        for name, (original, _, mirrored) in INPUTS.items():
            assert Path(name).read_bytes() == original
            if mirrored:
                for mirror in M:
                    assert (mirror / Path(name).relative_to(B)).read_bytes() == original
        publication_observations.append(dict(pass_number=number + 1, observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), copies=3))
        if number == 0:
            time.sleep(1)
    print(json.dumps(dict(root_readback=dict(path=str(destinations[0]), bytes=len(raw), sha256=h(raw)), publication_readback_observations=publication_observations)))


if __name__ == '__main__':
    main()
