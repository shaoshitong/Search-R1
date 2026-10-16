#!/usr/bin/env python3
"""Future actual185 curve readback; source preparation never asserts readiness.

Adapted from root_curve180_actual_readback.py (22278B,
d8dadba4aa9cbe7111a8b8fbcc71399846e183afcd4182c02e1a3b4f96b18bd3).
Invoke only after actual185 cost, unique collector/owner, independent ROOT_DEV185
and curve185 publication have completed. ROOT_FULL185 is deliberately not read.

python3 -B work/tree-speed-repair-20261008/root_curve185_actual_readback.py \
  --pins "$ACTUAL_ROOT_CURVE185_PINS" --pins-bytes "$ACTUAL_ROOT_CURVE185_PINS_BYTES" \
  --pins-sha256 "$ACTUAL_ROOT_CURVE185_PINS_SHA256"

The future actual pin JSON requires schema=dev185_root_curve_actual_readback_input_pins_v1,
through_outer=185, source_pins mapping exactly the four curve185 products in NAMES
and ROOT_DEV185_NATIVE_COST_ACTUAL_READBACK.json to {path,bytes,sha256}.
All unknown185 bytes/SHA values must be supplied from actual readback. The curve
build manifest is itself bound by the new publication and immutable cost binding;
it must bind each original first qualifying FULL181..185 and its native/CUM.
Only ROOT_CURVE_DEV_COST_THROUGH185_ACTUAL_READBACK.json may be created here.
No SSH, collector main, training, evaluation, compile, test, review, canonical
write, cold reload, HF185 acceptance, or final200/51713 claim is performed.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import re
import time

W = Path(__file__).resolve().parents[2]
B = W / 'outputs/tree-nested-branch-handoff-20261013'
M = [W / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
     Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
NAMES = ('LEARNING_CURVE_COMPLETE_THROUGH185.json', 'LEARNING_CURVE_COMPLETE_THROUGH185.csv',
         'LEARNING_CURVE_THROUGH185_LOCAL_PUBLICATION.json', 'LEARNING_CURVE_THROUGH185_COST_SUPPLEMENT_BINDING.json')
ROOT_DEV = 'ROOT_DEV185_NATIVE_COST_ACTUAL_READBACK.json'
OUTPUT = 'ROOT_CURVE_DEV_COST_THROUGH185_ACTUAL_READBACK.json'
SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
ENTRY = '1edaf50bb8e14de51db936e0859e369792011233'
COLLECTOR_SHA = '545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0'
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
    assert manifest['schema'] == 'dev185_root_curve_actual_readback_input_pins_v1' and manifest['through_outer'] == 185
    assert set(manifest['source_pins']) == set(NAMES) | {ROOT_DEV}
    for name, ref in manifest['source_pins'].items():
        bound(ref, name)
    rootdev = read(ROOT_DEV)
    assert rootdev['status'] == 'PASS_ACTUAL_DEV185_LOCAL_ARTIFACT_READBACK' and rootdev['outer'] == 185
    assert rootdev['inconsistencies'] == [] and rootdev['two_pass_local_input_and_mirror_bytes_verified'] is True
    assert rootdev['prior36_rows_exact_object_equal'] is True and rootdev['prior36_control_original_bytes_SHA_preserved'] is True
    assert rootdev['all_original_DEV180_row_parts_exact_bytes_SHA_preserved'] is True
    for name, ref in rootdev['source_pins'].items():
        bound(ref, name)
    curve, publication, binding = read(NAMES[0]), read(NAMES[2]), read(NAMES[3])
    curve_ref, csv_ref = pin(NAMES[0]), pin(NAMES[1])
    old_ref = pin('LEARNING_CURVE_COMPLETE_THROUGH180.json', 'b0c412c7d41e134ad603e90c4457f76d2beb1bcc32e184bea7389da3ab509c48', 1092710)
    old_csv_ref = pin('LEARNING_CURVE_COMPLETE_THROUGH180.csv', '5a906b78adb56111939abc8fd3650f3b7dcfb1235f67c227ec655213637b12cc', 194906)
    old = read(old_ref['path'])
    assert curve['training_points'][:180] == old['training_points'] and curve['completed_dev512_points'][:36] == old['completed_dev512_points']
    assert len(curve['training_points']) == curve['training_point_count'] == 185
    assert len(curve['completed_dev512_points']) == curve['completed_dev512_point_count'] == 37
    assert segments(NAMES[0], 'training_points')[:180] == segments(old_ref['path'], 'training_points')
    assert segments(NAMES[0], 'completed_dev512_points')[:36] == segments(old_ref['path'], 'completed_dev512_points')
    oldcsv, newcsv = INPUTS[str(locate(old_csv_ref['path']))][0], INPUTS[str(locate(NAMES[1]))][0]
    assert newcsv.startswith(oldcsv) and len(list(csv.DictReader(io.StringIO(newcsv.decode('utf8'))))) == 222
    assert binding['through_outer'] == 185 and binding['counts'] == publication['counts'] == dict(training=185, actual_completed_dev512=37, csv_data_rows=222)
    assert binding['actual_remote_incremental_collection_invocations'] == binding['new_entry_main_invocations_this_task_by_collector_executor'] == 1
    assert binding['preceding_local_pending_invocations'] == 0 and binding['preceding_pending_remote_operations'] is False
    assert binding['ROOT_FULL185_used_as_input'] is False and publication['ROOT_FULL185_used_as_input'] is False
    assert binding['original_wall_CPU_double_added'] is False and binding['supplemental_totals_added_to_curve_or_CUM'] is False
    assert binding['HF150_boundary_advanced'] is False and publication['HF150_boundary_advanced'] is False
    refs = {key: bound(ref) for key, ref in binding['references'].items()}
    assert refs['curve'] == curve_ref and refs['csv'] == csv_ref and refs['curve_publication'] == pin(NAMES[2])
    assert refs['root_dev_native_cost_readback'] == pin(ROOT_DEV)
    assert refs['full185'] == pin('FULL_OUTER185_PROGRESS_RECEIPT.json')
    build_manifest_ref = bound(binding['build_input_manifest'])
    assert publication['build_input_manifest'] == build_manifest_ref
    build_manifest = read(build_manifest_ref['path'])
    assert build_manifest['schema'] == 'dev185_actual_curve_build_input_pins_v1' and build_manifest['through_outer'] == 185
    required = {ROOT_DEV} | {f'FULL_OUTER{o}_PROGRESS_RECEIPT.json' for o in range(181, 186)} | {f'NATIVE_HISTORY_{o}_METADATA_READBACK.json' for o in range(181, 186)} | {f'CUMULATIVE_COST_THROUGH{o}.json' for o in range(181, 186)}
    assert set(build_manifest['source_pins']) == required
    for name, ref in build_manifest['source_pins'].items():
        bound(ref, name)
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
    for outer in range(181, 186):
        point = next(x for x in curve['training_points'] if x['outer'] == outer)
        full_name = f'FULL_OUTER{outer}_PROGRESS_RECEIPT.json'
        full = read(full_name)
        full_ref = pin(full_name, point['source_receipt_sha256'])
        assert full['completed_outer'] == point['outer'] == outer and full['source_commit'] == point['source_commit'] == SOURCE
        assert full['config_sha256'] == point['config_sha256'] == CONFIG
        assert full['actual_training_pid'] == 503255 and full['actual_training_invocation_id'] == 'a313bda919834d24b4c5d491698d39ce'
        poll_ref = bound(full['input_bindings']['poll'])
        poll_path = locate(poll_ref['path'])
        assert poll_path.parent == B and poll_path.name.startswith('NATIVE5_CONTINUE_POLL_')
        assert locate(point['source_poll']) == locate(full['source_poll']) == poll_path
        assert point['source_poll_sha256'] == full['source_poll_sha256'] == poll_ref['sha256']
        poll = read(poll_path)
        assert re.search(r'epoch\s+\d+,\s*step\s+'+str(outer+1)+r'\b',(poll.get('logs') or {}).get('stage.log','')), 'Original actual next-stage marker required; saved cursor alone is not entry'
        metrics = [x for x in poll['iteration_metrics'] if int(x['step']) == outer]
        assert len(metrics) == 1 and metrics[0] == full['iteration_metrics'] and not poll['progress'].get('errors')
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
        native_ref = bound(full['input_bindings']['native_readback'], f'NATIVE_HISTORY_{outer}_METADATA_READBACK.json')
        native = read(native_ref['path'])
        assert native['outer'] == outer and native['archive_matches_original_manifest'] is True
        if outer < 185:
            assert full['actual_dev512'] is None and point['current_outer_completed_dev512'] is None
        new.append(dict(outer=outer, full=full_ref, poll=poll_ref, native_metadata=native_ref,
                        cumulative_cost=point['cumulative_cost_source'], original_first_FULL_input_binding_used=True))
    full = read('FULL_OUTER185_PROGRESS_RECEIPT.json')
    actual = full['actual_dev512']
    dev = next(x for x in curve['completed_dev512_points'] if x['outer'] == 185)
    assert dev['rows'] == actual['rows'] == 512 and actual['metrics']['val/qa_count/all'] == 512
    assert actual['metrics']['val/qa_count/nq'] == actual['metrics']['val/qa_count/hotpotqa'] == 256
    assert actual['status'] == dev['status'] == 'VALIDATION_COST_COMPLETE'
    assert dev['ledger_sha256'] == actual['ledger_sha256'] and dev['ledger_path'] == actual['path']
    assert dev['f1'] == actual['metrics']['val/f1/all'] and dev['em'] == actual['metrics']['val/em/all']
    assert dev['nq_f1'] == actual['metrics']['val/f1/nq'] and dev['hotpotqa_f1'] == actual['metrics']['val/f1/hotpotqa']
    sp = W / 'work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py'
    pin(sp, COLLECTOR_SHA, 27911)
    spec = importlib.util.spec_from_file_location('root_curve185_published_decoder_only', sp)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    summary = read('DEV5_185_NATIVE_COST_SUPPLEMENT.json')
    prior_ref = pin('DEV5_180_NATIVE_COST_SUPPLEMENT.json', 'ec65e3de925f37684c66eed27d916d88dcaa287d2303fee11ee70fe3756f65f9', 7096)
    prior_summary = read(prior_ref['path'])
    rows, oldrows = decoder.rows_from_summary(B, M, summary), decoder.rows_from_summary(B, M, prior_summary)
    assert len(rows) == 37 and len(oldrows) == 36 and rows[:-1] == oldrows
    parts = [bound(ref) for ref in binding['dev_native_cost_rows_part_references']]
    prior_parts = [pin(ref['file'], ref['sha256'], ref['bytes']) for ref in prior_summary['rows']['parts']]
    prior_owner_ref = pin('DEV5_180_NATIVE_COST_LOCAL_RECEIPT.json', 'f780787865ff843a27a904fc9a096da2d715393d8095dc93a8c4782f173aa5d8', 49992)
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
    control = read(refs['dev185_original_control_summary']['path'])
    assert last['outer'] == 185 and last['ledger_sha256'] == dev['ledger_sha256'] == control['ledger_sha256']
    assert control['metrics'] == actual['metrics']
    assert refs['dev185_original_control_summary'] == controls[-1]
    costparts = [bound(ref) for ref in binding['cumulative_cost_part_references']]
    cum = read('CUMULATIVE_COST_THROUGH185.json')
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
    cost = read('COST_OUTER185_LOCAL_RECEIPT.json')
    assert costrefs == [bound(ref) for ref in cost['mirrored_artifacts'].values()]
    assert len(costrefs) == cost['mirrored_artifact_count'] == rootdev['actual_ref_counts']['cost_mirrored_artifacts']
    owner = read('DEV5_185_NATIVE_COST_LOCAL_RECEIPT.json')
    assert owner['collector_source_commit'] == ENTRY and owner['collector_source_sha256'] == COLLECTOR_SHA
    assert owner['same_entry_prior_confirmed_main_invocations'] == owner['same_entry_prior_confirmed_remote_incremental_invocations'] == 5
    assert owner['same_entry_executor_attested_cumulative_main_invocations'] == owner['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == 6
    assert owner['new_entry_main_invocations_this_task'] == owner['actual_remote_incremental_collection_invocations'] == 1
    assert owner['preceding_local_pending_invocations'] == owner['remote_collector_exit_code'] == 0 and owner['preceding_pending_remote_operations'] is False
    for key, value in binding['collector_invocation_attestations'].items():
        assert owner[key] == value
    source_ref = pin(Path(__file__).resolve())
    observations = two_pass()
    result = dict(schema='root_curve_and_dev_native_cost_through185_actual_paged_readback_v2',
        status='PASS_ACTUAL_CURVE185_LOCAL_ARTIFACT_READBACK', created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        executor='/root', outer=185, scope='Actual local immutable-source and own first-FULL original-poll reconciliation after successful actual185 owner and independent ROOT_DEV185. No new remote read/evaluation/test/code review/tensor access.',
        actual_input_manifest=manifest_ref, reader_source=source_ref, references=refs, curve185=curve_ref, csv185=csv_ref,
        old_curve180=old_ref, old_csv180=old_csv_ref, old180_training_dev_objects_raw_fragments_and_CSV194906_prefix_preserved=True,
        new181_185_bindings=new, actual_completed_dev185=dev, new_dev5_185=refs['dev_native_cost'],
        rows_storage=summary['rows'].get('storage', 'inline_list') if isinstance(summary['rows'], dict) else 'inline_list',
        actual_decoded_rows=len(rows), all_complete_rows_parts=parts, cumulative_cost_complete_field_parts=costparts,
        cost_outer_actual_artifact_references=costrefs, old36_dev_control_rows_unchanged=True,
        old36_control_original_bytes_SHA_preserved=True, old180_all_row_parts_preserved=prior_parts,
        actual_dev185_control=refs['dev185_original_control_summary'], root_dev185_actual_readback=pin(ROOT_DEV),
        cost_supplement_binding=pin(NAMES[3]), native=refs['native185_metadata'], actual_native_counters=last['counters'],
        observer_wall_seconds=last['wall_seconds'], trainer_process_cpu_seconds=last['trainer_process_cpu_seconds'],
        fullmetric_dev_seconds=full['normal_dev512_wall_seconds'], wall_CPU_added_again=False,
        metadata_collector_CPU_is_separate=True, precise_engine_emitted_tokens=None, scoring_call_event_total=None,
        counts=dict(training=185, actual_completed_dev512=37, csv_data_rows=222),
        same_entry_executor_attested_cumulative_main_invocations=6, same_entry_executor_attested_cumulative_remote_incremental_invocations=6,
        invocation_count_scope='Source1ed SHA-bound executor attestations160+165+170+175+180+185, corroborated by artifacts; no independent global process audit.',
        both_mirrors_actual_byte_readback=True, actual_input_readback_observations=observations,
        cold_load_actual_native_outer=5, cold_load_actual_adam_each=100, HF_permanent_milestone_outer=150,
        HF185_reload_claimed=False, ROOT_FULL185_used_as_input=False, remote_operations=False,
        canonical_modified=False, goal_status='active', goal_complete=False)
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
