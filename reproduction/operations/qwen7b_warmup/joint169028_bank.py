"""Independent five-bank metadata join after full same-HF G10 acceptance.

Never samples, loads tensors, modifies124194, or submits GPU work. Every new
bank must already have producer full-payload/retained-row/physical-source
acceptance. This join freshly rechecks metadata, indices and source sidecars;
the eventual training reader again hashes every packed payload.
"""
import argparse
import fcntl
import gzip
import json
from pathlib import Path
import time

from . import replay_probe_layers as r
from . import joint124194_bank as old

A = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
TARGET = A / 'joint169028-native-training-bank-v1'
COUNTS = dict(current124194=124194, HPC3_first9344=9344, donor_only14080=14080,
              HPC3_next14592=14592, HPC3_final6818=6818)
QID_SHA = 'aa14bce45fc6a793789369d6f33c84edf3c1d79e601fec45882ec193b1e4c58d'
STATUS = 'FROZEN_FIVE_BANK_REAL_G10_169028_NATIVE_JOINT_INPUTS_ACCEPTED'
NEW_FLAGS = ('physical_execution_prefix_and_retained_row_SHA_binding_complete',
             'producer_full_payload_SHA_preserved', 'metadata_join_rehashes_tensor_payload',
             'all_retained_rows_reconstructed_SHA_and_finite_verified')


def checked(ref):
    return r.read(r.checked_file(ref)['path'])


def validate_link(bound, chosen, feature):
    """Identity join is required: source projection and capture orders can differ."""
    if (any(bound.get(k) != v for k, v in chosen.items())
            or bound['trajectory_id'] != chosen['trajectory_id']
            or bound['qid'] != feature['qid']
            or bound['prefix_body_sha256'] != feature['prefix_sha256']
            or bound['feature_retained_row_sha256'] != feature['retained_sha256']
            or bound['feature_manifest'] != feature['manifest']
            or bound['feature_index'] != feature['index']
            or not old.hexadecimal(bound['canonical_prefix_row_sha256'])):
        raise ValueError('Chosen physical execution/canonical row/retained feature link differs')


def gzip_rows(ref):
    with gzip.open(r.checked_file(ref)['path'], 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def verify_source_links(done, groups, manifests):
    expected = {t: g['qid'] for g in groups for t in g['trajectory_ids']}
    prep = checked(done['preparation'])
    raw = checked(done['raw_complete'])
    raw_ref = raw['outputs']['groups']
    raw_by_qid = {g['qid']: g for g in checked(raw_ref)['train']}
    for g in groups:
        source = raw_by_qid.get(g['qid'])
        if source is None or any(source[k] != g[k] for k in ('question_hash', 'trajectory_ids', 'labels')):
            raise ValueError('Feature bank changed the selected whole own-G10 source truth')
    feature_rows = {}
    for ref in manifests:
        manifest = checked(ref)
        index = manifest['files']['index.jsonl']
        with Path(r.checked_file(index)['path']).open() as stream:
            for line in stream:
                row = json.loads(line)
                tid = row['trajectory_id']
                if tid in feature_rows or expected.get(tid) != row['qid']:
                    raise ValueError('Missing/repeated/foreign own feature in source-bound join')
                feature_rows[tid] = dict(qid=row['qid'], prefix_sha256=row['source_prefix_sha256'],
                    retained_sha256=row['retained_feature_sha256'], manifest=ref, index=index)
    side_ref = done['outputs']['source-bound-retained-rows.json']
    sides = checked(side_ref)
    if len(sides) != len(prep['selections']) or len(sides) != len(done['shard_completes']) or len(sides) != 4:
        raise ValueError('All four physical-source shard proofs required')
    seen, executions, physical = set(), {}, {}
    for selection_ref, side, cap_ref in zip(prep['selections'], sides, done['shard_completes']):
        selection, cap = checked(selection_ref), checked(cap_ref)
        if cap['source_selection'] != selection_ref:
            raise ValueError('Capture does not bind the selected physical input')
        wanted = {v['trajectory_id']: v['qid'] for v in selection['trajectories']}
        chosen = {}
        for row in gzip_rows(selection['chosen_physical_prefix_bindings']):
            tid = row['trajectory_id']
            if tid in chosen:
                raise ValueError('Repeated chosen physical execution identity')
            chosen[tid] = row
        if set(chosen) != set(wanted):
            raise ValueError('Selected physical trajectory source set differs')
        shard_seen = set()
        for row in gzip_rows(side):
            tid = row['trajectory_id']
            if tid in seen or tid in shard_seen or wanted.get(tid) != row['qid'] or tid not in feature_rows:
                raise ValueError('Source sidecar repeats or substitutes a trajectory')
            validate_link(row, chosen[tid], feature_rows[tid])
            for key in ('physical_prefix', 'physical_seal'):
                ref = row[key]
                if ref['path'] in physical and physical[ref['path']] != ref:
                    raise ValueError('One physical source path has different sealed bytes')
                physical[ref['path']] = ref
            kind = row['chosen_execution']
            executions[kind] = executions.get(kind, 0) + 1
            shard_seen.add(tid)
        if shard_seen != set(wanted) or len(shard_seen) != cap['trajectories']:
            raise ValueError('Actual complete shard lost selected own trajectories')
        seen.update(shard_seen)
    if seen != set(expected) or set(feature_rows) != set(expected) or executions.get('original_canonical_old5') != len(groups) * 5:
        raise ValueError('Full own-G10 old5/new5 physical-source coverage differs')
    for ref in physical.values():
        if Path(ref['path']).stat().st_size != ref['bytes']:
            raise ValueError('Producer-verified original physical file changed size')
    r.checked_file(done['source_resolution_contract'])
    return dict(own_rows=len(seen), execution_counts=executions, physical_files=len(physical),
                metadata_sidecars_full_SHA_rechecked=True, all_physical_retained_links_rechecked=True,
                large_original_files_rehashed_in_this_metadata_join=False)


def run(config_path, config_sha):
    config_ref = r.checked_file(dict(path=str(config_path), bytes=Path(config_path).stat().st_size, sha256=config_sha))
    c = r.read(config_ref['path'])
    if (c['target'] != str(TARGET) or c['role'] != 'full169028_sameHF_source_resolved_native_joint'
            or c['questions'] != 169028 or c['canonical_qid_sha256'] != QID_SHA
            or [v['key'] for v in c['banks']] != list(COUNTS)
            or {v['key']: v['questions'] for v in c['banks']} != COUNTS):
        raise ValueError('Exact independent full169028 cohort configuration required')
    missing = [v['complete']['path'] for v in c['banks'] if not Path(v['complete']['path']).exists()]
    if missing:
        return dict(status='WAIT_FOR_ACTUAL_FULL169028_FEATURE_BANKS', missing=missing, GPU_submitted=False)
    TARGET.mkdir(exist_ok=True)
    with (TARGET / 'join.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (TARGET / 'complete.json').exists():
            done = r.read(TARGET / 'complete.json')
            if done['config'] != config_ref or done['status'] != STATUS:
                raise ValueError('Existing full-data join configuration differs')
            for ref in done['outputs'].values():
                r.checked_file(ref)
            return done
        exclusion = checked(c['evaluation_exclusions'])
        if exclusion['role'] != 'joint_training_fixed_evaluation_identity_exclusions' or len(exclusion['groups']) != 6656:
            raise ValueError('True512/2048/4096 identity-only exclusions required')
        # Recheck the canonical source fingerprint without loading another full copy.
        r.checked_file(c['canonical_groups'])
        banks, refs, lineage = {}, [], []
        for item in c['banks']:
            done = checked(item['complete'])
            n = COUNTS[item['key']]
            if done['questions'] != n or done['trajectories'] != n * 10 or done['actor_updates'] != 0:
                raise ValueError('Full source bank count/actor identity differs')
            if item['key'] == 'current124194':
                if item['complete']['sha256'] != 'a8fe2c2ea823087d50ac86b283eb361b51ae7ba5ba0571860193214338200801' or done['status'] != old.STATUS:
                    raise ValueError('Actual original124194 bank is immutable')
            else:
                expected_status = ('DONOR14080_OWN_G10_LAYER26_SOURCE_RESOLVED_FEATURE_BANK_ACCEPTED'
                    if item['key'] == 'donor_only14080' else
                    'FIXED_HPC3_TAIL%d_OWN_G10_LAYER26_SOURCE_RESOLVED_FEATURE_BANK_ACCEPTED' % n)
                if (done['status'] != expected_status or done['capture_layer'] != 26
                        or done['prefix_tokens'] != 20 or done['question_max_tokens'] != 273
                        or done['model_revision'] != r.REVISION or any(done.get(k) is not True for k in NEW_FLAGS)
                        or done['probe_updates'] != 0 or done['new_responses_generated'] != 0):
                    raise ValueError('Actual full source-resolved sameHF L26/P20 producer acceptance required')
                for key in ('qualified_runtime_publication', 'preparation', 'raw_complete', 'source_resolution_contract'):
                    r.checked_file(done[key])
            group_ref, feature_ref = done['outputs']['groups.json'], done['outputs']['features-layer26.json']
            groups, features = checked(group_ref)['train'], checked(feature_ref)
            proof = verify_source_links(done, groups, features) if item['key'] != 'current124194' else dict(original_qualified_bank_preserved=True)
            banks[item['key']] = groups
            refs.extend(features)
            lineage.append(dict(key=item['key'], complete=item['complete'], groups=group_ref,
                features=feature_ref, physical_source_identity_audit=proof))
            r.atomic(TARGET / 'progress.json', dict(stage='ACTUAL_FULL169028_SOURCE_BANKS_ADVANCING',
                banks=len(lineage), questions=sum(len(v) for v in banks.values()), epoch=time.time()))
        groups, summary = old.merge_groups(banks, exclusion['groups'], expected_counts=COUNTS)
        if len(groups) != 169028 or old.digest(sorted(g['qid'] for g in groups)) != QID_SHA:
            raise ValueError('Merged actual canonical169028 ID set differs')
        features, feature_proof = old.verify_features(refs, groups,
            lambda value: r.atomic(TARGET / 'progress.json', value))
        old.immutable(TARGET / 'groups.json', dict(train=groups))
        old.immutable(TARGET / 'features-layer26.json', features)
        old.immutable(TARGET / 'source-refs.json', lineage)
        outputs = {name: r.fingerprints(TARGET / name) for name in ('groups.json', 'features-layer26.json', 'source-refs.json')}
        done = dict(status=STATUS, config=config_ref, outputs=outputs, **summary, **feature_proof,
            groups=outputs['groups.json'], features_layer26=outputs['features-layer26.json'],
            canonical_groups=c['canonical_groups'], canonical_qid_sha256=QID_SHA,
            evaluation_exclusions=c['evaluation_exclusions'], source_refs=lineage,
            whole_G5_source_selection_before_packed_union=True, current124194_fit_expanded=False,
            actor_updates=0, probe_updates=0, GPU_submitted=False, final_outcomes_read=False, epoch=time.time())
        old.immutable(TARGET / 'complete.json', done)
        return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.config_sha256), sort_keys=True))
