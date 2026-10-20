"""Fresh verify the extra REAL G5 event, then join old5+new5 without relabeling.

Features stay in their independent original/extra G5 sources; the G10 manifest
only binds ten actual trajectory IDs and their own stored binary strict EM.
"""
import argparse
import gzip
import json
from pathlib import Path
import time

from .extra_g5_prepare import ROLE, REVISION, atomic, canonical_sha, file_proof, read, remote, sha, validate_config


def checked(item):
    path = Path(item['path'])
    if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
        raise ValueError('Immutable generation payload SHA/size differs: '+str(path))
    return path


def verify_extra_shard(base, shard):
    base = remote(base)
    c = validate_config(read(base/'config.json'))
    own_root = base/'sampling'/('shard-%02d' % shard)
    generic_path = own_root/'complete.json'
    generic = read(generic_path)
    if generic['run_id'] != c['run_id'] or generic['model_revision'] != REVISION:
        raise ValueError('Generic G5 collector completion belongs to a different phase')
    qplan = read(base/'inputs/questions.json')
    expected = {r['qid']:r for r in qplan['groups'][shard::4]}
    accepted, seen_tids, features, batch_proofs = {}, set(), [], []
    for raw_path in generic['batches']:
        path = Path(raw_path)
        if not str(path.resolve()).startswith(str(own_root.resolve())+'/batches/'):
            raise ValueError('Batch receipt escapes this extra event/shard')
        batch = read(path)
        input_path = base/'inputs'/('shard-%02d' % shard)/(batch['batch']+'.json')
        source = read(input_path)
        if batch['input_sha256'] != sha(input_path) or batch['run_id'] != c['run_id'] or batch['actor_optimizer_updates'] != 0:
            raise ValueError('Extra batch input/run/actor-update binding differs')
        inputs = {r['qid']:r for r in source['rows']}
        prefix, full = batch['prefix'], batch['full']
        for phase, keys in ((prefix, ('states', 'prefix', 'actor_hidden')), (full, ('full', 'own_em'))):
            if phase['run_id'] != c['run_id'] or phase['actor_version'] != 'official7b-'+REVISION:
                raise ValueError('Extra phase actor/run identity differs')
            if phase.get('sampling_attempt_id') != c['sampling_attempt_id']:
                raise ValueError('Extra phase lost its independent native seed namespace')
            if {k:phase[k] for k in c['original_protocol_contract']} != c['original_protocol_contract']:
                raise ValueError('Extra decoder/interaction/dtype contract differs from original actor rollouts')
            for key in keys:
                checked(phase[key])
        if prefix['sampling_contract'] != full['sampling_contract'] or prefix['interaction_protocol'] != full['interaction_protocol']:
            raise ValueError('Extra prefix and full continuation protocols differ')
        feature = read(prefix['actor_hidden']['path'])
        if feature['status'] != 'LOSSLESS_NATIVE_FEATURES_COMPLETE' or feature['feature_layer'] != 26 or feature['prefix_tokens'] != 20:
            raise ValueError('Extra source is not an independently completed nativeG5 feature event')
        for item in feature['files'].values():
            checked(item)
        features.append(prefix['actor_hidden'])
        prefix_rows, em_rows, full_rows = {}, {}, {}
        with gzip.open(prefix['prefix']['path'], 'rt') as f:
            for line in f:
                row = json.loads(line)
                tid = row['trajectory_id']
                if tid in prefix_rows:
                    raise ValueError('Repeated extra prefix trajectory')
                prefix_rows[tid] = row
        with Path(full['own_em']['path']).open() as f:
            for line in f:
                row = json.loads(line)
                tid, label = row['trajectory_id'], row['strict_em']
                if tid in em_rows or type(label) is not int or label not in (0, 1):
                    raise ValueError('Duplicate or invalid extra own-EM')
                em_rows[tid] = row
        with gzip.open(full['full']['path'], 'rt') as f:
            for line in f:
                row = json.loads(line)
                tid = row['trajectory_id']
                if tid in full_rows:
                    raise ValueError('Repeated extra complete trajectory')
                if not row['behavior_turns'] or not row['sampling_journal']:
                    raise ValueError('Extra answer lacks actual native generation/retrieval journal')
                if len(row['response_token_ids']) != len(row['response_info_mask']):
                    raise ValueError('Extra complete response/token-mask length differs')
                full_rows[tid] = row
        if set(prefix_rows) != set(em_rows) or set(prefix_rows) != set(full_rows):
            raise ValueError('Extra prefix/full/own-EM trajectory sets differ')
        grouped = {}
        for tid, row in prefix_rows.items():
            qid, response_index = row['qid'], row['response_index']
            if tid in seen_tids or qid not in inputs or qid not in expected or tid in expected[qid]['trajectory_ids']:
                raise ValueError('Unexpected/reused original or extra trajectory identity')
            seen_tids.add(tid)
            full_row, em = full_rows[tid], em_rows[tid]
            if row['actor_version'] != 'official7b-'+REVISION or full_row['policy_snapshot_id'] != row['actor_version']:
                raise ValueError('Extra native features and answer belong to different actors')
            if response_index not in c['extra_response_indices'] or full_row['response_index'] != response_index:
                raise ValueError('Extra response indices differ from the declared independentG5 event')
            if row['split'] != 'train' or full_row['split'] != 'train' or full_row['qid'] != qid or em['qid'] != qid:
                raise ValueError('Validation/final or another question entered extra training event')
            if row['question_hash'] != inputs[qid]['question_hash'] or full_row['question_hash'] != inputs[qid]['question_hash']:
                raise ValueError('Extra question identity changed')
            if row['initial_prompt_token_ids'] != inputs[qid]['initial_prompt_token_ids'] or full_row['initial_prompt_token_ids'] != row['initial_prompt_token_ids']:
                raise ValueError('Extra prompt differs from original actual official prompt tokens')
            if full_row['prefix_sha256'] != row['prefix_sha256'] or em['prefix_sha256'] != row['prefix_sha256']:
                raise ValueError('Extra continuation does not belong to the retained native prefix')
            if full_row['strict_em'] != em['strict_em']:
                raise ValueError('Extra own EM differs from its actual full trajectory label')
            from reproduction.prefix_selection200.rollout import seed_for
            old_seeds = set(qplan['original_native_seeds_by_qid'][qid])
            for chunk in full_row['sampling_journal']:
                expected_seed = seed_for(c['run_id'], 1, tid, chunk['logical_turn_index'],
                    chunk['native_chunk_index'], c['seed'], sampling_attempt_id=c['sampling_attempt_id'])
                if chunk['seed'] != expected_seed or chunk['sampling']['seed'] != expected_seed:
                    raise ValueError('Extra actual registered native seed differs from new event derivation')
                if expected_seed in old_seeds:
                    raise ValueError('Actual extra seed collides with an original same-question seed; preserve outputs and use a new attempt')
            grouped.setdefault(qid, []).append((response_index, tid, em['strict_em']))
        if set(grouped) != set(inputs):
            raise ValueError('Incomplete extra question coverage in batch')
        for qid, values in grouped.items():
            values.sort()
            if [v[0] for v in values] != c['extra_response_indices'] or qid in accepted:
                raise ValueError('Incomplete or repeated extra G5 question')
            accepted[qid] = dict(qid=qid, question_hash=inputs[qid]['question_hash'],
                trajectory_ids=[v[1] for v in values], labels=[v[2] for v in values],
                response_indices=c['extra_response_indices'])
        batch_proofs.append(file_proof(path))
    if set(accepted) != set(expected) or len(seen_tids) != 5*len(expected):
        raise ValueError('Incomplete extra shard question/trajectory coverage')
    return dict(status='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED', role=ROLE,
        shard=shard, run_id=c['run_id'], model_revision=REVISION, sampling_attempt_id=c['sampling_attempt_id'],
        questions=len(accepted), trajectories=len(seen_tids), groups=[accepted[qid] for qid in expected],
        generic_collector_complete=file_proof(generic_path), batches=batch_proofs,
        independent_G5_feature_sources=features, actor_updates=0, original_G5_modified=False,
        synthetic_EM=False, time=time.time())


def merge(base):
    base = remote(base)
    c = validate_config(read(base/'config.json'))
    destination = base/'bank/complete.json'
    if destination.exists():
        raise ValueError('G10 bank already accepted; verify it instead of overwriting')
    for key in ('original_bank', 'original_bank_complete', 'original_config', 'final_plan'):
        checked(c[key])
    original = read(c['original_bank']['path'])
    by_qid = {g['qid']:g for g in original['train']}
    plan = read(base/'inputs/questions.json')
    for item in plan['original_batches']+plan['original_input_batches']:
        checked(item)
    qids = [g['qid'] for g in plan['groups']]
    if canonical_sha(qids) != plan['ordered_qid_sha256']:
        raise ValueError('Pilot ordered question plan changed')
    extra, shard_proofs, feature_sources = {}, [], []
    for shard in range(4):
        result = verify_extra_shard(base, shard)
        receipt_path = base/'sampling'/('shard-%02d' % shard)/'extra-g5-complete.json'
        if not receipt_path.exists():
            raise ValueError('Sampler has not published its own extra event completion')
        old = read(receipt_path)
        if old['groups'] != result['groups'] or old['run_id'] != c['run_id']:
            raise ValueError('Published extra event completion differs from fresh payload audit')
        shard_proofs.append(file_proof(receipt_path))
        feature_sources.extend(result['independent_G5_feature_sources'])
        for group in result['groups']:
            if group['qid'] in extra:
                raise ValueError('Extra question repeats between shards')
            extra[group['qid']] = group
    if set(extra) != set(qids):
        raise ValueError('Not every declared pilot question has its new real G5')
    g10, seen = [], set()
    for qid in qids:
        old, new = by_qid[qid], extra[qid]
        declared = next(g for g in plan['groups'] if g['qid'] == qid)
        if old != declared or old['question_hash'] != new['question_hash']:
            raise ValueError('Original G5 was altered or extra question binding differs')
        tids, labels = old['trajectory_ids']+new['trajectory_ids'], old['labels']+new['labels']
        if len(tids) != 10 or len(set(tids)) != 10 or seen.intersection(tids):
            raise ValueError('Joined G10 has missing/repeated old/new trajectory IDs')
        if len(labels) != 10 or any(type(y) is not int or y not in (0, 1) for y in labels):
            raise ValueError('Joined G10 has invalid own binary labels')
        seen.update(tids)
        g10.append(dict(qid=qid, question_hash=old['question_hash'], trajectory_ids=tids, labels=labels,
            original_fixed5_trajectory_ids=old['trajectory_ids'], original_fixed5_labels=old['labels'],
            extra_response_indices=c['extra_response_indices'], original_generation_run_id=c['original_run_id'],
            extra_generation_run_id=c['run_id'], extra_sampling_attempt_id=c['sampling_attempt_id']))
    atomic(base/'bank/groups.json', dict(train=g10))
    # Keep the two events independent; existing G5 packing lifecycle is valid.
    atomic(base/'bank/extra-features.json', feature_sources)
    result = dict(status='PILOT_ACTUAL_ORIGINAL5_PLUS_EXTRA5_G10_SHA_OWN_EM_VERIFIED', role=ROLE,
        questions=len(g10), trajectories=10*len(g10), extra_real_trajectories=5*len(g10),
        ordered_qid_sha256=canonical_sha(qids), groups=file_proof(base/'bank/groups.json'),
        original_G5_bank=c['original_bank'], extra_G5_feature_sources=file_proof(base/'bank/extra-features.json'),
        original_feature_manifest=file_proof(Path(c['original_base'])/'bank/features.json'),
        extra_shards=shard_proofs, model_revision=REVISION, actor_updates=0,
        validation_or_final_questions_generated=False, synthetic_EM=False,
        original_fixed5_preserved=True, time=time.time())
    atomic(destination, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    a = p.parse_args()
    print(json.dumps(merge(a.base)), flush=True)
