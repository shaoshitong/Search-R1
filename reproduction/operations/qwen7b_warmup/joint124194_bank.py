"""Freeze the four accepted real-G10 banks for native joint pretraining.

Metadata/own-ID join only. Packed tensors remain remote and retain producer
qualifications; this join freshly checks manifest/index SHA and payload sizes,
not a second bulk tensor hash. Training's reader verifies retained row SHA.
No selection on labels, new responses, model loading or GPU submission.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import time

from . import replay_probe_layers as r
from .g10_grouping import _validate
from .replay_feature_reader import ReplayFeatureReader, hexadecimal

STATUS = 'FROZEN_FOUR_BANK_REAL_G10_124194_NATIVE_JOINT_INPUTS_ACCEPTED'
A = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
TARGET = A / 'joint124194-native-training-bank-v1'
COUNTS = dict(existing58432=58432, recovered1216=1216, audited243=31010, audited262=33536)
SUBSET_STATUS = 'JOINT_TRAIN_SUBSET_REAL_G10_LAYER26_FEATURE_BANK_ACCEPTED'
RAW_STATUS = 'RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'


def immutable(path, value):
    if path.exists():
        if r.read(path) != value:
            raise ValueError('Conflicting immutable metadata join artifact: ' + str(path))
    else:
        r.atomic(path, value)


def digest(value):
    return hashlib.sha256(r.canonical(value)).hexdigest()


def merge_groups(banks, exclusions, expected_counts=COUNTS, expected_domains=None,
                 expected_text_repeats=None):
    """Authentic own-label merge; canonical repeated training texts are retained."""
    if set(banks) != set(expected_counts):
        raise ValueError('Exactly the frozen four bank roles required')
    blocked = {g['qid'] for g in exclusions}
    hashes = {g['question_hash'] for g in exclusions}
    qids, tids, seen_hashes, output = set(), set(), set(), []
    domains, k_counts = Counter(), Counter()
    for key in expected_counts:
        groups = banks[key]
        _validate(groups, True)
        if len(groups) != expected_counts[key]:
            raise ValueError('Actual whole source cohort count differs: ' + key)
        for group in groups:
            q, h, ids = group['qid'], group['question_hash'], group['trajectory_ids']
            if not hexadecimal(h) or q in blocked or h in hashes:
                raise ValueError('Invalid hash or true external evaluation identity in fit')
            if q in qids or tids.intersection(ids):
                raise ValueError('Question or trajectory identity reused across banks')
            domain = json.loads(q)[0]
            if domain not in ('nq', 'hotpotqa'):
                raise ValueError('Only the canonical training domains are permitted')
            own = {name: group[name] for name in ('qid', 'question_hash', 'trajectory_ids', 'labels')}
            output.append(own)
            qids.add(q); tids.update(ids); seen_hashes.add(h)
            domains[domain] += 1; k_counts[sum(group['labels'])] += 1
    repeats = len(output) - len(seen_hashes)
    if expected_domains is not None and dict(domains) != expected_domains:
        raise ValueError('Frozen canonical domain coverage differs')
    if expected_text_repeats is not None and repeats != expected_text_repeats:
        raise ValueError('Frozen train-only text repetition count differs')
    output.sort(key=lambda g: g['qid'])
    identity = [[g['qid'], g['question_hash'], g['trajectory_ids']] for g in output]
    return output, dict(questions=len(output), trajectories=len(tids), domains=dict(domains),
        canonical_train_repeated_text_questions=repeats, distinct_question_hashes=len(seen_hashes),
        K10_histogram={str(k): k_counts[k] for k in range(11)},
        Mixed_capable_questions=sum(n for k, n in k_counts.items() if 0 < k < 10),
        canonical_ordered_qid_hash_own_tids_sha256=digest(identity),
        ordered_qid_sha256=digest([g['qid'] for g in output]),
        own_EM_labels_preserved=True, label_based_identity_selection=False,
        external_evaluation_qid_or_hash_overlap=0, former_internal5120_is_training=True,
        original_warmup_unseen_question_claim=False)


def subset_lineage(item, done):
    prep = r.read(r.checked_file(done['preparation'])['path'])
    config = r.read(r.checked_file(prep['config'])['path'])
    identity = r.read(r.checked_file(done['identity_plan'])['path'])
    accepted = r.read(r.checked_file(done['raw_complete'])['path'])
    if (done['raw_complete'] != prep['raw_complete'] or config['raw_complete'] != done['raw_complete'] or
            accepted['status'] != RAW_STATUS or accepted['strict_EM_mismatches'] or accepted['actor_updates'] or
            not done['all_own_IDs_strictEM_and_metadata_SHA_verified'] or
            identity['summary']['labels_consulted'] or identity['summary']['scores_consulted']):
        raise ValueError('Actual ownEM and label-blind subset lineage required')
    if item.get('receiver_snapshot'):
        receiver = Path(accepted['receiver_complete']['path']).parent
        if receiver.name != item['receiver_snapshot']:
            raise ValueError('Different later source snapshot cannot drift frozen971 cohort')
        catalog = r.read(r.checked_file(accepted['receiver_catalog'])['path'])
        source = catalog['source_snapshot']
        if (source['snapshot_id'] != item['receiver_snapshot'] or
                source['frozen_sealed_batches'] != item['source_batches'] or
                source['questions'] != item['questions'] or source['trajectories'] != item['questions'] * 5 or
                source['seed'] != 20261021 or source['source_run_id'] != '70188b8e50fd4b45bc420ce58014f46e' or
                source['revision'] != r.REVISION):
            raise ValueError('Fixed authentic newseed5 source snapshot binding differs')
        if item.get('source_complete_sha256'):
            # Exporter seals canonical JSON plus newline; HF embeds that exact value.
            actual = hashlib.sha256(r.canonical(source) + b'\n').hexdigest()
            if actual != item['source_complete_sha256']:
                raise ValueError('Frozen exported source completion differs')
    return dict(preparation=done['preparation'], identity_plan=done['identity_plan'],
                raw_complete=done['raw_complete'])


def verify_features(refs, groups, progress=None):
    expected = {t: g['qid'] for g in groups for t in g['trajectory_ids']}
    seen, paths, manifests, packed = set(), set(), [], 0
    validator = object.__new__(ReplayFeatureReader)
    validator.dimension, validator.layer = r.DIM, 26
    for ref in refs:
        if ref['path'] in paths:
            raise ValueError('Feature manifest reused across cohort banks')
        paths.add(ref['path'])
        manifest = r.read(r.checked_file(ref)['path'])
        if (manifest['status'] != 'LOSSLESS_HF_REPLAY_FEATURES_COMPLETE' or
                manifest['model_revision'] != r.REVISION or manifest['model'] != r.MODEL or
                manifest['source_runtime'] != 'huggingface_frozen_causal_prefill' or
                manifest['layout'] != 'lossless_question_base_xor_policy_v1' or
                manifest['capture_layer'] != 26 or manifest['capture_kind'] != r.CAPTURE_KIND or
                manifest['dimension'] != r.DIM or manifest['prefix_tokens'] != 20 or
                manifest['question_max_tokens'] != 273 or manifest['actor_optimizer_updates'] != 0):
            raise ValueError('Actual frozen HF L26/P20 geometry/provenance differs')
        for item in manifest['files'].values():
            p = r.remote(item['path'])
            if p.stat().st_size != item['bytes']:
                raise ValueError('Producer-qualified packed feature size changed')
            packed += item['bytes']
        index = r.checked_file(manifest['files']['index.jsonl'])
        count = 0
        with Path(index['path']).open() as stream:
            for line in stream:
                row = json.loads(line); validator.validate_row(row, manifest)
                tid = row['trajectory_id']
                if tid in seen or expected.get(tid) != str(row['qid']):
                    raise ValueError('Missing, repeated or foreign own feature identity')
                seen.add(tid); count += 1
        if count != manifest['count']:
            raise ValueError('Actual own index count differs')
        manifests.append(ref)
        if progress is not None and (len(manifests) % 128 == 0 or len(manifests) == len(refs)):
            progress(dict(stage='ACTUAL_MANIFEST_INDEX_ID_COVERAGE_ADVANCING', manifests=len(manifests),
                          expected_manifests=len(refs), own_trajectory_rows=len(seen), expected_rows=len(expected),
                          packed_bytes=packed, epoch=time.time()))
    if seen != set(expected):
        raise ValueError('Every own trajectory needs its actual L26 feature')
    return manifests, dict(manifests=len(manifests), own_trajectory_rows=len(seen), packed_bytes=packed,
        payload_sizes_and_index_SHA_freshly_verified=True,
        payload_tensor_SHA_rehashed_in_this_metadata_join=False,
        producer_full_payload_SHA_and_reader_qualifications_preserved=True,
        training_reader_reconstructs_and_checks_own_retained_row_SHA=True)


def run(config_path, config_sha):
    config_ref = r.checked_file(dict(path=str(config_path), bytes=Path(config_path).stat().st_size, sha256=config_sha))
    config = r.read(config_ref['path'])
    if (config['role'] != 'fixed971_four_real_G10_banks_native_joint' or
            config['target'] != str(TARGET) or config['questions'] != 124194 or
            config['source_frozen_batches'] != 971 or
            {x['key']: x['questions'] for x in config['banks']} != COUNTS):
        raise ValueError('Only the fixed current124194 canonical cohort is authorized here')
    missing = [x['complete']['path'] for x in config['banks'] if not Path(x['complete']['path']).exists()]
    if missing:
        return dict(status='WAIT_FOR_ACTUAL_SUBSET_COMPLETIONS', missing=missing, GPU_submitted=False,
                    questions_ready_claim=False, actor_updates=0)
    TARGET.mkdir(exist_ok=True)
    with (TARGET/'join.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (TARGET/'complete.json').exists():
            done = r.read(TARGET/'complete.json')
            if done['config'] != config_ref or done['status'] != STATUS:
                raise ValueError('Existing bank source binding differs')
            for ref in done['outputs'].values(): r.checked_file(ref)
            return done
        exclusion_ref = r.checked_file(config['evaluation_exclusions'])
        excluded = r.read(exclusion_ref['path'])
        if (excluded['role'] != 'joint_training_fixed_evaluation_identity_exclusions' or
                len(excluded['groups']) != 6656 or len({g['qid'] for g in excluded['groups']}) != 6656):
            raise ValueError('Actual diagnostic512/selection2048/final4096 identity-only exclusions required')
        banks, refs, sources = {}, [], []
        for item in config['banks']:
            complete_ref = r.checked_file(item['complete']); done = r.read(complete_ref['path'])
            if done['questions'] != item['questions'] or done['trajectories'] != item['questions'] * 10 or done['actor_updates'] != 0:
                raise ValueError('Actual entire subset completion differs')
            if item['key'] == 'existing58432':
                if done['status'] != 'THREE_ACCEPTED_BANKS_58432_INTERNAL5120_FIT53312_APPEND728_METADATA_ACCEPTED':
                    raise ValueError('Actual original whole58432 bank acceptance required')
                group_ref = done['outputs']['merged-bank/groups.json']
            else:
                if done['status'] != SUBSET_STATUS or done['prefix_tokens'] != 20 or done['capture_layer'] != 26:
                    raise ValueError('Actual qualified native-joint feature subset completion required')
                group_ref = done['outputs']['groups.json']
                subset_lineage(item, done)
            group_ref = r.checked_file(group_ref)
            feature_ref = r.checked_file(done['outputs']['features-layer26.json'])
            banks[item['key']] = r.read(group_ref['path'])['train']
            refs.extend(r.read(feature_ref['path']))
            sources.append(dict(key=item['key'], complete=complete_ref, groups=group_ref, features=feature_ref))
            r.atomic(TARGET/'progress.json', dict(stage='ACTUAL_ACCEPTED_BANK_METADATA_READ', banks=len(sources),
                questions=sum(len(values) for values in banks.values()), actor_updates=0, epoch=time.time()))
        groups, summary = merge_groups(banks, excluded['groups'], expected_domains=config['domains'],
                                       expected_text_repeats=config['canonical_train_repeated_text_questions'])
        features, proof = verify_features(refs, groups, lambda value: r.atomic(TARGET/'progress.json', value))
        immutable(TARGET/'groups.json', dict(train=groups))
        immutable(TARGET/'features-layer26.json', features)
        immutable(TARGET/'source-refs.json', sources)
        outputs = {name: r.fingerprints(TARGET/name) for name in ('groups.json', 'features-layer26.json', 'source-refs.json')}
        done = dict(status=STATUS, config=config_ref, **summary, **proof,
            outputs=outputs, groups=outputs['groups.json'], features_layer26=outputs['features-layer26.json'],
            source_refs=sources, evaluation_exclusions=exclusion_ref, source_frozen_batches=971,
            complete169028_claim=False,
            actor_updates=0, probe_updates=0, GPU_submitted=False, final_outcomes_read=False, epoch=time.time())
        immutable(TARGET/'complete.json', done)
        return done


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--config-sha256', required=True)
    a = p.parse_args()
    try:
        result = run(a.config, a.config_sha256)
    except BaseException as error:
        if TARGET.exists():
            r.atomic(TARGET/'failure.json', dict(type=type(error).__name__, message=str(error), epoch=time.time()))
        raise
    print(json.dumps(result, sort_keys=True), flush=True)
