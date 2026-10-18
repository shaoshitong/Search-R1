"""Read actual101/151 raw Prefix->G5 continuation and selected PPO projection.

Run remotely on CPU. Only the small result crosses SSH; no model/features are
loaded, no inference/scoring/training is performed. This supplements, rather
than replaces, transition/checkpoint/QA acceptance. Early terminal prefixes
are checked as short real token streams, not invented padded policy tokens.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4')
RUNTIME = '2000d46a530c64d78be7802bf11175a68b65effe'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, choices=(101, 151), required=True)
    args = parser.parse_args()
    n = args.outer
    root = BASE / 'outers' / ('outer-%06d' % n)
    if not (root / 'complete.json').is_file():
        print(json.dumps(dict(status='PENDING_ACTUAL_POST_TRANSITION_OUTER', outer=n)))
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['DYNRPO_PROGRESSIVE_PREFIX'] = 'cp50_prefix20_40_80_three_epoch_v1'
    os.environ['DYNRPO_PREFIX_TOKENS'] = str(40 if n == 101 else 80)
    sys.path.insert(0, str(BASE / 'repo'))
    from reproduction.prefix_selection200.progressive_prefix import actor_prefix
    from reproduction.prefix_selection200.interaction_protocol import CONTRACT
    c = read(BASE / 'preparation/launch-config.json')
    j = read(root / 'complete.json')
    joint_sha = sha(root / 'complete.json')
    assert c['source_commit'] == j['runtime_source_commit'] == RUNTIME
    assert j['run_id'] == c['run_id'] and j['execution_id'] == c['execution_id']
    proof = BASE / 'preparation' / ('prefix-continuity-%06d.json' % n)
    if proof.exists():
        old = read(proof)
        assert old['source_joint_sha256'] == sha(root / 'complete.json')
        assert old['operator_sha256'] == sha(__file__)
        print(json.dumps(dict(status='EXISTING_PREFIX_CONTINUITY_PROOF', path=str(proof), sha256=sha(proof))))
        return
    prefix = read(root / 'prefix/complete.json')
    pool = read(root / 'full-pool/complete.json')
    selected = read(root / 'full/complete.json')
    score = read(root / 'probe-score/manifest.json')
    top = read(root / 'probe-score/pool-selection.json')
    final_path = root / 'probe-score/actual-em-selection.json'
    final = read(final_path)
    expected_prefix = actor_prefix(n)
    assert expected_prefix == (40 if n == 101 else 80)
    assert prefix['head_prefix_tokens'] == expected_prefix and prefix['questions'] == 5120
    assert pool['questions'] == 1024 and selected['questions'] == 512
    assert prefix['actor_version'] == pool['actor_version'] == selected['actor_version'] == j['actor']['sampling_actor_version']
    assert score['head_max_tokens'] == 273 + expected_prefix
    assert prefix['interaction_protocol'] == pool['interaction_protocol'] == selected['interaction_protocol'] == CONTRACT
    assert top['outcome_labels_used'] is False and final['outcome_labels_used'] is True
    scores_path = root / 'probe-score/candidate-scores.jsonl'
    assert sha(scores_path) == score['candidate_scores_sha256']
    candidates = list(rows(scores_path))
    assert len(candidates) == 5120 and len({r['qid'] for r in candidates}) == 5120
    expected = sorted(candidates, key=lambda r: (-r['q'], r['candidate_ordinal']))[:1024]
    assert top['selected'] == expected
    ids = {tid for g in expected for tid in g['trajectory_ids']}
    assert len(ids) == 5120
    verified = []

    def checked(item):
        p = Path(item['path'])
        assert BASE in p.parents and p.resolve() == p and p.stat().st_size == item['bytes']
        h = sha(p)
        assert h == item['sha256']
        verified.append(dict(path=str(p), bytes=p.stat().st_size, sha256=h))
        return rows(p)

    originals = {r['trajectory_id']: r for item in pool['shards'] for r in checked(item)}
    assert set(originals) == ids
    labels = list(checked(pool['own_em']))
    assert len(labels) == 5120 and {r['trajectory_id'] for r in labels} == ids
    for label in labels:
        original = originals[label['trajectory_id']]
        assert all(label[key] == original[key] for key in
                   ('qid', 'strict_em', 'actor_version', 'prefix_sha256', 'prefix_scan_sha256', 'selection_sha256'))
    prefix_records = {}
    for rank in prefix['ranks']:
        assert rank['actor_hidden_layer'] == 27 and rank['independent_encoder_forward'] is False
        assert rank['actual_compute_dtype'] == 'torch.bfloat16' and rank['native_hidden_flush'] is True
        for r in checked(rank['prefix']):
            if r['trajectory_id'] in ids:
                assert r['trajectory_id'] not in prefix_records
                prefix_records[r['trajectory_id']] = r
    assert set(prefix_records) == ids
    checked_ids = set()
    lengths = Counter()
    flushes = 0
    discarded = Counter()
    for rank in prefix['ranks']:
        for state in checked(rank['states']):
            tid = state['trajectory_id']
            if tid not in ids:
                continue
            assert tid not in checked_ids
            checked_ids.add(tid)
            p, full = prefix_records[tid], originals[tid]
            assert p['qid'] == state['group']['qid'] == full['qid']
            assert p['response_index'] == state['response_index'] == full['response_index']
            assert all(r['actor_version'] == prefix['actor_version'] and r['outer'] == n and r['run_id'] == c['run_id']
                       for r in (p, state, full))
            assert len(full['response_token_ids']) == len(full['response_info_mask']) == len(full['response_log_probs'])
            count = p['retained_policy_length']
            assert 0 <= count <= expected_prefix and (count == expected_prefix or p['terminal'])
            assert p['future_information_included'] is False
            assert p['prefix_sha256'] == state['prefix_sha256'] == full['prefix_sha256']
            assert full['prefix_scan_sha256'] == prefix['scan_sha256']
            assert p['initial_prompt_token_ids'] == state['prompt'] == full['initial_prompt_token_ids']
            policy = [token for token, mask in zip(full['response_token_ids'], full['response_info_mask']) if mask]
            assert policy[:count] == p['policy_token_ids']
            positions = []
            for turn in full['behavior_turns']:
                take = min(turn['kept_token_count'], count - len(positions))
                start = len(turn['actual_request_token_ids'])
                positions.extend(range(start, start + take))
                if len(positions) == count:
                    break
            assert positions == p['policy_position_ids']
            size = len(state['response'])
            assert full['response_token_ids'][:size] == state['response']
            assert full['response_info_mask'][:size] == state['info_mask']
            assert full['response_log_probs'][:size] == state['log_probs']
            assert full['behavior_turns'][:len(state['turns'])] == state['turns']
            prior_chunks = [chunk for turn in state['turns'] for chunk in turn['actual_sampling_chunks']]
            if state['action'] is not None:
                action = state['action']
                turn = full['behavior_turns'][len(state['turns'])]
                assert turn['actual_request_token_ids'] == action['actual_request_token_ids']
                for key in ('raw_generated_token_ids', 'selected_logprobs', 'actual_sampling_chunks'):
                    assert turn[key][:len(action[key])] == action[key]
                prior_chunks += action['actual_sampling_chunks']
            assert full['sampling_journal'][:len(prior_chunks)] == prior_chunks
            for chunk in prior_chunks:
                flush = chunk['sampling'].get('actor_hidden_flush')
                if flush:
                    assert flush['lookahead_token_not_passed_to_ppo_or_continuation'] is True
                    assert flush['lookahead_hidden_not_passed_to_probe'] is True
                    assert flush['captured_after_last_retained_token'] is True
                    extra = flush['discarded_sampled_lookahead_tokens']
                    assert type(extra) is int and extra in (0, 1)
                    discarded[extra] += 1
                    flushes += 1
            lengths[count] += 1
    assert checked_ids == ids
    projected = [r for item in selected['shards'] for r in checked(item)]
    wanted = {tid for g in final['selected'] for tid in g['trajectory_ids']}
    assert len(projected) == 2560 and {r['trajectory_id'] for r in projected} == wanted
    actual_groups = []
    for group in expected:
        labels = [originals[tid]['strict_em'] for tid in group['trajectory_ids']]
        assert all(type(label) is int and label in (0, 1) for label in labels)
        k = sum(labels)
        actual_groups.append(dict(group, actual_correct_count=k, actual_mixed=0 < k < 5))
    ranked = sorted(actual_groups, key=lambda r: (not r['actual_mixed'], -r['q'], r['candidate_ordinal']))[:512]
    assert final['selected'] == ranked
    added = {'full_pool_origin_selection_sha256', 'full_pool_complete_sha256',
             'full_pool_own_em_sha256', 'predicted_selection_sha256', 'actor_selection_outcome_labels_used'}
    final_sha, pool_sha = sha(final_path), sha(root / 'full-pool/complete.json')
    for row in projected:
        original = originals[row['trajectory_id']]
        assert set(row) == set(original) | added
        assert all(row[key] == value for key, value in original.items() if key != 'selection_sha256')
        assert row['selection_sha256'] == final_sha
        assert row['full_pool_origin_selection_sha256'] == original['selection_sha256'] == pool['selection_sha256']
        assert row['full_pool_complete_sha256'] == pool_sha
        assert row['full_pool_own_em_sha256'] == pool['own_em']['sha256']
        assert row['predicted_selection_sha256'] == final['predicted_selection_sha256']
        assert row['actor_selection_outcome_labels_used'] is True
    assert sha(root / 'complete.json') == joint_sha
    result = dict(status='ACTUAL_POST_TRANSITION_PREFIX_CONTINUATION_AND_PPO_PROJECTION_PASS',
                  outer=n, checked_unix=time.time(), runtime_commit=RUNTIME, operator_sha256=sha(__file__),
                  source_joint_sha256=joint_sha, prefix_tokens=expected_prefix,
                  checked_prefix_trajectories=5120, selected_projected_trajectories=2560,
                  retained_length_histogram=dict(sorted(lengths.items())), discarded_lookahead_flushes=flushes,
                  discarded_lookahead_token_count_histogram=dict(sorted(discarded.items())),
                  original_tokens_logprobs_masks_positions_and_sampling_chunks_preserved=True,
                  projected_scientific_payload_exactly_equal=True, selection_bookkeeping_rebound_explicitly=True,
                  final_actual_mixed_priority_rank_recomputed_from_stored_strict_labels=True,
                  native_hidden_tensor_payloads_read=False,
                  padded_feature_tensors_independently_checked=False, QA_or_200_goal_accepted=False,
                  verified_raw_files=verified)
    with proof.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'], path=str(proof), sha256=sha(proof),
                          retained_length_histogram=result['retained_length_histogram'], discarded_lookahead_flushes=flushes)))


if __name__ == '__main__':
    main()
