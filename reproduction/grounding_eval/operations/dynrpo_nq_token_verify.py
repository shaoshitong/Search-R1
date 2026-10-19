"""Independently flatten native request chunks and verify the NQ token audit."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    summary_path, output = Path(args.summary), Path(args.output)
    assert not output.exists()
    summary = json.loads(summary_path.read_text())
    assert summary['state'] == 'DYNRPO_NQ_ACTUAL_TOKEN_COST_FULL_READ_VERIFIED'
    counts_path = Path(summary['per_question_counts_path'])
    assert sha(counts_path) == summary['per_question_counts_sha256']
    expected = {r['evaluation_ordinal']: r for r in
                (json.loads(line) for line in counts_path.open())}
    assert len(expected) == 3610
    totals, seen, request_ids, proofs, correct = Counter(), set(), set(), [], 0
    for proof in summary['source_rank_proofs']:
        raw_path = Path(proof['full_output_path'])
        measured = sha(raw_path)
        assert measured == proof['full_output_sha256']
        rank_rows = 0
        with gzip.open(raw_path, 'rt') as f:
            for line in f:
                # Native outputs are canonical JSON; inspect only the NQ data,
                # while the entire compressed source is independently hashed.
                if '"data_source":"nq"' not in line:
                    continue
                row = json.loads(line)
                assert row['data_source'] == 'nq'
                ordinal = row['evaluation_ordinal']
                assert ordinal in expected and ordinal not in seen
                assert ordinal % 4 == proof['rank']
                assert row['qid'] == expected[ordinal]['qid']
                seen.add(ordinal); rank_rows += 1; correct += row['em']
                chunks = [c for t in row['behavior_turns'] for c in t['actual_sampling_chunks']]
                for c in chunks:
                    assert c['actor_version'] == summary['actor_version']
                    assert c['request_id'] not in request_ids
                    request_ids.add(c['request_id'])
                inp = sum(len(c['actual_request_token_ids']) for c in chunks)
                gen = sum(len(c['raw_generated_token_ids']) for c in chunks)
                kept = sum(row['response_info_mask'])
                obs = sum(len(t['observation_token_ids']) for t in row['behavior_turns'])
                correction = sum(len(t['correction_token_ids']) for t in row['behavior_turns'])
                prompt = len(row['initial_prompt_token_ids'])
                assert len(row['response_token_ids']) == kept + obs + correction
                direct = dict(cumulative_model_input_tokens=inp, raw_generated_tokens=gen,
                    input_plus_raw_output_tokens=inp+gen, initial_prompt_tokens=prompt,
                    model_generated_tokens=kept, observation_tokens=obs,
                    environment_correction_tokens=correction, retained_environment_tokens=obs+correction,
                    trajectory_total_tokens=prompt+len(row['response_token_ids']), actual_generation_requests=len(chunks))
                assert all(expected[ordinal].get(k, 0) == v for k, v in direct.items())
                totals.update(direct)
        assert rank_rows == proof['NQ_rows']
        proofs.append(dict(rank=proof['rank'], NQ_rows=rank_rows, full_output_sha256=measured))
    assert seen == set(expected) and correct == summary['correct']
    for k, value in totals.items():
        stat = summary['token_usage'][k]
        assert stat['total'] == value and stat['measured_questions'] == 3610
        assert stat['mean'] == value/3610
    result = dict(state='DYNRPO_NQ_SECOND_INDEPENDENT_RECOUNT_VERIFIED', at=time.time(),
        summary_path=str(summary_path), summary_sha256=sha(summary_path),
        per_question_counts_sha256=sha(counts_path), source_rank_proofs=proofs,
        canonical_question_gold_order_sha256=summary['canonical_question_gold_order_sha256'],
        questions=len(seen), correct=correct, totals=dict(totals),
        mean_input_plus_raw_output_tokens=totals['input_plus_raw_output_tokens']/3610,
        verifier_sha256=sha(__file__), gpu_used=False, training_mutated=False,
        source_outputs_modified=False)
    with output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
