"""Independent final-Parquet content/token arithmetic audit; no GPU or rescoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import pyarrow.parquet as pq


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--source', required=True); a.add_argument('--data', required=True)
    a.add_argument('--output', required=True); args = a.parse_args()
    expected_sha = '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
    assert sha(args.data) == expected_sha
    data = pq.read_table(args.data).to_pylist()
    source = [json.loads(s) for s in Path(args.source).read_text().splitlines()]
    assert len(data) == 51713 and len(source) == 3610
    totals = dict.fromkeys(('initial_prompt_tokens', 'raw_generated_tokens',
        'model_generated_tokens', 'observation_tokens', 'trajectory_total_tokens',
        'cumulative_model_input_tokens'), 0)
    observations = 0
    for ordinal, row in enumerate(source):
        original = data[ordinal]
        assert original['data_source'] == row['data_source'] == 'nq'
        assert row['evaluation_ordinal'] == ordinal
        assert row['qid'] == str(original['extra_info']['index'])
        prompt = original['prompt'][0]['content']
        question = prompt[list(re.finditer(r'Question:\s*', prompt))[-1].end():].strip()
        assert row['question'] == question
        gold = original['reward_model']['ground_truth']['target']
        assert row['gold'] == ([gold] if isinstance(gold, str) else list(gold))
        assert row['source_shard_sha256'] == expected_sha
        saved = dict(row); checksum = saved.pop('source_record_hash')
        assert hashlib.sha256(json.dumps(saved, sort_keys=True, separators=(',', ':'),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest() == checksum
        assert len(row['prompt_token_ids']) == row['initial_prompt_tokens']
        assert len(row['response_token_ids']) == len(row['response_policy_mask']) == row['trajectory_response_tokens']
        assert row['model_generated_tokens'] == sum(row['response_policy_mask'])
        assert row['observation_tokens'] == row['trajectory_response_tokens'] - row['model_generated_tokens']
        assert row['trajectory_total_tokens'] == row['initial_prompt_tokens'] + row['trajectory_response_tokens']
        assert row['cumulative_model_input_tokens'] == sum(len(t['actual_request_token_ids']) for t in row['behavior_turns'])
        assert row['raw_generated_tokens'] == sum(len(t['raw_generated_token_ids']) for t in row['behavior_turns'])
        assert row['actual_retrieval_calls'] == len(row['seen_docs'])
        observations += row['actual_retrieval_calls']
        for key in totals:
            totals[key] += row[key]
    report = dict(at=time.time(), questions=3610, ordinals=[0, 3609],
        original_data_questions=51713, data_sha256=expected_sha,
        exact_question_gold_occurrence_match=True, all_record_hashes_valid=True,
        token_arithmetic_all_valid=True, actual_retrieval_calls=observations,
        source=str(Path(args.source)), source_bytes=Path(args.source).stat().st_size,
        source_sha256=sha(args.source), token_totals=totals,
        token_means={k: v / 3610 for k, v in totals.items()},
        cached_input_tokens=None, cached_input_tokens_reason='Native sampler does not expose cache hit counts')
    with Path(args.output).open('x') as f:
        json.dump(report, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
