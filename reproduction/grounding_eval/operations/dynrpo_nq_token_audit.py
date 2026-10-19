"""Count real native NQ request/completion tokens without generating or training.

Run on the original HPC3 shared storage. Full trajectories and per-question
counts remain remote; only the final small audit is intended for local export.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


DATA_SHA = '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
ORDER_SHA = '23c0a8307f38a7a8d1cbd9e71fb115b7bd091421a3d3b10952fb71c8210b551c'
SOURCE_COMPLETE_SHA = '970e596d983f383ca1fdb73afeea091ea27dc7c2a2c3d30474320f1928a80af8'
RUN = 'cc0de51f2d68495c9664aa888074839c'
ACTOR = RUN + '-actor0200'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def validate_trajectory(row, actor, all_request_ids):
    prompt, response = row['initial_prompt_token_ids'], row['response_token_ids']
    assert len(response) == row['response_token_count'] == len(row['response_info_mask'])
    assert len(row['behavior_turns']) == row['turns'] and row['turns'] > 0
    kept_ids, mask = [], []
    stats = Counter(initial_prompt_tokens=len(prompt))
    reasons, aborts = Counter(), 0
    for turn in row['behavior_turns']:
        assert turn['actor_version'] == turn['policy_snapshot_id'] == actor
        context = prompt + kept_ids
        assert turn['actual_request_token_ids'] == context
        generated = []
        for chunk in turn['actual_sampling_chunks']:
            assert chunk['actor_version'] == chunk['policy_snapshot_id'] == actor
            assert chunk['phase'] == 'evaluate'
            rid = chunk['request_id']
            assert rid not in all_request_ids
            all_request_ids.add(rid)
            assert chunk['sampling']['registered_request_id'] == rid
            assert chunk['sampling']['temperature'] == 0 and chunk['sampling']['n'] == 1
            assert chunk['actual_request_token_ids'] == context + generated
            raw = chunk['raw_generated_token_ids']
            assert len(raw) == len(chunk['selected_logprobs']) <= chunk['max_tokens']
            assert all(math.isfinite(p) and p <= 1e-6 for p in chunk['selected_logprobs'])
            stats['cumulative_model_input_tokens'] += len(chunk['actual_request_token_ids'])
            stats['raw_generated_tokens'] += len(raw)
            stats['actual_generation_requests'] += 1
            reasons[str(chunk['finish_reason'])] += 1
            aborts += int(chunk['native_action_boundary_abort'])
            generated.extend(raw)
        assert generated == turn['raw_generated_token_ids']
        assert len(generated) == turn['raw_generated_token_count'] <= 500
        cut = turn['prefix_cut']['kept_token_ids']
        assert cut == generated[:len(cut)] and len(cut) == turn['kept_token_count']
        observation, correction = turn['observation_token_ids'], turn['correction_token_ids']
        assert not (observation and correction)
        assert len(observation) + len(correction) <= 500
        kept_ids.extend(cut + observation + correction)
        mask.extend([1] * len(cut) + [0] * (len(observation) + len(correction)))
        stats['model_generated_tokens'] += len(cut)
        stats['observation_tokens'] += len(observation)
        stats['environment_correction_tokens'] += len(correction)
    assert kept_ids == response and mask == row['response_info_mask']
    stats['retained_environment_tokens'] = stats['observation_tokens'] + stats['environment_correction_tokens']
    stats['trajectory_total_tokens'] = len(prompt) + len(response)
    assert stats['trajectory_total_tokens'] == len(prompt) + stats['model_generated_tokens'] + stats['retained_environment_tokens']
    stats['input_plus_raw_output_tokens'] = stats['cumulative_model_input_tokens'] + stats['raw_generated_tokens']
    # Independent flat-list arithmetic avoids accidentally counting one logical
    # action once when several actual native requests exist.
    chunks = [c for t in row['behavior_turns'] for c in t['actual_sampling_chunks']]
    assert stats['cumulative_model_input_tokens'] == sum(len(c['actual_request_token_ids']) for c in chunks)
    assert stats['raw_generated_tokens'] == sum(len(c['raw_generated_token_ids']) for c in chunks)
    return dict(stats), reasons, aborts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base, data, out = map(Path, (args.base, args.data, args.output))
    assert base.name == 'dynrpo-prefix5-thread5-20261017-r4'
    assert not out.exists() and base.resolve() not in out.resolve().parents
    out.mkdir(parents=True)
    started = time.time()
    output_storage = subprocess.check_output(['findmnt', '-T', str(out), '-J'], text=True)
    storage = json.loads(output_storage)
    assert sha(data) == DATA_SHA
    source = base / 'final-evaluation'
    assert sha(source / 'complete.json') == SOURCE_COMPLETE_SHA
    complete = read(source / 'complete.json')
    assert complete['run_id'] == RUN and complete['actor_version'] == ACTOR
    assert complete['qa_count'] == 51713 and complete['actual_compute_dtype'] == 'torch.bfloat16'
    assert all(v == 0 for v in complete['actual_worker_exit_codes'].values())
    assert complete['question_source_sha256'] == DATA_SHA
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    questions = pq.read_table(data).to_pylist()
    assert len(questions) == 51713
    nq = [(i, r) for i, r in enumerate(questions) if r['data_source'] == 'nq']
    assert len(nq) == 3610
    expected, order = {}, []
    for local_ordinal, (ordinal, row) in enumerate(nq):
        qid = row.get('qid', row.get('__g1_question_id'))
        if qid is None:
            qid = json.dumps([str(row['data_source']), str((row.get('extra_info') or {}).get('index', ordinal))])
        messages = row['prompt']
        text = '\n'.join(m['content'] for m in messages)
        question = text[list(re.finditer(r'Question:\s*', text))[-1].end():].strip()
        gold = row['reward_model']['ground_truth']['target']
        gold = [gold] if isinstance(gold, str) else list(gold)
        expected[ordinal] = dict(qid=str(qid), prompt=messages, nq_ordinal=local_ordinal)
        order.append(['nq', local_ordinal, question, sorted(gold)])
    content_sha = hashlib.sha256(canonical(order)).hexdigest()
    assert content_sha == ORDER_SHA
    model = Path(complete['actor_model_dir'])
    assert model.is_dir() and model.name == 'huggingface' and 'outer-0200-attempt-' in str(model)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    token_meta = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in model.iterdir()
                  if p.name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json')}
    reports, items, seen, all_requests = [], [], set(), set()
    reasons, aborts = Counter(), 0
    for rank in range(4):
        receipt_info = complete['rank_receipts'][rank]
        receipt_path = Path(receipt_info['path'])
        assert sha(receipt_path) == receipt_info['sha256']
        receipt = read(receipt_path)
        assert receipt['rank'] == rank and receipt['run_id'] == RUN and receipt['actor_version'] == ACTOR
        assert receipt['actor_model_dir'] == str(model)
        full = Path(receipt['full_output']['path'])
        measured_sha = sha(full)
        assert measured_sha == receipt['full_output']['sha256'] == complete['output_shards'][rank]['sha256']
        rank_count = rank_raw = rank_requests = rank_nq = 0
        stored_rows = {r['evaluation_ordinal']: r for r in receipt['rows']}
        with gzip.open(full, 'rt') as f:
            for line in f:
                row = json.loads(line)
                ordinal = row['evaluation_ordinal']
                assert ordinal % 4 == rank
                assert all(row[k] == v for k, v in stored_rows[ordinal].items())
                chunks = [c for t in row['behavior_turns'] for c in t['actual_sampling_chunks']]
                rank_count += 1; rank_raw += sum(len(c['raw_generated_token_ids']) for c in chunks)
                rank_requests += len(chunks)
                if row['data_source'] != 'nq':
                    continue
                assert ordinal in expected and ordinal not in seen
                seen.add(ordinal); rank_nq += 1
                spec = expected[ordinal]
                assert row['qid'] == spec['qid']
                prompt = tokenizer.apply_chat_template(spec['prompt'], add_generation_prompt=True, tokenize=False)
                ids = tokenizer.encode(prompt, add_special_tokens=False)[-2048:]
                assert ids == row['initial_prompt_token_ids']
                stats, finishes, aborted = validate_trajectory(row, ACTOR, all_requests)
                reasons.update(finishes); aborts += aborted
                items.append(dict(evaluation_ordinal=ordinal, nq_ordinal=spec['nq_ordinal'], qid=row['qid'],
                                  em=row['em'], actual_retrieval_calls=row['actual_retrieval_calls'], **stats))
        assert rank_count == receipt['question_count'] == receipt['qa_count']
        assert rank_raw == receipt['actual_sampled_tokens'] and rank_requests == receipt['actual_generation_requests']
        proof = dict(rank=rank, all_dataset_rows=rank_count, NQ_rows=rank_nq,
                     all_dataset_actual_generation_requests=rank_requests,
                     all_dataset_raw_generated_tokens=rank_raw,
                     full_output_path=str(full), full_output_bytes=full.stat().st_size, full_output_sha256=measured_sha,
                     rank_receipt_path=str(receipt_path), rank_receipt_sha256=receipt_info['sha256'])
        reports.append(proof)
        write(out / 'progress.json', dict(state='RECOUNTING_EXISTING_OUTPUTS', at=time.time(),
              completed_ranks=rank+1, NQ_questions=len(items), target=3610, latest_rank=proof))
    assert seen == set(expected) and len(items) == 3610
    items.sort(key=lambda r: r['nq_ordinal'])
    assert sum(r['em'] for r in items) / 3610 == complete['per_source']['nq']['em']
    fields = ['initial_prompt_tokens', 'cumulative_model_input_tokens', 'raw_generated_tokens',
              'model_generated_tokens', 'observation_tokens', 'environment_correction_tokens',
              'retained_environment_tokens', 'trajectory_total_tokens', 'input_plus_raw_output_tokens',
              'actual_generation_requests']
    costs = {}
    for key in fields:
        values = [r.get(key, 0) for r in items]; ordered = sorted(values)
        costs[key] = dict(total=sum(values), mean=sum(values)/3610, min=min(values), max=max(values),
                          median=statistics.median(values), p95_nearest_rank=ordered[math.ceil(.95*3610)-1], measured_questions=3610)
    per_question = out / 'NQ-token-counts.jsonl'
    with per_question.open('x') as f:
        for row in items:
            f.write(canonical(row).decode() + '\n')
        f.flush(); os.fsync(f.fileno())
    result = dict(state='DYNRPO_NQ_ACTUAL_TOKEN_COST_FULL_READ_VERIFIED', at=time.time(), elapsed_seconds=time.time()-started,
          method='DynRPO', source_session='01a0fcbe-15f8-7600-ae84-9487d924560a', checkpoint_iteration=200,
          run_id=RUN, actor_version=ACTOR, actor_model_dir=str(model), source_evaluation_root=str(source),
          source_complete_sha256=SOURCE_COMPLETE_SHA, source_dataset_sha256=DATA_SHA,
          canonical_question_gold_order_sha256=content_sha, NQ_questions=3610, correct=sum(r['em'] for r in items),
          NQ_EM=sum(r['em'] for r in items)/3610, token_usage=costs,
          actual_retrieval_calls=sum(r['actual_retrieval_calls'] for r in items),
          native_finish_reason_counts=dict(reasons), native_action_boundary_aborts=aborts,
          evaluation_contract=complete['evaluation_contract'], source_rank_proofs=reports,
          tokenizer_and_generation_files=token_meta, per_question_counts_path=str(per_question),
          per_question_counts_sha256=sha(per_question), output_storage=storage,
          script_sha256=sha(__file__), model_inference_rerun=False, gpu_used=False, training_mutated=False,
          measurement_scope='Exact cumulative actual request input IDs plus committed native raw completion IDs on the fixed3610 NQ occurrences. Input history counts again on each actual request. No CF/probe/training/pilot tokens; no cache discount or hardware FLOP estimate.',
          comparison_limit='Same dataset and accounting definition as the six-method table. DynRPO uses its native stream-first-action4096/BF16 evaluation adapter; legacy adapters can generate longer discarded tails. These counts do not isolate model-policy savings from stopping implementation.')
    write(out / 'summary.json', result)
    write(out / 'progress.json', dict(state='COMPLETE', at=time.time(), NQ_questions=3610))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
