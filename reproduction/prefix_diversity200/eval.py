"""Eight-TP1 BF16 QA using the identical training interaction state machine."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .actor import atomic_json, canonical, read, remote, sha, utc

from .interaction import BUDGET, contract

CONTRACT = contract(temperature=0., responses=1)


def score_tokens(tokenizer, prompt, response, mask, gold):
    from search_g1.signals import decode_segments, contexts
    from reproduction.common8.calibration_evaluation import _f1
    from verl.utils.reward_score.qa_em import em_check
    text, ranges = decode_segments(tokenizer, response, mask)
    ctx = contexts(tokenizer.decode(prompt), text, ranges)
    answer = ctx['answer'] if ctx is not None else None
    aliases = [gold] if isinstance(gold, str) else list(gold)
    return dict(em=int(answer is not None and em_check(answer, aliases)), f1=_f1(answer, aliases),
                answer_present=answer is not None,
                answer_sha256=hashlib.sha256(answer.encode()).hexdigest() if answer is not None else None)


def load_questions(path, expected_count):
    path = remote(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        original = pq.read_table(path).to_pylist()
    else:
        document = read(path); original = document['rows'] if isinstance(document, dict) else document
    if len(original) != expected_count or expected_count not in (512, 51713):
        raise ValueError('Evaluation must use the exact fixed512 or complete51713 canonical questions')
    rows = []
    for ordinal, row in enumerate(original):
        extra = row.get('extra_info') or {}
        qid = row.get('qid', row.get('__g1_question_id'))
        if qid is None:
            qid = json.dumps([str(row['data_source']), str(extra.get('index', ordinal))])
        rows.append(dict(row, qid=str(qid), evaluation_ordinal=ordinal))
    if len({row['qid'] for row in rows}) != len(rows):
        raise ValueError('Canonical evaluation question identities repeat')
    return rows


def build_evaluation_engine(config, model, seed, free, total, *, factory=None, cleanup=None):
    """One constructor per process; a CUDA-free supervisor owns bounded retries."""
    if factory is None:
        from vllm import LLM
        factory = LLM
    reserve = float(config.get('evaluation_free_reserve_gib', 2)) * 1024 ** 3
    cap = float(config.get('evaluation_utilization_cap', .95))
    requested = config.get('evaluation_memory_utilization', 'auto')
    utilization = min(cap, (free-reserve)/total) if requested=='auto' else float(requested)
    utilization *= float(config.get('evaluation_initialization_utilization_scale',1))
    if not 0 < utilization <= 1:
        raise ValueError('Actual available VRAM cannot supply the configured evaluation reservation')
    limits = dict(max_num_seqs=int(config.get('evaluation_max_num_seqs',1024)),
        max_num_batched_tokens=int(config.get('evaluation_max_batched_tokens',32768)))
    llm = factory(model=str(model),tokenizer=str(model),trust_remote_code=False,
        dtype='bfloat16',tensor_parallel_size=1,max_model_len=4096,seed=seed,
        gpu_memory_utilization=utilization,swap_space=0,
        enforce_eager=bool(config.get('evaluation_enforce_eager',False)),
        enable_chunked_prefill=True,enable_prefix_caching=True,disable_async_output_proc=True,
        disable_log_stats=True,**limits)
    return llm,utilization,limits,[]


def worker(config_path, model_dir, output_dir, questions_path, expected_count, rank):
    from reproduction.prefix20_online.rollout import _configure_one_visible_gpu
    from .rollout import (initial_state, start_action, next_request, append_chunk,
        finish_action, stream_action_present, native_snapshot, registered_sampling)
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    import re
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    config = read(remote(config_path)); device = _configure_one_visible_gpu(rank)
    model_dir, output = remote(model_dir), remote(output_dir)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    questions = load_questions(questions_path, expected_count)[rank::8]
    free, total = torch.cuda.mem_get_info(); seed = int(config.get('seed', 1729)) + rank * 104729
    from .evaluation_initialization import initialization_status, initialization_error
    try:
        llm, utilization, limits, initialization_oom_attempts = build_evaluation_engine(config, model_dir, seed, free, total)
    except Exception as error:
        actual_oom=isinstance(error,torch.cuda.OutOfMemoryError) or (isinstance(error,RuntimeError)
            and ('CUDA out of memory' in str(error) or 'CUDA error: out of memory' in str(error)))
        initialization_status(config_path,model_dir,questions_path,expected_count,rank,
            failure=initialization_error(error,actual_oom))
        raise
    initialization_status(config_path,model_dir,questions_path,expected_count,rank)
    engine = llm.llm_engine
    generation = read(model_dir / 'generation_config.json')
    stop = generation.get('eos_token_id', tokenizer.eos_token_id)
    stop = [stop] if isinstance(stop, int) else list(stop)
    terminal_ids = sorted(set(stop + [tokenizer.eos_token_id]))
    recipe = GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500, num_gpus=8, search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe, is_validation=True)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(agent_workers=limits['max_num_seqs'],
        max_inflight_trajectories=limits['max_num_seqs'], retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    matched = re.search(r'actor([0-9]+)$', config['actor_version'])
    outer = int(matched.group(1)) if matched else 0
    states = [initial_state(config, outer, dict(row, candidate_ordinal=row['evaluation_ordinal']), 0, tokenizer) for row in questions]
    waiting, ready, pending, tools, finished = deque(states), deque(), {}, {}, []
    active = 0; requests = 0; started = time.monotonic(); budget = dict(BUDGET)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output.mkdir(parents=True, exist_ok=True); full = output / ('rank-%d-full.jsonl.gz' % rank)
    if full.exists(): raise FileExistsError('Never overwrite an existing evaluation shard')
    pool = ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency)

    def finish_environment(state, cut, environment, stream):
        nonlocal active
        finish_action(state, cut, environment, tokenizer, budget)
        if not state['terminal']:
            ready.append(state); return
        row = state['group']
        measured = score_tokens(tokenizer, state['prompt'], state['response'], state['info_mask'], row['reward_model']['ground_truth']['target'])
        item = dict(qid=row['qid'], data_source=row['data_source'], evaluation_ordinal=row['evaluation_ordinal'], **measured,
            actual_retrieval_calls=state['retrieval_calls'], valid_actions=sum(t['valid_action'] for t in state['turns']),
            turns=len(state['turns']), response_token_count=len(state['response']))
        stream.write(canonical(dict(item, initial_prompt_token_ids=state['prompt'], response_token_ids=state['response'],
            response_info_mask=state['info_mask'], response_log_probs=state['log_probs'], behavior_turns=state['turns'])).decode() + '\n')
        finished.append(item); active -= 1
        if len(finished) % 16 == 0:
            stream.flush()
            atomic_json(output / ('rank-%d-progress.json' % rank), dict(rank=rank, completed_questions=len(finished),
                expected_questions=len(questions), active=active, pending_generation=len(pending), pending_retrieval=len(tools), utc=utc()), replace=True)

    try:
        with gzip.open(full, 'xt', encoding='utf8', compresslevel=1) as stream:
            while len(finished) < len(questions):
                while waiting and active < limits['max_num_seqs']:
                    ready.append(waiting.popleft()); active += 1
                while ready:
                    state = ready.popleft(); start_action(state, budget)
                    request = next_request(state, 'evaluation', config)
                    if request is None: raise ValueError('Unfinished QA action has no actual budget')
                    params = SamplingParams(n=1, temperature=0., top_p=1., top_k=-1,
                        max_tokens=request['max_tokens'], stop=[], stop_token_ids=stop, ignore_eos=False,
                        logprobs=1, detokenize=False, seed=request['seed'])
                    engine.add_request(request['request_id'], {'prompt_token_ids':request['actual_request_token_ids']}, params)
                    pending[request['request_id']] = state, request, registered_sampling(engine, request, params); requests += 1
                if engine.has_unfinished_requests():
                    for result in engine.step():
                        rid = str(result.request_id)
                        if rid not in pending: continue
                        state, request, sampling = pending[rid]
                        if len(result.outputs)!=1 or list(result.prompt_token_ids)!=request['actual_request_token_ids']:
                            raise ValueError('Actual QA native request identity changed')
                        native = result.outputs[0]; raw = list(native.token_ids)
                        seen_action = stream_action_present(state['action']['raw_generated_token_ids']+raw, decode)
                        aborted = bool(seen_action and not result.finished)
                        if not result.finished and not seen_action: continue
                        snapshot = native_snapshot(native)
                        if aborted: engine.abort_request(rid)
                        pending.pop(rid)
                        cut = append_chunk(state, request, snapshot['raw_generated_token_ids'], snapshot['selected_logprobs'], sampling,
                            finish_reason=snapshot['finish_reason'], stop_reason=snapshot['stop_reason'], terminal_ids=terminal_ids,
                            decode=decode, action_boundary_abort=aborted)
                        if cut is None: ready.append(state)
                        elif cut['native_terminal']:
                            finish_environment(state, cut, ('',True,int(cut['action'] is not None),0), stream)
                        elif cut['action']=='search' and not state['action']['final_turn']:
                            tools[pool.submit(execute_one_action, manager, cut['execution_text'], True, retriever)] = state, cut
                        else:
                            finish_environment(state, cut, execute_one_action(manager, cut['execution_text'], not state['action']['final_turn'], retriever), stream)
                done = [future for future in tools if future.done()]
                for future in done:
                    state, cut = tools.pop(future); finish_environment(state, cut, future.result(), stream)
                if not ready and not engine.has_unfinished_requests() and tools and not done: time.sleep(.001)
    finally: pool.shutdown(wait=True, cancel_futures=True)
    finished.sort(key=lambda row:row['evaluation_ordinal'])
    receipt = dict(schema_version=2, rank=rank, run_id=config['run_id'], actor_model_dir=str(model_dir), actor_version=config['actor_version'],
        evaluation_contract=CONTRACT, question_source_sha256=sha(questions_path), question_count=len(finished), qa_count=len(finished), rows=finished,
        full_output=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)), device_binding=device,
        actual_compute_dtype=str(engine.model_config.dtype), actual_generation_requests=requests, seed=seed,
        gpu_memory_utilization=utilization, actual_scheduler_limits=limits, initialization_oom_attempts=initialization_oom_attempts,
        elapsed_seconds=time.monotonic()-started, completed_utc=utc())
    atomic_json(output / ('rank-%d-complete.json'%rank), receipt); return receipt


def launch(config_path, model_dir, output, questions_path, expected_count=512):
    config_path, model_dir, output, questions_path = map(remote, (config_path, model_dir, output, questions_path))
    config = read(config_path); target = output / 'complete.json'
    expected_sha=config['validation_questions_sha256'] if expected_count==512 else config['final_questions_sha256']
    if expected_count not in (512,51713) or sha(questions_path)!=expected_sha:
        raise ValueError('QA must use the original pinned canonical questions without a new split')
    canonical_questions=load_questions(questions_path,expected_count)
    existing=read(target) if target.exists() else None
    output.mkdir(parents=True, exist_ok=True)
    from .evaluation_initialization import supervise
    supervision=supervise(config_path,model_dir,output,questions_path,expected_count)
    receipts = [read(output / ('rank-%d-complete.json' % rank)) for rank in range(8)]
    for rank,item in enumerate(receipts):
        expected_rows=canonical_questions[rank::8]
        if (item['rank']!=rank or item['run_id']!=config['run_id']
                or item['actor_model_dir']!=str(model_dir) or item['actor_version']!=config['actor_version']
                or item['question_source_sha256']!=expected_sha
                or item['seed']!=int(config.get('seed',1729))+rank*104729
                or item['question_count']!=len(expected_rows) or item['qa_count']!=len(expected_rows)
                or [(r['evaluation_ordinal'],r['qid'],r['data_source']) for r in item['rows']]
                    !=[(r['evaluation_ordinal'],r['qid'],r['data_source']) for r in expected_rows]):
            raise ValueError('Actual QA rank receipt lost canonical question/model/seed binding')
        shard=output/('rank-%d-full.jsonl.gz'%rank)
        if item['full_output']['path']!=str(shard) or item['full_output']['bytes']!=shard.stat().st_size:
            raise ValueError('QA complete receipt does not describe the retained full shard')
        if rank in supervision['reused_rank_receipts'] and item['full_output']['sha256']!=sha(shard):
            raise ValueError('Previously completed QA shard changed before explicit reuse')
    rows = sorted([row for item in receipts for row in item['rows']], key=lambda row: row['evaluation_ordinal'])
    if ([row['evaluation_ordinal'] for row in rows] != list(range(expected_count))
            or len({row['qid'] for row in rows}) != expected_count
            or any(item['actor_version'] != config['actor_version'] or item['evaluation_contract'] != CONTRACT
                   or item['actual_compute_dtype'] != 'torch.bfloat16' for item in receipts)):
        raise ValueError('Fixed greedy BF16 shared-interaction evaluation omitted, duplicated or changed a canonical question')
    sources = Counter(row['data_source'] for row in rows)
    if expected_count == 512 and sources != dict(nq=256, hotpotqa=256):
        raise ValueError('Fixed512 must retain original256NQ+256Hotpot')
    source_metrics = {}
    for source in sorted(sources):
        subset = [row for row in rows if row['data_source'] == source]
        source_metrics[source] = dict(qa_count=len(subset), em=sum(row['em'] for row in subset) / len(subset),
            f1=sum(row['f1'] for row in subset) / len(subset))
    result = dict(schema_version=1, run_id=config['run_id'], actor_version=config['actor_version'],
        actor_model_dir=str(model_dir), evaluation_contract=CONTRACT, qa_count=len(rows),
        question_source_sha256=sha(questions_path), em=sum(row['em'] for row in rows) / len(rows),
        f1=sum(row['f1'] for row in rows) / len(rows), per_source=source_metrics,
        output_shards=[item['full_output'] for item in receipts],
        rank_receipts=[dict(path=str(output / ('rank-%d-complete.json' % r)),
                            sha256=sha(output / ('rank-%d-complete.json' % r))) for r in range(8)],
        completed_utc=utc(), **supervision)
    if existing is not None:
        keys=('run_id','actor_version','actor_model_dir','evaluation_contract','qa_count',
            'question_source_sha256','em','f1','per_source','output_shards','rank_receipts')
        if any(existing.get(key)!=result[key] for key in keys):
            raise ValueError('Existing QA aggregate does not match the independently revalidated canonical rank receipts')
        return existing
    atomic_json(target, result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--questions', required=True); parser.add_argument('--expected-count', type=int, default=512)
    parser.add_argument('--rank', type=int); args = parser.parse_args()
    if args.rank is None: launch(args.config, args.model, args.output, args.questions, args.expected_count)
    else: worker(args.config, args.model, args.output, args.questions, args.expected_count, args.rank)


if __name__ == '__main__': main()
