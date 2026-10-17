"""Eight-TP1 QA using the same BF16 interaction contract as training.

Greedy n1 changes sampling only. Context budgeting, information wrappers,
early-final decisions, first-complete-action streaming and native EOS trimming
are shared with the actual prefix/continuation sampler. Strict scoring is
unchanged; original canonical questions and full outputs remain remote.
"""
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
from .interaction_contract import (BUDGET, INTERACTION_CONTRACT, evaluation_contract, first_action_cut, final_turn,
    interaction_outcome, stream_action_present)

CONTRACT = evaluation_contract()


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


def _cleanup_failed_initialization():
    """Release only this TP1 process's failed constructor, never another task."""
    import gc
    import torch
    from vllm.distributed import parallel_state
    parallel_state.destroy_model_parallel()
    if hasattr(parallel_state, 'destroy_distributed_environment'):
        parallel_state.destroy_distributed_environment()
    gc.collect(); torch.cuda.empty_cache()


def build_evaluation_engine(config, model, seed, free, total, *, factory=None, cleanup=None):
    """Fill available VRAM; lower cache/concurrency only after actual constructor OOM."""
    import torch
    if factory is None:
        from vllm import LLM
        factory = LLM
    if cleanup is None: cleanup = _cleanup_failed_initialization
    reserve = float(config.get('evaluation_free_reserve_gib', 2)) * 1024 ** 3
    cap = float(config.get('evaluation_utilization_cap', .95))
    requested = config.get('evaluation_memory_utilization', 'auto')
    utilization = min(cap, (free - reserve) / total) if requested == 'auto' else float(requested)
    if not 0 < utilization <= 1:
        raise ValueError('Actual available VRAM cannot supply the configured evaluation reservation')
    limits = dict(max_num_seqs=int(config.get('evaluation_max_num_seqs', 1024)),
                  max_num_batched_tokens=int(config.get('evaluation_max_batched_tokens', 32768)))
    attempts = []
    while True:
        try:
            llm = factory(model=str(model), tokenizer=str(model), trust_remote_code=False,
                dtype='bfloat16', tensor_parallel_size=1, max_model_len=4096, seed=seed,
                gpu_memory_utilization=utilization, swap_space=0,
                enforce_eager=bool(config.get('evaluation_enforce_eager', False)),
                # Same native token-output commit semantics as training.
                enable_chunked_prefill=bool(config.get('rollout_enable_chunked_prefill', True)),
                enable_prefix_caching=True, disable_async_output_proc=True, disable_log_stats=True, **limits)
            return llm, utilization, limits, attempts
        except Exception as error:
            message = str(error)
            actual_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                isinstance(error, RuntimeError) and ('CUDA out of memory' in message or 'CUDA error: out of memory' in message))
            attempts.append(dict(gpu_memory_utilization=utilization, limits=dict(limits),
                error_type=type(error).__name__, actual_cuda_oom=actual_oom))
            if not actual_oom or len(attempts) >= 4:
                raise
            # The exception traceback otherwise retains its failed LLM/model
            # constructor frames and GPU tensors through the following retry.
            error.__traceback__ = None
            cleanup()
            utilization *= .85
            limits['max_num_seqs'] = max(32, limits['max_num_seqs'] // 2)
            limits['max_num_batched_tokens'] = max(4096, limits['max_num_batched_tokens'] // 2)


def evaluation_stream_snapshot(result, expected_request, terminal_ids, decode, abort_request):
    """Commit the same first-action token boundary used by native training.

    Copy metadata before abort releases/mutates the vLLM sequence. An unfinished
    partial action remains active; EOS ends the trajectory without a tool call.
    """
    if len(result.outputs) != 1 or list(result.prompt_token_ids) != expected_request:
        raise ValueError('Actual evaluation request/completion identity differs')
    native = result.outputs[0]
    raw = list(native.token_ids)
    if not result.finished and not stream_action_present(raw, decode):
        return None
    finish_reason, stop_reason = native.finish_reason, native.stop_reason
    cut = first_action_cut(raw, decode, terminal_ids=terminal_ids,
        native_stopped=finish_reason == 'stop')
    aborted = not result.finished
    snapshot = dict(raw_generated_token_ids=raw, finish_reason=finish_reason,
        stop_reason=stop_reason, native_action_boundary_abort=aborted, prefix_cut=cut)
    if aborted:
        abort_request(str(result.request_id))
    return snapshot


def worker(config_path, model_dir, output_dir, questions_path, expected_count, rank):
    from reproduction.prefix20_online.rollout import _configure_one_visible_gpu, prompt_tokens
    config = read(remote(config_path)); device = _configure_one_visible_gpu(rank)
    backend = config.get('rollout_attention_backend')
    if backend is not None:
        if backend != 'XFORMERS': raise ValueError('Unsupported configured native attention backend')
        os.environ['VLLM_ATTENTION_BACKEND'] = backend
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    model_dir, output = remote(model_dir), remote(output_dir)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    questions = load_questions(questions_path, expected_count)[rank::8]
    free, total = torch.cuda.mem_get_info()
    seed = int(config.get('seed', 1729)) + rank * 104729
    llm, utilization, limits, initialization_oom_attempts = build_evaluation_engine(
        config, model_dir, seed, free, total)
    engine = llm.llm_engine
    generation = read(model_dir / 'generation_config.json')
    stop = generation.get('eos_token_id', tokenizer.eos_token_id)
    stop = [stop] if isinstance(stop, int) else list(stop)
    terminal_ids = sorted(set(stop + [tokenizer.eos_token_id]))
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    params = SamplingParams(n=1, temperature=0., top_p=1., top_k=-1, max_tokens=500,
        stop_token_ids=stop, ignore_eos=False, logprobs=None, detokenize=False)
    recipe = GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500, num_gpus=8, search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe, is_validation=True)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(agent_workers=limits['max_num_seqs'],
        max_inflight_trajectories=limits['max_num_seqs'],
        retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    waiting = deque(dict(group=row, prompt=prompt_tokens(tokenizer, row['prompt']), response=[],
                         mask=[], turns=[], ordinal=0, retrieval_calls=0, valid_actions=0)
                    for row in questions)
    ready, pending, tools, finished = deque(), {}, {}, []
    active = 0; requests = 0; started = time.monotonic()
    budget = dict(BUDGET)
    output.mkdir(parents=True, exist_ok=True)
    full = output / ('rank-%d-full.jsonl.gz' % rank)
    if full.exists(): raise FileExistsError('Never overwrite an existing evaluation shard')
    pool = ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency)

    def finish_environment(state, environment, stream):
        nonlocal active
        text, done, valid, searched = environment
        cut = state.pop('cut'); kept = cut['kept_token_ids']
        final = state['final_turn']
        transition = interaction_outcome(tokenizer, len(state['request']), len(kept),
            final, cut, environment, budget)
        observation = transition['observation_token_ids']
        turn = state.pop('turn'); turn.update(observation_token_ids=observation,
            observation_clipped=transition['observation_clipped'],
            info_mask=[1] * len(kept) + [0] * len(observation), final_turn=transition['terminal'])
        state['turns'].append(turn); state['response'].extend(kept + observation)
        state['mask'].extend([1] * len(kept) + [0] * len(observation))
        state['retrieval_calls'] += transition['retrieval_calls']
        state['valid_actions'] += int(valid)
        if len(state['prompt']) + len(state['response']) > budget['total_tokens']:
            raise ValueError('Actual retained QA context exceeds shared4096-token budget')
        if not transition['terminal']:
            state['ordinal'] += 1; ready.append(state); return
        row = state['group']
        measured = score_tokens(tokenizer, state['prompt'], state['response'], state['mask'],
                                row['reward_model']['ground_truth']['target'])
        item = dict(qid=row['qid'], data_source=row['data_source'],
            evaluation_ordinal=row['evaluation_ordinal'], **measured,
            actual_retrieval_calls=state['retrieval_calls'], valid_actions=state['valid_actions'],
            turns=len(state['turns']), response_token_count=len(state['response']))
        stream.write(canonical(dict(item, initial_prompt_token_ids=state['prompt'],
            response_token_ids=state['response'], response_info_mask=state['mask'],
            behavior_turns=state['turns'])).decode() + '\n')
        finished.append(item); active -= 1
        if len(finished) % 16 == 0:
            stream.flush()
            atomic_json(output / ('rank-%d-progress.json' % rank), dict(rank=rank,
                completed_questions=len(finished), expected_questions=len(questions),
                active=active, pending_generation=len(pending), pending_retrieval=len(tools), utc=utc()), replace=True)

    try:
        with gzip.open(full, 'xt', encoding='utf8', compresslevel=1) as stream:
            while len(finished) < len(questions):
                while waiting and active < limits['max_num_seqs']:
                    ready.append(waiting.popleft()); active += 1
                while ready:
                    state = ready.popleft(); request = state['prompt'] + state['response']
                    state['final_turn'] = final_turn(len(request), state['ordinal'], budget)
                    rid = '%d-%d-%d' % (rank, state['group']['evaluation_ordinal'], state['ordinal'])
                    state['request'] = request; pending[rid] = state; requests += 1
                    engine.add_request(rid, {'prompt_token_ids': request}, params)
                if engine.has_unfinished_requests():
                    for result in engine.step():
                        rid = str(result.request_id)
                        if rid not in pending: continue
                        state = pending[rid]
                        snapshot = evaluation_stream_snapshot(result, state['request'], terminal_ids,
                            decode, engine.abort_request)
                        if snapshot is None: continue
                        pending.pop(rid)
                        cut = snapshot['prefix_cut']
                        state['cut'] = cut
                        state['turn'] = dict(snapshot, actual_request_token_ids=state['request'],
                            kept_token_ids=cut['kept_token_ids'], kept_token_count=cut['kept_token_count'],
                            ordinal=state['ordinal'], sampling=dict(temperature=0., top_p=1., top_k=-1, max_tokens=500, n=1))
                        if cut['native_terminal']:
                            finish_environment(state, ('', True, int(cut['action'] is not None), 0), stream)
                        elif cut['action'] == 'search' and not state['final_turn']:
                            tools[pool.submit(execute_one_action, manager, cut['execution_text'], True, retriever)] = state
                        else:
                            finish_environment(state, execute_one_action(manager, cut['execution_text'],
                                not state['final_turn'], retriever), stream)
                done_tools = [future for future in tools if future.done()]
                for future in done_tools: finish_environment(tools.pop(future), future.result(), stream)
                if not ready and not engine.has_unfinished_requests() and tools and not done_tools: time.sleep(.001)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    finished.sort(key=lambda row: row['evaluation_ordinal'])
    receipt = dict(schema_version=1, rank=rank, run_id=config['run_id'], actor_model_dir=str(model_dir),
        actor_version=config['actor_version'], evaluation_contract=CONTRACT, question_source_sha256=sha(questions_path),
        question_count=len(finished), qa_count=len(finished), rows=finished,
        full_output=dict(path=str(full), bytes=full.stat().st_size, sha256=sha(full)),
        device_binding=device, actual_compute_dtype=str(engine.model_config.dtype),
        actual_generation_requests=requests, seed=seed, gpu_memory_utilization=utilization,
        actual_scheduler_limits=limits, initialization_oom_attempts=initialization_oom_attempts,
        elapsed_seconds=time.monotonic() - started, completed_utc=utc())
    atomic_json(output / ('rank-%d-complete.json' % rank), receipt)
    return receipt


def launch(config_path, model_dir, output, questions_path, expected_count=512):
    config_path, model_dir, output, questions_path = map(remote, (config_path, model_dir, output, questions_path))
    config = read(config_path); target = output / 'complete.json'
    if target.exists():
        existing = read(target)
        if (existing['actor_version'] != config['actor_version'] or existing['qa_count'] != expected_count
                or existing['question_source_sha256'] != sha(questions_path)
                or existing.get('evaluation_contract') != CONTRACT):
            raise ValueError('Existing fixed evaluation receipt differs')
        return existing
    output.mkdir(parents=True, exist_ok=True); processes, streams = [], []
    try:
        for rank in range(8):
            if (output / ('rank-%d-complete.json' % rank)).exists(): continue
            stream = (output / ('rank-%d.log' % rank)).open('ab'); streams.append(stream)
            process = subprocess.Popen([sys.executable, '-m', 'reproduction.prefix_selection200.eval',
                '--config', str(config_path), '--model', str(model_dir), '--output', str(output),
                '--questions', str(questions_path), '--expected-count', str(expected_count), '--rank', str(rank)],
                stdout=stream, stderr=subprocess.STDOUT)
            processes.append((rank, process))
        while any(process.poll() is None for _, process in processes):
            failed = {rank: p.returncode for rank, p in processes if p.poll() not in (None, 0)}
            if failed: raise RuntimeError('Owned fixed evaluation worker failed: ' + json.dumps(failed))
            time.sleep(1)
        if any(p.returncode != 0 for _, p in processes): raise RuntimeError('Fixed evaluation worker failed')
    finally:
        for _, process in processes:
            if process.poll() is None: process.terminate()
        deadline = time.monotonic() + 60
        for _, process in processes:
            if process.poll() is None:
                try: process.wait(timeout=max(.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        for stream in streams: stream.close()
    receipts = [read(output / ('rank-%d-complete.json' % rank)) for rank in range(8)]
    rows = sorted([row for item in receipts for row in item['rows']], key=lambda row: row['evaluation_ordinal'])
    if ([row['evaluation_ordinal'] for row in rows] != list(range(expected_count))
            or len({row['qid'] for row in rows}) != expected_count
            or any(item['actor_version'] != config['actor_version'] or item['evaluation_contract'] != CONTRACT
                   or item['actual_compute_dtype'] != 'torch.bfloat16' for item in receipts)):
        raise ValueError('Fixed greedy shared-BF16 evaluation omitted, duplicated or changed a canonical question')
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
        completed_utc=utc(), actual_worker_exit_codes={str(r): p.returncode for r, p in processes})
    atomic_json(target, result); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--questions', required=True); parser.add_argument('--expected-count', type=int, default=512)
    parser.add_argument('--rank', type=int); args = parser.parse_args()
    if args.rank is None: launch(args.config, args.model, args.output, args.questions, args.expected_count)
    else: worker(args.config, args.model, args.output, args.questions, args.expected_count, args.rank)


if __name__ == '__main__': main()
