"""Eight independent frozen-official vLLM engines for one fresh probe epoch.

The entry point completes ONLY the requested epoch; the pipeline then encodes
its twenty retained policy positions and fits the head before requesting the
next epoch. It never updates the actor, generates the fixed diagnostic again,
precomputes future epochs, or transports trajectory/model bytes to the Mac.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .pool import atomic, canonical, digest, immutable, prepare, read

REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prompt_tokens(tokenizer, messages, max_start_length=2048):
    # Exactly RLHFDataset's official chat-template and add_special_tokens=False.
    # Native generation selects the last max_start_length original prompt IDs.
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tokenizer(text, add_special_tokens=False)['input_ids'])[-max_start_length:]


def strict_label(tokenizer, response_ids, info_mask, ground_truth):
    from search_g1.signals import decode_segments, contexts
    from verl.utils.reward_score.qa_em import em_check
    text, authored = decode_segments(tokenizer, response_ids, info_mask)
    context = contexts('', text, authored)
    return int(context is not None and em_check(context['answer'], ground_truth['target']))


def compact_prefix(turns, trajectory_id, limit=20):
    from reproduction.prefix_only_correctness_ab.features import retained_segments
    return retained_segments(turns, trajectory_id, limit)


def _configure_one_visible_gpu(rank):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    visible = [value for value in visible if value]
    if len(visible) != 8 or not 0 <= rank < 8:
        raise ValueError('Fresh rollout needs exactly eight actually assigned visible GPUs')
    # No distributed actor/optimizer group is needed for these eight TP1 engines.
    os.environ['CUDA_VISIBLE_DEVICES'] = visible[rank]
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(key, None)
    return visible[rank]


def worker(config_path, epoch, rank):
    config = read(config_path); selected_device = _configure_one_visible_gpu(rank)
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag
    from reproduction.common8.context_budget import final_stage, observation_room, observation_tokens
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from types import SimpleNamespace

    epoch_root = Path(config['base']) / 'epochs' / ('epoch-%04d' % epoch)
    bundle = read(epoch_root / 'questions.json'); selection = read(epoch_root / 'selection.json')
    if bundle['epoch'] != epoch or bundle['selection_sha256'] != digest(epoch_root / 'selection.json'):
        raise ValueError('Question bundle is not bound to the current epoch selection')
    if config['model_revision'] != REVISION:
        raise ValueError('Fresh teacher must be the unchanged official model revision')
    model = str(Path(config['model_dir']).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    free, total = torch.cuda.mem_get_info()
    reserve = int(config.get('rollout_free_reserve_gib', 8)) * 1024 ** 3
    requested = config.get('rollout_memory_utilization', 'auto')
    utilization = min(.85, (free - reserve) / total) if requested == 'auto' else float(requested)
    if not .1 <= utilization <= .95:
        raise ValueError('Actual free GPU memory cannot fit the frozen teacher with requested reserve')
    limits = dict(max_num_seqs=int(config.get('rollout_max_num_seqs', 512)),
                  max_num_batched_tokens=int(config.get('rollout_max_batched_tokens', 32768)))
    seed = int(config.get('seed', 1729)) + epoch * 1009 + rank * 104729
    llm = LLM(model=model, tokenizer=model, trust_remote_code=False, dtype='bfloat16',
              tensor_parallel_size=1, max_model_len=4096, seed=seed,
              gpu_memory_utilization=utilization, swap_space=0,
              enforce_eager=bool(config.get('rollout_enforce_eager', False)),
              enable_chunked_prefill=True, enable_prefix_caching=True,
              disable_log_stats=True, **limits)
    engine = llm.llm_engine
    generation_config = read(Path(model) / 'generation_config.json')
    stop = generation_config.get('eos_token_id', tokenizer.eos_token_id)
    stop = [stop] if isinstance(stop, int) else list(stop)
    # No PPO probability recomputation or logprob capture: labels depend only on
    # complete trajectory strict EM, and the actor remains immutable.
    params = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
                            max_tokens=500, stop_token_ids=stop, ignore_eos=False,
                            logprobs=None, detokenize=False)
    recipe = GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
                              max_response_length=500, max_obs_length=500, num_gpus=8,
                              search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(
        agent_workers=limits['max_num_seqs'], max_inflight_trajectories=limits['max_num_seqs'],
        retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    budget = dict(total_tokens=4096, action_tokens=500, answer_reserve_tokens=500)
    special = sorted(set(tokenizer.all_special_ids))
    snapshot = config['run_id'] + '-official-' + REVISION
    groups = bundle['rows'][rank::8]
    if len(groups) != 640:
        raise ValueError('Each of eight real engines needs exactly640 G5 groups')
    states = []
    for group in groups:
        prompt = prompt_tokens(tokenizer, group['prompt'])
        if not prompt:
            raise ValueError('Original official prompt is empty')
        for response_index in range(5):
            tid = '%s-epoch%04d-%s-g%d' % (config['run_id'], epoch,
                    hashlib.sha256(group['qid'].encode()).hexdigest()[:24], response_index)
            states.append(dict(trajectory_id=tid, group=group, response_index=response_index,
                prompt=prompt, response=[], info_mask=[], turns=[], ordinal=0, retrieval_calls=0))
    ready = deque(); waiting = deque(states); pending = {}; tools = {}
    completed = 0; correct = 0; active = 0; peak = 0; requests_count = 0; policy_tokens = 0
    output = epoch_root / 'rollout'; output.mkdir(exist_ok=True)
    full_path = output / ('rank-%d-full.jsonl.gz' % rank)
    prefix_path = output / ('rank-%d-prefix20.jsonl.gz' % rank)
    progress_path = output / ('rank-%d-progress.json' % rank)
    if full_path.exists() or prefix_path.exists():
        raise ValueError('Unfinished or already completed rollout shard exists; inspect before any replay')
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency,
                              thread_name_prefix='prefix20-retrieval')

    def publish_progress(stage):
        atomic(progress_path, dict(stage=stage, epoch=epoch, rank=rank, completed_trajectories=completed,
            expected_trajectories=3200, correct_trajectories=correct, active_trajectories=active, pending_generation=len(pending),
            pending_retrieval=len(tools), peak_inflight=peak, generation_requests=requests_count,
            elapsed_seconds=time.monotonic()-started, utc=utc(), device=selected_device,
            gpu_memory_utilization=utilization, scheduler_limits=limits, seed=seed))

    def finish_environment(state, environment, full_stream, prefix_stream):
        nonlocal active, completed, correct
        text, done, valid, is_search = environment
        kept = state.pop('cut')['kept_token_ids']; request = state['request']
        room = observation_room(len(request), len(kept), budget)
        observation, clipped = ([], False) if state['final_turn'] else observation_tokens(
            tokenizer, text, min(500, room))
        turn = state.pop('raw_turn')
        turn.update(observation_token_ids=observation if is_search else [],
                    correction_token_ids=observation if not is_search else [],
                    final_turn=bool(state['final_turn'] or done),
                    observation_clipped=clipped)
        state['turns'].append(turn)
        state['response'].extend(kept + observation)
        state['info_mask'].extend([1] * len(kept) + [0] * len(observation))
        state['retrieval_calls'] += int(is_search and not state['final_turn'])
        if len(state['prompt']) + len(state['response']) > 4096:
            raise ValueError('Actual retained context exceeds the original4096token budget')
        if not state['final_turn'] and not done:
            state['ordinal'] += 1; ready.append(state); return
        label = strict_label(tokenizer, state['response'], state['info_mask'], state['group']['reward_model']['ground_truth'])
        group = state['group']; segments, ids, positions = compact_prefix(state['turns'], state['trajectory_id'], 20)
        if not ids:
            raise ValueError('A generated trajectory contains no retained policy token')
        common = dict(schema_version=1, epoch=epoch, split='train',
            trajectory_id=state['trajectory_id'], qid=group['qid'], question_hash=group['question_hash'],
            normalized_question_hash=group['normalized_question_hash'], source_index=group['source_index'],
            pool_cycle=group['pool_cycle'], response_index=state['response_index'], strict_em=label,
            policy_snapshot_id=snapshot, model_revision=REVISION, rollout_seed=seed)
        full_record = dict(common, behavior_turns=state['turns'], response_token_ids=state['response'],
                           response_info_mask=state['info_mask'], retrieval_calls=state['retrieval_calls'])
        full_stream.write(canonical(full_record).decode() + '\n')
        prefix_record = dict(common, prefix_segments=segments, policy_token_ids=ids,
                             policy_position_ids=positions, original_retained_policy_length=sum(
                                 turn['kept_token_count'] for turn in state['turns']),
                             first20_feature_context_includes_future_tokens=False,
                             full_trajectory_record_sha256=hashlib.sha256(canonical(full_record)).hexdigest())
        prefix_stream.write(canonical(prefix_record).decode() + '\n')
        active -= 1; completed += 1; correct += label
        if completed % 100 == 0:
            full_stream.flush(); prefix_stream.flush(); publish_progress('FRESH_ROLLOUT_ADVANCING')

    try:
        with gzip.open(full_path, 'xt', encoding='utf8', compresslevel=1) as full_stream, \
             gzip.open(prefix_path, 'xt', encoding='utf8', compresslevel=1) as prefix_stream:
            publish_progress('OFFICIAL_ENGINE_READY')
            while completed < len(states):
                while waiting and active < limits['max_num_seqs']:
                    ready.append(waiting.popleft()); active += 1
                peak = max(peak, active)
                while ready:
                    state = ready.popleft(); request = state['prompt'] + state['response']
                    state['final_turn'] = final_stage(len(request), state['ordinal'], 4, budget)
                    if len(request) + 500 > 4096:
                        raise ValueError('Original trajectory context cannot fit the next500token action')
                    rid = '%s-request%d' % (state['trajectory_id'], state['ordinal'])
                    state['request'] = request
                    pending[rid] = state; requests_count += 1
                    engine.add_request(rid, {'prompt_token_ids': request}, params)
                if engine.has_unfinished_requests():
                    for result in engine.step():
                        if not result.finished:
                            continue
                        rid = str(result.request_id); state = pending.pop(rid)
                        if len(result.outputs) != 1 or list(result.prompt_token_ids) != state['request']:
                            raise ValueError('Actual native completion identity or count changed')
                        native = result.outputs[0]; raw = list(native.token_ids); policy_tokens += len(raw)
                        cut = raw_prefix_to_first_closing_tag(raw,
                            lambda tokens: tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                            special_token_ids=special,
                            executor_decode=lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                            effective_stop_token_ids=stop)
                        state['cut'] = cut
                        state['raw_turn'] = dict(trajectory_id=state['trajectory_id'], turn_index=state['ordinal'],
                            request_id=rid, actual_request_token_ids=state['request'],
                            raw_generated_token_ids=raw, raw_generated_token_count=len(raw),
                            kept_token_count=cut['kept_token_count'], prefix_cut=cut,
                            finish_reason=native.finish_reason, stop_reason=native.stop_reason,
                            policy_snapshot_id=snapshot)
                        action, _ = manager.postprocess_predictions([cut['execution_text']])
                        if action[0] == 'search' and not state['final_turn']:
                            future = pool.submit(execute_one_action, manager, cut['execution_text'], True, retriever)
                            tools[future] = state
                        else:
                            finish_environment(state, execute_one_action(manager, cut['execution_text'],
                                not state['final_turn'], retriever), full_stream, prefix_stream)
                done_tools = [future for future in tools if future.done()]
                for future in done_tools:
                    finish_environment(tools.pop(future), future.result(), full_stream, prefix_stream)
                if not ready and not engine.has_unfinished_requests() and tools and not done_tools:
                    time.sleep(.001)
            if pending or tools or waiting or ready:
                raise ValueError('Fresh epoch has incomplete real trajectories')
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for path in (full_path, prefix_path):
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    result = dict(schema_version=1, epoch=epoch, rank=rank, questions=640, trajectories=completed,
        device=selected_device, model_revision=REVISION, frozen_actor_updates=0,
        strict_correct_trajectories=correct, strict_incorrect_trajectories=completed-correct,
        source_model_dir=model, seed=seed, pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id), selection_sha256=digest(epoch_root/'selection.json'),
        questions_sha256=digest(epoch_root/'questions.json'),
        prefix=dict(path=str(prefix_path), sha256=digest(prefix_path), bytes=prefix_path.stat().st_size, count=completed),
        full=dict(path=str(full_path), sha256=digest(full_path), bytes=full_path.stat().st_size, count=completed),
        actual_scheduler_limits={key:getattr(engine.scheduler_config, key) for key in limits},
        gpu_memory_utilization=utilization, initial_free_gpu_bytes=free, initial_total_gpu_bytes=total,
        retained_head_prefix_tokens=20, raw_sampled_tokens=policy_tokens, generation_requests=requests_count,
        peak_inflight_trajectories=peak, elapsed_seconds=time.monotonic()-started, completed_utc=utc(),
        sampling=dict(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=500, n=1,
                      ignore_eos=False, stop_token_ids=stop, logprobs=None),
        context=dict(max_start_length=2048, max_prompt_length=4096, max_response_length=500,
                     max_obs_length=500, max_turns=4), same_original_prompt_chat_template=True)
    immutable(output/('rank-%d-complete.json' % rank), result)
    publish_progress('FRESH_ROLLOUT_RANK_COMPLETE')
    print(json.dumps(result), flush=True)
    return result


def launch_epoch(config_path, epoch):
    config = read(config_path)
    if epoch < 2:
        return prepare(config, epoch)
    proof = prepare(config, epoch); epoch_root = Path(proof['selection_path']).parent
    target = epoch_root/'complete.json'
    if target.exists():
        saved = read(target)
        if saved['epoch'] != epoch or saved['selection_sha256'] != proof['selection_sha256']:
            raise ValueError('Existing fresh epoch receipt differs')
        return saved
    rollout = epoch_root/'rollout'; rollout.mkdir(exist_ok=True)
    processes = []; streams = []
    try:
        for rank in range(8):
            receipt = rollout/('rank-%d-complete.json' % rank)
            if receipt.exists():
                saved = read(receipt)
                if saved['selection_sha256'] != proof['selection_sha256'] or saved['trajectories'] != 3200:
                    raise ValueError('Existing shard belongs to a different selection')
                for key in ('prefix', 'full'):
                    item=saved[key]
                    if digest(item['path']) != item['sha256'] or Path(item['path']).stat().st_size != item['bytes']:
                        raise ValueError('Existing completed rollout shard changed')
                continue
            stream = (rollout/('rank-%d.log' % rank)).open('ab'); streams.append(stream)
            command = [sys.executable, '-m', 'reproduction.prefix20_online.rollout', '--config', str(config_path),
                       '--epoch', str(epoch), '--rank', str(rank)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((rank, process))
        while any(process.poll() is None for _, process in processes):
            failed = {rank: process.returncode for rank, process in processes
                      if process.poll() not in (None, 0)}
            if failed:
                raise RuntimeError('Actual frozen rollout worker failed: ' + json.dumps(failed))
            time.sleep(1)
        codes = {rank: process.returncode for rank, process in processes}
        if any(code != 0 for code in codes.values()):
            raise RuntimeError('Actual frozen rollout worker failed: ' + json.dumps(codes))
    finally:
        # This entry point owns only these children. Never signal other jobs or
        # leave own vLLM allocations behind after a partial worker failure.
        for rank, process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 60
        for rank, process in processes:
            if process.poll() is None:
                try: process.wait(timeout=max(.01, deadline-time.monotonic()))
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        for stream in streams:
            stream.close()
    ranks = [read(rollout/('rank-%d-complete.json' % rank)) for rank in range(8)]
    pad_tokens = {row['pad_token_id'] for row in ranks}
    if len(pad_tokens) != 1 or any(type(value) is not int or value < 0 for value in pad_tokens):
        raise ValueError('Actual official tokenizer pad IDs differ across rollout engines')
    pad_token_id = next(iter(pad_tokens))
    result = dict(schema_version=1, role='fresh_official_5120_G5_prefix20_epoch', epoch=epoch,
        questions=5120, trajectories=sum(row['trajectories'] for row in ranks),
        model_revision=config['model_revision'], model_files_sha256=config['model_files_sha256'],
        model_dir=config['model_dir'], pad_token_id=pad_token_id,
        run_id=config['run_id'], source_commit=config.get('source_commit'),
        selection_sha256=proof['selection_sha256'], questions_sha256=proof['questions_sha256'],
        questions_path=proof['questions_path'], pool_sha256=proof['pool_sha256'],
        canonical_train_sha256=config['canonical_train_sha256'],
        fixed_test_qids_sha256=proof['fixed_test_qids_sha256'],
        fixed_test_normalized_text_sha256=proof['fixed_test_normalized_text_sha256'],
        shards=[row['prefix'] for row in ranks], raw_shards=[row['full'] for row in ranks],
        ranks=ranks, frozen_actor_updates=0, fixed_test_regenerated=False,
        head_prefix_tokens=20, completed_utc=utc(),
        question_repeat_policy='without_replacement_until_pool_exhausted_then_reshuffle',
        trajectory_repeat_policy='new_complete_rollout_every_epoch_from2',
        next_epoch_precomputed=False)
    if result['trajectories'] != 25600:
        raise ValueError('Exactly5120completeG5groups are required')
    immutable(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='ranks'}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True);parser.add_argument('--epoch',required=True,type=int)
    parser.add_argument('--rank',type=int); args=parser.parse_args()
    if args.rank is None: launch_epoch(Path(args.config).resolve(),args.epoch)
    else: worker(Path(args.config).resolve(),args.epoch,args.rank)


if __name__=='__main__': main()
