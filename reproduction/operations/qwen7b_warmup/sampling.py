"""Native reference sampling worker for independent full-data 7B warmup.

Interaction, first-action execution, hidden flush, same-trajectory continuation
and strict-EM logic are copied from the pinned reference worker. New adapters
supply two admitted GPUs, complete unselected G5 and durable bounded batches.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import atexit
from reproduction.prefix20_online.pool import atomic, canonical, digest, immutable, read
from reproduction.prefix_selection200.rollout import (root_dir, phase_output, initial_state,
    retained_length, next_request, prefix_record as reference_prefix_record, discard_hidden_lookahead, file_item,
    _build_llm as reference_build_llm, iter_rows, seed_for)
from reproduction.prefix20_online.rollout import strict_label, utc
from reproduction.prefix_selection200.interaction_protocol import (ACTUAL_COMPUTE_DTYPE,
    ACTION_TOKENS, BOUNDARY, BUDGET, COMPUTE_DTYPE, CONTRACT as INTERACTION_CONTRACT,
    MAX_OBSERVATION_TOKENS, MAX_SEARCH_TURNS, MAX_START_TOKENS, append_chunk,
    finish_action, native_snapshot, registered_sampling, start_action,
    stream_action_present, terminal_token_ids)
from reproduction.prefix_selection200 import prefix_diversity
from reproduction.prefix_selection200.vllm_ports import bounded_vllm_port_namespace as reference_namespace
from reproduction.prefix_selection200.contract import G, PREFIX_TOKENS, SAMPLING_CONTRACT

WORLD = 1
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
CONTRACT = SAMPLING_CONTRACT
_CACHED = {}


def prefix_record(state):
    row=reference_prefix_record(state)
    row['split']=state['group']['split']
    row.pop('prefix_sha256')
    row['prefix_sha256']=hashlib.sha256(canonical(row)).hexdigest()
    return row


def _configure_one_visible_gpu(rank):
    if 'device_binding' in _CACHED:
        return _CACHED['device_binding']
    import csv, io
    visible = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x]
    assert len(visible) == 2 and len(set(visible)) == 2 and rank == 0
    rows = list(csv.reader(io.StringIO(subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True))))
    mapping = {row[0].strip():row[1].strip() for row in rows}
    from reproduction.common8.native_allocation import cuda_uuids
    uid = cuda_uuids(2)[0]
    token = visible[0]
    assert uid in mapping.values()
    index = next(key for key,value in mapping.items() if value == uid)
    # Keep Slurm's two-device CUDA namespace unchanged. TP1 uses CUDA ordinal0;
    # the separate retriever child sees only the second assigned device.
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        os.environ.pop(key,None)
    result = dict(physical_gpu_index=int(index),gpu_uuid=uid,originally_assigned_device=token)
    _CACHED['device_binding'] = result
    return result


@contextmanager
def bounded_vllm_port_namespace(config, rank, *, outer, phase):
    if 'namespace' in _CACHED:
        yield _CACHED['namespace']
    else:
        lease = reference_namespace(config,rank,outer=outer,phase=phase)
        namespace = lease.__enter__()
        _CACHED.update(namespace=namespace, namespace_lease=lease)
        atexit.register(lambda: lease.__exit__(None,None,None))
        yield namespace


def _build_llm(config, model, seed, free, total):
    if 'llm' not in _CACHED:
        # Eager from birth permits authentic native hidden capture in later batches.
        _CACHED['llm'] = reference_build_llm(dict(config,_capture_actor_hidden=True),model,seed,free,total)
    return _CACHED['llm']


def validate_rank_receipt(config_path, config, outer, phase, rank, path):
    row = read(path)
    assert row['run_id'] == config['run_id'] and row['actor_version'] == config['actor_version']
    assert row['config_sha256'] == digest(config_path) and row['phase'] == phase and row['rank'] == rank
    expected = len(read(root_dir(config,outer)/'questions.json')['rows'][rank::WORLD])*5
    assert row['trajectories'] == expected
    for key in (('states','prefix','actor_hidden') if phase == 'prefix' else ('full','own_em')):
        item = row[key]
        assert Path(item['path']).stat().st_size == item['bytes'] and digest(item['path']) == item['sha256']
    return row, None


def _states(config, outer, rank, phase, tokenizer, selection_path):
    root = root_dir(config,outer)
    groups = read(root/'questions.json')['rows'][rank::WORLD]
    if phase == 'prefix':
        return [initial_state(config,outer,row,j,tokenizer) for row in groups for j in range(5)]
    prefix = read(root/'prefix/complete.json')
    shard = prefix['ranks'][rank]['states']
    assert digest(shard['path']) == shard['sha256']
    states = list(iter_rows(shard['path']))
    expected = {str(group['qid']) for group in groups}
    assert len(states) == len(groups)*5 and {str(s['group']['qid']) for s in states} == expected
    for state in states:
        state['prefix_scan_sha256'] = prefix['scan_sha256']
        state['selection_sha256'] = digest(selection_path)
    return states

def worker(config_path, outer, phase, rank, selection_path=None):
    config = read(config_path)
    assert config["model_revision"] == REVISION and config["warmup_only"] is True
    root = root_dir(config, outer)
    output = phase_output(config,outer,phase)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / ('rank-%d-complete.json' % rank)
    if receipt_path.exists():
        saved,_reuse=validate_rank_receipt(config_path,config,outer,phase,rank,receipt_path)
        return saved
    device_binding = _configure_one_visible_gpu(rank)
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    model = str(Path(config['actor_model_dir']).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    selection_path = Path(selection_path or root / 'probe-score' / 'selection.json')
    states = _states(config, outer, rank, phase, tokenizer, selection_path)
    free, total = torch.cuda.mem_get_info()
    seed = seed_for(config['run_id'], outer, 'engine', 0, rank, config.get('seed', 1729),
                    sampling_attempt_id=config.get('sampling_attempt_id'))
    capture_hidden = phase == 'prefix' and config.get('actor_hidden_native', False)
    diverse = phase=='prefix' and prefix_diversity.enabled(config)
    with bounded_vllm_port_namespace(config, rank, outer=outer, phase=phase) as port_namespace:
        print('VLLM_PORT_NAMESPACE=' + json.dumps(port_namespace.receipt(), sort_keys=True), flush=True)
        llm, utilization, limits, oom_attempts = _build_llm(dict(config, _capture_actor_hidden=capture_hidden), model, seed, free, total)
    engine = llm.llm_engine
    if str(engine.model_config.dtype) != ACTUAL_COMPUTE_DTYPE:
        raise ValueError('Actual rollout engine must use the shared BF16 train/QA protocol')
    hidden_capture = hidden_writer = None
    if capture_hidden:
        from .native_capture import ActorHiddenCapture
        from .packed_features import NativeFeatureWriter
        hidden_capture = ActorHiddenCapture(engine, config['actor_version'])
        hidden_writer = NativeFeatureWriter(output, rank, [row['trajectory_id'] for row in states],
            selected_capacity=len(states)//2 if diverse else None)
    hidden_flushes = {}
    generation_path = Path(model) / 'generation_config.json'
    generation = read(generation_path) if generation_path.exists() else {}
    stop, terminal_ids = terminal_token_ids(generation, tokenizer)
    recipe = GenerationConfig(max_turns=MAX_SEARCH_TURNS, max_start_length=MAX_START_TOKENS,
                              max_prompt_length=BUDGET['total_tokens'], max_response_length=ACTION_TOKENS,
                              max_obs_length=MAX_OBSERVATION_TOKENS, num_gpus=WORLD,
                              search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(agent_workers=limits['max_num_seqs'],
        max_inflight_trajectories=limits['max_num_seqs'], retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    budget = dict(BUDGET)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    waiting, ready, pending, tools = deque(states), deque(), {}, {}
    completed, active, peak, requests, sampled = 0, 0, 0, 0, 0
    retained=0;diverse_pending={};diverse_proofs=[]
    diversity_path=output/('rank-%d-prefix-diversity.jsonl'%rank)
    diversity_stream=diversity_path.open('x') if diverse else None
    started = time.monotonic()
    progress_path = output / ('rank-%d-progress.json' % rank)
    names = ('states.jsonl.gz', 'prefix.jsonl.gz') if phase == 'prefix' else ('full.jsonl.gz', 'own-em.jsonl')
    paths = [output / ('rank-%d-%s' % (rank, name)) for name in names]
    if any(path.exists() for path in paths):
        raise ValueError('Uncommitted phase shard exists: preserve and reconcile it before retrying')
    pool = ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency, thread_name_prefix='prefix-selection-tools')

    def progress(stage):
        atomic(progress_path, dict(stage=stage, phase=phase, outer=outer, rank=rank, utc=utc(),
            actor_version=config['actor_version'], completed_trajectories=completed, expected_trajectories=len(states),
            retained_prefixes=retained if phase=='prefix' else None,prefix_candidates_per_question=prefix_diversity.draws(config) if phase=='prefix' else None,
            active_trajectories=active, pending_generation=len(pending), pending_retrieval=len(tools),
            peak_inflight_trajectories=peak, generation_requests=requests, actual_sampled_tokens=sampled,
            elapsed_seconds=time.monotonic()-started, device_binding=device_binding,
            actual_engine_enforce_eager=bool(engine.model_config.enforce_eager),
            actual_engine_enable_prefix_caching=bool(engine.cache_config.enable_prefix_caching),
            gpu_memory_utilization=utilization, scheduler_limits=limits, initialization_oom_attempts=oom_attempts))

    def save(state, first_stream, second_stream):
        nonlocal completed, active, retained
        if phase == 'prefix':
            row = prefix_record(state)
            native_features = hidden_capture.finish(row) if hidden_capture is not None else None
            if diverse:
                siblings=diverse_pending.setdefault(str(row['qid']),[]);siblings.append((state,row,native_features))
                if len(siblings)<10:
                    completed+=1;active-=1
                    return
                siblings.sort(key=lambda item:item[1]['response_index'])
                selected,proof=prefix_diversity.select([item[1]for item in siblings])
                proof.update(qid=row['qid'],outer=outer,rank=rank)
                diversity_stream.write(canonical(proof).decode()+'\n');diverse_proofs.append(proof)
                accepted=[]
                for slot,index in enumerate(selected):
                    member,original,features=siblings[index]
                    member.update(sampled_response_index=member['response_index'],response_index=slot,
                        diversity_selection_sha256=proof['selection_sha256'])
                    accepted.append((member,prefix_record(member),features))
                del diverse_pending[str(row['qid'])]
            else:accepted=[(state,row,native_features)]
            for member,prefix,features in accepted:
                if hidden_capture is not None:prefix.update(hidden_writer.write(prefix,features),feature_rank=rank)
                member['prefix_sha256']=prefix['prefix_sha256'];retained+=1
                first_stream.write(canonical(member).decode()+'\n');second_stream.write(canonical(prefix).decode()+'\n')
        else:
            group = state['group']
            label = strict_label(tokenizer, state['response'], state['info_mask'], group['reward_model']['ground_truth'])
            row = dict(schema_version=1, run_id=config['run_id'], outer=outer, actor_version=config['actor_version'],
                policy_snapshot_id=config['actor_version'], trajectory_id=state['trajectory_id'], qid=group['qid'],
                question_hash=group['question_hash'], split=group['split'], candidate_ordinal=state['candidate_ordinal'],
                response_index=state['response_index'], initial_prompt_token_ids=state['prompt'],
                response_token_ids=state['response'], response_info_mask=state['info_mask'],
                response_log_probs=state['log_probs'], behavior_turns=state['turns'],
                sampling_journal=[chunk for turn in state['turns'] for chunk in turn['actual_sampling_chunks']],
                sampling_contract=CONTRACT, action_boundary=BOUNDARY, interaction_protocol=INTERACTION_CONTRACT, strict_em=label,
                reward_model=group['reward_model'], data_source=group['data_source'],
                extra_info=group.get('extra_info', {}), prefix_sha256=state['prefix_sha256'],
                prefix_scan_sha256=state['prefix_scan_sha256'], selection_sha256=state['selection_sha256'],
                retrieval_calls=state['retrieval_calls'], terminal_reason=state['terminal_reason'])
            first_stream.write(canonical(row).decode() + '\n')
            second_stream.write(canonical({key: row[key] for key in ('trajectory_id', 'qid', 'strict_em',
                'actor_version', 'prefix_sha256', 'prefix_scan_sha256', 'selection_sha256')}).decode() + '\n')
        completed += 1; active -= 1
        if completed % 100 == 0:
            first_stream.flush(); second_stream.flush(); progress('NATIVE_PHASE_ADVANCING')

    def schedule(state, streams):
        if state['terminal'] or (phase == 'prefix' and retained_length(state) >= PREFIX_TOKENS):
            save(state, *streams)
        else:
            ready.append(state)

    def environment_done(state, cut, result, streams):
        finish_action(state, cut, result, tokenizer, budget)
        schedule(state, streams)

    try:
        opener2 = gzip.open if phase == 'prefix' else open
        with gzip.open(paths[0], 'xt', encoding='utf8', compresslevel=1) as first_stream, \
             opener2(paths[1], 'xt', encoding='utf8') as second_stream:
            streams = first_stream, second_stream
            progress('NATIVE_ENGINE_READY')
            while completed < len(states):
                while waiting and active < limits['max_num_seqs']:
                    state = waiting.popleft(); active += 1; schedule(state, streams)
                peak = max(peak, active)
                while ready:
                    state = ready.popleft(); start_action(state, budget)
                    request = next_request(state, phase, config)
                    if request is None:
                        raise ValueError('Unfinished logical action has no available native quota')
                    params = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
                        max_tokens=request['max_tokens'] + int(capture_hidden), stop=[], stop_token_ids=stop, ignore_eos=False,
                        logprobs=1, detokenize=False, seed=request['seed'])
                    engine.add_request(request['request_id'], {'prompt_token_ids': request['actual_request_token_ids']}, params)
                    sampling = registered_sampling(engine, request, params)
                    if hidden_capture is not None:
                        hidden_capture.register(state, request)
                    pending[request['request_id']] = state, request, sampling
                    requests += 1
                if engine.has_unfinished_requests():
                    for result in engine.step():
                        rid = str(result.request_id)
                        if rid not in pending:
                            continue
                        state, request, sampling = pending[rid]
                        if len(result.outputs) != 1 or list(result.prompt_token_ids) != request['actual_request_token_ids']:
                            raise ValueError('Actual native output request identity changed')
                        native = result.outputs[0]; raw = list(native.token_ids)
                        cumulative = state['action']['raw_generated_token_ids'] + raw
                        seen_action = stream_action_present(cumulative, decode)
                        # Processing the final retained token requires one more
                        # native forward than sampling it. Freeze the boundary
                        # snapshot, run one decode step, and discard its sampled
                        # lookahead. That token never enters the state or head.
                        flushing = rid in hidden_flushes
                        if capture_hidden and not flushing and not result.finished and (
                                seen_action or len(raw) >= request['max_tokens']):
                            hidden_flushes[rid] = native_snapshot(native)
                            continue
                        aborted = bool(not result.finished and (seen_action or flushing))
                        if not flushing and not result.finished and not seen_action:
                            continue
                        actual_snapshot = native_snapshot(native)
                        snapshot = actual_snapshot
                        if flushing:
                            snapshot, flush_receipt = discard_hidden_lookahead(hidden_flushes.pop(rid), actual_snapshot)
                            sampling = dict(sampling, actor_hidden_flush=flush_receipt)
                        if aborted:
                            engine.abort_request(rid)
                        if hidden_capture is not None:
                            hidden_capture.forget_request(rid)
                        pending.pop(rid); sampled += len(actual_snapshot['raw_generated_token_ids'])
                        cut = append_chunk(state, request, snapshot['raw_generated_token_ids'], snapshot['selected_logprobs'], sampling,
                            finish_reason=snapshot['finish_reason'],
                            stop_reason=snapshot['stop_reason'], terminal_ids=terminal_ids, decode=decode,
                            action_boundary_abort=aborted)
                        if cut is None:
                            schedule(state, streams)
                        elif cut['native_terminal']:
                            environment_done(state, cut, ('', True, int(cut['action'] is not None), 0), streams)
                        elif cut['action'] == 'search' and not state['action']['final_turn']:
                            future = pool.submit(execute_one_action, manager, cut['execution_text'], True, retriever)
                            tools[future] = state, cut
                        else:
                            environment_done(state, cut, execute_one_action(manager, cut['execution_text'],
                                             not state['action']['final_turn'], retriever), streams)
                done = [future for future in tools if future.done()]
                for future in done:
                    state, cut = tools.pop(future)
                    environment_done(state, cut, future.result(), streams)
                if not ready and not engine.has_unfinished_requests() and tools and not done:
                    time.sleep(.001)
            if waiting or ready or pending or tools:
                raise ValueError('Completed native phase leaves incomplete own trajectories')
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if hidden_capture is not None:
            hidden_capture.close()
        if diversity_stream is not None:
            diversity_stream.flush();os.fsync(diversity_stream.fileno());diversity_stream.close()
    if diverse_pending:raise ValueError('Incomplete ten-prefix sibling group')
    feature_path = hidden_writer.finish() if hidden_writer is not None else None
    result = dict(schema_version=1, phase=phase, outer=outer, rank=rank, run_id=config['run_id'],
        actor_version=config['actor_version'], actor_model_dir=model, config_sha256=digest(config_path),
        sampling_contract=CONTRACT, action_boundary=BOUNDARY, interaction_protocol=INTERACTION_CONTRACT,
        actual_compute_dtype=str(engine.model_config.dtype), sampling_attempt_id=config.get('sampling_attempt_id'),
        vllm_port_namespace=port_namespace.receipt(),
        native_output_commit='synchronous_token_ids_and_selected_logprobs_snapshot_before_action_abort',
        trajectories=retained if phase=='prefix' else completed, questions=(retained if phase=='prefix' else completed)//G, device_binding=device_binding,
        pad_token_id=int(tokenizer.pad_token_id), eos_token_id=int(tokenizer.eos_token_id),
        native_terminal_ids=terminal_ids, gpu_memory_utilization=utilization,
        actual_engine_enforce_eager=bool(engine.model_config.enforce_eager),
        actual_engine_enable_prefix_caching=bool(engine.cache_config.enable_prefix_caching),
        initial_free_gpu_bytes=free, initial_total_gpu_bytes=total, initialization_oom_attempts=oom_attempts,
        scheduler_limits=limits, engine_seed=seed, actual_sampled_tokens=sampled, generation_requests=requests,
        elapsed_seconds=time.monotonic()-started, completed_utc=utc())
    for key, path in zip(('states', 'prefix') if phase == 'prefix' else ('full', 'own_em'), paths):
        result[key] = file_item(path, retained if phase=='prefix' else completed)
    if feature_path is not None:
        result.update(actor_hidden=file_item(feature_path, retained),
            actor_hidden_layer=26, actor_hidden_boundary='post_block_hidden_plus_residual',
            actor_hidden_forward_calls=hidden_capture.calls, actor_hidden_captured_rows=hidden_capture.captured_rows,
            actor_hidden_forward_and_capture_seconds=hidden_capture.elapsed_seconds,
            independent_encoder_forward=False, native_hidden_flush=True)
    if diverse:
        if retained!=len(states)//2 or completed!=len(states):raise ValueError('Ten-to-five prefix accounting differs')
        result.update(prefix_candidates_generated=completed,prefixes_discarded=completed-retained,
            prefix_diversity=file_item(diversity_path,len(diverse_proofs)),
            diversity_statistics=dict(questions=len(diverse_proofs),
                mean_unique_before=sum(p['unique_input_prefixes']for p in diverse_proofs)/len(diverse_proofs),
                mean_unique_after=sum(p['unique_retained_prefixes']for p in diverse_proofs)/len(diverse_proofs),
                mean_pairwise_before=sum(p['mean_pairwise_distance_before']for p in diverse_proofs)/len(diverse_proofs),
                mean_pairwise_after=sum(p['mean_pairwise_distance_after']for p in diverse_proofs)/len(diverse_proofs),
                probe_evaluated_prefixes=retained,discarded_probe_evaluations=0))
    if phase == 'continue':
        result.update(selection_sha256=digest(selection_path), prefix_scan_sha256=read(root/'prefix'/'complete.json')['scan_sha256'])
    immutable(receipt_path, result); progress('NATIVE_RANK_PHASE_COMPLETE')
    return result
