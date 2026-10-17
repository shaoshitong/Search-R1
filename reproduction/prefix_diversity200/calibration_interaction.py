"""Scheduled reward calibration through the actual train/QA state machine."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from .interaction import BUDGET


def provenance(state, tokenizer):
    from search_g1.signals import decode_segments
    from search_g1.calibration_evidence import validate_provenance
    blocks = []
    for turn in state['turns']:
        pieces = [('model', turn['prefix_cut']['kept_token_ids']),
                  ('evidence', turn['observation_token_ids']),
                  ('correction', turn['correction_token_ids'])]
        for role, ids in pieces:
            if not ids:
                continue
            if blocks and blocks[-1][0] == role:
                blocks[-1][1].extend(ids)
            else:
                blocks.append((role, list(ids)))
    text = ''; spans = dict(model=[], evidence=[], correction=[])
    for role, ids in blocks:
        begin = len(text); text += tokenizer.decode(ids)
        spans[role].append([begin, len(text)])
    actual_text, actual_model = decode_segments(tokenizer, state['response'], state['info_mask'])
    if text != actual_text or spans['model'] != actual_model:
        raise ValueError('Calibration text provenance differs from actual retained policy/environment segments')
    validate_provenance(text, spans['model'], spans['evidence'], spans['correction'])
    return text, spans['model'], state['retrieval_calls'], spans['evidence'], spans['correction']


def rollouts(llm, prompts, tokenizer, sample, config):
    from vllm import SamplingParams
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from .rollout import (initial_state_from_tokens, start_action, next_request, append_chunk,
        finish_action, stream_action_present, native_snapshot, registered_sampling)
    engine = llm.llm_engine
    if engine.model_config.max_model_len != BUDGET['total_tokens']:
        raise ValueError('Calibration engine must use the shared4096 context')
    generation = json.loads((Path(config['calibration_model'])/'generation_config.json').read_text())
    stops = generation.get('eos_token_id', tokenizer.eos_token_id)
    stops = [stops] if isinstance(stops, int) else list(stops)
    terminals = sorted(set(stops + [tokenizer.eos_token_id]))
    maximum = int(engine.scheduler_config.max_num_seqs)
    recipe = GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500, num_gpus=1, search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe, is_validation=not sample)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(agent_workers=maximum,
        max_inflight_trajectories=maximum,
        retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    cfg = dict(config, run_id=config['run_id']+'-reward-calibration')
    group_size = int(config['calibration_responses_per_question'])
    outer = int(config['calibration_outer'])
    prompt_ids = config['calibration_prompt_token_ids']
    if len(prompt_ids)*group_size != len(prompts):
        raise ValueError('Calibration native prompt IDs omit an original G5 group')
    states = [initial_state_from_tokens(cfg, outer, dict(prompt=prompt, candidate_ordinal=i//group_size),
                            i % group_size, prompt_ids[i//group_size]) for i, prompt in enumerate(prompts)]
    waiting, ready, pending, tools = deque(states), deque(), {}, {}
    active = finished = 0
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pool = ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency)

    def finish_environment(state, cut, environment):
        nonlocal active, finished
        finish_action(state, cut, environment, tokenizer, BUDGET)
        if state['terminal']:
            finished += 1; active -= 1
        else:
            ready.append(state)

    try:
        while finished < len(states):
            while waiting and active < maximum:
                ready.append(waiting.popleft()); active += 1
            while ready:
                state = ready.popleft(); start_action(state, BUDGET)
                request = next_request(state, 'calibration', cfg)
                if request is None:
                    raise ValueError('Unfinished calibration action has no native budget')
                params = SamplingParams(n=1, temperature=1. if sample else 0., top_p=1., top_k=-1,
                    max_tokens=request['max_tokens'], stop=[], stop_token_ids=stops, ignore_eos=False,
                    logprobs=1, detokenize=False, seed=request['seed'])
                engine.add_request(request['request_id'], {'prompt_token_ids':request['actual_request_token_ids']}, params)
                pending[request['request_id']] = state, request, registered_sampling(engine, request, params)
            if engine.has_unfinished_requests():
                for result in engine.step():
                    rid = str(result.request_id)
                    if rid not in pending:
                        continue
                    state, request, sampling = pending[rid]
                    if len(result.outputs) != 1 or list(result.prompt_token_ids) != request['actual_request_token_ids']:
                        raise ValueError('Calibration native request identity changed')
                    native = result.outputs[0]
                    seen = stream_action_present(state['action']['raw_generated_token_ids']+list(native.token_ids), decode)
                    aborted = bool(seen and not result.finished)
                    if not result.finished and not seen:
                        continue
                    snapshot = native_snapshot(native)
                    if aborted:
                        engine.abort_request(rid)
                    pending.pop(rid)
                    cut = append_chunk(state, request, snapshot['raw_generated_token_ids'], snapshot['selected_logprobs'], sampling,
                        finish_reason=snapshot['finish_reason'], stop_reason=snapshot['stop_reason'],
                        terminal_ids=terminals, decode=decode, action_boundary_abort=aborted)
                    if cut is None:
                        ready.append(state)
                    elif cut['native_terminal']:
                        finish_environment(state, cut, ('', True, int(cut['action'] is not None), 0))
                    elif cut['action']=='search' and not state['action']['final_turn']:
                        tools[pool.submit(execute_one_action, manager, cut['execution_text'], True, retriever)] = state, cut
                    else:
                        finish_environment(state, cut, execute_one_action(manager, cut['execution_text'],
                            not state['action']['final_turn'], retriever))
            done = [future for future in tools if future.done()]
            for future in done:
                state, cut = tools.pop(future); finish_environment(state, cut, future.result())
            if not ready and not engine.has_unfinished_requests() and tools and not done:
                time.sleep(.001)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    rows = [provenance(state, tokenizer) for state in states]
    return tuple([row[index] for row in rows] for index in range(5))
