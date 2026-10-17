"""Grounding-calibration rollouts using the canonical train/QA interaction.

Only the explicit --interaction-config dump path uses this adapter. The
closed-book and evidence-deletion label recipes remain separate interventions.
No sampled policy token is decoded/re-encoded to construct a later request.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .interaction_protocol import (ACTUAL_COMPUTE_DTYPE, ACTION_TOKENS, BUDGET,
    CONTRACT, MAX_OBSERVATION_TOKENS, MAX_SEARCH_TURNS, MAX_START_TOKENS,
    append_chunk, finish_action, native_snapshot, registered_sampling,
    start_action, stream_action_present, terminal_token_ids)
from .rollout import next_request


def rollouts(llm, prompts, tokenizer, sample, config):
    from vllm import SamplingParams
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_g1.calibration_evidence import validate_provenance

    if config.get('interaction_protocol', CONTRACT) != CONTRACT:
        raise ValueError('Grounding calibration must use the canonical train/QA interaction protocol')
    engine = llm.llm_engine
    if str(engine.model_config.dtype) != ACTUAL_COMPUTE_DTYPE or engine.model_config.max_model_len != BUDGET['total_tokens']:
        raise ValueError('Grounding calibration requires actual BF16 and4096 native engine bounds')
    generation = json.loads((Path(config['calibration_model']) / 'generation_config.json').read_text())
    stops, terminals = terminal_token_ids(generation, tokenizer)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    recipe = GenerationConfig(max_turns=MAX_SEARCH_TURNS, max_start_length=MAX_START_TOKENS,
        max_prompt_length=BUDGET['total_tokens'], max_response_length=ACTION_TOKENS,
        max_obs_length=MAX_OBSERVATION_TOKENS, num_gpus=1, search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe, is_validation=not sample)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(agent_workers=1024,
        max_inflight_trajectories=1024,
        retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    states = [dict(run_id=config['run_id'], outer=int(config.get('grounding_refit_outer', 0)),
        trajectory_id='grounding-calibration-%d' % index,
        actor_version=config.get('actor_version', 'grounding:' + config['calibration_model']),
        prompt=list(tokenizer.encode(text, add_special_tokens=False))[-MAX_START_TOKENS:],
        response=[], info_mask=[], log_probs=[], turns=[], ordinal=0, chunks_started=0,
        action=None, retrieval_calls=0, terminal=False, terminal_reason=None,
        text='', model=[], evidence=[], correction=[]) for index, text in enumerate(prompts)]
    if any(not state['prompt'] for state in states):
        raise ValueError('Grounding calibration official initial request is empty')
    stats = dict(protocol=CONTRACT, actual_compute_dtype=str(engine.model_config.dtype),
        actual_generation_requests=0, native_action_boundary_aborts=0, early_final_actions=0,
        native_terminal_ids=terminals, max_request_tokens=0, max_retained_context_tokens=0)
    with ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency) as pool:
        for _ordinal in range(MAX_SEARCH_TURNS + 1):
            active = [index for index, state in enumerate(states) if not state['terminal']]
            if not active:
                break
            pending, cuts = {}, {}
            for index in active:
                state = states[index]; start_action(state, BUDGET)
                request = next_request(state, 'grounding', config)
                params = SamplingParams(n=1, temperature=1. if sample else 0., top_p=1., top_k=-1,
                    max_tokens=request['max_tokens'], stop=[], stop_token_ids=stops,
                    ignore_eos=False, logprobs=1, detokenize=False, seed=request['seed'])
                rid = request['request_id']
                engine.add_request(rid, {'prompt_token_ids': request['actual_request_token_ids']}, params)
                pending[rid] = index, request, registered_sampling(engine, request, params)
                stats['actual_generation_requests'] += 1
                stats['max_request_tokens'] = max(stats['max_request_tokens'], len(request['actual_request_token_ids']))
                stats['early_final_actions'] += int(state['action']['final_turn'] and state['ordinal'] < MAX_SEARCH_TURNS)
            while engine.has_unfinished_requests():
                for result in engine.step():
                    rid = str(result.request_id)
                    if rid not in pending:
                        continue
                    index, request, sampling = pending[rid]; state = states[index]
                    if len(result.outputs) != 1 or list(result.prompt_token_ids) != request['actual_request_token_ids']:
                        raise ValueError('Grounding native output changed its original request identity')
                    native = result.outputs[0]
                    seen = stream_action_present(state['action']['raw_generated_token_ids'] + list(native.token_ids), decode)
                    if not result.finished and not seen:
                        continue
                    snapshot = native_snapshot(native)
                    aborted = bool(not result.finished and seen)
                    if aborted:
                        engine.abort_request(rid); stats['native_action_boundary_aborts'] += 1
                    pending.pop(rid)
                    cut = append_chunk(state, request, snapshot['raw_generated_token_ids'], snapshot['selected_logprobs'],
                        sampling, finish_reason=snapshot['finish_reason'], stop_reason=snapshot['stop_reason'],
                        terminal_ids=terminals, decode=decode, action_boundary_abort=aborted)
                    if cut is None:
                        raise ValueError('Unpaused grounding action ended without action/EOS/500-token quota')
                    cuts[index] = cut
            if pending or set(cuts) != set(active):
                raise ValueError('Grounding native stream omitted an actual trajectory action')
            environments = {}
            for index in active:
                state, cut = states[index], cuts[index]
                if cut['native_terminal']:
                    environments[index] = ('', True, int(cut['action'] is not None), 0)
                else:
                    environments[index] = pool.submit(execute_one_action, manager, cut['execution_text'],
                                                      not state['action']['final_turn'], retriever)
            for index in active:
                state, cut = states[index], cuts[index]; result = environments[index]
                environment = result if isinstance(result, tuple) else result.result()
                finish_action(state, cut, environment, tokenizer, BUDGET)
                turn = state['turns'][-1]
                begin = len(state['text']); state['text'] += decode(cut['kept_token_ids'])
                state['model'].append([begin, len(state['text'])])
                for key, tokens in (('evidence', turn['observation_token_ids']), ('correction', turn['correction_token_ids'])):
                    if tokens:
                        begin = len(state['text']); state['text'] += decode(tokens)
                        state[key].append([begin, len(state['text'])])
                stats['max_retained_context_tokens'] = max(stats['max_retained_context_tokens'],
                                                           len(state['prompt']) + len(state['response']))
    if any(not state['terminal'] for state in states):
        raise ValueError('Grounding interaction exhausted its turns without a real terminal state')
    for state in states:
        validate_provenance(state['text'], state['model'], state['evidence'], state['correction'])
    llm._g1_calibration_interaction_stats = stats
    llm._g1_calibration_initial_prompt_texts = [tokenizer.decode(state['prompt'], skip_special_tokens=False,
        clean_up_tokenization_spaces=False) for state in states]
    return tuple([[state[key] for state in states] for key in ('text', 'model', 'retrieval_calls', 'evidence', 'correction')])
