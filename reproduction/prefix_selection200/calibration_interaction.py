"""Opt-in calibration rollouts with the exact new training/QA interaction.

Only the scheduled grounding refitter selects this path. Historical dump callers
keep their original default and their closed-book/counterfactual label recipe.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .interaction_contract import BUDGET, final_turn, interaction_outcome


def rollouts(llm, prompts, tokenizer, sample, config):
    from vllm import SamplingParams
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_g1.calibration_evidence import validate_provenance
    from .eval import evaluation_stream_snapshot

    generation = json.loads((Path(config['calibration_model']) / 'generation_config.json').read_text())
    stops = generation.get('eos_token_id', tokenizer.eos_token_id)
    stops = [stops] if isinstance(stops, int) else list(stops)
    terminals = sorted(set(stops + [tokenizer.eos_token_id]))
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    params = SamplingParams(n=1, temperature=1. if sample else 0., top_p=1., top_k=-1,
        max_tokens=500, stop_token_ids=stops, ignore_eos=False, detokenize=False)
    recipe = GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500, num_gpus=1, search_url=config['search_url'], topk=3)
    manager = LLMGenerationManager(tokenizer, None, recipe, is_validation=not sample)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig(agent_workers=1024,
        max_inflight_trajectories=1024,
        retrieval_concurrency=int(config.get('retrieval_threads_per_engine', 16)),
        tool_timeout=float(config.get('retrieval_timeout', 60)), tool_retries=2, engine_step_budget=8))
    states = [dict(prompt=list(tokenizer.encode(text, add_special_tokens=False))[-2048:], response=[],
        mask=[], gen='', model=[], evidence=[], correction=[], calls=0, done=False) for text in prompts]
    engine = llm.llm_engine
    with ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency) as pool:
        for ordinal in range(5):
            active = [index for index, state in enumerate(states) if not state['done']]
            if not active:
                break
            pending = {}; snapshots = {}
            for index in active:
                state = states[index]; request = state['prompt'] + state['response']
                state['request'] = request; state['final'] = final_turn(len(request), ordinal)
                rid = 'grounding-calibration-%d-%d' % (ordinal, index)
                pending[rid] = index
                engine.add_request(rid, {'prompt_token_ids': request}, params)
            while engine.has_unfinished_requests():
                for result in engine.step():
                    rid = str(result.request_id)
                    if rid not in pending:
                        continue
                    index = pending[rid]; state = states[index]
                    snapshot = evaluation_stream_snapshot(result, state['request'], terminals, decode, engine.abort_request)
                    if snapshot is not None:
                        snapshots[index] = snapshot['prefix_cut']; pending.pop(rid)
            if pending or set(snapshots) != set(active):
                raise ValueError('Calibration native stream omitted an actual trajectory action')
            futures = {}
            for index in active:
                state = states[index]; cut = snapshots[index]
                if cut['native_terminal']:
                    futures[index] = ('', True, int(cut['action'] is not None), 0)
                else:
                    futures[index] = pool.submit(execute_one_action, manager, cut['execution_text'],
                                                 not state['final'], retriever)
            for index in active:
                state = states[index]; cut = snapshots[index]; result = futures[index]
                environment = result if isinstance(result, tuple) else result.result()
                transition = interaction_outcome(tokenizer, len(state['request']), cut['kept_token_count'],
                    state['final'], cut, environment)
                ids = cut['kept_token_ids']; observation = transition['observation_token_ids']
                begin = len(state['gen']); state['gen'] += decode(ids)
                state['model'].append([begin, len(state['gen'])])
                begin = len(state['gen']); state['gen'] += decode(observation)
                role = 'evidence' if environment[3] else 'correction'
                if observation:
                    state[role].append([begin, len(state['gen'])])
                state['response'].extend(ids + observation)
                state['mask'].extend([1] * len(ids) + [0] * len(observation))
                state['calls'] += transition['retrieval_calls']; state['done'] = transition['terminal']
                if len(state['prompt']) + len(state['response']) > BUDGET['total_tokens']:
                    raise ValueError('Calibration retained context exceeds the shared4096 contract')
    for state in states:
        validate_provenance(state['gen'], state['model'], state['evidence'], state['correction'])
    return tuple([[state[key] for state in states] for key in ('gen', 'model', 'calls', 'evidence', 'correction')])
