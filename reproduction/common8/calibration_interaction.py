"""Reward calibration using this run's actual train/QA token transitions.

Only opt-in scheduled refits use this adapter. Closed-book and evidence-deletion
labels, hidden extraction and grouped CV remain in ts_probe_dump unchanged.
Native completion followed by the existing dual-decode cut is intentional:
both this experiment's trainer and QA use that exact same boundary policy.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .context_budget import protocol, final_stage, observation_room, observation_tokens
from .behavior_trace import raw_prefix_to_first_closing_tag


ROLE = 'common8_calibration_matches_train_qa_v1'


def settings(model, search_url):
    return dict(role=ROLE, model=str(model), search_url=search_url,
        max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500,
        action_boundary='native_completion_then_existing_dual_decode_cut',
        eos_policy='existing_shared_execute_one_action', topk=3)


def rollouts(llm, prompts, tokenizer, sample, config):
    from vllm import SamplingParams
    from search_g1.async_rollout_runtime import AsyncRolloutConfig, BoundedRetriever, execute_one_action
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_g1.calibration_evidence import validate_provenance
    if config != settings(config['model'], config['search_url']):
        raise ValueError('Unbound calibration interaction settings')
    recipe = GenerationConfig(**{k:config[k] for k in ('max_turns','max_start_length',
        'max_prompt_length','max_response_length','max_obs_length','search_url','topk')}, num_gpus=1)
    manager = LLMGenerationManager(tokenizer, None, recipe, is_validation=not sample)
    budget = protocol(recipe)
    retriever = BoundedRetriever(manager, AsyncRolloutConfig())
    generation = json.loads((Path(config['model'])/'generation_config.json').read_text())
    stops = generation.get('eos_token_id', tokenizer.eos_token_id)
    stops = [stops] if isinstance(stops, int) else list(stops)
    stops = sorted(set(stops+[tokenizer.eos_token_id]))
    params = SamplingParams(n=1, temperature=1. if sample else 0., top_p=1., top_k=-1,
        max_tokens=500, stop_token_ids=stops, ignore_eos=False, detokenize=False)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    executor = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    states = [dict(prompt=list(tokenizer.encode(p, add_special_tokens=False))[-2048:],
        response=[], gen='', model=[], evidence=[], correction=[], calls=0, done=False) for p in prompts]
    with ThreadPoolExecutor(max_workers=retriever.config.retrieval_concurrency) as pool:
        for ordinal in range(5):
            active = [s for s in states if not s['done']]
            if not active:
                break
            requests = [s['prompt']+s['response'] for s in active]
            if any(len(ids)+500>budget['total_tokens'] for ids in requests):
                raise ValueError('Calibration action cannot fit preserved context')
            outputs = llm.generate([{'prompt_token_ids':ids} for ids in requests], params, use_tqdm=False)
            if len(outputs) != len(active):
                raise ValueError('Calibration omitted real trajectories')
            futures = []
            for state, request, result in zip(active, requests, outputs):
                if list(result.prompt_token_ids) != request or len(result.outputs) != 1:
                    raise ValueError('Calibration actual request identity changed')
                cut = raw_prefix_to_first_closing_tag(list(result.outputs[0].token_ids), decode,
                    executor_decode=executor, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, special_token_ids=tokenizer.all_special_ids,
                    effective_stop_token_ids=stops)
                final = final_stage(len(request), ordinal, 4, budget)
                state.update(cut=cut, final=final, request=request)
                futures.append(pool.submit(execute_one_action, manager, cut['execution_text'], not final, retriever))
            for state, future in zip(active, futures):
                text, done, valid, search = future.result()
                ids = state['cut']['kept_token_ids']
                limit = min(500, observation_room(len(state['request']), len(ids), budget))
                observation, _ = ([], False) if state['final'] else observation_tokens(tokenizer, text, limit)
                start = len(state['gen']); state['gen'] += decode(ids)
                state['model'].append([start, len(state['gen'])])
                start = len(state['gen']); state['gen'] += decode(observation)
                if observation:
                    state['evidence' if search else 'correction'].append([start, len(state['gen'])])
                state['response'].extend(ids+observation)
                state['calls'] += int(search and not state['final'])
                state['done'] = bool(state['final'] or done)
                if len(state['prompt'])+len(state['response']) > budget['total_tokens']:
                    raise ValueError('Calibration retained context exceeds total budget')
    for state in states:
        validate_provenance(state['gen'], state['model'], state['evidence'], state['correction'])
    return tuple([[s[key] for s in states] for key in ('gen','model','calls','evidence','correction')])
