"""Old content semantics and FP32 evaluation, retaining independent async phases.

Recipe old-context-fp32-async is not a byte-identical old synchronous recipe.
Only evaluation callers import this controller; training runtime is untouched.
Overlength old requests are actual failures, never silently cropped/reserved.
"""

RECIPE_ID = 'old-context-fp32-async'


def content_policy(config):
    if (config.max_start_length, config.max_prompt_length,
            config.max_response_length, config.max_obs_length, config.max_turns) != (2048,4096,500,500,4):
        raise ValueError('Compatible evaluation content dimensions changed')
    return dict(role=RECIPE_ID, total_tokens=4096, action_tokens=500,
        answer_reserve_tokens=None,
        observation_policy='old_first500_tokens_without_wrapper_preservation',
        final_policy='ordinal4_only_no_context_early_final')


def request_policy(prefix_length, ordinal, max_turns=4, request_limit=4096):
    if not 0 <= ordinal <= max_turns or prefix_length > request_limit:
        raise ValueError('rolling_window_would_change_actor_context')
    return ordinal == max_turns


def check_response_append(response_length, generated_length, observation_length, limit=4096):
    # Deliberately excludes the original prompt, as the actual pre-d00 loop did.
    # A subsequent full request can therefore fail its separate <=4096 check.
    if min(response_length, generated_length, observation_length) < 0 or response_length + generated_length + observation_length > limit:
        raise ValueError('response_prefix_budget_would_drop_actual_tokens')


def run(manager, gen_batch, initial_input_ids, captured=None, *, service=None, tool_runner=None, config=None):
    import torch
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    import copy, time, uuid
    from search_g1.async_rollout_runtime import (AsyncRolloutConfig, Trajectory,
        RayPhaseService, BoundedRetriever, execute_one_action)
    from reproduction.common8.behavior_trace import BehaviorTraceError, make_turn
    from reproduction.common8.generation_behavior import (
        _observation_ids, _require, assemble_output, validated_raw_turn)
    if captured is None:
        captured = []
    options = (config or AsyncRolloutConfig.from_manager(manager)).validated()
    recipe, tokenizer = manager.config, manager.tokenizer
    budget = content_policy(recipe)
    _require(recipe.max_turns == 4 and not recipe.no_think_rl, 'generation_recipe_changed')
    identity = gen_batch.meta_info.get('behavior_identity')
    _require(isinstance(identity, dict) and identity.get('validation') is True and
             gen_batch.meta_info.get('do_sample') is False, 'evaluation_only_greedy_required')
    row_ids = gen_batch.non_tensor_batch.get('behavior_row_id')
    _require(isinstance(identity, dict) and row_ids is not None and len(row_ids) == len(gen_batch),
             'missing_complete_row_identity')
    row_ids = row_ids.tolist()
    repeats = 1 if identity['validation'] else 5
    _require(row_ids == identity['trajectory_ids'] and len(set(row_ids)) == len(row_ids) and
             len(row_ids) == len(identity['parent_group_ids']) * repeats,
             'async_initial_trajectory_group_or_order_changed')
    _require(gen_batch.meta_info.get('recompute_log_prob') is False,
             'per_turn_HF_recompute_must_not_precede_context_audit')
    pad = tokenizer.pad_token_id
    special = sorted(set(int(value) for value in tokenizer.all_special_ids))
    prompt_limit = recipe.max_start_length
    left = initial_input_ids[:, -prompt_limit:].detach().cpu().long()
    attention = gen_batch.batch['attention_mask'][:, -left.shape[1]:].detach().cpu().long()
    _require(left.shape == attention.shape and bool(((attention == 0) | (attention == 1)).all()),
             'initial_prompt_shape_or_mask_changed')
    prompts = [[int(value) for value, valid in zip(row.tolist(), mask.tolist()) if valid]
               for row, mask in zip(left, attention)]
    _require(all(prompt and pad not in prompt for prompt in prompts), 'actual_prompt_contains_pad_or_is_empty')
    states = [Trajectory(index, row_id, identity['parent_group_ids'][index // repeats], index % repeats, prompts[index])
              for index, row_id in enumerate(row_ids)]
    _require(type(recipe.num_gpus) is int and recipe.num_gpus > 0, 'invalid_async_generation_world')
    service = service or RayPhaseService(manager.actor_rollout_wg, recipe.num_gpus, options.engine_step_budget)
    retriever = BoundedRetriever(manager, options)
    tool_runner = tool_runner or (lambda text, do_search: execute_one_action(manager, text, do_search, retriever))
    phase_id = uuid.uuid4().hex
    snapshot = dict(phase_id=phase_id, policy_snapshot_id=identity['policy_snapshot_id'],
                    behavior_identity=copy.deepcopy(identity), expected_trajectory_count=len(states))
    pending, tools = {}, {}
    ready, unstarted = deque(), deque(states)
    active_limit = min(options.agent_workers, options.max_inflight_trajectories)
    active_count = completed_count = generation_calls = tool_calls = 0
    active_peak = 0
    sampling, final_meta = None, {}
    phase_open = False
    start = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=options.retrieval_concurrency, thread_name_prefix='g1-retrieval')

    def submit(state):
        nonlocal generation_calls
        tokens = state.prompt + state.response
        final = request_policy(len(tokens), state.ordinal, recipe.max_turns, recipe.max_prompt_length)
        request_id = phase_id + '-request-' + uuid.uuid4().hex
        call_id = identity['policy_snapshot_id'] + '-call-' + uuid.uuid4().hex
        meta = {name: gen_batch.meta_info[name] for name in
                ('eos_token_id', 'pad_token_id', 'do_sample', 'validate') if name in gen_batch.meta_info}
        meta.update(recompute_log_prob=False, behavior_call_id=call_id,
                    behavior_rollout_ordinal=state.ordinal, policy_snapshot_id=identity['policy_snapshot_id'])
        request = dict(request_id=request_id, prompt_token_ids=tokens, row_id=state.row_id,
            question_id=state.question_id, response_index=state.response_index,
            rollout_ordinal=state.ordinal, rollout_call_id=call_id,
            policy_snapshot_id=identity['policy_snapshot_id'], max_tokens=budget['action_tokens'], meta_info=meta)
        state.request, state.final_turn, state.status = request, final, 'generating'
        pending[request_id] = state
        service.submit(request, state.index % recipe.num_gpus)
        generation_calls += 1

    def complete_environment(state, environment):
        nonlocal active_count, completed_count
        text, done, valid, is_search = environment
        _require(isinstance(text, str) and type(done) in (int, bool) and done in (0, 1) and
                 type(valid) is int and valid in (0, 1) and type(is_search) is int and is_search in (0, 1),
                 'async_environment_return_changed')
        kept = state.cut['kept_token_ids']
        if state.final_turn:
            tokens, clipped = [], False
        else:
            # No reserve/room clipping or wrapper reconstruction: exactly the
            # pre-d00 tokenizer -> first500 tokens -> remove pad IDs path.
            rows = _observation_ids(manager, [text])
            tokens, clipped = rows[0], False
        check_response_append(len(state.response), len(kept), len(tokens), recipe.max_prompt_length)
        captured.append(dict(role='actual_environment_event', rollout_call_id=state.request['rollout_call_id'],
            request_id=state.request['request_id'], row_ids=[state.row_id],
            observation_or_correction_token_ids=[tokens], is_search=[is_search], final_turn=state.final_turn))
        turn = make_turn(state.raw_record, trajectory_id=state.row_id, turn_index=state.ordinal,
            kept_token_count=state.cut['kept_token_count'], special_token_ids=special,
            observation_token_ids=tokens if is_search else [], correction_token_ids=[] if is_search else tokens,
            final_turn=state.final_turn or bool(done), prefix_cut=state.cut)
        state.turns.append(turn)
        state.response.extend(kept + tokens)
        state.probabilities.extend(state.raw_record['selected_logprobs'][:len(kept)] + [0.0] * len(tokens))
        state.info_mask.extend([1] * len(kept) + [0] * len(tokens))
        state.clipped_observations += int(clipped)
        state.early_final |= bool(state.final_turn and state.ordinal < recipe.max_turns)
        state.valid_actions += valid
        state.valid_searches += is_search
        state.actual_retrieval += is_search if not state.final_turn else 0
        if not state.final_turn and not done:
            state.turns_stat += 1
        if state.final_turn or done:
            state.status = 'complete'; active_count -= 1; completed_count += 1
        else:
            state.ordinal += 1; state.status = 'ready'; ready.append(state)

    try:
        phase_open = True
        service.begin(snapshot)
        while completed_count < len(states):
            while unstarted and active_count < active_limit:
                ready.append(unstarted.popleft()); active_count += 1
            active_peak = max(active_peak, active_count)
            while ready:
                submit(ready.popleft())
            for value in service.poll(timeout=0.01):
                request_id = value['request_id']
                _require(request_id in pending, 'async_unknown_or_duplicate_request_completion')
                state = pending.pop(request_id)
                output = value['output']
                _require(len(output) == 1 and 'sampling_behavior' in output.non_tensor_batch and
                         len(output.non_tensor_batch['sampling_behavior']) == 1,
                         'async_generation_single_row_return_changed')
                record = output.non_tensor_batch['sampling_behavior'][0]
                captured.append(copy.deepcopy(record))
                _require(record['request_id'] == request_id and record['row_id'] == state.row_id,
                         'async_actual_request_row_identity_changed')
                cut, sampling = validated_raw_turn(manager, output, 0, record,
                    request_tokens=state.request['prompt_token_ids'], row_id=state.row_id, identity=identity,
                    call_id=state.request['rollout_call_id'], ordinal=state.ordinal, sampling=sampling)
                final_meta.update(output.meta_info)
                state.raw_record, state.cut, state.status = record, cut, 'processing_tool'
                tool_calls += 1
                actions, _ = manager.postprocess_predictions([cut['execution_text']])
                if actions[0] == 'search' and not state.final_turn:
                    future = pool.submit(tool_runner, cut['execution_text'], True)
                    tools[future] = state
                else:
                    # Answers, invalid actions, and final-stage no-search never
                    # wait behind another trajectory's retrieval queue.
                    complete_environment(state, tool_runner(cut['execution_text'], not state.final_turn))
            finished = [future for future in tools if future.done()]
            if not pending and not ready and tools and not finished:
                finished = list(wait(list(tools), timeout=0.01, return_when=FIRST_COMPLETED).done)
            for future in finished:
                state = tools.pop(future)
                complete_environment(state, future.result())
        _require(not pending and not tools and not unstarted and all(state.status == 'complete' for state in states),
                 'async_incomplete_native_outer')
        phase_metrics = service.end(abort=False); phase_open = False
    except BaseException:
        if phase_open:
            try:
                service.end(abort=True)
            except Exception as cleanup_error:
                raise RuntimeError('Generation phase cleanup failed; refuse actor update') from cleanup_error
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    final_meta.update(async_rollout=dict(role='independent_native_trajectory_state_machine',
        phase_id=phase_id, completed_trajectories=len(states), generation_requests=generation_calls,
        environment_actions=tool_calls, peak_inflight_trajectories=active_peak,
        retrieval_concurrency=options.retrieval_concurrency,
        tool_timeout=options.tool_timeout, tool_retries=options.tool_retries,
        phase_elapsed_seconds=time.monotonic() - start, phase_metrics=phase_metrics,
        result_order='original_trajectory_ids', actor_updates_during_phase=0))
    return assemble_output(row_ids=row_ids, identity=identity, budget=budget, left=left,
        attention=attention, pad=pad, responses=[state.response for state in states],
        probabilities=[state.probabilities for state in states], info_masks=[state.info_mask for state in states],
        turns=[state.turns for state in states], final_meta=final_meta, sampling=sampling,
        clipped_observations=[state.clipped_observations for state in states],
        early_final_turns=[state.early_final for state in states], turns_stats=[state.turns_stat for state in states],
        active=[False] * len(states), valid_actions=[state.valid_actions for state in states],
        valid_searches=[state.valid_searches for state in states], actual_retrieval=[state.actual_retrieval for state in states],
        active_counts=[len(states), 0])
