"""Independent G1 trajectories inside one resident, immutable sampling phase.

Only the phase begin/end perform collective model transitions. Rank-local
engine steps continuously accept the next requests while bounded retrieval
threads wait independently. A failed request rejects the complete outer; it is
never converted into a zero-scored or missing response.
"""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from dataclasses import dataclass, field
import math
import time
import uuid

import torch

from reproduction.common8.behavior_trace import BehaviorTraceError, make_turn
from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
from reproduction.common8.context_budget import final_stage, observation_room, protocol
from reproduction.common8.generation_behavior import (
    _observation_ids, _require, assemble_output, validated_raw_turn,
)


@dataclass(frozen=True)
class AsyncRolloutConfig:
    agent_workers: int = DEFAULT_ROLLOUT_PHASE['agent_workers']
    max_inflight_trajectories: int = DEFAULT_ROLLOUT_PHASE['max_inflight_trajectories']
    retrieval_concurrency: int = DEFAULT_ROLLOUT_PHASE['retrieval_concurrency']
    tool_timeout: float = DEFAULT_ROLLOUT_PHASE['tool_timeout']
    tool_retries: int = DEFAULT_ROLLOUT_PHASE['tool_retries']
    engine_step_budget: int = DEFAULT_ROLLOUT_PHASE['engine_step_budget']

    @classmethod
    def from_manager(cls, manager):
        values = {name: getattr(manager.config, name, DEFAULT_ROLLOUT_PHASE[name])
                  for name in cls.__dataclass_fields__}
        return cls(**values).validated()

    def validated(self):
        _require(all(type(value) is int and value > 0 for value in
                     (self.agent_workers, self.max_inflight_trajectories, self.retrieval_concurrency, self.engine_step_budget)),
                 'invalid_async_concurrency')
        _require(type(self.tool_retries) is int and 0 <= self.tool_retries <= 10 and
                 type(self.tool_timeout) in (int, float) and math.isfinite(self.tool_timeout) and
                 self.tool_timeout > 0, 'invalid_bounded_tool_retry_policy')
        return self


@dataclass
class Trajectory:
    index: int
    row_id: str
    question_id: str
    response_index: int
    prompt: list
    ordinal: int = 0
    status: str = 'ready'
    response: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)
    info_mask: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    turns_stat: int = 1
    valid_actions: int = 0
    valid_searches: int = 0
    actual_retrieval: int = 0
    clipped_observations: int = 0
    early_final: bool = False
    request: dict = None
    raw_record: dict = None
    cut: dict = None
    final_turn: bool = False


class BoundedRetriever:
    """Same G1 payload/doc formatting, with finite transport attempts and timeouts."""
    def __init__(self, manager, config):
        self.manager, self.config = manager, config

    def search(self, query):
        import requests
        urls = [value.strip() for value in str(self.manager.config.search_url).split(',') if value.strip()]
        _require(bool(urls), 'missing_async_retriever_url')
        payload = dict(queries=[query], topk=self.manager.config.topk, return_scores=True)
        last_error = None
        # Each worker owns its Session. No shared mutable requests connection pool.
        with requests.Session() as session:
            session.trust_env = False
            for _ in range(self.config.tool_retries + 1):
                for url in urls:
                    try:
                        response = session.post(url, json=payload,
                            timeout=(min(10.0, self.config.tool_timeout), self.config.tool_timeout))
                        response.raise_for_status()
                        result = response.json()['result']
                        _require(type(result) is list and len(result) == 1 and type(result[0]) is list,
                                 'async_retrieval_result_count_changed')
                        return self.manager._passages2string(result[0])
                    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
                        last_error = error
        raise BehaviorTraceError('async_retrieval_attempts_exhausted:' + type(last_error).__name__)


def execute_one_action(manager, text, do_search, retriever):
    """Apply the upstream action parser/environment wording to one trajectory."""
    actions, contents = manager.postprocess_predictions([text])
    action = actions[0]
    if action == 'answer':
        return '', True, 1, 0
    if action == 'search':
        result = retriever.search(contents[0]) if do_search else ''
        return '\n\n<information>' + result.strip() + '</information>\n\n', False, 1, 1
    return ('\nMy previous action is invalid. '
            'If I want to search, I should put the query between <search> and </search>. '
            'If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n',
            False, 0, 0)


class RayPhaseService:
    """Wait-any adapter for collective phase transitions and local engine ticks."""
    def __init__(self, worker_group, world_size, engine_step_budget=8):
        self.group, self.world_size = worker_group, world_size
        self.engine_step_budget = engine_step_budget
        self.queues = [deque() for _ in range(world_size)]
        self.pending_counts = [0] * world_size
        self.refs = {}
        self.handle = None

    def begin(self, snapshot_meta):
        self.handle = {key: snapshot_meta[key] for key in ('phase_id', 'policy_snapshot_id')}
        handles = self.group.begin_generation_phase(snapshot_meta)
        _require(type(handles) is list and len(handles) == self.world_size and
                 all(handle == handles[0] for handle in handles), 'async_phase_handles_diverged')
        self.handle = handles[0]
        _require(self.handle['phase_id'] == snapshot_meta['phase_id'] and
                 self.handle['policy_snapshot_id'] == snapshot_meta['policy_snapshot_id'],
                 'async_phase_snapshot_changed')

    def submit(self, request, rank):
        _require(self.handle is not None and 0 <= rank < self.world_size,
                 'async_submit_outside_generation_phase')
        self.queues[rank].append(request)

    def poll(self, timeout=0.01):
        import ray
        running = set(self.refs.values())
        for rank in range(self.world_size):
            if rank in running or not (self.queues[rank] or self.pending_counts[rank]):
                continue
            queued = list(self.queues[rank]); self.queues[rank].clear()
            # Colocated Ray methods use this fixed role prefix; this is the
            # rank-local endpoint, never WorkerGroup.generate_sequences.
            endpoint = self.group._workers[rank].actor_rollout_advance_generation_phase
            ref = endpoint.remote(self.handle, queued, max_steps=self.engine_step_budget)
            self.refs[ref] = rank
        if not self.refs:
            return []
        ready, _ = ray.wait(list(self.refs), num_returns=1, timeout=timeout)
        completed = []
        for ref in ready:
            rank = self.refs.pop(ref)
            value = ray.get(ref)
            _require(type(value) is dict and type(value['completed']) is list and
                     type(value['pending_request_ids']) is list,
                     'async_rank_tick_return_changed')
            self.pending_counts[rank] = len(value['pending_request_ids'])
            completed.extend(value['completed'])
        return completed

    def end(self, abort=False):
        if self.handle is None:
            return []
        import ray
        # Drain in-flight rank calls before all ranks enter the closing boundary.
        if self.refs:
            refs = list(self.refs); self.refs.clear()
            failures = []
            for ref in refs:
                try:
                    ray.get(ref)
                except Exception as error:
                    failures.append(error)
            if failures and not abort:
                raise failures[0]
        if not abort:
            _require(not any(self.pending_counts) and not any(self.queues),
                     'async_phase_closed_with_pending_requests')
        result = self.group.end_generation_phase(self.handle, abort=abort)
        self.handle = None
        return result


def run(manager, gen_batch, initial_input_ids, *, service=None, tool_runner=None, config=None):
    """Native entry point; tests inject a tiny engine/tool without GPU execution."""
    captured = []
    try:
        return _run(manager, gen_batch, initial_input_ids, captured, service=service,
                    tool_runner=tool_runner, config=config)
    except Exception as error:
        from reproduction.common8.behavior_runtime import failed_generation_evidence
        identity = gen_batch.meta_info.get('behavior_identity')
        if isinstance(identity, dict):
            failed_generation_evidence(identity, captured, type(error).__name__)
        raise


def _run(manager, gen_batch, initial_input_ids, captured, *, service=None, tool_runner=None, config=None):
    options = (config or AsyncRolloutConfig.from_manager(manager)).validated()
    recipe, tokenizer = manager.config, manager.tokenizer
    budget = protocol(recipe)
    _require(recipe.max_turns == 4 and not recipe.no_think_rl, 'generation_recipe_changed')
    identity = gen_batch.meta_info.get('behavior_identity')
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
    prompt_limit = min(recipe.max_start_length, budget['total_tokens'] - budget['answer_reserve_tokens'])
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
    reuse_weights = gen_batch.meta_info.get('snapshot_weight_reuse', False)
    _require(type(reuse_weights) is bool, 'invalid_snapshot_weight_reuse')
    if reuse_weights:
        retain = gen_batch.meta_info.get('retain_snapshot_weights')
        _require(type(retain) is bool, 'invalid_snapshot_weight_retention')
        snapshot.update(snapshot_weight_reuse=True, retain_snapshot_weights=retain)
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
        final = final_stage(len(tokens), state.ordinal, recipe.max_turns, budget)
        _require(state.ordinal <= recipe.max_turns and len(tokens) + budget['action_tokens'] <= budget['total_tokens'],
                 'invalid_preplanned_total_context_budget')
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
        room = observation_room(len(state.request['prompt_token_ids']), len(kept), budget)
        if state.final_turn:
            tokens, clipped = [], False
        else:
            rows, flags = _observation_ids(manager, [text], [room])
            tokens, clipped = rows[0], flags[0]
        _require(len(state.prompt) + len(state.response) + len(kept) + len(tokens) <= budget['total_tokens'],
                 'invalid_total_context_plan')
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
        policy_snapshot_id=identity['policy_snapshot_id'], snapshot_weight_reuse=reuse_weights,
        retain_snapshot_weights=snapshot.get('retain_snapshot_weights', False),
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
