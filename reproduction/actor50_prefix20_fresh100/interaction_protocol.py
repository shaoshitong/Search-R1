"""Canonical new train/QA interaction protocol, with unchanged strict scoring.

Only sampling temperature and prefix pauses differ between train and greedy QA.
Both execute the first complete action seen in the native token stream, retain
original token IDs, and plan environment input before the next model request.
The common8 context-budget implementation is the authorized training source of
truth. No sampled model token is re-encoded or cropped to create context room.
"""
import math
import re

from reproduction.common8.context_budget import final_stage, observation_room, observation_tokens
from .contract import SAMPLING_CONTRACT

PROTOCOL_ID = 'new_context4096_action500_reserve500_stream_first_action_bf16_v1'
COMPUTE_DTYPE = 'bfloat16'
ACTUAL_COMPUTE_DTYPE = 'torch.bfloat16'
MAX_START_TOKENS = 2048
ACTION_TOKENS = 500
MAX_OBSERVATION_TOKENS = 500
MAX_SEARCH_TURNS = 4
BUDGET = dict(total_tokens=4096, action_tokens=ACTION_TOKENS, answer_reserve_tokens=500)
BOUNDARY = 'first_complete_action_seen_in_actual_chunk_stream'
ACTION = re.compile(r'<(search|answer)>(.*?)</\1>', re.DOTALL)
CONTRACT = dict(protocol_id=PROTOCOL_ID, kind='shared_new_context_stream_bf16',
    compute_dtype=COMPUTE_DTYPE, train_compute_dtype=COMPUTE_DTYPE,
    actor_master_dtype='float32', actor_loss_dtype='float32',
    max_action_tokens=ACTION_TOKENS, max_start_tokens=MAX_START_TOKENS,
    max_request_tokens=BUDGET['total_tokens'] - ACTION_TOKENS,
    max_context_tokens=BUDGET['total_tokens'], max_model_tokens=BUDGET['total_tokens'],
    max_observation_tokens=MAX_OBSERVATION_TOKENS, max_search_turns=MAX_SEARCH_TURNS,
    answer_reserve_tokens=BUDGET['answer_reserve_tokens'], early_final=True,
    observation_truncation='clip_before_input_preserve_information_wrapper',
    action_boundary=BOUNDARY, natural_eos='terminal_without_environment_correction',
    asynchronous_generation=True,
    native_output_commit='synchronous_token_ids_and_selected_logprobs_snapshot_before_action_abort')


def terminal_token_ids(generation, tokenizer):
    """Use the same generation-config and tokenizer EOS policy in both roles."""
    stop = generation.get('eos_token_id', tokenizer.eos_token_id)
    stop = [stop] if isinstance(stop, int) else list(stop)
    if not stop or any(type(token) is not int for token in stop):
        raise ValueError('Actual generation stop-token IDs are missing or invalid')
    terminal = sorted(set(stop + [tokenizer.eos_token_id]))
    return stop, terminal


def first_action_cut(token_ids, decode, *, terminal_ids=(), native_stopped=False):
    """Find the first complete executable action without looking at future IDs.

    A token straddling the closing tag remains whole. Execution text is cut at
    the closing tag, so its token suffix cannot create a second tool action.
    Only an actual trailing native EOS/stop token is omitted from PPO policy
    positions, as in the existing retained-token contract.
    """
    raw = list(token_ids)
    terminal = bool(raw and raw[-1] in set(terminal_ids))
    usable = raw[:-1] if terminal else raw
    text = decode(usable)
    match = ACTION.search(text)
    if match is None:
        return dict(kept_token_ids=usable, kept_token_count=len(usable),
                    execution_text=text, action=None, boundary_found=False,
                    native_terminal=bool(terminal or native_stopped), discarded_token_count=len(raw)-len(usable))
    # Decode original IDs only; do not tokenize decoded text to manufacture IDs.
    # Presence of a complete ASCII closing tag is monotonic over byte-BPE
    # prefixes. Binary search preserves the whole boundary token without an
    # O(action_length**2) tokenizer-decode loop on every completed action.
    lo, keep = 1, len(usable)
    while lo < keep:
        middle = (lo + keep) // 2
        if ACTION.search(decode(usable[:middle])) is not None:
            keep = middle
        else:
            lo = middle + 1
    execution = decode(usable[:keep])
    actual = ACTION.search(execution)
    return dict(kept_token_ids=usable[:keep], kept_token_count=keep,
                execution_text=execution[:actual.end()], action=actual.group(1),
                boundary_found=True, native_terminal=bool(terminal or native_stopped),
                discarded_token_count=len(raw)-keep)


def stream_action_present(token_ids, decode):
    # Qwen byte-BPE cannot use more than one token per ASCII byte in these
    # closing tags. Avoid decoding an entire growing500-token action for every
    # native decode step of every concurrent request; full regex parsing is
    # needed only after a closing tag actually enters the observed suffix.
    tail = decode(token_ids[-32:])
    if '</search>' not in tail and '</answer>' not in tail:
        return False
    return ACTION.search(decode(token_ids)) is not None


def start_action(state, budget):
    if state['action'] is not None:
        return
    request = state['prompt'] + state['response']
    if len(request) + budget['action_tokens'] > budget['total_tokens']:
        raise ValueError('Real context cannot accommodate a logical500-token action')
    state['action'] = dict(turn_index=state['ordinal'], actual_request_token_ids=request,
                           raw_generated_token_ids=[], selected_logprobs=[], actual_sampling_chunks=[],
                           final_turn=final_stage(len(request), state['ordinal'], MAX_SEARCH_TURNS, budget))


def append_chunk(state, request, raw_ids, log_probs, sampling, *, finish_reason,
                 stop_reason, terminal_ids, decode, action_boundary_abort=False):
    """Append actual sampler evidence; return an action cut only at real events."""
    action = state['action']
    raw_ids, log_probs = list(raw_ids), list(log_probs)
    if len(raw_ids) != len(log_probs) or any(not math.isfinite(p) or p > 1e-6 for p in log_probs):
        raise ValueError('Actual native selected-token log probabilities are missing/nonfinite')
    if (request['actual_request_token_ids'] != action['actual_request_token_ids'] + action['raw_generated_token_ids']
            or len(raw_ids) > request['max_tokens']):
        raise ValueError('Native continuation request or quota differs from retained state')
    chunk = dict(request, raw_generated_token_ids=raw_ids, selected_logprobs=log_probs,
                 sampling=sampling, finish_reason=finish_reason, stop_reason=stop_reason,
                 native_action_boundary_abort=bool(action_boundary_abort), actor_version=state['actor_version'],
                 policy_snapshot_id=state['actor_version'], logprob_semantics='selected_token_logprob_at_actual_sampling_call')
    action['actual_sampling_chunks'].append(chunk)
    action['raw_generated_token_ids'].extend(raw_ids)
    action['selected_logprobs'].extend(log_probs)
    stopped = finish_reason == 'stop'
    cut = first_action_cut(action['raw_generated_token_ids'], decode,
                           terminal_ids=terminal_ids, native_stopped=stopped)
    exhausted = len(action['raw_generated_token_ids']) >= ACTION_TOKENS
    if cut['boundary_found'] or cut['native_terminal'] or exhausted:
        return cut
    if len(raw_ids) != request['max_tokens']:
        raise ValueError('Native completion ended without an actual action, EOS or requested quota')
    return None


def finish_action(state, cut, environment, tokenizer, budget):
    action = state['action']
    text, done, valid, is_search = environment
    natural_eos = cut['native_terminal']
    room = observation_room(len(action['actual_request_token_ids']), cut['kept_token_count'], budget)
    observation, clipped = ([], False) if action['final_turn'] or natural_eos else observation_tokens(
        tokenizer, text, min(MAX_OBSERVATION_TOKENS, room))
    terminal = bool(action['final_turn'] or done or natural_eos)
    turn = dict(action, trajectory_id=state['trajectory_id'], policy_snapshot_id=state['actor_version'],
                actor_version=state['actor_version'], sampling_contract=SAMPLING_CONTRACT,
                raw_generated_token_count=len(action['raw_generated_token_ids']),
                kept_token_count=cut['kept_token_count'], prefix_cut=cut,
                observation_token_ids=observation if is_search else [],
                correction_token_ids=observation if not is_search else [],
                observation_clipped=clipped, final_turn=terminal,
                valid_action=int(valid), is_search=int(is_search),
                finish_reason=action['actual_sampling_chunks'][-1]['finish_reason'],
                stop_reason=action['actual_sampling_chunks'][-1]['stop_reason'])
    kept = cut['kept_token_ids']
    state['response'].extend(kept + observation)
    state['info_mask'].extend([1] * len(kept) + [0] * len(observation))
    state['log_probs'].extend(action['selected_logprobs'][:len(kept)] + [0.] * len(observation))
    state['turns'].append(turn)
    state['retrieval_calls'] += int(is_search and not action['final_turn'] and not natural_eos)
    state['action'] = None
    if len(state['prompt']) + len(state['response']) > budget['total_tokens']:
        raise ValueError('Actual retained context exceeds4096 tokens')
    state['terminal'] = terminal
    if terminal:
        state['terminal_reason'] = ('natural_eos' if natural_eos else
                                    'answer' if done else 'final_action_budget')
    else:
        state['ordinal'] += 1


def _sampling_dict(params):
    fields = ('n', 'best_of', 'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty',
              'frequency_penalty', 'repetition_penalty', 'ignore_eos', 'logprobs', 'max_tokens',
              'min_tokens', 'stop', 'stop_token_ids', 'include_stop_str_in_output', 'seed', 'detokenize')
    value = {key: getattr(params, key) for key in fields if hasattr(params, key)}
    value['do_sample'] = value.get('temperature', 1.) != 0
    value['logprob_semantics'] = 'selected_token_logprob_at_actual_sampling_call'
    return value


def registered_sampling(engine, request, caller):
    """Read the actual vLLM0.6.3 registered group, after EOS config merging.

    The input SamplingParams is cloned by LLMEngine. Saving just the caller
    would incorrectly describe that clone as actual execution metadata.
    Reading its newly queued group neither advances nor modifies sampling.
    """
    schedulers = engine.scheduler
    if not isinstance(schedulers, (list, tuple)):
        schedulers = [schedulers]
    groups = (group for scheduler in schedulers
              for name in ('waiting', 'running', 'swapped')
              for group in getattr(scheduler, name, ()))
    for group in groups:
        if group.request_id != request['request_id']:
            continue
        if list(group.prompt_token_ids) != request['actual_request_token_ids']:
            raise ValueError('Actual registered native group has different original request IDs')
        actual = _sampling_dict(group.sampling_params)
        actual['effective_all_stop_token_ids'] = sorted(group.sampling_params.all_stop_token_ids)
        actual['caller_sampling'] = _sampling_dict(caller)
        actual['registered_request_id'] = request['request_id']
        actual['metadata_source'] = 'actual_vllm_registered_sequence_group_after_generation_config_merge'
        return actual
    raise ValueError('New native request is missing from the actual vLLM scheduler queues')


def native_snapshot(native):
    """Copy matched native IDs/LP before external abort releases its sequence.

    vLLM0.6.3 intermediate outputs freeze token IDs but expose the live
    sequence LP list. Async stepping can advance that shared list after the
    token snapshot, mixing commit clocks. This engine therefore uses
    synchronous output commit (not serial generation).
    Neither missing probabilities nor future tokens are fabricated or fitted.
    """
    tokens=list(native.token_ids)
    rows=None if native.logprobs is None else list(native.logprobs)
    if rows is None or len(rows) != len(tokens):
        raise ValueError('Native vLLM selected-token behavior probability mismatch: token_count=%d logprob_count=%s finish_reason=%r' %
                         (len(tokens),None if rows is None else len(rows),native.finish_reason))
    values = []
    for token, row in zip(tokens, rows):
        selected = row.get(token)
        if selected is None:
            raise ValueError('Actual selected native token is absent from logprob row')
        values.append(float(selected.logprob if hasattr(selected, 'logprob') else selected))
    return dict(raw_generated_token_ids=tokens,selected_logprobs=values,
                finish_reason=native.finish_reason,stop_reason=native.stop_reason)
