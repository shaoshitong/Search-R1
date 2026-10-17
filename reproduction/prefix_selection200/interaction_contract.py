"""One token/context/action contract shared by training and QA.

This module has no model, GPU or environment dependencies. Training may pause
at a retained20-token prefix, but both paths execute the first complete action
seen in their actual native token stream and use the same context transition.
"""
import re

BUDGET = dict(total_tokens=4096, action_tokens=500, answer_reserve_tokens=500)
BOUNDARY = 'first_complete_action_seen_in_actual_chunk_stream'
ACTION = re.compile(r'<(search|answer)>(.*?)</\1>', re.DOTALL)
INTERACTION_CONTRACT = dict(
    kind='unified_context4096_bf16_stream_action_v1', compute_dtype='bfloat16',
    max_action_tokens=500, max_start_tokens=2048, max_request_tokens=4096,
    max_model_tokens=4096, max_observation_tokens=500, max_search_turns=4,
    answer_reserve_tokens=500, early_final=True,
    observation_truncation='clip_before_input_preserve_information_wrapper',
    action_boundary=BOUNDARY, eos_policy='omit_native_terminal_id_and_end_trajectory',
    sampled_action_tokens='original_ids_through_first_complete_action_boundary_token',
    strict_answer_scoring=True)


def evaluation_contract():
    return dict(INTERACTION_CONTRACT, temperature=0., top_p=1., top_k=-1, n=1,
        asynchronous_generation=True, native_output_commit=
        'synchronous_token_ids_snapshot_before_action_abort')


def final_turn(request_length, ordinal, budget=BUDGET):
    from reproduction.common8.context_budget import final_stage
    if type(ordinal) is not int or not 0 <= ordinal <= 4:
        raise ValueError('An interaction has four search-capable turns and one final turn')
    if request_length + budget['action_tokens'] > budget['total_tokens']:
        raise ValueError('Real context cannot accommodate a logical500-token action')
    return final_stage(request_length, ordinal, 4, budget)


def interaction_outcome(tokenizer, request_length, generated_length, final, cut, environment, budget=BUDGET):
    from reproduction.common8.context_budget import observation_room, observation_tokens
    text, done, valid, is_search = environment
    natural_eos = bool(cut['native_terminal'])
    room = observation_room(request_length, generated_length, budget)
    observation, clipped = ([], False) if final or natural_eos else observation_tokens(
        tokenizer, text, min(500, room))
    terminal = bool(final or done or natural_eos)
    return dict(observation_token_ids=observation, observation_clipped=clipped,
                terminal=terminal, natural_eos=natural_eos, valid_action=int(valid),
                is_search=int(is_search), retrieval_calls=int(is_search and not final and not natural_eos))


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
