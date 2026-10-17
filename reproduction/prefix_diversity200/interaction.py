"""Shared authorized training/QA interaction and execution contract."""

ROLE = 'prefix_selection_shared_bf16_context4096_action500_reserve500_stream_boundary_v2'
BUDGET = dict(total_tokens=4096, action_tokens=500, answer_reserve_tokens=500)


def contract(temperature, responses):
    return dict(kind=ROLE, compute_dtype='bfloat16', temperature=temperature,
        top_p=1., top_k=-1, n=responses, max_action_tokens=500, max_start_tokens=2048,
        max_request_tokens=4096, max_model_tokens=4096, max_observation_tokens=500,
        max_search_turns=4, answer_reserve_tokens=500, early_final=True,
        observation_truncation='clip_body_preserve_information_wrapper',
        action_boundary='first_complete_action_seen_in_actual_chunk_stream',
        synchronous_native_output_commit=True, scoring='unchanged_strict_qa_em_and_f1')
