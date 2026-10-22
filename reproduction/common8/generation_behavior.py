"""Common8 generation with actual raw-token and behavior-probability transport.

Other profiles keep the upstream controller. This path never retokenizes model
output, removes a trajectory, or permits a cropped request to be optimized with
a different context. Its private evidence does not constitute GPU admission.
"""
import copy
import uuid

import numpy as np
import torch

from verl import DataProto
from reproduction.common8.behavior_trace import (
    BehaviorTraceError, document_sha256, make_turn, raw_prefix_to_first_closing_tag,
)
from reproduction.common8.sampling_capture import validate_effective_stop_provenance
from reproduction.common8.context_budget import protocol, final_stage, observation_room, observation_tokens


def _require(condition, reason):
    if not condition:
        raise BehaviorTraceError('Common8 generation rejected: ' + reason)


def _left_pad(rows, pad):
    _require(bool(rows) and all(row for row in rows), 'empty_actual_request')
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, -len(row):] = torch.tensor(row, dtype=torch.long)
        attention[index, -len(row):] = 1
    positions = (attention.cumsum(-1) - 1) * attention
    return dict(input_ids=ids, attention_mask=attention, position_ids=positions)


def prefix_text_decoders(tokenizer):
    """Keep Qwen fast-tokenizer text while avoiding repeated list conversion.

    Prefix tokens have already been validated as Python integer lists. With
    cleanup disabled, the pinned Qwen2TokenizerFast implementation returns its
    native decoder text directly. Other tokenizers retain their existing API.
    """
    if (type(tokenizer).__name__ == 'Qwen2TokenizerFast' and
            type(tokenizer).__module__ == 'transformers.models.qwen2.tokenization_qwen2_fast' and
            getattr(tokenizer, 'is_fast', False)):
        decode = tokenizer.backend_tokenizer.decode
        return (lambda tokens: decode(tokens, skip_special_tokens=False),
                lambda tokens: decode(tokens, skip_special_tokens=True))
    return (lambda tokens: tokenizer.decode(tokens, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False),
            lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False))


def generate_padded(manager, active, captured_records=None):
    """Pad tensor and row identity together, and trim both after real generation."""
    size, world = len(active), manager.config.num_gpus
    _require(size > 0 and type(world) is int and world > 0, 'invalid_active_world_size')
    count = (-size) % world
    values = {name: value for name, value in active.batch.items()}
    identities = active.non_tensor_batch['behavior_row_id'].tolist()
    if count:
        values = {name: torch.cat((value, value[:1].repeat(count, *([1] * (value.ndim - 1)))))
                  for name, value in values.items()}
        identities += ['padding-' + uuid.uuid4().hex for _ in range(count)]
    call = DataProto.from_dict(values, non_tensors={'behavior_row_id': identities},
                               meta_info=copy.deepcopy(active.meta_info))
    result = manager.actor_rollout_wg.generate_sequences(call)
    if captured_records is not None and 'sampling_behavior' in result.non_tensor_batch:
        # Preserve an offending real return before row-count/identity checks.
        captured_records.extend(copy.deepcopy(result.non_tensor_batch['sampling_behavior'].tolist()))
    _require(len(result) == size + count, 'actual_output_row_count_changed')
    _require('sampling_behavior' in result.non_tensor_batch and
             len(result.non_tensor_batch['sampling_behavior']) == len(result),
             'missing_row_aligned_actual_behavior')
    records = result.non_tensor_batch['sampling_behavior'].tolist()
    _require([record.get('row_id') for record in records] == identities,
             'actual_output_row_identity_changed')
    # DataProto.__getitem__ returns an Item; construct a real batch explicitly.
    return DataProto.from_dict({name: value[:size] for name, value in result.batch.items()},
                non_tensors={name: value[:size] for name, value in result.non_tensor_batch.items()},
                meta_info=copy.deepcopy(result.meta_info))


def validated_raw_turn(manager, output, local, record, *, request_tokens, row_id, identity, call_id, ordinal, sampling):
    """Shared native-token/probability/stop checks for batch and independent requests."""
    tokenizer = manager.tokenizer
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    special = sorted(set(int(value) for value in tokenizer.all_special_ids))
    _require(record['actual_request_token_ids'] == request_tokens and
             record['policy_snapshot_id'] == identity['policy_snapshot_id'] and
             record['rollout_call_id'] == call_id and record['rollout_ordinal'] == ordinal and
             record['eos_token_id'] == eos and record['pad_token_id'] == pad,
             'actual_sampling_identity_or_request_changed')
    current = record['sampling']
    if sampling is None:
        sampling = copy.deepcopy(current)
    _require(current == sampling, 'actual_sampler_changed_between_turns_or_ranks')
    count = record['raw_generated_token_count']
    _require(type(count) is int and count == len(record['raw_generated_token_ids']) and
             len(record['selected_logprobs']) == count,
             'actual_pre_padding_raw_count_changed')
    _require(output.batch['responses'][local, :count].tolist() == record['raw_generated_token_ids'] and
             output.batch['sampling_mask'][local].sum().item() == count,
             'raw_tensor_or_sampling_mask_changed')
    original = torch.tensor(record['selected_logprobs'], dtype=torch.float32)
    _require(torch.equal(output.batch['sampling_log_probs'][local, :count].cpu(), original),
             'native_selected_probability_transport_changed')
    _require('effective_stop_provenance' in record and 'request_id' in record,
             'missing_actual_group_stop_provenance')
    proof = record['effective_stop_provenance']
    try:
        stop_ids, seq_eos = validate_effective_stop_provenance(proof,
            request_id=record['request_id'], actual_request_token_ids=request_tokens,
            sampling=current, eos_token_id=eos, pad_token_id=pad)
    except (ValueError, TypeError, KeyError) as error:
        raise BehaviorTraceError('invalid_effective_stop_provenance') from error
    config_proof = proof['generation_config']
    _require(proof['rollout_call_id'] == call_id and proof['row_id'] == row_id and
             config_proof['run_id'] == identity['run_id'] and
             config_proof['slurm_job_id'] == identity['slurm_job_id'] and
             config_proof['source_map_sha256'] == document_sha256(identity['source_sha256']),
             'actual_group_stop_identity_changed')
    visible_decode, executor_decode = prefix_text_decoders(tokenizer)
    cut = raw_prefix_to_first_closing_tag(record['raw_generated_token_ids'], visible_decode,
        eos_token_id=eos, pad_token_id=pad, special_token_ids=special,
        executor_decode=executor_decode,
        effective_stop_token_ids=sorted(stop_ids))
    # Keep primary EOS termination semantics. An alternate configured
    # terminal requires a matching native reason, or actual sequence EOS.
    if cut['trailing_eos_excluded']:
        _require(record['finish_reason'] == 'stop' and
                 record['stop_reason'] in (None, eos), 'unproven_actual_EOS_stop')
    elif cut.get('trailing_actual_stop_excluded', False):
        terminal = record['raw_generated_token_ids'][-1]
        _require(record['finish_reason'] == 'stop' and
                 (type(record['stop_reason']) is int and record['stop_reason'] == terminal or
                  record['stop_reason'] is None and seq_eos == terminal),
                 'unproven_actual_terminal_stop')
    return cut, sampling


def _observation_ids(manager, texts, limits=None):
    if limits is not None:
        rows = [observation_tokens(manager.tokenizer, text, min(int(manager.config.max_obs_length), limit))
                for text, limit in zip(texts, limits)]
        return [row[0] for row in rows], [row[1] for row in rows]
    # Keep the original environment tokenization and max_obs_length=500.
    padded = manager._process_next_obs(texts).detach().cpu()
    pad = manager.tokenizer.pad_token_id
    return [[int(value) for value in row.tolist() if value != pad] for row in padded]


def run(manager, gen_batch, initial_input_ids):
    records = []
    try:
        return _run(manager, gen_batch, initial_input_ids, records)
    except Exception as error:
        from reproduction.common8.behavior_runtime import failed_generation_evidence
        identity = gen_batch.meta_info.get('behavior_identity')
        if isinstance(identity, dict):
            # Record tokens only in the native private namespace. Public failure
            # text names the invariant; it never includes model or question text.
            failed_generation_evidence(identity, records, type(error).__name__)
        raise


def _run(manager, gen_batch, initial_input_ids, captured_records):
    config, tokenizer = manager.config, manager.tokenizer
    budget = protocol(config)
    _require(config.max_turns == 4 and not config.no_think_rl, 'generation_recipe_changed')
    row_ids = gen_batch.non_tensor_batch.get('behavior_row_id')
    identity = gen_batch.meta_info.get('behavior_identity')
    _require(row_ids is not None and len(row_ids) == len(gen_batch) and
             len(set(row_ids.tolist())) == len(gen_batch) and isinstance(identity, dict),
             'missing_complete_row_identity')
    row_ids = row_ids.tolist()
    _require(row_ids == identity['trajectory_ids'], 'initial_trajectory_order_changed')
    _require(gen_batch.meta_info.get('recompute_log_prob') is False,
             'per_turn_HF_recompute_must_not_precede_context_audit')
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    special = sorted(set(int(value) for value in tokenizer.all_special_ids))
    prompt_limit = min(config.max_start_length, budget['total_tokens'] - budget['answer_reserve_tokens'])
    left = initial_input_ids[:, -prompt_limit:].detach().cpu().long()
    attention = gen_batch.batch['attention_mask'][:, -left.shape[1]:].detach().cpu().long()
    _require(left.shape == attention.shape and bool(((attention == 0) | (attention == 1)).all()),
             'initial_prompt_shape_or_mask_changed')
    prompts = [[int(value) for value, valid in zip(row.tolist(), mask.tolist()) if valid]
               for row, mask in zip(left, attention)]
    _require(all(prompt and pad not in prompt for prompt in prompts), 'actual_prompt_contains_pad_or_is_empty')
    # First vLLM request and final HF prompt now use the same original 2048-token
    # prefix selection. Later requests are not silently cropped to a new context.
    responses = [[] for _ in row_ids]
    probabilities = [[] for _ in row_ids]
    info_masks = [[] for _ in row_ids]
    turns = [[] for _ in row_ids]
    active = [True] * len(row_ids)
    turns_stats = [1] * len(row_ids)
    valid_actions, valid_searches, actual_retrieval = ([0] * len(row_ids) for _ in range(3))
    active_counts = [len(row_ids)]
    final_meta = {}
    clipped_observations = [0] * len(row_ids)
    early_final_turns = [False] * len(row_ids)
    sampling = None
    for ordinal in range(config.max_turns + 1):
        indices = [index for index, value in enumerate(active) if value]
        if not indices:
            break
        # Normal and early-final trajectories share the same fixed500 sampler.
        # Separate their environment execution so one long row cannot end others.
        pending = indices
        stages = {index: final_stage(len(prompts[index]) + len(responses[index]),
                                    ordinal, config.max_turns, budget) for index in pending}
        for final_turn in (False, True):
            indices = [index for index in pending if stages[index] == final_turn]
            if not indices:
                continue
            requests = [prompts[index] + responses[index] for index in indices]
            _require(all(len(request) + budget['action_tokens'] <= budget['total_tokens'] for request in requests),
                     'invalid_preplanned_total_context_budget')
            call_id = identity['policy_snapshot_id'] + '-call-' + uuid.uuid4().hex
            call_meta = {name: gen_batch.meta_info[name] for name in
                         ('eos_token_id', 'pad_token_id', 'do_sample', 'validate')
                         if name in gen_batch.meta_info}
            call_meta.update(recompute_log_prob=False, behavior_call_id=call_id,
                             behavior_rollout_ordinal=ordinal,
                             policy_snapshot_id=identity['policy_snapshot_id'])
            active_batch = DataProto.from_dict(_left_pad(requests, pad),
                non_tensors={'behavior_row_id': [row_ids[index] for index in indices]},
                meta_info=call_meta)
            output = generate_padded(manager, active_batch, captured_records)
            final_meta.update(output.meta_info)
            records = output.non_tensor_batch['sampling_behavior'].tolist()
            cuts = []
            texts = [''] * len(row_ids)
            for local, index in enumerate(indices):
                record = records[local]
                cut, sampling = validated_raw_turn(manager, output, local, record,
                    request_tokens=requests[local], row_id=row_ids[index], identity=identity,
                    call_id=call_id, ordinal=ordinal, sampling=sampling)
                cuts.append(cut)
                texts[index] = cut['execution_text']
            next_obs, dones, valid_action, is_search = manager.execute_predictions(
                texts, tokenizer.pad_token, torch.tensor([index in indices for index in range(len(row_ids))], dtype=torch.bool),
                do_search=not final_turn)
            rooms = [0] * len(row_ids)
            for local, index in enumerate(indices):
                rooms[index] = observation_room(len(requests[local]), len(cuts[local]['kept_token_ids']), budget)
            if final_turn:
                obs_ids, clipped = [[] for _ in row_ids], [False] * len(row_ids)
            else:
                obs_ids, clipped = _observation_ids(manager, next_obs, rooms)
            captured_records.append(dict(role='actual_environment_event', rollout_call_id=call_id,
                row_ids=[row_ids[index] for index in indices],
                observation_or_correction_token_ids=[obs_ids[index] for index in indices],
                is_search=[int(is_search[index]) for index in indices], final_turn=final_turn))
            for local, index in enumerate(indices):
                record, cut = records[local], cuts[local]
                kept, environment = cut['kept_token_ids'], obs_ids[index]
                _require(len(prompts[index]) + len(responses[index]) + len(kept) + len(environment) <= budget['total_tokens'],
                         'invalid_total_context_plan')
                clipped_observations[index] += int(clipped[index])
                early_final_turns[index] |= bool(final_turn and ordinal < config.max_turns)
                observation = environment if is_search[index] else []
                correction = environment if not is_search[index] else []
                turn = make_turn(record, trajectory_id=row_ids[index], turn_index=ordinal,
                    kept_token_count=cut['kept_token_count'], special_token_ids=special,
                    observation_token_ids=observation, correction_token_ids=correction,
                    final_turn=final_turn or bool(dones[index]), prefix_cut=cut)
                turns[index].append(turn)
                responses[index].extend(kept + environment)
                probabilities[index].extend(record['selected_logprobs'][:len(kept)] + [0.0] * len(environment))
                info_masks[index].extend([1] * len(kept) + [0] * len(environment))
                valid_actions[index] += int(valid_action[index])
                valid_searches[index] += int(is_search[index])
                actual_retrieval[index] += int(is_search[index]) if not final_turn else 0
                active[index] = not (final_turn or bool(dones[index]))
            if not final_turn:
                # Preserve the legacy statistic, including inactive rows' done flag.
                for index in indices:
                    if not dones[index]:
                        turns_stats[index] += 1
            active_counts.append(sum(active))
    return assemble_output(row_ids=row_ids, identity=identity, budget=budget, left=left,
        attention=attention, pad=pad, responses=responses, probabilities=probabilities,
        info_masks=info_masks, turns=turns, final_meta=final_meta, sampling=sampling,
        clipped_observations=clipped_observations, early_final_turns=early_final_turns,
        turns_stats=turns_stats, active=active, valid_actions=valid_actions, valid_searches=valid_searches,
        actual_retrieval=actual_retrieval, active_counts=active_counts)


def assemble_output(*, row_ids, identity, budget, left, attention, pad, responses, probabilities,
                    info_masks, turns, final_meta, sampling, clipped_observations, early_final_turns,
                    turns_stats, active, valid_actions, valid_searches, actual_retrieval, active_counts):
    """Restore canonical input order after all complete native trajectories finish."""
    _require(sampling is not None and all(turns), 'missing_actual_rollout')
    width = max(map(len, responses))
    _require(width > 0, 'no_actual_response_tokens')
    response_tensor = torch.full((len(row_ids), width), pad, dtype=torch.long)
    response_attention = torch.zeros_like(response_tensor)
    model_mask = torch.zeros_like(response_tensor)
    selected = torch.zeros((len(row_ids), width), dtype=torch.float32)
    for index, response in enumerate(responses):
        response_tensor[index, :len(response)] = torch.tensor(response, dtype=torch.long)
        response_attention[index, :len(response)] = 1
        model_mask[index, :len(response)] = torch.tensor(info_masks[index], dtype=torch.long)
        selected[index, :len(response)] = torch.tensor(probabilities[index], dtype=torch.float32)
    mask = torch.cat((attention, response_attention), dim=1)
    values = dict(prompts=left, responses=response_tensor,
        responses_with_info_mask=torch.where(model_mask.bool(), response_tensor, pad),
        input_ids=torch.cat((left, response_tensor), dim=1), attention_mask=mask,
        info_mask=torch.cat((attention, model_mask), dim=1),
        position_ids=(mask.cumsum(-1) - 1) * mask,
        sampling_log_probs=selected, sampling_mask=model_mask.bool())
    final_meta.update(behavior_identity=copy.deepcopy(identity), behavior_sampling=sampling,
        behavior_context_policy=budget, clipped_observation_stats=clipped_observations,
        early_final_turn_stats=early_final_turns,
        turns_stats=turns_stats, active_mask=active, valid_action_stats=valid_actions,
        valid_search_stats=valid_searches, actual_retrieval_stats=actual_retrieval)
    print('ACTIVE_TRAJ_NUM:', active_counts)
    object_turns = np.empty(len(turns), dtype=object)
    object_turns[:] = turns
    return DataProto.from_dict(values, non_tensors={
        'behavior_trajectory_id': row_ids, 'behavior_turns': object_turns,
    }, meta_info=final_meta)
