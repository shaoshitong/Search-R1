"""Logical500-token actions assembled from unmodified native prefix/suffix calls.

The logical sampling map is explicitly a policy/budget contract, never a claim
that a20-token native request had max_tokens500. Every real request retains its
own complete sampler, stop provenance, token IDs and selected log probabilities.
"""
import copy
import math

ROLE='prefix20_composite_logical_action_v1'
BUDGET_SEMANTICS='logical_action_cap_with_unmodified_native_chunk_samplers'


def logical_sampling(actual):
    result=copy.deepcopy(actual)
    result['max_tokens']=500
    result['budget_semantics']=BUDGET_SEMANTICS
    return result


def make_turn(records,*,trajectory_id,turn_index,special_token_ids,prefix_cut,
              observation_token_ids=(),correction_token_ids=(),final_turn=False):
    if not records:raise ValueError('PREFIX20_LOGICAL_ACTION_NEEDS_NATIVE_CALLS')
    first=records[0]
    result=dict(role=ROLE,trajectory_id=trajectory_id,turn_index=turn_index,
        rollout_call_id=first['rollout_call_id'],policy_snapshot_id=first['policy_snapshot_id'],
        eos_token_id=first['eos_token_id'],pad_token_id=first['pad_token_id'],
        sampling=logical_sampling(first['sampling']),special_token_ids=list(special_token_ids),
        native_chunks=copy.deepcopy(records),prefix_cut=copy.deepcopy(prefix_cut),
        kept_token_count=prefix_cut['kept_token_count'],
        observation_token_ids=list(observation_token_ids),correction_token_ids=list(correction_token_ids),
        final_turn=bool(final_turn))
    verify_turn(result)
    return result


def verify_turn(turn):
    from reproduction.common8.behavior_trace import (
        _require,_tokens,_name,canonical_bytes,document_sha256,_verify_dual_cut,
        _NATIVE_STOP_FIELDS,
    )
    from reproduction.common8.behavior_probability import verify_sampling
    from reproduction.common8.sampling_capture import validate_effective_stop_provenance
    fields={'role','trajectory_id','turn_index','rollout_call_id','policy_snapshot_id',
            'eos_token_id','pad_token_id','sampling','special_token_ids','native_chunks',
            'prefix_cut','kept_token_count','observation_token_ids','correction_token_ids','final_turn'}
    _require(type(turn) is dict and set(turn)==fields and turn['role']==ROLE,'prefix20_logical_turn_schema_changed')
    for name in ('trajectory_id','rollout_call_id','policy_snapshot_id'):_name(turn[name],name)
    _require(type(turn['turn_index']) is int and 0<=turn['turn_index']<5 and
             type(turn['final_turn']) is bool,'prefix20_invalid_logical_turn')
    policy=turn['sampling']
    _require(policy.get('budget_semantics')==BUDGET_SEMANTICS,'prefix20_missing_logical_budget_semantics')
    verify_sampling(policy)
    chunks=turn['native_chunks']
    _require(isinstance(chunks,list) and 1<=len(chunks)<=21,'prefix20_missing_or_excess_native_chunks')
    request0=None;raw=[];probabilities=[];calls=set();request_ids=set();stop_set=None;previous_length=0
    for ordinal,chunk in enumerate(chunks):
        _require(type(chunk) is dict and _NATIVE_STOP_FIELDS<=set(chunk) and
                 set(chunk)-_NATIVE_STOP_FIELDS<={'schema_version','row_id','rollout_ordinal'},
                 'prefix20_native_chunk_schema_changed')
        for name in ('request_id','rollout_call_id'):
            _name(chunk.get(name),name)
        _require(chunk['row_id']==turn['trajectory_id'] and chunk['rollout_ordinal']==turn['turn_index'] and
                 chunk['policy_snapshot_id']==turn['policy_snapshot_id'] and
                 chunk['eos_token_id']==turn['eos_token_id'] and chunk['pad_token_id']==turn['pad_token_id'],
                 'prefix20_native_chunk_identity_changed')
        _require(chunk['request_id'] not in request_ids and chunk['rollout_call_id'] not in calls,
                 'prefix20_duplicate_native_call')
        calls.add(chunk['rollout_call_id']);request_ids.add(chunk['request_id'])
        request=_tokens(chunk['actual_request_token_ids'],'actual_request_token_ids')
        if request0 is None:request0=request
        _require(request==request0+raw,'prefix20_native_suffix_request_changed')
        actual=chunk['sampling'];maximum=actual.get('max_tokens')
        _require(type(maximum) is int and 1<=maximum<=500-len(raw),
                 'prefix20_native_chunk_exceeds_remaining_action_budget')
        _require(canonical_bytes(logical_sampling(actual))==canonical_bytes(policy),
                 'prefix20_native_sampling_policy_changed')
        tokens=_tokens(chunk['raw_generated_token_ids'],'raw_generated_token_ids')
        values=chunk['selected_logprobs']
        _require(type(chunk['raw_generated_token_count']) is int and chunk['raw_generated_token_count']==len(tokens) and
                 len(tokens)<=maximum and len(values)==len(tokens) and
                 all(type(p) in (int,float) and math.isfinite(p) and p<=0 for p in values),
                 'prefix20_native_token_or_logprob_count_changed')
        proof=chunk['effective_stop_provenance']
        stops,seq_eos=validate_effective_stop_provenance(proof,request_id=chunk['request_id'],
            actual_request_token_ids=request,sampling=actual,eos_token_id=turn['eos_token_id'],pad_token_id=turn['pad_token_id'])
        stops=set(stops)
        _require(proof['row_id']==turn['trajectory_id'] and proof['rollout_call_id']==chunk['rollout_call_id'],
                 'prefix20_actual_stop_caller_changed')
        if stop_set is None:stop_set=stops
        _require(stops==stop_set,'prefix20_effective_stops_changed_between_chunks')
        if ordinal<len(chunks)-1:
            _require(len(tokens)==maximum and chunk['finish_reason']=='length' and
                     not stops.intersection(tokens),'prefix20_continuation_after_short_or_terminal_chunk')
            previous_length+=len(tokens)
        raw.extend(tokens);probabilities.extend(values)
    _require(turn['rollout_call_id']==chunks[0]['rollout_call_id'],'prefix20_logical_call_binding_changed')
    cut=turn['prefix_cut'];count=turn['kept_token_count']
    _require(type(count) is int and previous_length<=count<=len(raw),'prefix20_selected_prefix_dropped')
    _verify_dual_cut(cut,raw,count,set(turn['special_token_ids'])|{turn['eos_token_id'],turn['pad_token_id']},
                     turn['eos_token_id'],turn['pad_token_id'],stop_set)
    if cut['trailing_eos_excluded'] or cut.get('trailing_actual_stop_excluded',False):
        terminal=raw[-1];last=chunks[-1]
        _require(last['finish_reason']=='stop' and
                 (last['stop_reason']==terminal or last['stop_reason'] is None and seq_eos==terminal),
                 'prefix20_unproven_terminal_stop')
    observation=_tokens(turn['observation_token_ids'],'observation_token_ids')
    correction=_tokens(turn['correction_token_ids'],'correction_token_ids')
    _require(not (observation and correction) and not (turn['final_turn'] and (observation or correction)),
             'prefix20_invalid_environment_after_action')
    return request0,raw,probabilities,observation+correction


def provenance_records(turn):
    return turn['native_chunks'] if turn.get('role')==ROLE else [turn]
