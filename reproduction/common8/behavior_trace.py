"""CPU protocol for actual per-call tokens; no model/GPU/healthy admission.

Only native private artifacts may contain the full journal. Public reports
contain scalar/hash metadata. No token is retokenized, no failed trajectory is
filtered, and an audit failure rejects the complete original512/G5 outer.
The masked-mean formula is unchanged; its numerical token count need not match
the old decoded/retokenized implementation.
"""
import copy
import hashlib
import json
import math
import os
import stat
from pathlib import Path

SCHEMA_VERSION = 1
TRACE_ROLE = 'common8_actual_behavior_trace'
AUDIT_ROLE = 'common8_behavior_context_audit'
LOGPROB_SEMANTICS = 'selected_token_logprob_at_actual_sampling_call'
DECODE_PROJECTION_ROLE = 'deterministic_tokenizer_text_projection_not_utf8_validity_proof'
DUAL_PROJECTION_ROLE = 'raw_authored_tokens_and_upstream_skip_special_executor_v1'
SAMPLING_FIELDS = frozenset((
    'n', 'best_of', 'do_sample', 'temperature', 'top_p', 'top_k', 'min_p',
    'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'ignore_eos',
    'logprobs', 'min_tokens', 'logits_processors', 'logprob_semantics', 'max_tokens',
))
_ACTOR_FIELDS = frozenset(('trajectory_id', 'input_ids', 'attention_mask', 'info_mask',
                         'position_ids', 'prompt_width'))
_NATIVE_FIELDS = frozenset(('rollout_call_id', 'actual_request_token_ids', 'raw_generated_token_ids',
    'raw_generated_token_count', 'selected_logprobs', 'finish_reason', 'stop_reason',
    'eos_token_id', 'pad_token_id', 'sampling', 'policy_snapshot_id'))
_TURN_FIELDS = _NATIVE_FIELDS | frozenset(('trajectory_id', 'turn_index', 'kept_token_count',
    'observation_token_ids', 'correction_token_ids', 'final_turn', 'special_token_ids', 'prefix_cut'))
_NATIVE_STOP_FIELDS = _NATIVE_FIELDS | frozenset(('request_id', 'effective_stop_provenance'))
_TURN_STOP_FIELDS = _TURN_FIELDS | frozenset(('request_id', 'effective_stop_provenance'))
_DUAL_CUT_FIELDS = frozenset(('kept_token_count', 'kept_token_ids', 'closing_tag', 'execution_text',
    'decoded_retained_prefix', 'retained_trailing_whitespace', 'token_boundary_whitespace_retained',
    'trailing_eos_excluded', 'raw_generated_token_count', 'raw_tokens_sha256', 'retained_tokens_sha256',
    'retained_token_boundary_suffix', 'token_boundary_suffix_classification', 'decode_projection_role',
    'decoded_full_generation', 'full_decode_replacement_count', 'retained_decode_replacement_count',
    'dual_decode_projection_role', 'decoded_executor_full_generation', 'decoded_executor_retained_prefix',
    'decoded_executor_previous_prefix', 'executor_full_decode_replacement_count',
    'executor_retained_decode_replacement_count', 'included_nonterminal_special_positions',
    'effective_stop_token_ids_sha256'))
_DUAL_ACTUAL_STOP_CUT_FIELDS = _DUAL_CUT_FIELDS | frozenset((
    'trailing_actual_stop_excluded', 'excluded_trailing_stop_token_id'))
_TRACE_FIELDS = frozenset(('schema_version', 'role', 'run_id', 'source_sha256', 'slurm_job_id',
    'logical_update', 'policy_snapshot_id', 'sampling', 'parent_group_ids', 'trajectories', 'actor_rows'))


class BehaviorTraceError(ValueError):
    """Controlled rejection. Messages contain no raw tokens or question text."""


def _require(condition, reason):
    if not condition:
        raise BehaviorTraceError(reason)


def canonical_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError) as error:
        raise BehaviorTraceError('noncanonical_trace_value') from error


def document_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _list(value, name):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    _require(type(value) in (list, tuple), 'invalid_sequence:' + name)
    return list(value)


def _tokens(value, name):
    result = _list(value, name)
    _require(all(type(token) is int and token >= 0 for token in result), 'invalid_token_ids:' + name)
    return result


def _mask(value, name):
    result = _list(value, name)
    _require(all(type(item) in (int, bool) and item in (0, 1) for item in result), 'invalid_mask:' + name)
    return [int(item) for item in result]


def _name(value, name):
    _require(isinstance(value, str) and bool(value) and len(value) <= 512, 'invalid_identity:' + name)
    return value


def _sampling(value):
    _require(type(value) is dict and SAMPLING_FIELDS <= set(value), 'incomplete_actual_sampling_metadata')
    _require(value['logprob_semantics'] == LOGPROB_SEMANTICS, 'unknown_selected_logprob_semantics')
    _require(type(value['do_sample']) is bool and type(value['ignore_eos']) is bool,
             'invalid_actual_sampling_boolean')
    for key in ('n', 'top_k', 'logprobs', 'max_tokens'):
        _require(type(value[key]) is int, 'invalid_actual_sampling_integer:' + key)
    _require(value['n'] == 1 and value['logprobs'] == 1, 'not_single_actual_selected_logprob')
    _require(value['max_tokens'] == 500, 'actual_generation_budget_not_original500')
    for key in ('temperature', 'top_p', 'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty'):
        _require(type(value[key]) in (int, float) and math.isfinite(value[key]), 'invalid_sampling_number:' + key)
    _require(value['best_of'] is None or type(value['best_of']) is int, 'invalid_actual_best_of')
    _require(value['min_tokens'] is None or type(value['min_tokens']) is int, 'invalid_actual_min_tokens')
    _require(value['logits_processors'] is None or type(value['logits_processors']) is list,
             'invalid_actual_logits_processors')
    canonical_bytes(value)
    return copy.deepcopy(value)


def raw_prefix_to_first_closing_tag(raw_generated_token_ids, decode, *, eos_token_id,
                                   pad_token_id, special_token_ids, executor_decode=None,
                                   effective_stop_token_ids=None):
    """Locate a decoded closing tag, then take only the original token prefix.

    The caller's decoder must retain special tokens and disable cleanup. A
    closing tag inside a token retains that complete sampled token, including
    any suffix. Only the executor receives the original character projection;
    context, loss masks and reward decoding keep the full authored token.
    With no tag only one genuine trailing terminal token is removed; alternate
    configured stops require the dual decoder and authenticated stop provenance.
    Internal special tokens are never silently removed. Actual raw length comes from vLLM before
    padding; this function never guesses that length from EOS/pad IDs.
    Deterministic decode may be lossy. Replacement characters remain verbatim,
    with an explicit census; full decoded text must contain the selected prefix
    exactly, without extending that token prefix to repair UTF-8.
    """
    if executor_decode is not None:
        return _dual_raw_prefix(raw_generated_token_ids, decode, executor_decode,
            eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            special_token_ids=special_token_ids, effective_stop_token_ids=effective_stop_token_ids)
    _require(effective_stop_token_ids is None, 'stop_proof_without_dual_decode')
    raw = _tokens(raw_generated_token_ids, 'raw_generated_token_ids')
    _require(type(eos_token_id) is int and type(pad_token_id) is int, 'missing_actual_eos_pad_identity')
    special = set(_tokens(special_token_ids, 'special_token_ids')) | {eos_token_id, pad_token_id}
    trailing_eos = bool(raw and raw[-1] == eos_token_id)
    eligible = raw[:-1] if trailing_eos else raw
    text = decode(eligible)
    _require(isinstance(text, str), 'invalid_decoder_result')
    # Match the upstream decoded-response boundary: a search closing tag
    # takes precedence anywhere in the text. The executor separately chooses
    # the first complete action in that retained text; do not reorder actions
    # or discard an earlier answer from the original sampled token prefix.
    closing = '</search>' if '</search>' in text else '</answer>' if '</answer>' in text else None
    count, retained_text, execution, suffix = len(eligible), text, text, ''
    if closing is not None:
        # Bounded500-token generation makes this independent CPU scan small;
        # it avoids assumptions about monotone Unicode prefix decoding.
        for count in range(1, len(eligible) + 1):
            candidate = decode(eligible[:count])
            _require(isinstance(candidate, str), 'invalid_decoder_result')
            if closing in candidate:
                end = candidate.find(closing) + len(closing)
                remainder = candidate[end:]
                _require(text.startswith(candidate), 'ambiguous_decoding_boundary')
                retained_text, execution, suffix = candidate, candidate[:end], remainder
                break
        else:
            raise BehaviorTraceError('closing_tag_without_raw_prefix_boundary')
    kept = eligible[:count]
    _require(not special.intersection(kept), 'included_special_or_pad_token')
    classification = 'none' if not suffix else 'whitespace' if suffix.isspace() else 'nonwhitespace'
    whitespace = suffix if classification == 'whitespace' else ''
    return dict(kept_token_count=count, kept_token_ids=kept, closing_tag=closing,
                execution_text=execution, decoded_retained_prefix=retained_text,
                retained_token_boundary_suffix=suffix, token_boundary_suffix_classification=classification,
                decode_projection_role=DECODE_PROJECTION_ROLE, decoded_full_generation=text,
                full_decode_replacement_count=text.count('\ufffd'),
                retained_decode_replacement_count=retained_text.count('\ufffd'),
                retained_trailing_whitespace=whitespace,
                token_boundary_whitespace_retained=bool(whitespace),
                trailing_eos_excluded=trailing_eos, raw_generated_token_count=len(raw),
                raw_tokens_sha256=document_sha256(raw), retained_tokens_sha256=document_sha256(kept))


def _dual_raw_prefix(raw_generated_token_ids, decode, executor_decode, *, eos_token_id,
                     pad_token_id, special_token_ids, effective_stop_token_ids):
    """Only the upstream executor projection skips special symbols.

    One trailing configured terminal token is excluded from the eligible actor
    and executor prefix, including an actual stop that shares the padding ID.
    The raw tokens and selected LP journal remain intact; all other retained
    tokens, LP and loss/context masks stay unchanged. Reward decodes that
    authored prefix with nonterminal special symbols visible. Stop provenance
    is authenticated before this call, termination before action execution,
    and both again in the completed trace audit. This is not equivalence to the
    old retokenized future context or cleaned reward.
    """
    raw = _tokens(raw_generated_token_ids, 'raw_generated_token_ids')
    _require(type(eos_token_id) is int and type(pad_token_id) is int, 'missing_actual_eos_pad_identity')
    special = set(_tokens(special_token_ids, 'special_token_ids')) | {eos_token_id, pad_token_id}
    _require(effective_stop_token_ids is not None, 'missing_effective_stop_provenance')
    stops = sorted(set(_tokens(effective_stop_token_ids, 'effective_stop_token_ids')))
    _require(eos_token_id in stops, 'tokenizer_eos_not_actual_stop')
    trailing = bool(raw and raw[-1] == eos_token_id)
    trailing_actual_stop = bool(raw and raw[-1] != eos_token_id and raw[-1] in stops)
    eligible = raw[:-1] if trailing or trailing_actual_stop else raw
    visible, projected = decode(eligible), executor_decode(eligible)
    _require(isinstance(visible, str) and isinstance(projected, str), 'invalid_decoder_result')
    closing = '</search>' if '</search>' in projected else '</answer>' if '</answer>' in projected else None
    count = len(eligible)
    retained_visible, retained_projected, execution, suffix = visible, projected, projected, ''
    if closing is not None:
        for count in range(1, len(eligible) + 1):
            candidate = executor_decode(eligible[:count])
            _require(isinstance(candidate, str), 'invalid_decoder_result')
            if closing in candidate:
                retained_projected, retained_visible = candidate, decode(eligible[:count])
                _require(isinstance(retained_visible, str) and visible.startswith(retained_visible) and
                         projected.startswith(candidate), 'ambiguous_decoding_boundary')
                end = candidate.find(closing) + len(closing)
                execution, suffix = candidate[:end], candidate[end:]
                break
        else:
            raise BehaviorTraceError('closing_tag_without_raw_prefix_boundary')
    kept = eligible[:count]
    _require(not (set(stops) | {eos_token_id, pad_token_id}).intersection(kept),
             'included_actual_stop_or_pad_token')
    positions = [index for index, token in enumerate(kept) if token in special]
    classification = 'none' if not suffix else 'whitespace' if suffix.isspace() else 'nonwhitespace'
    previous = executor_decode(eligible[:max(0, count - 1)])
    _require(isinstance(previous, str), 'invalid_decoder_result')
    cut = dict(kept_token_count=count, kept_token_ids=kept, closing_tag=closing, execution_text=execution,
        decoded_retained_prefix=retained_visible, retained_token_boundary_suffix=suffix,
        token_boundary_suffix_classification=classification,
        retained_trailing_whitespace=suffix if classification == 'whitespace' else '',
        token_boundary_whitespace_retained=classification == 'whitespace',
        trailing_eos_excluded=trailing, raw_generated_token_count=len(raw),
        raw_tokens_sha256=document_sha256(raw), retained_tokens_sha256=document_sha256(kept),
        decode_projection_role=DECODE_PROJECTION_ROLE, decoded_full_generation=visible,
        full_decode_replacement_count=visible.count('\ufffd'),
        retained_decode_replacement_count=retained_visible.count('\ufffd'),
        dual_decode_projection_role=DUAL_PROJECTION_ROLE,
        decoded_executor_full_generation=projected, decoded_executor_retained_prefix=retained_projected,
        decoded_executor_previous_prefix=previous,
        executor_full_decode_replacement_count=projected.count('\ufffd'),
        executor_retained_decode_replacement_count=retained_projected.count('\ufffd'),
        included_nonterminal_special_positions=positions,
        effective_stop_token_ids_sha256=document_sha256(stops))
    if trailing_actual_stop:
        cut.update(trailing_actual_stop_excluded=True, excluded_trailing_stop_token_id=raw[-1])
    return cut


def _verified_effective_stops(turn):
    from reproduction.common8.sampling_capture import validate_effective_stop_provenance
    proof = turn['effective_stop_provenance']
    try:
        stops, seq_eos = validate_effective_stop_provenance(proof,
            request_id=turn['request_id'], actual_request_token_ids=turn['actual_request_token_ids'],
            sampling=turn['sampling'], eos_token_id=turn['eos_token_id'], pad_token_id=turn['pad_token_id'])
    except (ValueError, TypeError, KeyError) as error:
        raise BehaviorTraceError('invalid_effective_stop_provenance') from error
    _require(proof['rollout_call_id'] == turn['rollout_call_id'] and proof['row_id'] == turn['trajectory_id'],
             'effective_stop_call_or_row_changed')
    return set(stops), seq_eos


def _verify_dual_cut(cut, raw, count, special, eos, pad, stops):
    _require(type(cut) is dict and set(cut) in (_DUAL_CUT_FIELDS, _DUAL_ACTUAL_STOP_CUT_FIELDS) and
             cut['dual_decode_projection_role'] == DUAL_PROJECTION_ROLE and
             cut['decode_projection_role'] == DECODE_PROJECTION_ROLE,
             'dual_decode_proof_schema_changed')
    _require(type(cut['kept_token_count']) is int and cut['kept_token_count'] == count and
             _tokens(cut['kept_token_ids'], 'cut_kept_token_ids') == raw[:count] and
             type(cut['raw_generated_token_count']) is int and cut['raw_generated_token_count'] == len(raw) and
             cut['raw_tokens_sha256'] == document_sha256(raw) and
             cut['retained_tokens_sha256'] == document_sha256(raw[:count]), 'raw_prefix_proof_changed')
    _require(not (stops | {eos, pad}).intersection(raw[:count]), 'included_actual_stop_or_pad_token')
    _require(cut['effective_stop_token_ids_sha256'] == document_sha256(sorted(stops)) and
             cut['included_nonterminal_special_positions'] == [i for i, token in enumerate(raw[:count]) if token in special] and
             all(type(i) is int for i in cut['included_nonterminal_special_positions']),
             'included_nonterminal_special_proof_changed')
    _require(type(cut['trailing_eos_excluded']) is bool and
             cut['trailing_eos_excluded'] == bool(raw and raw[-1] == eos), 'trailing_eos_proof_changed')
    trailing_actual_stop = set(cut) == _DUAL_ACTUAL_STOP_CUT_FIELDS
    if trailing_actual_stop:
        _require(type(cut['trailing_actual_stop_excluded']) is bool and
                 cut['trailing_actual_stop_excluded'] is True and
                 type(cut['excluded_trailing_stop_token_id']) is int and
                 bool(raw) and cut['excluded_trailing_stop_token_id'] == raw[-1] and
                 raw[-1] != eos and raw[-1] in stops, 'trailing_actual_stop_proof_changed')
    for full_key, retained_key, full_count, retained_count in (
        ('decoded_full_generation', 'decoded_retained_prefix', 'full_decode_replacement_count', 'retained_decode_replacement_count'),
        ('decoded_executor_full_generation', 'decoded_executor_retained_prefix',
         'executor_full_decode_replacement_count', 'executor_retained_decode_replacement_count')):
        full, retained = cut[full_key], cut[retained_key]
        _require(isinstance(full, str) and isinstance(retained, str) and full.startswith(retained),
                 'ambiguous_decoding_boundary')
        _require(type(cut[full_count]) is int and type(cut[retained_count]) is int and
                 cut[full_count] == full.count('\ufffd') and cut[retained_count] == retained.count('\ufffd'),
                 'deterministic_decode_projection_or_replacement_census_changed')
    projected, retained, previous = (cut[key] for key in ('decoded_executor_full_generation',
                                    'decoded_executor_retained_prefix', 'decoded_executor_previous_prefix'))
    closing = '</search>' if '</search>' in projected else '</answer>' if '</answer>' in projected else None
    _require(cut['closing_tag'] == closing and isinstance(previous, str), 'full_decode_action_projection_changed')
    suffix = cut['retained_token_boundary_suffix']
    _require(isinstance(suffix, str) and isinstance(cut['execution_text'], str) and
             retained == cut['execution_text'] + suffix, 'decoded_executor_cut_suffix_changed')
    classification = 'none' if not suffix else 'whitespace' if suffix.isspace() else 'nonwhitespace'
    _require(cut['token_boundary_suffix_classification'] == classification and
             cut['retained_trailing_whitespace'] == (suffix if classification == 'whitespace' else '') and
             type(cut['token_boundary_whitespace_retained']) is bool and
             cut['token_boundary_whitespace_retained'] == (classification == 'whitespace'),
             'decoded_suffix_classification_changed')
    if closing:
        _require(count > 0 and closing in retained and closing not in previous and
                 cut['execution_text'] == retained[:retained.find(closing) + len(closing)] and
                 cut['execution_text'].endswith(closing), 'execution_is_not_original_character_projection')
    else:
        _require(count == len(raw) - int(cut['trailing_eos_excluded'] or trailing_actual_stop) and not suffix and
                 retained == projected and cut['decoded_retained_prefix'] == cut['decoded_full_generation'],
                 'unproven_no_tag_truncation')


def make_turn(sampling_record=None, *, trajectory_id, turn_index, kept_token_count,
              special_token_ids, observation_token_ids=(), correction_token_ids=(),
              final_turn=False, prefix_cut=None, **native_fields):
    """Accept a real row-aligned sampling_behavior dict or explicit native fields.

    No absent sampler value/count/logprob is inferred. Only four native transport
    metadata keys (schema/request/row/ordinal) may accompany its exact fields.
    """
    if sampling_record is not None:
        _require(type(sampling_record) is dict and not native_fields, 'ambiguous_native_sampling_record')
        native_schema = _NATIVE_STOP_FIELDS if 'effective_stop_provenance' in sampling_record else _NATIVE_FIELDS
        extra = set(sampling_record) - native_schema
        _require(extra <= {'schema_version', 'request_id', 'row_id', 'rollout_ordinal'}, 'unknown_native_sampling_field')
        if 'schema_version' in sampling_record:
            _require(type(sampling_record['schema_version']) is int and sampling_record['schema_version'] == 1,
                     'native_sampling_schema_changed')
        if 'row_id' in sampling_record:
            _require(sampling_record['row_id'] == trajectory_id, 'native_sampling_row_mismatch')
        if 'rollout_ordinal' in sampling_record:
            _require(type(sampling_record['rollout_ordinal']) is int and sampling_record['rollout_ordinal'] == turn_index,
                     'native_sampling_turn_mismatch')
        _require(native_schema <= set(sampling_record), 'missing_native_sampling_field')
        native_fields = {key: copy.deepcopy(sampling_record[key]) for key in native_schema}
    _require(set(native_fields) in (_NATIVE_FIELDS, _NATIVE_STOP_FIELDS), 'missing_or_extra_native_sampling_field')
    result = dict(native_fields, trajectory_id=_name(trajectory_id, 'trajectory_id'), turn_index=turn_index,
                  kept_token_count=kept_token_count, special_token_ids=_tokens(special_token_ids, 'special_token_ids'),
                  observation_token_ids=_tokens(observation_token_ids, 'observation_token_ids'),
                  correction_token_ids=_tokens(correction_token_ids, 'correction_token_ids'),
                  final_turn=final_turn, prefix_cut=copy.deepcopy(prefix_cut))
    _verify_turn(result)
    return result


def _verify_turn(turn):
    if isinstance(turn,dict) and turn.get('role')=='prefix20_composite_logical_action_v1':
        from search_g1.prefix20_split import verify_turn
        return verify_turn(turn)
    _require(type(turn) is dict and set(turn) in (_TURN_FIELDS, _TURN_STOP_FIELDS), 'turn_schema_changed')
    _name(turn['trajectory_id'], 'trajectory_id'); _name(turn['rollout_call_id'], 'rollout_call_id')
    _name(turn['policy_snapshot_id'], 'policy_snapshot_id')
    _require(type(turn['turn_index']) is int and 0 <= turn['turn_index'] < 5, 'invalid_turn_index')
    _require(type(turn['final_turn']) is bool, 'invalid_final_turn')
    request = _tokens(turn['actual_request_token_ids'], 'actual_request_token_ids')
    raw = _tokens(turn['raw_generated_token_ids'], 'raw_generated_token_ids')
    _require(bool(request) and type(turn['raw_generated_token_count']) is int and
             turn['raw_generated_token_count'] == len(raw), 'actual_raw_count_or_request_missing')
    probabilities = _list(turn['selected_logprobs'], 'selected_logprobs')
    _require(len(probabilities) == len(raw) and all(type(value) in (int, float) and
             math.isfinite(value) and value <= 0 for value in probabilities), 'invalid_actual_selected_logprobs')
    _require(type(turn['eos_token_id']) is int and turn['eos_token_id'] >= 0 and
             type(turn['pad_token_id']) is int and turn['pad_token_id'] >= 0, 'invalid_actual_eos_pad_identity')
    _require(isinstance(turn['finish_reason'], str) and bool(turn['finish_reason']) and
             (turn['stop_reason'] is None or type(turn['stop_reason']) in (int, str)), 'missing_actual_finish_stop_reason')
    count = turn['kept_token_count']
    _require(type(count) is int and 0 <= count <= len(raw), 'invalid_kept_raw_prefix_count')
    special = set(_tokens(turn['special_token_ids'], 'special_token_ids')) | {turn['eos_token_id'], turn['pad_token_id']}
    cut = turn['prefix_cut']
    dual = isinstance(cut, dict) and cut.get('dual_decode_projection_role') == DUAL_PROJECTION_ROLE
    _require(dual == (set(turn) == _TURN_STOP_FIELDS), 'dual_decode_without_actual_stop_provenance')
    if dual:
        stops, seq_eos = _verified_effective_stops(turn)
        _verify_dual_cut(cut, raw, count, special, turn['eos_token_id'], turn['pad_token_id'], stops)
    else:
        _require(not special.intersection(raw[:count]), 'included_special_or_pad_token')
    if dual:
        pass
    elif cut is None:
        expected = len(raw) - int(bool(raw and raw[-1] == turn['eos_token_id']))
        _require(count == expected, 'unproven_raw_prefix_truncation')
    else:
        expected_fields = {'kept_token_count', 'kept_token_ids', 'closing_tag', 'execution_text',
            'decoded_retained_prefix', 'retained_trailing_whitespace', 'token_boundary_whitespace_retained',
            'trailing_eos_excluded', 'raw_generated_token_count', 'raw_tokens_sha256', 'retained_tokens_sha256'}
        new_fields = {'retained_token_boundary_suffix', 'token_boundary_suffix_classification'}
        projection_fields = {'decode_projection_role', 'decoded_full_generation',
                             'full_decode_replacement_count', 'retained_decode_replacement_count'}
        _require(type(cut) is dict and set(cut) in (expected_fields, expected_fields | new_fields,
                                                 expected_fields | new_fields | projection_fields) and
                 type(cut['kept_token_count']) is int and cut['kept_token_count'] == count and
                 _tokens(cut['kept_token_ids'], 'cut_kept_token_ids') == raw[:count] and
                 type(cut['raw_generated_token_count']) is int and cut['raw_generated_token_count'] == len(raw) and
                 cut['raw_tokens_sha256'] == document_sha256(raw) and
                 cut['retained_tokens_sha256'] == document_sha256(raw[:count]), 'raw_prefix_proof_changed')
        _require(type(cut['trailing_eos_excluded']) is bool and
                 cut['trailing_eos_excluded'] == bool(raw and raw[-1] == turn['eos_token_id']), 'trailing_eos_proof_changed')
        whitespace = cut['retained_trailing_whitespace']
        # Legacy proofs are readable only with their original whitespace-only
        # semantics. New proofs classify arbitrary suffixes without disguising
        # them as whitespace, removing their tokens, or cleaning reward text.
        suffix = cut.get('retained_token_boundary_suffix', whitespace)
        _require(isinstance(suffix, str) and isinstance(whitespace, str), 'invalid_decoded_suffix')
        classification = 'none' if not suffix else 'whitespace' if suffix.isspace() else 'nonwhitespace'
        if new_fields <= set(cut):
            _require(cut['token_boundary_suffix_classification'] == classification,
                     'decoded_suffix_classification_changed')
        else:
            _require(classification != 'nonwhitespace', 'legacy_suffix_is_not_whitespace')
        _require(whitespace == (suffix if classification == 'whitespace' else '') and
                 type(cut['token_boundary_whitespace_retained']) is bool and
                 cut['token_boundary_whitespace_retained'] == (classification == 'whitespace') and
                 isinstance(cut['execution_text'], str) and isinstance(cut['decoded_retained_prefix'], str) and
                 cut['decoded_retained_prefix'] == cut['execution_text'] + suffix,
                 'decoded_cut_suffix_changed')
        if projection_fields <= set(cut):
            full = cut['decoded_full_generation']
            _require(cut['decode_projection_role'] == DECODE_PROJECTION_ROLE and isinstance(full, str) and
                     type(cut['full_decode_replacement_count']) is int and
                     type(cut['retained_decode_replacement_count']) is int and
                     cut['full_decode_replacement_count'] == full.count('\ufffd') and
                     cut['retained_decode_replacement_count'] == cut['decoded_retained_prefix'].count('\ufffd'),
                     'deterministic_decode_projection_or_replacement_census_changed')
            _require(full.startswith(cut['decoded_retained_prefix']), 'ambiguous_decoding_boundary')
            projected_closing = '</search>' if '</search>' in full else '</answer>' if '</answer>' in full else None
            _require(cut['closing_tag'] == projected_closing, 'full_decode_action_projection_changed')
            if projected_closing is None:
                _require(full == cut['decoded_retained_prefix'], 'no_tag_decode_projection_changed')
        else:
            # Old exact schemas keep their original no-replacement meaning.
            _require('\ufffd' not in cut['decoded_retained_prefix'], 'legacy_ambiguous_decoding_boundary')
        _require(cut['closing_tag'] in (None, '</search>', '</answer>'), 'unknown_closing_tag')
        if cut['closing_tag'] is not None:
            closing = cut['closing_tag']
            decoded = cut['decoded_retained_prefix']
            _require(closing in decoded and cut['execution_text'] == decoded[:decoded.find(closing) + len(closing)] and
                     closing == ('</search>' if '</search>' in decoded else '</answer>'),
                     'execution_is_not_original_character_projection')
            _require(count > 0 and cut['execution_text'].endswith(closing),
                     'closing_execution_text_changed')
        else:
            _require(count == len(raw) - int(cut['trailing_eos_excluded']) and not suffix, 'unproven_no_tag_truncation')
    observation = _tokens(turn['observation_token_ids'], 'observation_token_ids')
    correction = _tokens(turn['correction_token_ids'], 'correction_token_ids')
    _require(not (observation and correction), 'ambiguous_environment_segment_order')
    _require(not turn['final_turn'] or not (observation or correction), 'environment_after_final_turn')
    sampling = _sampling(turn['sampling'])
    _require(len(raw) <= sampling['max_tokens'], 'actual_raw_generation_exceeds_bound500')
    if dual and cut.get('trailing_actual_stop_excluded', False):
        terminal = raw[-1]
        _require(sampling['ignore_eos'] is False and turn['finish_reason'] == 'stop' and
                 (type(turn['stop_reason']) is int and turn['stop_reason'] == terminal or
                  turn['stop_reason'] is None and seq_eos == terminal),
                 'unproven_native_actual_stop_termination')
    if raw and raw[-1] == turn['eos_token_id'] and count < len(raw):
        _require(sampling['ignore_eos'] is False and turn['finish_reason'] == 'stop' and
                 turn['stop_reason'] in (None, turn['eos_token_id']), 'unproven_native_eos_termination')
    return request, raw, probabilities, observation + correction


def make_actor_row(*, trajectory_id, input_ids, attention_mask, info_mask, position_ids, prompt_width):
    row = dict(trajectory_id=_name(trajectory_id, 'trajectory_id'), input_ids=_tokens(input_ids, 'input_ids'),
               attention_mask=_mask(attention_mask, 'attention_mask'), info_mask=_mask(info_mask, 'info_mask'),
               position_ids=_tokens(position_ids, 'position_ids'), prompt_width=prompt_width)
    _actor_parts(row)
    return row


def _actor_parts(row):
    _require(type(row) is dict and set(row) == _ACTOR_FIELDS, 'actor_row_schema_changed')
    _name(row['trajectory_id'], 'trajectory_id')
    tokens = _tokens(row['input_ids'], 'input_ids')
    attention, info = _mask(row['attention_mask'], 'attention_mask'), _mask(row['info_mask'], 'info_mask')
    positions = _tokens(row['position_ids'], 'position_ids'); width = row['prompt_width']
    _require(len(tokens) == len(attention) == len(info) == len(positions) and type(width) is int and
             0 < width < len(tokens), 'invalid_actual_actor_shapes')
    _require(all(model <= active for model, active in zip(info, attention)), 'model_mask_outside_attention')
    _require(attention[:width] == sorted(attention[:width]) and
             attention[width:] == sorted(attention[width:], reverse=True), 'actor_padding_or_internal_attention_hole')
    _require([pos for pos, active in zip(positions, attention) if active] == list(range(sum(attention))),
             'actual_actor_positions_not_request_origin_zero')
    _require(all(info[index] == attention[index] for index in range(width)), 'actual_prompt_model_mask_changed')
    prompt = [token for token, active in zip(tokens[:width], attention[:width]) if active]
    response = [token for token, active in zip(tokens[width:], attention[width:]) if active]
    model_mask = [model for model, active in zip(info[width:], attention[width:]) if active]
    _require(bool(prompt), 'empty_actual_actor_prompt')
    return prompt, response, model_mask


def _audit_trajectory(turns, row, sampling, snapshot):
    prompt, response, mask = _actor_parts(row)
    _require(type(turns) is list and 1 <= len(turns) <= 5, 'missing_or_excess_actual_turns')
    cursor, selected, count, calls = 0, [], 0, set()
    for index, turn in enumerate(turns):
        request, raw, logprobs, environment = _verify_turn(turn)
        _require(turn['trajectory_id'] == row['trajectory_id'] and turn['turn_index'] == index and
                 turn['policy_snapshot_id'] == snapshot and canonical_bytes(turn['sampling']) == canonical_bytes(sampling),
                 'turn_identity_sampling_or_policy_changed')
        from search_g1.prefix20_split import provenance_records
        current_calls=[record['rollout_call_id'] for record in provenance_records(turn)]
        _require(not calls.intersection(current_calls), 'duplicate_trajectory_rollout_call')
        calls.update(current_calls)
        _require(not turn['final_turn'] or index == len(turns) - 1, 'turn_after_final_rollout')
        # Equality at the start of each turn plus exact raw prefix equality
        # implies equality for every included autoregressive token in that turn.
        _require(request == prompt + response[:cursor], 'actual_request_actor_context_mismatch')
        kept = raw[:turn['kept_token_count']]; end = cursor + len(kept)
        _require(response[cursor:end] == kept, 'raw_model_prefix_retokenized_dropped_or_changed')
        _require(mask[cursor:end] == [1] * len(kept), 'actual_model_prefix_loss_mask_changed')
        selected.extend(logprobs[:len(kept)]); cursor = end; count += len(kept)
        end = cursor + len(environment)
        _require(response[cursor:end] == environment, 'observation_or_correction_context_changed')
        _require(mask[cursor:end] == [0] * len(environment), 'observation_or_correction_optimized')
        selected.extend([0.0] * len(environment)); cursor = end
    _require(cursor == len(response) and count > 0, 'final_actor_has_untraced_or_no_model_tokens')
    physical = row['input_ids'][row['prompt_width']:]
    attention = row['attention_mask'][row['prompt_width']:]
    aligned = iter(selected)
    projection = [next(aligned) if active else 0.0 for active in attention]
    included = [bool(a and m) for a, m in zip(attention, row['info_mask'][row['prompt_width']:])]
    _require(len(projection) == len(physical), 'response_projection_shape_changed')
    return count, projection, included


def make_trace(*, run_id, source_sha256, slurm_job_id, logical_update, policy_snapshot_id,
               sampling, parent_group_ids, trajectories, actor_rows):
    return dict(schema_version=SCHEMA_VERSION, role=TRACE_ROLE, run_id=run_id,
                source_sha256=copy.deepcopy(source_sha256), slurm_job_id=slurm_job_id,
                logical_update=logical_update, policy_snapshot_id=policy_snapshot_id,
                sampling=copy.deepcopy(sampling), parent_group_ids=list(parent_group_ids),
                trajectories=copy.deepcopy(trajectories), actor_rows=copy.deepcopy(actor_rows))


def _outer(trace, *, expected_run_id, expected_source_sha256, expected_slurm_job_id,
           expected_logical_update, expected_parent_group_ids, expected_trajectory_ids,
           expected_policy_snapshot_id, expected_sampling, actor_rows=None, expected_generation_producers=None):
    fields = _TRACE_FIELDS | ({'generation_producers'} if expected_generation_producers is not None else set())
    _require(type(trace) is dict and set(trace) == fields and
             type(trace['schema_version']) is int and trace['schema_version'] == SCHEMA_VERSION and trace['role'] == TRACE_ROLE,
             'outer_trace_schema_changed')
    for name, value in (('run_id', expected_run_id), ('slurm_job_id', expected_slurm_job_id),
                        ('policy_snapshot_id', expected_policy_snapshot_id)):
        if name=='slurm_job_id' and value is None:
            from reproduction.common8.topology import execution_job_id
            run=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
            _require(run.get('execution_backend')=='ssh_local' and execution_job_id(run) is None and
                     run['run_id']==expected_run_id and run['source_sha256']==expected_source_sha256,
                     'outer_local_execution_binding_changed')
        else:
            _name(value, name)
        _require(trace[name] == value, 'outer_bound_identity_changed:' + name)
    _require(type(expected_logical_update) is int and expected_logical_update >= 1 and
             type(trace['logical_update']) is int and trace['logical_update'] == expected_logical_update,
             'outer_logical_update_changed')
    source = trace['source_sha256']
    _require(type(source) is dict and bool(source) and canonical_bytes(source) == canonical_bytes(expected_source_sha256),
             'outer_actual_source_changed')
    for name, value in source.items():
        _require(isinstance(name, str) and bool(name) and isinstance(value, str) and len(value) == 64 and
                 all(x in '0123456789abcdef' for x in value), 'invalid_source_hash')
    parents, ids = _list(expected_parent_group_ids, 'expected_parent_group_ids'), _list(expected_trajectory_ids, 'expected_trajectory_ids')
    _require(len(parents) == 512 and all(isinstance(item, str) and bool(item) for item in parents) and
             trace['parent_group_ids'] == parents and len(ids) == 2560 and len(set(ids)) == 2560 and
             all(isinstance(item, str) and bool(item) for item in ids), 'not_original_complete512_G5_outer')
    sampling = _sampling(expected_sampling)
    _require(canonical_bytes(trace['sampling']) == canonical_bytes(sampling), 'outer_actual_sampling_changed')
    trajectories = trace['trajectories']; rows = trace['actor_rows'] if actor_rows is None else actor_rows
    _require(type(trajectories) is list and type(rows) is list and len(trajectories) == len(rows) == 2560,
             'outer_missing_trajectory_or_actor_row')
    by_id = {}
    token_metadata = None
    producers = expected_generation_producers or [dict(run_id=expected_run_id,
        slurm_job_id=expected_slurm_job_id, source_map_sha256=document_sha256(expected_source_sha256))]
    if expected_generation_producers is not None:
        _require(trace.get('generation_producers') == producers and bool(producers) and
                 all(p['run_id'] == expected_run_id for p in producers), 'historical_generation_producers_changed')
    for index, trajectory in enumerate(trajectories):
        _require(type(trajectory) is dict and set(trajectory) == {'trajectory_id', 'parent_occurrence_index', 'sample_index', 'turns'} and
                 trajectory['trajectory_id'] == ids[index] and type(trajectory['parent_occurrence_index']) is int and
                 trajectory['parent_occurrence_index'] == index // 5 and type(trajectory['sample_index']) is int and
                 trajectory['sample_index'] == index % 5, 'original_GRPO_group_or_occurrence_changed')
        _require(type(trajectory['turns']) is list and bool(trajectory['turns']), 'missing_actual_turns')
        for turn in trajectory['turns']:
            _require(type(turn) is dict, 'turn_schema_changed')
            from search_g1.prefix20_split import provenance_records
            for record in provenance_records(turn):
                if 'effective_stop_provenance' in record:
                    provenance = record['effective_stop_provenance']['generation_config']
                    _require(any(all(provenance[key] == p[key] for key in
                             ('run_id','slurm_job_id','source_map_sha256')) for p in producers),
                             'effective_stop_run_source_binding_changed')
            current = dict(eos_token_id=turn['eos_token_id'], pad_token_id=turn['pad_token_id'],
                           special_token_ids=sorted(set(_tokens(turn['special_token_ids'], 'special_token_ids'))))
            if token_metadata is None:
                token_metadata = current
            _require(current == token_metadata, 'actual_tokenizer_metadata_changed_between_rows_or_turns')
        by_id[trajectory['trajectory_id']] = trajectory
    row_ids = [row.get('trajectory_id') for row in rows if type(row) is dict]
    _require(len(row_ids) == len(set(row_ids)) == 2560 and set(row_ids) == set(ids), 'actor_rows_dropped_duplicated_or_unknown')
    counts, widths, projections, masks, mask_hashes = [], [], [], [], []
    for row in rows:
        count, projection, included = _audit_trajectory(by_id[row['trajectory_id']]['turns'], row, sampling, expected_policy_snapshot_id)
        counts.append(count); widths.append(len(projection)); projections.append(projection); masks.append(included)
        mask_hashes.append(document_sha256(dict(attention_mask=row['attention_mask'][row['prompt_width']:],
                                               info_mask=row['info_mask'][row['prompt_width']:])))
    summary = dict(schema_version=SCHEMA_VERSION, role=AUDIT_ROLE, run_id=expected_run_id,
        source_sha256=copy.deepcopy(source), source_map_sha256=document_sha256(source), slurm_job_id=expected_slurm_job_id,
        logical_update=expected_logical_update, policy_snapshot_id=expected_policy_snapshot_id,
        trace_sha256=document_sha256(trace), actor_rows_sha256=document_sha256(rows), sampling_sha256=document_sha256(sampling),
        tokenizer_metadata_sha256=document_sha256(token_metadata),
        trajectory_ids=row_ids, actor_token_counts=counts, response_widths=widths, response_mask_sha256=mask_hashes,
        parent_count=512, response_count=2560, included_token_count=sum(counts), context_verified=True,
        raw_prefix_verified=True, selected_logprob_alignment_verified=True, whole_outer_required=True,
        production_admission=False, healthy_checkpoint_accepted=False, physical_execution_authenticated=False,
        scope='Exact recorded token/context/mask structure only; no GPU probability or healthy-policy acceptance')
    return summary, projections, masks


def audit_outer(trace, **bindings):
    """Return hashes/scalars only; any mismatch rejects the complete outer."""
    try:
        return _outer(trace, **bindings)[0]
    except BehaviorTraceError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise BehaviorTraceError('missing_or_malformed_complete_outer') from error


def project_sampling_logprobs(trace, **bindings):
    """Private aligned vectors reconstructed from raw selected LP, never HF LP."""
    try:
        audit, probabilities, mask = _outer(trace, **bindings)
        return dict(audit=audit, trajectory_ids=audit['trajectory_ids'],
                    sampling_log_probs=probabilities, sampling_mask=mask)
    except BehaviorTraceError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise BehaviorTraceError('missing_or_malformed_complete_outer') from error


def _exclusive_private_file(path, content):
    """Simple owned private artifact primitive, tested with CPU fixture bytes."""
    path = Path(path)
    parent = path.parent
    _require(parent.is_dir() and parent.resolve() == parent and not parent.is_symlink() and parent.stat().st_uid == os.getuid() and
             stat.S_IMODE(parent.stat().st_mode) == 0o700, 'not_owned_private_artifact_directory')
    directory = os.open(str(parent), os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        details = os.fstat(directory)
        _require(details.st_uid == os.getuid() and stat.S_IMODE(details.st_mode) == 0o700,
                 'not_owned_private_artifact_directory')
        descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.fsync(directory)
    finally:
        os.close(directory)


def _evidence_value(value):
    # A failed sampler may itself return NaN/Inf. Preserve that failure as a
    # typed marker rather than lose the raw token evidence or emit invalid JSON.
    # Such transformed bytes are never a successful trace/audit proof.
    if type(value) is float and not math.isfinite(value):
        label = 'nan' if math.isnan(value) else ('inf' if value > 0 else '-inf')
        return {'__behavior_trace_nonfinite_float__': label}, True
    if type(value) in (list, tuple):
        converted = [_evidence_value(item) for item in value]
        return [item[0] for item in converted], any(item[1] for item in converted)
    if type(value) is dict:
        converted = {key: _evidence_value(item) for key, item in value.items()}
        return {key: item[0] for key, item in converted.items()}, any(item[1] for item in converted.values())
    return value, False


def write_private_evidence(base, trace, *, attempt_id=None):
    """Preserve complete or partial real failure evidence before any audit.

    This verifies disk/env identity but does not claim a real GPU allocation;
    that admission belongs to the existing native resource gate before callers
    invoke this writer. No trace completeness, context or healthy-policy claim
    is inferred here. A nonfinite-float marker makes a failed record valid JSON
    and explicitly prevents treating its byte hash as an accepted trace hash.
    """
    base = Path(base)
    try:
        run = json.loads((base / 'logs/run-manifest.json').read_text())
    except (OSError,ValueError) as error:
        raise BehaviorTraceError('missing_or_invalid_run_manifest') from error
    from reproduction.common8.topology import execution_job_id, validate_base
    validate_base(base,run)
    _require(type(trace) is dict and type(trace.get('logical_update')) is int and trace['logical_update'] >= 1 and
             run['run_id'] == trace.get('run_id') and run['source_sha256'] == trace.get('source_sha256') and
             'slurm_job_id' in trace and
             execution_job_id(run) == trace['slurm_job_id'], 'raw_trace_execution_run_identity_changed')
    artifacts = base / 'artifacts'
    _require(artifacts.is_dir() and artifacts.resolve() == artifacts and
             artifacts.stat().st_uid == os.getuid(), 'raw_trace_artifact_symlink_or_foreign_owner')
    parent = artifacts / 'behavior-traces'
    _require(not parent.is_symlink() and parent.resolve() == parent, 'raw_trace_artifact_symlink')
    value, transformed = _evidence_value(trace)
    content = canonical_bytes(value)
    parent.mkdir(mode=0o700, exist_ok=True)
    _require(stat.S_IMODE(parent.stat().st_mode) == 0o700, 'raw_trace_artifact_directory_not_private')
    if run.get('policy_safety_mode')=='inline_first_five':
        _require(isinstance(attempt_id,str) and len(attempt_id)==32 and
                 all(char in '0123456789abcdef' for char in attempt_id), 'missing_formal_trace_attempt_identity')
        _require(trace.get('policy_snapshot_id')==run['run_id']+':'+str(trace['logical_update'])+':policy-'+attempt_id,
                 'formal_trace_attempt_snapshot_changed')
        path=parent/('attempt-'+attempt_id+'.json')
    else:
        _require(attempt_id is None,'legacy_trace_cannot_admit_formal_attempt')
        path = parent / ('outer-' + str(trace['logical_update']) + '.json')
    _exclusive_private_file(path, content)
    return dict(private_trace_path=str(path), private_trace_sha256=hashlib.sha256(content).hexdigest(),
                serialization_transformed=transformed, raw_trace_must_not_be_published=True,
                production_admission=False, healthy_checkpoint_accepted=False,
                physical_execution_authenticated=False, context_verified=False)


def write_private_journal(base, trace, **bindings):
    """Preserve real private bytes, then independently audit the full outer.

    A rejected audit leaves its raw journal intact. The exclusive filename
    prevents replacement of earlier successful or failed evidence.
    """
    evidence = write_private_evidence(base, trace)
    summary = audit_outer(trace, **bindings)
    _require(not evidence['serialization_transformed'] and
             evidence['private_trace_sha256'] == summary['trace_sha256'], 'accepted_trace_bytes_changed')
    return dict(summary, **{key: value for key, value in evidence.items() if key != 'context_verified'})
