"""Read saved Easy2560 selection/native-token evidence on CPU, without Torch.

The caller authenticates the successful update and its original run/job segment
before calling verify_attempt. This module does not assemble restore history,
run models, regenerate answers, write files, or change training admission. One
selected trace is read at a time on its source CPU; only small hashes/counters
are returned. Selection is proved against saved strict labels, not recomputed
EM of deleted unselected raw trajectories or independently measured GPU LP.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct

from reproduction.common8 import behavior_trace as trace_reader


STATUS = 'EASY2560_SAVED_SELECTION_NATIVE_TRACE_CPU_PASS'
CANDIDATES, SELECTED, RESPONSES = 2560, 512, 5
MAX_SMALL, MAX_TRACE = 128 * 1024**2, 2 * 1024**3
JOURNAL_POLICY = 'sampling_journal_no_extra_hf_forward'
JOURNAL_DENOMINATOR = 'actual_selected_sampling_log_probs_after_all_token_context_audit'
HF_DENOMINATOR = 'actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate'


def need(condition, message):
    if not condition:
        raise ValueError('EASY2560_SELECTION_' + message)


def _sha(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('EASY2560_SELECTION_NONFINITE_JSON')


def _float(value):
    result = float(value)
    need(math.isfinite(result), 'NONFINITE_JSON')
    return result


def _read(path, maximum):
    # Reuse the shared-filesystem stable-fd check, including timestamp-only
    # settlement handling. No trace is copied or serialized to a new file.
    from reproduction.easy2560_joint4.final_cpu_acceptance import _verify_stable_fd_snapshot
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path, 'EVIDENCE_PATH_NOT_CANONICAL')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, 'EVIDENCE_TYPE_OR_SIZE')
        raw = stream.read(maximum + 1); sha = hashlib.sha256(raw).hexdigest()
        _verify_stable_fd_snapshot(stream, path, before, sha, len(raw), check=need)
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)
    need(type(value) is dict, 'EVIDENCE_OBJECT_REQUIRED')
    return value, dict(path=str(path), bytes=len(raw), sha256=sha)


def _sampling(value):
    """The pure-Python equivalent of the native probability gate's sampler check."""
    need(type(value) is dict, 'SAMPLING_OBJECT_REQUIRED')
    neutral = dict(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                   presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0)
    for key, expected in neutral.items():
        actual = value.get(key)
        need(type(actual) in (int, float) and math.isfinite(actual) and actual == expected,
             'NONNEUTRAL_SAMPLING_' + key)
    need(value.get('do_sample') is True and value.get('ignore_eos') is False, 'SAMPLING_MODE_DIFFERS')
    for key, expected in (('n', 1), ('logprobs', 1), ('max_tokens', 500)):
        need(type(value.get(key)) is int and value[key] == expected, 'SAMPLING_INTEGER_DIFFERS_' + key)
    for key, allowed in (('min_tokens', (None, 0)), ('best_of', (None, 1))):
        need(key in value and (value[key] is None or type(value[key]) is int and value[key] in allowed),
             'SAMPLING_OPTION_DIFFERS_' + key)
    need('logits_processors' in value and value['logits_processors'] in (None, []) and
         'logit_bias' in value and value['logit_bias'] in (None, {}), 'CUSTOM_SAMPLING_FILTER')
    need(all(key in value and value[key] is None for key in
             ('allowed_token_ids', 'guided_decoding', 'truncate_prompt_tokens')), 'EXTERNAL_SAMPLING_FILTER')
    need(value.get('logprob_semantics') == trace_reader.LOGPROB_SEMANTICS, 'LOGPROB_SEMANTICS_DIFFERS')


def _tensor_hash(rows, dtype, shape):
    """Match _tensor_sha256's dtype/shape header and x86 native tensor bytes.

    Float packing performs the same storage-dtype rounding as torch.tensor;
    masks/tokens were explicitly cast to int64 by the native gate. Hash each row
    separately so rebuilding a whole probability tensor needs no second copy.
    """
    codes = {'torch.int64': 'q', 'torch.bool': '?', 'torch.float32': 'f', 'torch.float64': 'd'}
    need(dtype in codes, 'UNSUPPORTED_TENSOR_DTYPE')
    h = hashlib.sha256(trace_reader.canonical_bytes(dict(dtype=dtype, shape=shape)))
    for row in rows:
        h.update(struct.pack('<' + codes[dtype] * len(row), *row))
    return h.hexdigest()


def _policy(gate, run):
    need({'actual_frozen_old_all_finite', 'hf_recomputation_performed', 'hf_parity_accepted',
          'max_abs_discrepancy', 'tolerance', 'hf_dtype', 'denominator', 'tensor_sha256'} <= set(gate),
         'COMPLETE_PROBABILITY_POLICY_REQUIRED')
    need(gate.get('actual_frozen_old_all_finite') is True, 'FROZEN_OLD_NOT_FINITE')
    policy = run.get('behavior_probability_policy')
    need(policy in (None, JOURNAL_POLICY), 'UNKNOWN_PROBABILITY_POLICY')
    if policy == JOURNAL_POLICY:
        need(gate.get('behavior_probability_policy') == policy and
             gate.get('hf_recomputation_performed') is False and gate.get('hf_parity_accepted') is None and
             gate.get('max_abs_discrepancy') is None and gate.get('tolerance') is None and
             gate.get('hf_dtype') is None and gate.get('denominator') == JOURNAL_DENOMINATOR and
             set(gate['tensor_sha256']) == {'sampling_log_probs', 'actor_mask', 'attention_mask'},
             'JOURNAL_POLICY_DIFFERS')
    else:
        difference = gate.get('max_abs_discrepancy')
        need(gate.get('behavior_probability_policy') is None and
             gate.get('hf_recomputation_performed') is True and gate.get('hf_parity_accepted') is True and
             type(difference) in (int, float) and math.isfinite(difference) and
             0 <= difference <= gate.get('tolerance', -1) == 0.05 and
             gate.get('hf_dtype') in ('torch.float32', 'torch.float64') and
             gate.get('denominator') == HF_DENOMINATOR and
             set(gate['tensor_sha256']) == {'sampling_log_probs', 'actor_mask', 'attention_mask', 'hf_old_log_probs'} and
             _sha(gate['tensor_sha256']['hf_old_log_probs']), 'SAVED_HF_POLICY_DIFFERS')


def _verify_attempt(base, run, update, expected_job):
    base = Path(base)
    need(base.is_absolute() and base.resolve() == base, 'BASE_NOT_CANONICAL')
    run_id, commit, source = run['run_id'], run['source_commit'], run['source_sha256']
    need(_sha(run_id, 32) and _sha(commit, 40) and type(source) is dict and source and
         all(type(name) is str and name and _sha(sha) for name, sha in source.items()), 'EXPECTED_SOURCE_REQUIRED')
    need(isinstance(expected_job, str) and expected_job.isdigit(), 'EXPECTED_JOB_REQUIRED')
    need(run.get('easy2560_selection') is True and run.get('training_world_size') == 4 and
         run.get('data_selection_mode') == 'mixed_first' and run.get('question_probe') is None and
         run.get('selection_candidate_questions') == CANDIDATES and run.get('selection_selected_questions') == SELECTED and
         run.get('selection_fixed_candidate_budget') is True and run.get('all_correct_virtual_zero') is False,
         'FIXED_EASY_RECIPE_REQUIRED')
    step, aid, index = update['step'], update['attempt_id'], update['attempt_index']
    writes, before, after = (update[k] for k in
                            ('actual_optimizer_writes', 'optimizer_writes_before', 'optimizer_writes_after'))
    need(type(step) is int and 1 <= step <= 200 and _sha(aid, 32) and type(index) is int and index >= step and
         update['run_id'] == run_id and type(writes) is int and 1 <= writes <= 20 and
         type(before) is int and before >= 0 and type(after) is int and after - before == writes,
         'AUTHENTICATED_SUCCESSFUL_UPDATE_REQUIRED')
    need(_sha(update['behavior_probability_gate_sha256']) and _sha(update['behavior_trace_sha256']),
         'UPDATE_EVIDENCE_SHA_REQUIRED')
    snapshot = run_id + ':' + str(step) + ':policy-' + aid
    audit, audit_record = _read(base/'artifacts/mixed-selection'/(aid + '.json'), MAX_SMALL)
    need(type(audit['step']) is int and audit['step'] == step and
         audit['source_commit'] == commit and audit['policy_snapshot_id'] == snapshot and
         audit.get('fixed_candidate_budget') is True and audit['attempted_candidates'] == CANDIDATES and
         audit['responses_per_group'] == RESPONSES and audit['selected_groups'] == SELECTED,
         'SELECTION_IDENTITY_OR_BUDGET_DIFFERS')
    retry = audit['pool_retry']
    need(type(retry) is int and retry >= 0 and type(audit['discarded_insufficient_pools']) is list and
         len(audit['discarded_insufficient_pools']) == retry and
         audit['pool_path'] == str(base/'artifacts/mixed-candidate-pools'/(aid + '-pool-' + str(retry))),
         'SUCCESSFUL_POOL_BINDING_DIFFERS')
    groups = audit['groups']
    need(type(groups) is list and len(groups) == CANDIDATES and
         all(type(g['ordinal']) is int and g['ordinal'] == i for i, g in enumerate(groups)), 'CANDIDATE_ORDINALS_DIFFER')
    question_ids, contents, all_ids = set(), set(), set()
    mixed, correct, chosen, parents, upstream = [], [], [], {}, {}
    for ordinal, group in enumerate(groups):
        qid, content = group['question_id'], group['content_sha256']
        need(type(qid) is str and qid and _sha(content), 'MISSING_QUESTION_IDENTITY')
        need(qid not in question_ids and content not in contents, 'DUPLICATE_CANDIDATE_QUESTION')
        question_ids.add(qid); contents.add(content)
        labels, ids = group['strict_correctness'], group['trajectory_ids']
        need(type(labels) is list and len(labels) == RESPONSES and
             all(type(value) is int and value in (0, 1) for value in labels), 'INVALID_STRICT_LABELS')
        category = 'mixed' if 0 < sum(labels) < RESPONSES else 'all_correct' if sum(labels) == RESPONSES else 'all_wrong'
        need(group['category'] == category and type(group['selected']) is bool, 'CATEGORY_OR_SELECTED_FLAG_DIFFERS')
        if category == 'mixed': mixed.append(ordinal)
        if category == 'all_correct': correct.append(ordinal)
        if group['selected']: chosen.append(ordinal)
        tag = 'selected_' + ('mixed' if category == 'mixed' else 'all_correct_fallback') if group['selected'] else 'excluded'
        need(group['source'] == tag, 'SELECTION_SOURCE_LABEL_DIFFERS')
        need(type(ids) is list and len(ids) == RESPONSES and all(type(tid) is str for tid in ids), 'INCOMPLETE_GROUP_IDS')
        prefix = run_id + '-' + str(step) + '-attempt-' + aid + '-'
        occurrence = str(retry * CANDIDATES + ordinal)
        match = re.fullmatch(re.escape(prefix + occurrence + '-') + '([0-9a-f]{64})-sample-0', ids[0])
        need(match is not None, 'ORIGINAL_CANDIDATE_ID_DIFFERS')
        parent = occurrence + '-' + match.group(1)
        need(ids == [prefix + parent + '-sample-' + str(sample) for sample in range(RESPONSES)] and
             not all_ids.intersection(ids), 'ORIGINAL_G5_IDS_DIFFER')
        all_ids.update(ids); parents[ordinal] = parent
        upstream.update({tid: match.group(1) for tid in ids})
    expected = sorted(mixed[:SELECTED] if len(mixed) >= SELECTED else mixed + correct[:SELECTED-len(mixed)])
    need(len(expected) == SELECTED and chosen == expected, 'NOT_EXACT_MIXED_FIRST_ORDINAL512')
    selected_mixed = sum(groups[i]['category'] == 'mixed' for i in chosen)
    need(type(audit['selected_mixed']) is int and audit['selected_mixed'] == selected_mixed and
         type(audit['selected_all_correct_fallback']) is int and
         audit['selected_all_correct_fallback'] == SELECTED-selected_mixed, 'SELECTED_CATEGORY_COUNTS_DIFFER')
    ids = [tid for i in chosen for tid in groups[i]['trajectory_ids']]
    gate, gate_record = _read(base/'logs'/('behavior-probability-attempt-' + aid + '.json'), MAX_SMALL)
    need(gate_record['sha256'] == update['behavior_probability_gate_sha256'] and
         type(gate.get('schema_version')) is int and gate['schema_version'] == 1 and
         gate.get('role') == 'common8_behavior_probability_gate' and gate.get('accepted') is True and
         gate['run_id'] == run_id and gate['source_sha256'] == source and gate['slurm_job_id'] == expected_job and
         type(gate['logical_update']) is int and gate['logical_update'] == step and gate['attempt_id'] == aid and
         type(gate['attempt_index']) is int and gate['attempt_index'] == index and
         gate['policy_snapshot_id'] == snapshot and gate['trajectory_count'] == SELECTED*RESPONSES and
         gate['trajectory_ids'] == ids and gate['upstream_group_sha256'] == {tid: upstream[tid] for tid in ids},
         'GATE_UPDATE_OR_ORIGINAL_GROUP_BINDING_DIFFERS')
    _sampling(gate['sampling']); _policy(gate, run)
    trace, trace_record = _read(base/'artifacts/behavior-traces'/('attempt-' + aid + '.json'), MAX_TRACE)
    need(trace_record['sha256'] == update['behavior_trace_sha256'] == gate['trace_sha256'] == gate['private_trace_sha256'],
         'TRACE_SHA_DIFFERS')
    need([row['trajectory_id'] for row in trace['actor_rows']] == ids, 'ACTOR_ROW_ORDER_DIFFERS')
    request_ids = set()
    for trajectory in trace['trajectories']:
        for turn in trajectory['turns']:
            # Easy2560 full rollouts use complete native captures, not a legacy
            # trace missing actual request-group provenance or a prefix wrapper.
            need(type(turn) is dict and set(turn) == trace_reader._TURN_STOP_FIELDS,
                 'NATIVE_CAPTURE_PROVENANCE_REQUIRED')
            proof = turn['effective_stop_provenance']; generation = proof['generation_config']
            need(proof['rollout_call_id'] == turn['rollout_call_id'] and proof['row_id'] == trajectory['trajectory_id'] and
                 generation['source_commit'] == commit and turn['request_id'] not in request_ids,
                 'NATIVE_REQUEST_SOURCE_OR_ID_DIFFERS')
            request_ids.add(turn['request_id'])
    projected = trace_reader.project_sampling_logprobs(trace, expected_run_id=run_id,
        expected_source_sha256=source, expected_slurm_job_id=expected_job, expected_logical_update=step,
        expected_parent_group_ids=[parents[i] for i in chosen], expected_trajectory_ids=ids,
        expected_policy_snapshot_id=snapshot, expected_sampling=gate['sampling'])
    rebuilt = projected['audit']
    need(type(gate['actor_token_counts']) is list and len(gate['actor_token_counts']) == SELECTED*RESPONSES and
         all(type(count) is int and count > 0 for count in gate['actor_token_counts']) and
         type(gate['included_actor_token_count']) is int, 'INVALID_ACTOR_TOKEN_COUNTS')
    need(rebuilt['trace_sha256'] == trace_record['sha256'] and rebuilt['actor_rows_sha256'] == gate['actor_rows_sha256'] and
         trace_reader.document_sha256(rebuilt) == gate['audit_sha256'] and rebuilt['trajectory_ids'] == ids and
         rebuilt['actor_token_counts'] == gate['actor_token_counts'] and
         rebuilt['included_token_count'] == gate['included_actor_token_count'], 'REBUILT_CONTEXT_AUDIT_DIFFERS')
    rows, probabilities, masks = trace['actor_rows'], projected['sampling_log_probs'], projected['sampling_mask']
    widths = {len(row['input_ids'])-row['prompt_width'] for row in rows}
    need(len(widths) == 1 and len({row['prompt_width'] for row in rows}) == 1 and
         len({len(row['input_ids']) for row in rows}) == 1, 'NONRECTANGULAR_ACTOR_TENSORS')
    width = widths.pop(); dtype = gate['sampling_dtype']
    need(dtype in ('torch.float32', 'torch.float64'), 'UNSUPPORTED_SAMPLING_DTYPE')
    whole = dict(sampling_log_probs=_tensor_hash(probabilities, dtype, [len(ids), width]),
                 actor_mask=_tensor_hash([row['info_mask'][row['prompt_width']:] for row in rows], 'torch.int64', [len(ids), width]),
                 attention_mask=_tensor_hash([row['attention_mask'][row['prompt_width']:] for row in rows], 'torch.int64', [len(ids), width]))
    need(all(gate['tensor_sha256'].get(key) == value for key, value in whole.items()), 'REBUILT_SAMPLING_TENSOR_SHA_DIFFERS')
    row_hashes = {}
    for row, values, mask in zip(rows, probabilities, masks):
        fields = {name: (row[name], 'torch.int64') for name in ('input_ids', 'attention_mask', 'position_ids', 'info_mask')}
        fields.update(responses=(row['input_ids'][row['prompt_width']:], 'torch.int64'),
                      old_log_probs=(values, dtype), sampling_log_probs=(values, dtype), sampling_mask=(mask, 'torch.bool'))
        row_hashes[row['trajectory_id']] = trace_reader.document_sha256({name: _tensor_hash([data], kind, [len(data)])
                                                                       for name, (data, kind) in fields.items()})
    need(gate['row_tensor_sha256'] == row_hashes, 'REBUILT_ACTOR_ROW_SHA_DIFFERS')
    return dict(schema_version=1, status=STATUS, run_id=run_id, source_commit=commit, source_job=expected_job,
        step=step, attempt_id=aid, attempt_index=index, pool_retry=retry, distinct_candidate_questions=CANDIDATES,
        candidate_question_ids_unique=True, candidate_content_hashes_unique=True,
        exact_mixed_first_ordinal_selection=True, selected_groups=SELECTED, selected_trajectories=len(ids),
        selected_mixed=selected_mixed, selected_all_correct_fallback=SELECTED-selected_mixed,
        selected_ordinals_sha256=trace_reader.document_sha256(chosen),
        selected_trajectory_ids_sha256=trace_reader.document_sha256(ids), native_requests=len(request_ids),
        included_actor_tokens=rebuilt['included_token_count'], native_token_context_mask_logprob_rebuilt=True,
        gate_row_tensor_hashes_rebuilt=True, sampling_dtype=dtype, evidence=dict(selection=audit_record, gate=gate_record, trace=trace_record),
        saved_strict_labels_recomputed_from_raw=False, unselected_raw_required=False,
        actual_gpu_probabilities_recomputed=False, hf_parity_recomputed=False,
        new_training_updates=0, gpu_allocations=0, model_reloads=0,
        scope='one authenticated successful pool; selection against saved strict labels and native selected-token/LP to actor-row byte hashes; no unselected EM, HF/GPU probability or physical execution remeasurement')


def verify_attempt(base, run, completed_update, *, expected_job):
    """Audit one caller-authenticated successful update in its source run/job.

    The caller must prove the completed-update ledger and restore-history chain;
    this function's small summary does not establish final200 completeness.
    """
    try:
        return _verify_attempt(base, run, completed_update, expected_job)
    except trace_reader.BehaviorTraceError as error:
        raise ValueError('EASY2560_SELECTION_NATIVE_TRACE_REJECTED_' + str(error)) from error
    except (KeyError, TypeError, OverflowError, struct.error, json.JSONDecodeError) as error:
        raise ValueError('EASY2560_SELECTION_MALFORMED_EVIDENCE') from error
