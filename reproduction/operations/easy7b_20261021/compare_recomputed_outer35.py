"""Optional CPU comparison of saved outer35 hashes; never a training gate.

Read only two selection and two probability-gate JSON files. Raw tensors,
private token traces, models, networks and production ledgers are not opened.
An unequal hash records a difference; it never requests restart or rollback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


OLD_BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022')
NEW_BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022')
RUN = '232f20e112ac4a978c5aba52a0367195'
OLD_JOB, NEW_JOB, STEP = '746125', '747219', 35
OLD_COMMIT = 'ebb6844db6a1ae739a06d5cc31d3510e7b4779b1'
NEW_COMMIT = 'b76d7a05245a5eb7adc75385e2db803c67acce43'
CANDIDATES, SELECTED, RESPONSES = 2560, 512, 5
MAX_SMALL = 128 * 1024**2
TENSOR_NAMES = ('sampling_log_probs', 'actor_mask', 'attention_mask')
LIMITS = dict(raw_pt_read=False, private_trace_read=False, token_ids_or_text_returned=False,
    unselected_candidate_token_or_logprob_hashes_available=False,
    independent_gpu_probability_or_tensor_remeasurement=False,
    separate_token_id_hash_available_in_small_gate=False,
    row_hash_is_combined_token_context_mask_old_and_sampling_logprob_hash=True,
    hashes_include_padding_dtype_and_shape=True,
    whole_json_trace_actor_rows_and_audit_hashes_compared=False,
    old_outer35_training_complete=False, new_outer35_training_completion_checked=False,
    restore_or_model_state_independently_checked=False, numerical_error_magnitude_available=False)


def need(value, message):
    if not value:
        raise ValueError('OUTER35_COMPARE_' + message)


def attempt(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{32}', value) is not None,
         'EXPLICIT_FULL_ATTEMPT_REQUIRED')
    return value


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def _read(path):
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_SMALL,
             'SMALL_REGULAR_JSON_REQUIRED')
        raw = stream.read(MAX_SMALL + 1)
        after = os.fstat(stream.fileno())
        need(len(raw) == before.st_size and
             (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             'JSON_CHANGED_DURING_READ')
    value = json.loads(raw, object_pairs_hook=_pairs)
    need(type(value) is dict, 'JSON_OBJECT_REQUIRED')
    return value, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _selection(value, aid, commit):
    need(value.get('step') == STEP and value.get('source_commit') == commit and
         value.get('policy_snapshot_id') == RUN + ':' + str(STEP) + ':policy-' + aid and
         value.get('fixed_candidate_budget') is True and
         value.get('attempted_candidates') == CANDIDATES and
         value.get('responses_per_group') == RESPONSES and value.get('selected_groups') == SELECTED,
         'SELECTION_IDENTITY_OR_BUDGET_DIFFERS')
    groups = value.get('groups')
    need(type(groups) is list and len(groups) == CANDIDATES, 'ALL_CANDIDATE_GROUPS_REQUIRED')
    retry = value.get('pool_retry')
    need(type(retry) is int and retry >= 0, 'POOL_RETRY_REQUIRED')
    ids, questions, contents, mixed, correct, chosen = set(), set(), set(), [], [], []
    for ordinal, group in enumerate(groups):
        need(type(group) is dict and type(group.get('ordinal')) is int and group['ordinal'] == ordinal,
             'ORDERED_GROUPS_REQUIRED')
        qid, content, labels = group.get('question_id'), group.get('content_sha256'), group.get('strict_correctness')
        need(isinstance(qid, str) and qid and qid not in questions and _hash(content) and content not in contents,
             'UNIQUE_QUESTION_IDENTITIES_REQUIRED')
        questions.add(qid); contents.add(content)
        need(type(labels) is list and len(labels) == RESPONSES and
             all(type(label) is int and label in (0, 1) for label in labels), 'STRICT_BINARY_G5_REQUIRED')
        category = 'mixed' if 0 < sum(labels) < RESPONSES else 'all_correct' if sum(labels) else 'all_wrong'
        need(group.get('category') == category and type(group.get('selected')) is bool,
             'SELECTION_CATEGORY_DIFFERS')
        if category == 'mixed': mixed.append(ordinal)
        if category == 'all_correct': correct.append(ordinal)
        if group['selected']: chosen.append(ordinal)
        tids = group.get('trajectory_ids')
        prefix = RUN + '-' + str(STEP) + '-attempt-' + aid + '-' + str(retry*CANDIDATES + ordinal) + '-'
        need(type(tids) is list and len(tids) == RESPONSES and isinstance(tids[0], str), 'G5_IDS_REQUIRED')
        match = re.fullmatch(re.escape(prefix) + '([0-9a-f]{64})-sample-0', tids[0])
        need(match is not None and tids == [prefix + match.group(1) + '-sample-' + str(i)
             for i in range(RESPONSES)] and not ids.intersection(tids), 'G5_ATTEMPT_IDS_DIFFER')
        ids.update(tids)
    expected = sorted(mixed[:SELECTED] if len(mixed) >= SELECTED else mixed + correct[:SELECTED-len(mixed)])
    need(len(expected) == SELECTED and chosen == expected, 'ACTUAL_MIXED_FIRST_SELECTION_DIFFERS')
    need(value.get('selected_mixed') == sum(groups[i]['category'] == 'mixed' for i in chosen) and
         value.get('selected_all_correct_fallback') == sum(groups[i]['category'] == 'all_correct' for i in chosen),
         'SELECTED_COUNTS_DIFFER')
    return groups, chosen


def _gate(value, aid, job, groups, chosen):
    ids = [tid for ordinal in chosen for tid in groups[ordinal]['trajectory_ids']]
    need(value.get('schema_version') == 1 and value.get('role') == 'common8_behavior_probability_gate' and
         value.get('accepted') is True and value.get('run_id') == RUN and value.get('slurm_job_id') == job and
         value.get('logical_update') == STEP and value.get('attempt_id') == aid and
         value.get('policy_snapshot_id') == RUN + ':' + str(STEP) + ':policy-' + aid and
         value.get('trajectory_count') == SELECTED*RESPONSES and value.get('trajectory_ids') == ids,
         'GATE_SELECTION_OR_EXECUTION_IDENTITY_DIFFERS')
    rows = value.get('row_tensor_sha256')
    need(type(rows) is dict and set(rows) == set(ids) and all(_hash(v) for v in rows.values()),
         'COMPLETE_COMBINED_ROW_HASHES_REQUIRED')
    tensors = value.get('tensor_sha256')
    need(type(tensors) is dict and set(tensors) == set(TENSOR_NAMES) and
         all(_hash(v) for v in tensors.values()) and value.get('actual_frozen_old_all_finite') is True and
         value.get('behavior_probability_policy') == 'sampling_journal_no_extra_hf_forward' and
         value.get('hf_recomputation_performed') is False and
         value.get('sampling_dtype') in ('torch.float32', 'torch.float64'), 'JOURNAL_TENSOR_HASHES_REQUIRED')
    counts = value.get('actor_token_counts')
    need(type(counts) is list and len(counts) == len(ids) and
         all(type(n) is int and n > 0 for n in counts) and
         type(value.get('included_actor_token_count')) is int and sum(counts) == value['included_actor_token_count'],
         'ACTOR_TOKEN_COUNTS_DIFFER')


def compare_documents(old_selection, old_gate, new_selection, new_gate, old_attempt, new_attempt):
    """Pure comparison; identity changes are not included in content hashes."""
    attempt(old_attempt); attempt(new_attempt)
    old, old_chosen = _selection(old_selection, old_attempt, OLD_COMMIT)
    new, new_chosen = _selection(new_selection, new_attempt, NEW_COMMIT)
    _gate(old_gate, old_attempt, OLD_JOB, old, old_chosen)
    _gate(new_gate, new_attempt, NEW_JOB, new, new_chosen)
    same_question = [a['question_id'] == b['question_id'] and a['content_sha256'] == b['content_sha256']
                     for a, b in zip(old, new)]
    common = sorted(set(old_chosen).intersection(new_chosen))
    comparable = [i for i in common if same_question[i]]
    equal = sum(old_gate['row_tensor_sha256'][old[i]['trajectory_ids'][sample]] ==
                new_gate['row_tensor_sha256'][new[i]['trajectory_ids'][sample]]
                for i in comparable for sample in range(RESPONSES))
    old_counts = dict(zip(old_gate['trajectory_ids'], old_gate['actor_token_counts']))
    new_counts = dict(zip(new_gate['trajectory_ids'], new_gate['actor_token_counts']))
    token_equal = sum(old_counts[old[i]['trajectory_ids'][sample]] == new_counts[new[i]['trajectory_ids'][sample]]
                      for i in comparable for sample in range(RESPONSES))
    compared = len(comparable)*RESPONSES
    return dict(status='COMPARISON_RECORDED', run_id=RUN, outer=STEP,
        old_job=OLD_JOB, new_job=NEW_JOB, old_attempt=old_attempt, new_attempt=new_attempt,
        candidate_questions=CANDIDATES, selected_questions_each=SELECTED,
        question_id_differences=sum(a['question_id'] != b['question_id'] for a,b in zip(old,new)),
        question_content_hash_differences=sum(a['content_sha256'] != b['content_sha256'] for a,b in zip(old,new)),
        strict_label_vector_differences_at_ordinal=sum(a['strict_correctness'] != b['strict_correctness'] for a,b in zip(old,new)),
        strict_label_vector_differences_same_question=sum(same_question[i] and a['strict_correctness'] != b['strict_correctness']
                                                        for i,(a,b) in enumerate(zip(old,new))),
        selection_flag_differences=sum(a['selected'] != b['selected'] for a,b in zip(old,new)),
        selected_ordinal_overlap=len(common), comparable_same_question_selected_rows=compared,
        mapped_combined_row_hash_equal=equal, mapped_combined_row_hash_different=compared-equal,
        all_selected_combined_row_hashes_equal=compared == SELECTED*RESPONSES and equal == compared,
        whole_tensor_hash_equal={key:old_gate['tensor_sha256'][key] == new_gate['tensor_sha256'][key] for key in TENSOR_NAMES},
        whole_tensor_row_alignment_equal=old_chosen == new_chosen and all(same_question[i] for i in old_chosen),
        sampling_dtype_equal=old_gate['sampling_dtype'] == new_gate['sampling_dtype'],
        mapped_actor_token_counts_equal=token_equal, mapped_actor_token_counts_different=compared-token_equal,
        old_included_actor_tokens=old_gate['included_actor_token_count'],
        new_included_actor_tokens=new_gate['included_actor_token_count'],
        old_pool_retry=old_selection['pool_retry'], new_pool_retry=new_selection['pool_retry'],
        limits=dict(LIMITS), training_gate=False, training_failure_inferred=False,
        training_modified=False, restart_or_rollback_requested=False)


def compare(old_attempt, new_attempt):
    attempt(old_attempt); attempt(new_attempt)
    paths = dict(old_selection=OLD_BASE/'artifacts/mixed-selection'/(old_attempt+'.json'),
        old_gate=OLD_BASE/'logs'/('behavior-probability-attempt-'+old_attempt+'.json'),
        new_selection=NEW_BASE/'artifacts/mixed-selection'/(new_attempt+'.json'),
        new_gate=NEW_BASE/'logs'/('behavior-probability-attempt-'+new_attempt+'.json'))
    missing = [name for name,path in paths.items() if not path.is_file()]
    if missing:
        return dict(status='PENDING_SAVED_JSON', missing=missing, outer=STEP,
            old_attempt=old_attempt, new_attempt=new_attempt, limits=dict(LIMITS), training_gate=False,
            training_failure_inferred=False, training_modified=False, restart_or_rollback_requested=False)
    loaded = {name:_read(path) for name,path in paths.items()}
    result = compare_documents(*(loaded[name][0] for name in paths), old_attempt, new_attempt)
    result['evidence'] = {name:record for name,(_,record) in loaded.items()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-attempt', required=True); parser.add_argument('--new-attempt', required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(args.old_attempt, args.new_attempt)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps(dict(status='COMPARISON_INPUT_OBSERVATION_ERROR', error=str(error),
            training_failure_inferred=False, training_gate=False, training_modified=False,
            restart_or_rollback_requested=False), sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0  # Differences and pending evidence are observations, not training failures.


if __name__ == '__main__': raise SystemExit(main())
