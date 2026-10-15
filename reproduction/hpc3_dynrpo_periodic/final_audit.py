"""Independent, fail-closed final audit for the periodic25 native Qwen run.

This is deliberately not imported by the immutable running trainer. Run it on
HPC3, from a separately published source, after the training allocation exits.
Use a real owned CPU Slurm allocation with at least64GiB (for example the
128GiB CPU debug allocation); a login shell's8GiB cgroup cannot load rank Adam.
It never starts a job, restores a learner, writes to a source experiment, or
downloads tensors. Only the small result may be exported to a control machine.

The legacy common8 acceptance is neither sufficient nor required here. Every
training segment and periodic transaction is inspected in its original owned
namespace. Missing raw outer receipts, including a CP200 saved just before a
failed dev200 evaluation, remain explicit incomplete evidence. This module has
no facility for constructing replacement update rows.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import signal
import socket
import subprocess
import time
from typing import Any


HEAD_KIND = 'qwen_native_periodic_lastblock_binary_v1'
REFIT_OUTERS = tuple(range(0, 200, 25))
SAVE_OUTERS = (5,) + tuple(range(25, 201, 25))
DEV_OUTERS = (5, 50, 100, 150, 200)
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
DEFAULT_BASE = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016')
TEACHER = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015/teacher-warmup-cache')
CANONICAL_DATA = Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b')
INITIAL_PREPARATION = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-native-thread5-20261015')
PAIR_STATUS = 'QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS'
SCHEMA = 'dynrpo-periodic25-independent-final-audit-v1'
SHA = re.compile(r'[0-9a-f]{64}')
STRICT_SCORER_FILES = ('search_g1/mixed_runtime.py', 'search_g1/signals.py',
                       'search_r1/qa_protocol.py', 'verl/utils/reward_score/qa_em.py')


class Incomplete(ValueError):
    """Evidence is absent, contradictory, unsafe, or outside the audit budget."""
    def __init__(self, code: str, path: Any = None, detail: Any = None):
        self.code, self.path, self.detail = code, None if path is None else str(path), detail
        super().__init__(code + (': ' + str(path) if path is not None else ''))

    def record(self):
        return dict(code=self.code, path=self.path, detail=self.detail)


def require(condition, code, path=None, detail=None):
    if not condition:
        raise Incomplete(code, path, detail)


def document_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def checkpoint_namespace(path, prefix=DEFAULT_BASE):
    """Recognize owned saved CP layouts without assuming a parents[] depth."""
    match = re.fullmatch('(' + re.escape(str(prefix)) + r'(?:-r[1-9][0-9]*)?)/checkpoints/'
        r'(?:search-g1-common8/)?actor/global_step_([1-9][0-9]*)', str(path))
    require(match is not None and int(match.group(2)) <= 200, 'CHECKPOINT_OUTSIDE_OWN_NAMESPACE', path)
    return Path(match.group(1))


def check_teacher_identity(identity, official_files, run):
    require(identity.get('schema') == 2 and identity.get('model_revision') == REVISION and
            identity.get('model_files_sha256') == official_files == run.get('initial_model_sha256') and
            identity.get('data_manifest_sha256') == DATA_SHA and
            identity.get('train_parquet_sha256') == run['input_sha256']['train.parquet'] and
            identity.get('teacher_frozen_evidence_required') is True and
            len(identity.get('questions', [])) == len({r.get('qid') for r in identity.get('questions', [])}) ==
            len({r.get('question_hash') for r in identity.get('questions', [])}) == 5120,
            'INITIAL_TEACHER_FULL_MODEL_OR_DATA_IDENTITY_CHANGED')


def exact_phases(phases, completed_outer, *, pending=False):
    """Require the full expected phase list, not merely a valid short prefix."""
    expected = [v for v in REFIT_OUTERS if v <= completed_outer]
    if pending and completed_outer in REFIT_OUTERS and completed_outer:
        expected.remove(completed_outer)
    require([p.get('outer') for p in phases] == expected, 'PERIODIC_PHASE_SET_INCOMPLETE',
            detail=dict(expected=expected, observed=[p.get('outer') for p in phases]))
    for index, phase in enumerate(phases):
        require(phase.get('fit_epochs') == 20 and
                phase.get('head_adam_steps_before') == index * 400 and
                phase.get('head_adam_steps_after') == (index + 1) * 400,
                'PERIODIC_PHASE_ADAM_OR_EPOCH_BUDGET', detail=phase)
        if index:
            require(phase.get('groups') == 5120 and phase.get('head_optimizer_updates') == 400 and
                    phase.get('replay_groups') == 0, 'PERIODIC_PHASE_POOL_OR_REPLAY_BUDGET')
    return 400 * len(expected)


def check_online_features(row, outer, writes, version):
    require(row.get('stage') == 'question_probe_selection_frozen' and row.get('step') == outer and
            row.get('head_version') == version and row.get('attempted_candidates') == 5120 and
            row.get('selected_groups') == 512 and row.get('prediction_before_label') is True and
            row.get('post_rollout_em_filter') is False, 'SELECTION_BOUNDARY_OR_HEAD_CHANGED')
    feature = row.get('online_features', {})
    require(feature.get('source') == 'current_student_online' and
            feature.get('student_completed_outer') == outer - 1 and
            feature.get('student_optimizer_writes') == writes and feature.get('questions') == 5120 and
            feature.get('feature_layer') == 27 and feature.get('dynamic_tensors_persisted') is False and
            feature.get('student_gradient_enabled') is False, 'CURRENT_STUDENT_RAM_FEATURE_PROVENANCE_MISSING')
    questions = row.get('questions', [])
    require(len(questions) == 5120 and len({q.get('qid') for q in questions}) == 5120 and
            len({q.get('content_sha256') for q in questions}) == 5120 and
            sum(q.get('selected') is True for q in questions) == 512,
            'SELECTION_5120_UNIQUE_TOP512_INCOMPLETE')
    require(all(isinstance(q.get('qid'), str) and q['qid'] and
                SHA.fullmatch(q.get('question_hash', '')) and
                SHA.fullmatch(q.get('content_sha256', '')) and
                type(q.get('predicted_mixed_probability')) in (int, float) and
                math.isfinite(q['predicted_mixed_probability']) and
                0 <= q['predicted_mixed_probability'] <= 1 for q in questions),
            'SELECTION_QUESTION_OR_PREDICTION_INVALID')


def check_evaluation(value, canonical, *, run_id, step, final=False):
    require(value.get('run_id') == run_id and value.get('step') == step and
            value.get('final') is final and value.get('split') == ('final' if final else 'validation'),
            'EVALUATION_RUN_BOUNDARY_CHANGED')
    count = 51713 if final else 512
    require(value.get('validation_sha256') == canonical['sha256'] and
            value.get('question_ids') == canonical['question_ids'] and
            value.get('data_sources') == canonical['data_sources'] and
            value.get('counts') == canonical['counts'] and len(value['question_ids']) == count,
            'EVALUATION_CANONICAL_ORDER_OR_SPLIT_CHANGED')
    em, f1 = value.get('em', []), value.get('f1', [])
    require(len(em) == len(f1) == count and
            all(type(a) in (int, float) and not isinstance(a, bool) and a in (0, 1) and
                type(b) in (int, float) and not isinstance(b, bool) and math.isfinite(b) and
                a <= b <= 1 for a, b in zip(em, f1)), 'EVALUATION_EM_F1_INCOMPLETE')
    require(value.get('correct') == sum(em) and value.get('weighted_em') == sum(em) / count and
            value.get('weighted_f1') == math.fsum(f1) / count, 'EVALUATION_AGGREGATES_CHANGED')
    return dict(questions=count, correct=sum(em), weighted_em=sum(em) / count,
                weighted_f1=math.fsum(f1) / count)


def check_counter_history(complete_rows, scheduler, prefix_writes):
    """Retained writes in aborted attempts are real; require their exact ledger.

    Adjacent COMPLETE rows need not have adjacent Adam counts. The durable
    BEGIN/ABORTED or BEGIN/PREPARED/COMPLETE history must explain every write.
    Existing strict verifiers additionally bind all opportunities and births.
    """
    require(complete_rows and type(prefix_writes) is int and prefix_writes >= 0,
            'COUNTER_HISTORY_PREFIX_MISSING')
    selected = scheduler.get('counter_history', [])
    groups = []
    for row in selected:
        if not groups or row.get('attempt_id') != groups[-1][0].get('attempt_id'):groups.append([])
        groups[-1].append(row)
    counter, logical, position = prefix_writes, complete_rows[0]['step'], 0
    seen, previous_index = set(), None
    for group in groups:
        first, last = group[0], group[-1]
        kinds = [r.get('kind') for r in group]
        require(kinds in (['BEGIN', 'ABORTED_ATTEMPT'], ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE']),
                'COUNTER_HISTORY_INCOMPLETE_ATTEMPT')
        attempt, ordinal = first.get('attempt_id'), first.get('attempt_index')
        require(attempt not in seen and type(ordinal) is int and ordinal > 0 and
                (previous_index is None or ordinal == previous_index + 1) and
                all(r.get('attempt_id') == attempt and r.get('attempt_index') == ordinal and
                    r.get('logical_update') == logical and r.get('writes_before') == counter for r in group) and
                first.get('writes_after') == counter, 'UNEXPLAINED_ADAM_COUNTER_GAP')
        seen.add(attempt);previous_index = ordinal
        delta, processed = last.get('actual_optimizer_writes'), last.get('opportunities_processed')
        require(type(delta) is int and type(processed) is int and 0 <= delta <= processed <= 20 and
                last.get('writes_after') == counter + delta, 'COUNTER_HISTORY_WRITE_DISPOSITION_CHANGED')
        complete = kinds[-1] == 'COMPLETE'
        require(complete == (processed == 20 and delta > 0) and
                last.get('scheduler_advances') == int(complete) and
                last.get('completed_outer_after') == (logical if complete else logical - 1),
                'COUNTER_HISTORY_INCOMPLETE_OUTER_ADVANCED')
        if complete:
            require(position < len(complete_rows), 'COUNTER_HISTORY_EXTRA_COMPLETE')
            actual = complete_rows[position]
            require((actual.get('step'), actual.get('attempt_id'), actual.get('attempt_index'),
                     actual.get('optimizer_writes_before'), actual.get('optimizer_writes_after')) ==
                    (logical, attempt, ordinal, counter, counter + delta),
                    'COUNTER_HISTORY_RAW_COMPLETE_ROW_CHANGED')
            position += 1;logical += 1
        counter += delta
    require(position == len(complete_rows) and scheduler.get('cumulative_writes') == counter,
            'COUNTER_HISTORY_CUMULATIVE_WRITES_UNPROVEN')
    return counter


def check_rng_schema(rng, *, worker, tensor_check):
    """Inspect real RNG payloads; never set global or CUDA RNG during audit."""
    require(isinstance(rng, dict) and all(k in rng for k in ('torch_cpu', 'numpy', 'python')) and
            tensor_check(rng['torch_cpu'], cpu_generator=True), 'SAVED_DRIVER_OR_WORKER_CPU_RNG_INVALID')
    numpy = rng['numpy']
    require(isinstance(numpy, (tuple, list)) and len(numpy) == 5 and numpy[0] == 'MT19937' and
            str(getattr(numpy[1], 'dtype', '')) == 'uint32' and len(numpy[1]) == 624 and
            type(numpy[2]) is int and 0 <= numpy[2] <= 624 and numpy[3] in (0, 1) and
            math.isfinite(float(numpy[4])), 'SAVED_NUMPY_RNG_INVALID')
    try:random.Random().setstate(rng['python'])
    except (ValueError, TypeError):raise Incomplete('SAVED_PYTHON_RNG_INVALID') from None
    if worker:
        require(all(k in rng for k in ('torch_cuda', 'actor_cuda_saved', 'rollout_cuda_saved',
                                      'request_generators', 'cuda_device', 'vllm_version', 'boundary')) and
                all(tensor_check(rng[k], cpu_generator=False) for k in
                    ('torch_cuda', 'actor_cuda_saved', 'rollout_cuda_saved')) and
                type(rng['cuda_device']) is int and 0 <= rng['cuda_device'] < 4 and
                rng['vllm_version'] == '0.6.3' and rng['boundary'] == 'idle synchronous generation engine' and
                isinstance(rng['request_generators'], dict), 'SAVED_CUDA_OR_SAMPLING_RNG_INVALID')
        for key, value in rng['request_generators'].items():
            require(isinstance(key, str) and key and isinstance(value, dict) and
                    re.fullmatch(r'cuda(?::[0-9]+)?', str(value.get('device', ''))) and
                    tensor_check(value.get('state'), cpu_generator=False), 'SAVED_REQUEST_GENERATOR_RNG_INVALID')


def check_cursor_schema(cursor, *, source_consumed, outer):
    """Validate the genuine sampler permutation and compact pending slice."""
    order = cursor.get('candidate_order', [])
    count, position, epoch = cursor.get('consumed_source_rows'), cursor.get('candidate_cursor'), cursor.get('epoch')
    require(len(order) == 169028 and all(type(v) is int for v in order) and
            sorted(order) == list(range(169028)) and type(count) is int and count == source_consumed and
            count >= outer * 5120 and type(position) is int and 0 <= position <= 169028 and
            type(epoch) is int and epoch >= 0, 'SAVED_CANONICAL_CURSOR_OR_PERMUTATION_INVALID')
    length, offset = cursor.get('pending_length'), cursor.get('pending_offset')
    require(type(length) is int and type(offset) is int and
            ((length == 0 and offset == 0) or (0 < length <= 512 and 0 <= offset < length)) and
            position >= length and count == epoch * 169028 + position - (length - offset),
            'SAVED_PENDING_CURSOR_ARITHMETIC_INVALID')
    if length:
        require(cursor.get('pending_indices') == order[position - length:position],
                'SAVED_PENDING_SOURCE_INDICES_DIFFER_FROM_CANONICAL_ORDER')


def check_inherited_phase(current, inherited, pre_sha):
    require(inherited.get('outer') == current.get('outer') and
            inherited.get('pre_receipt_sha256') == current.get('pre_receipt_sha256') == pre_sha and
            inherited.get('indices') == current.get('indices') and
            inherited.get('labels') == current.get('labels', [])[:len(inherited.get('labels', []))] and
            inherited.get('frozen_before') == current.get('frozen_before') and
            (inherited.get('feature_hashes') is None or inherited.get('feature_hashes') == current.get('feature_hashes')),
            'INHERITED_REFIT_POOL_FEATURES_OR_PENDING_CHECKPOINT_CHANGED')
    for name in ('minibatches', 'summaries'):
        previous = inherited.get('phase', {}).get(name, [])
        require(previous == current.get('phase', {}).get(name, [])[:len(previous)],
                'INHERITED_REFIT_ACTUAL_ADAM_HISTORY_DISCARDED')


def validate_audit_allocation(job, *, job_id, uid, hostname, nodes, cgroup_memory=None):
    require(str(job.get('JobId')) == str(job_id) and str(job_id).isdecimal() and
            job.get('JobState') == 'RUNNING' and
            re.fullmatch(r'[^()]+\(' + str(uid) + r'\)', job.get('UserId', '')) and
            hostname.split('.')[0] in [n.split('.')[0] for n in nodes],
            'AUDIT_REQUIRES_REAL_OWNED_RUNNING_CPU_SLURM_ALLOCATION')
    memory = re.search(r'(?:^|,)mem=([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:,|$)', job.get('AllocTRES', ''))
    require(memory is not None, 'AUDIT_ALLOCATED_MEMORY_UNAVAILABLE')
    multiplier = {'K':1024, 'M':1024**2, 'G':1024**3, 'T':1024**4, '':1024**2}
    allocated = int(float(memory.group(1)) * multiplier[memory.group(2)])
    require(allocated >= 64 * 1024**3 and (cgroup_memory is None or cgroup_memory >= 64 * 1024**3),
            'AUDIT_RANK_STATE_REQUIRES_AT_LEAST64GIB_ALLOCATED_MEMORY',
            detail=dict(allocated_bytes=allocated, actual_cgroup_limit_bytes=cgroup_memory))
    return dict(accepted=True, job_id=str(job_id), uid=uid, hostname=hostname,
        allocated_memory_bytes=allocated, actual_cgroup_limit_bytes=cgroup_memory,
        required_memory_bytes=64 * 1024**3, gpu_required=False, checked_unix=time.time(), producer_pid=os.getpid())


def audit_cpu_allocation():
    """Fresh Slurm and current-process cgroup checks before tensor imports."""
    job_id = os.environ.get('SLURM_JOB_ID', '')
    require(job_id.isdecimal(), 'AUDIT_MUST_NOT_LOAD_RANK_STATE_ON_LOGIN_NODE')
    result = subprocess.run(['scontrol', 'show', 'job', job_id, '-o'], text=True,
        capture_output=True, timeout=30, check=True)
    job = dict(part.split('=', 1) for part in result.stdout.split() if '=' in part)
    nodes = subprocess.run(['scontrol', 'show', 'hostnames', job.get('NodeList', '')], text=True,
        capture_output=True, timeout=30, check=True).stdout.split()
    limits = []
    for row in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = row.split(':', 2)
        if controllers == '':root, name = Path('/sys/fs/cgroup'), 'memory.max'
        elif 'memory' in controllers.split(','):root, name = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes'
        else:continue
        leaf = (root / relative.lstrip('/')).resolve()
        if not leaf.is_relative_to(root):continue
        for directory in (leaf,) + tuple(p for p in leaf.parents if p.is_relative_to(root)):
            path = directory / name
            if path.is_file():
                raw = path.read_text().strip()
                if raw.isdecimal() and int(raw) < 2**60:limits.append(int(raw))
    return validate_audit_allocation(job, job_id=job_id, uid=os.getuid(), hostname=socket.gethostname(),
        nodes=nodes, cgroup_memory=min(limits) if limits else None)


def resolve_refit_chunks(reader, backend, chain, directory, outer, current, pre_sha):
    """Reuse only authenticated same-outer parent prefix, never arbitrary files."""
    origin_bases = {segment['base'] / 'artifacts/question-probe/periodic-refits' / str(outer):
                    segment['base'] for segment in chain}
    require(Path(directory) in origin_bases, 'REFIT_DIRECTORY_HAS_NO_AUTHENTICATED_RUN', directory)
    roots = [Path(directory)]
    for segment in reversed(chain):
        root = segment['base'] / 'artifacts/question-probe/periodic-refits' / str(outer)
        if root in roots or not (segment['first'] - 1 <= outer <= segment['last']):continue
        if (root / 'latest-state.json').is_file():roots.append(root)
    inherited = {}
    chunks, files = [], []
    for number in range(10):
        chosen = None
        for root in roots:
            path = root / ('chunk-' + str(number)) / 'real-rollout-audit.json'
            if not path.is_file():continue
            if root != Path(directory):
                if root not in inherited:
                    pointer = reader.json(root / 'latest-state.json')
                    state_path = root / 'latest-state.pt'
                    require(reader.digest(state_path) == pointer['sha256'], 'INHERITED_REFIT_LATEST_STATE_SHA_CHANGED', state_path)
                    parent = backend.phase_state(state_path)
                    check_inherited_phase(current, parent, pre_sha)
                    ledger_path = root / 'update-ledger.json'
                    if ledger_path.is_file():
                        ledger = reader.json(ledger_path)
                        require(ledger['outer'] == outer and
                                ledger['possible_adam_steps'] <= parent['head_adam_steps'],
                                'INHERITED_REFIT_UNSAVED_ADAM_AFTER_HARD_KILL', ledger_path)
                    inherited[root] = parent
                require(len(inherited[root]['labels']) >= (number + 1) * 512,
                        'INHERITED_CHUNK_NOT_ACCEPTED_IN_SAVED_PHASE', path)
            chosen = path;break
        require(chosen is not None, 'AUTHENTICATED_REFIT_CHUNK_PREFIX_OR_SUFFIX_MISSING', directory, dict(chunk=number))
        chunks.append(reader.json(chosen));files.append(dict(path=str(chosen), sha256=reader.digest(chosen),
            source_base=str(origin_bases[root])))
    return chunks, files


def check_refit_documents(complete, state, chunks, pre, post, run, outer, before_head, after_head, *, chunk_runs=None):
    """Pure semantic checks on independently loaded native state and JSON."""
    index = REFIT_OUTERS.index(outer)
    require(complete.get('stage') == 'PERIODIC_PROBE_REFIT20_ACCEPTED' and complete.get('outer') == outer and
            complete.get('fit_epochs') == 20 and complete.get('groups') == 5120 and
            complete.get('trajectories') == 25600 and complete.get('head_optimizer_updates') == 400 and
            complete.get('head_adam_steps_before') == index * 400 and
            complete.get('head_adam_steps_after') == (index + 1) * 400 and
            complete.get('replay_groups') == 0 and
            all(complete.get(k) is False for k in ('student_updated', 'student_parameters_copied',
                'optimizer_reset', 'dynamic_feature_tensors_persisted')), 'REFIT_COMPLETE_BUDGET_OR_IDENTITY_CHANGED')
    require(pre.get('step') == post.get('step') == outer and pre.get('execution_phase') == 'PROBE_PENDING' and
            post.get('execution_phase') == 'STUDENT_READY' and
            pre.get('checkpoint') == post.get('checkpoint'), 'REFIT_NOT_PRECEDED_BY_AUTHENTICATED_CHECKPOINT')
    require(before_head['adam_steps'] == index * 400 and after_head['adam_steps'] == (index + 1) * 400 and
            before_head['fingerprint'] != after_head['fingerprint'], 'REFIT_ADAM_CONTINUITY_OR_PARAMETER_UPDATE_MISSING')
    require(state.get('outer') == outer and state.get('head_adam_steps') == after_head['adam_steps'] and
            state.get('head_fingerprint') == after_head['fingerprint'], 'REFIT_SAVED_LIVE_STATE_DIFFERS_FROM_POST_HEAD')
    phase = state.get('phase', {})
    require(phase.get('start_steps') == index * 400 and phase.get('epoch') == 20 and phase.get('batch') == 0 and
            phase.get('order') is None and len(phase.get('summaries', [])) == 20 and
            len(phase.get('minibatches', [])) == 400, 'REFIT_EPOCH_CHECKPOINT_HISTORY_INCOMPLETE')
    require([r.get('adam_steps') for r in phase['minibatches']] == list(range(index * 400 + 1, (index + 1) * 400 + 1)) and
            [(r.get('epoch'), r.get('batch')) for r in phase['minibatches']] ==
            [(e, b) for e in range(1, 21) for b in range(20)], 'REFIT_REAL_MINIBATCH_HISTORY_CHANGED')
    require(all(type(r.get('loss')) in (int, float) and math.isfinite(r['loss']) for r in phase['minibatches']),
            'REFIT_NONFINITE_MINIBATCH_LOSS')
    require([r.get('epoch') for r in phase['summaries']] == list(range(1, 21)) and
            all(r.get('training_groups') == 5120 and r.get('actual_head_adam_updates') == index * 400 + 20 * (i + 1)
                and r.get('target_statistic') == 'sample_std' and r.get('auxiliary_weight') == .3
                for i, r in enumerate(phase['summaries'])), 'REFIT_ALL_TWENTY_EPOCH_SUMMARIES_MISSING')
    labels, indices, hashes = state.get('labels', []), state.get('indices', []), state.get('feature_hashes', {})
    require(len(labels) == len(indices) == len(set(indices)) == 5120 and
            all(type(i) is int and 0 <= i < 169028 for i in indices) and
            len({r.get('qid') for r in labels}) == len({r.get('question_hash') for r in labels}) == 5120 and
            set(hashes) == {r['qid'] for r in labels} and all(SHA.fullmatch(v) for v in hashes.values()),
            'REFIT_DISTINCT_5120_POOL_OR_HASH_ONLY_FEATURES_MISSING')
    require(state.get('dynamic_tensor_fields_present') is False, 'REFIT_FEATURE_TENSOR_PERSISTED')
    require(len(chunks) == 10, 'REFIT_TEN_REAL_G5_CHUNKS_MISSING')
    chunk_runs = [run] * len(chunks) if chunk_runs is None else chunk_runs
    require(len(chunk_runs) == len(chunks), 'REFIT_CHUNK_RUN_LINEAGE_INCOMPLETE')
    observed, trajectories, protocol = [], set(), None
    for i, chunk in enumerate(chunks):
        origin_run = chunk_runs[i]
        require(all(origin_run.get(k) == run.get(k) for k in ('run_id', 'budget', 'input_sha256',
                'initial_model_sha256', 'model_revision', 'retriever_version', 'training_world_size',
                'precision_profile')) and origin_run.get('retriever_topk', 3) == run.get('retriever_topk', 3),
                'REFIT_CHUNK_SCIENTIFIC_RUN_IDENTITY_CHANGED')
        require(chunk.get('outer') == outer and chunk.get('questions') == 512 and
                chunk.get('trajectories') == 2560 and type(chunk.get('policy_tokens')) is int and
                chunk['policy_tokens'] > 0, 'REFIT_REAL_ROLLOUT_CHUNK_INCOMPLETE')
        identity = chunk.get('identity', {})
        parents = identity.get('parent_group_ids', [])
        require(identity.get('run_id') == run['run_id'] and identity.get('logical_update') == outer and
                identity.get('validation') is False and len(parents) == len(set(parents)) == 512 and
                len(identity.get('trajectory_ids', [])) == len(set(identity.get('trajectory_ids', []))) == 2560 and
                not trajectories.intersection(identity['trajectory_ids']), 'REFIT_REAL_TRAJECTORY_IDENTITY_CHANGED')
        trajectories.update(identity['trajectory_ids'])
        require(all(re.fullmatch(str(i * 512 + j) + r'-[0-9a-f]{64}', parent)
                    for j, parent in enumerate(parents)) and
                identity['trajectory_ids'] == [run['run_id'] + '-' + str(outer) + '-probe-refit-' + parent +
                    '-sample-' + str(sample) for parent in parents for sample in range(5)] and
                identity.get('source_sha256') == origin_run['source_sha256'] and
                re.fullmatch(re.escape(run['run_id'] + ':' + str(outer) + ':policy-') + r'[0-9a-f]{32}',
                    identity.get('policy_snapshot_id', '')) is not None,
                'REFIT_CHUNK_ORDER_SOURCE_OR_SNAPSHOT_CHANGED')
        if 'canonical_parent_group_ids' in state:
            require(parents == state['canonical_parent_group_ids'][i * 512:(i + 1) * 512],
                    'REFIT_CHUNK_CANONICAL_PARENT_ORDER_CHANGED')
        contract = chunk.get('contracts', {})
        scoring = contract.get('strict_scoring_contract', {})
        require(contract.get('run_id') == run['run_id'] and scoring.get('target') == 'strict_terminal_em' and
                scoring.get('responses_per_question') == 5 and scoring.get('std_ddof') == 1 and
                scoring.get('common_data_manifest_sha256') == DATA_SHA and
                document_sha(scoring) == contract.get('strict_scoring_contract_sha') and
                document_sha(contract.get('sampling_contract')) == contract.get('sampling_contract_sha') and
                document_sha(contract.get('retriever_contract')) == contract.get('retriever_version'),
                'REFIT_SAMPLING_SCORING_RETRIEVAL_CONTRACT_CHANGED')
        require(set(scoring.get('source_sha256', {})) == set(STRICT_SCORER_FILES) and all(origin_run['source_sha256'].get(k) == v
                for k, v in scoring['source_sha256'].items()), 'REFIT_STRICT_EM_SOURCE_CHANGED')
        require(contract['retriever_contract'].get('recorded_index_version') == run.get('retriever_version') and
                contract['retriever_contract'].get('topk') == run.get('retriever_topk', 3),
                'REFIT_RETRIEVER_IDENTITY_CHANGED')
        current = tuple(contract[k] for k in ('sampling_contract_sha', 'strict_scoring_contract_sha', 'retriever_version'))
        if protocol is None:
            protocol = current
        require(current == protocol, 'REFIT_CHUNK_PROTOCOLS_DIFFER')
        observed.extend(chunk.get('labels', []))
    require(observed == labels and len(trajectories) == 25600, 'REFIT_ALL_REAL_LABEL_GROUPS_NOT_PRESENT')
    for label in labels:
        vector = label.get('strict_em', [])
        require(len(vector) == 5 and all(type(x) is int and x in (0, 1) for x in vector) and
                label.get('correct_count') == sum(vector), 'REFIT_STRICT_G5_LABEL_INVALID')
    frozen = state.get('frozen_before', [])
    require(sorted(r.get('rank') for r in frozen) == list(range(4)) and
            all(r.get('completed_outer') == outer and SHA.fullmatch(r.get('parameter_sha256', '')) and
                SHA.fullmatch(r.get('scheduler_sha256', '')) and r.get('actual_optimizer_writes') ==
                pre['ranks'][r['rank']]['actual_optimizer_writes'] and
                r.get('adam_steps') == [r.get('actual_optimizer_writes')] for r in frozen),
            'REFIT_CURRENT_STUDENT_FROZEN_BOUNDARY_MISSING')


class Reader:
    """No symlink traversal; stable full hashes; bounded files, bytes and time."""
    def __init__(self, namespace_prefix=DEFAULT_BASE, *, max_seconds=7200, max_hash_bytes=2 * 1024**4):
        self.prefix = Path(namespace_prefix)
        self.deadline = time.monotonic() + max_seconds
        self.maximum_bytes, self.hashed_bytes, self.files_read = max_hash_bytes, 0, 0
        self.hash_cache = {}

    def tick(self):
        require(time.monotonic() < self.deadline, 'AUDIT_DEADLINE_EXCEEDED')
        require(self.files_read < 100000, 'AUDIT_FILE_COUNT_EXCEEDED')

    def namespace(self, base):
        base = Path(base)
        require(base.is_absolute() and base.resolve() == base and not base.is_symlink() and
                re.fullmatch(re.escape(str(self.prefix)) + r'(?:-r[1-9][0-9]*)?', str(base)) is not None and
                base.is_dir() and base.stat().st_uid == os.getuid(), 'FOREIGN_OR_UNOWNED_NAMESPACE', base)
        return base

    def file(self, path):
        self.tick()
        path = Path(path)
        require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
                'MISSING_OR_UNSAFE_FILE', path)
        self.files_read += 1
        return path

    def shared_input(self, path, canonical):
        """Permit only an explicitly pinned shared input alias, never outputs."""
        path, canonical = Path(path), Path(canonical)
        require(path.is_absolute() and path.name == canonical.name and path.resolve() == canonical and
                canonical.resolve() == canonical and canonical.is_file() and not canonical.is_symlink(),
                'SHARED_INPUT_ALIAS_TARGET_CHANGED', path, dict(expected=str(canonical)))
        return self.file(canonical)

    def json(self, path):
        path = self.file(path)
        require(path.stat().st_size <= 64 * 1024**2, 'JSON_EXCEEDS_BOUNDED_METADATA_SIZE', path)
        try:
            return json.loads(path.read_text())
        except (ValueError, UnicodeError) as error:
            raise Incomplete('INVALID_JSON', path, str(error)) from error

    def jsonl(self, path):
        path = self.file(path)
        require(path.stat().st_size <= 512 * 1024**2, 'JSONL_EXCEEDS_BOUNDED_METADATA_SIZE', path)
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        return rows

    def digest(self, path):
        path = self.file(path)
        before = path.stat()
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if key not in self.hash_cache:
            require(self.hashed_bytes + before.st_size <= self.maximum_bytes,
                    'AUDIT_HASH_BYTE_BUDGET_EXCEEDED', path)
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    self.tick(); digest.update(block); self.hashed_bytes += len(block)
            self.hash_cache[key] = digest.hexdigest()
        after = path.stat()
        require(key == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'FILE_CHANGED_DURING_AUDIT', path)
        return self.hash_cache[key]

    def record(self, row):
        path = self.file(row['path']);stat = path.stat()
        require(stat.st_ino == row['inode'] and stat.st_size == row['bytes'] and stat.st_mtime_ns == row['mtime_ns'] and
                self.digest(path) == row['full_read_sha256'], 'AUTHENTICATED_FILE_BINDING_CHANGED', path)


def tensor_tree_sha(value):
    """Deterministic CPU state fingerprint, not torch.save ZIP byte identity."""
    digest = hashlib.sha256()
    def visit(item):
        # Torch/numpy are deliberately lazy: the fixture suite needs neither.
        if hasattr(item, 'detach') and hasattr(item, 'dtype'):
            import torch
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
            raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast('B')
            for start in range(0, len(raw), 8 * 1024**2):digest.update(raw[start:start + 8 * 1024**2])
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda k:(type(k).__name__, str(k))):visit(key);visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode());digest.update(str(len(item)).encode())
            for entry in item:visit(entry)
        elif hasattr(item, 'tobytes') and hasattr(item, 'dtype'):
            digest.update(str(item.dtype).encode());digest.update(str(item.shape).encode());digest.update(item.tobytes())
        else:
            digest.update(type(item).__name__.encode());digest.update(repr(item).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


class NativeBackend:
    """Real remote-only parsers and existing strict numerical verifiers."""
    def __init__(self, reader, allocation=None):
        self.reader = reader
        self.allocation = allocation or audit_cpu_allocation()

    def require_allocation(self):
        require(self.allocation.get('accepted') is True and self.allocation.get('producer_pid') == os.getpid() and
                self.allocation.get('job_id') == os.environ.get('SLURM_JOB_ID') and
                self.allocation.get('uid') == os.getuid(), 'AUDIT_CPU_ALLOCATION_IDENTITY_CHANGED')
        self.reader.tick()

    @staticmethod
    def rng_tensor(value, *, cpu_generator=False):
        import torch
        if not (torch.is_tensor(value) and value.dtype == torch.uint8 and value.ndim == 1 and
                value.numel() > 0 and value.device.type == 'cpu'):return False
        if cpu_generator:
            try:torch.Generator(device='cpu').set_state(value)
            except (ValueError, RuntimeError):return False
        return True

    def driver_payload(self, state, receipt, *, boundary):
        # Original trainer driver.pt has no run_id field; its CP manifest/full
        # SHA supplies that binding. The post-evaluation boundary has run_id.
        require((not boundary or state.get('run_id') == receipt['run_id']) and
                state['completed_outer'] == receipt['step'] and
                state.get('attempt_progress') == receipt['attempt_progress'],
                'SAVED_DRIVER_BOUNDARY_IDENTITY_CHANGED')
        check_rng_schema(state['rng'], worker=False, tensor_check=self.rng_tensor)
        cursor = dict(state['candidate_stream']);pending = cursor.pop('pending_source')
        cursor['pending_length'] = 0 if pending is None else len(pending)
        if pending is not None:
            require('qprobe_source_index' in pending.batch, 'SAVED_PENDING_SOURCE_INDICES_MISSING')
            cursor['pending_indices'] = pending.batch['qprobe_source_index'].tolist()
        check_cursor_schema(cursor, source_consumed=receipt['driver_state']['source_consumed'], outer=receipt['step'])

    def run(self, base, run):
        from reproduction.common8.formal_contract import assert_formal_run, verify_formal_config
        from search_g1.qprobe_periodic_runtime import validate_config
        assert_formal_run(run);validate_config(run['question_probe'])
        resolved = self.reader.json(base / 'logs/resolved-training-config.json')
        proof = verify_formal_config(resolved['resolved_config'], run, base)
        require(resolved['run_id'] == run['run_id'] and
                resolved['resolved_config_sha256'] == proof['resolved_config_sha256'], 'RESOLVED_CONFIG_CHANGED', base)
        publication = self.reader.json(base / 'preparation/source-publication.json')
        require(publication.get('pushed_and_fetched') is True and publication['commit'] == run['source_commit'] and
                publication['remote_commit'] == run['source_commit'], 'TRAINING_SOURCE_NOT_PUBLISHED', base)
        source = base / 'source' / run['source_commit']
        require((base / 'repo').resolve() == source and source.resolve() == source, 'IMMUTABLE_SOURCE_ALIAS_CHANGED', base)
        for name, sha in run['source_sha256'].items():
            require(Path(name).is_relative_to('.') and '..' not in Path(name).parts and
                    self.reader.digest(source / name) == sha and publication['files'].get(name) == sha,
                    'BOUND_TRAINING_SOURCE_SHA_CHANGED', source / name)

    def canonical(self, base, split):
        from reproduction.common8.validation_identity import expected
        record, frame, ids = expected(base, split)
        return dict(sha256=record['sha256'], question_ids=ids, data_sources=frame['data_source'].tolist(),
                    counts=record['source_counts'])

    def pair(self, path):
        self.require_allocation()
        from search_g1.qprobe_checkpoint import read_source, validate_rank_state, boundary_worker_rng
        from verl.utils.checkpoint_integrity import validate_saved_checkpoint
        import torch
        _, receipt, checkpoint, base, run = read_source(path)
        for row in receipt['files'] + receipt['boundary_worker_files'] + receipt['reward_files'] + \
                [receipt['boundary_driver'], receipt['source_run_manifest'], receipt['reward_probe_manifest']] + \
                receipt.get('question_probe_files', []):
            self.reader.record(row)
        manifest = self.reader.json(checkpoint / 'training_state/manifest.json')
        for item in manifest['worker_files']:
            saved = torch.load(self.reader.file(item['path']), map_location='cpu', weights_only=False)
            require(saved['run_id'] == run['run_id'], 'CHECKPOINT_RANK_RUN_CHANGED', item['path'])
            actual = validate_rank_state(saved, item['rank'], 4, receipt['step'],
                expected_writes=receipt['ranks'][item['rank']]['actual_optimizer_writes'])
            require(actual == receipt['ranks'][item['rank']], 'CHECKPOINT_RANK_STATE_READBACK_DIFFERS', item['path'])
            check_rng_schema(saved['rng'], worker=True, tensor_check=self.rng_tensor)
            boundary = boundary_worker_rng(receipt, item['rank'], saved)
            check_rng_schema(boundary, worker=True, tensor_check=self.rng_tensor)
            del boundary
            del saved;gc.collect();self.reader.tick()
        for name, boundary in ((manifest['driver_file']['path'], False), (receipt['boundary_driver']['path'], True)):
            driver = torch.load(self.reader.file(name), map_location='cpu', weights_only=False)
            self.driver_payload(driver, receipt, boundary=boundary)
            del driver;gc.collect()
        owner = self.reader.namespace(checkpoint_namespace(checkpoint, self.reader.prefix))
        original_run = self.reader.json(owner / 'logs/run-manifest.json')
        require(original_run['run_id'] == run['run_id'] and
                original_run['initial_model_sha256'] == run['initial_model_sha256'] and
                checkpoint == Path(original_run['checkpoint_dir']) / 'actor' / ('global_step_' + str(receipt['step'])),
                'INHERITED_CHECKPOINT_CREATION_NAMESPACE_CHANGED', checkpoint)
        count = validate_saved_checkpoint(checkpoint, Path(run['initial_model']), original_run['started_unix'])
        require(count == receipt['hf_saved_tensor_count'], 'CHECKPOINT_HF_TENSOR_READBACK_DIFFERS', checkpoint)
        return receipt

    def head(self, path, *, raw=False):
        self.require_allocation()
        import torch
        from search_g1.qprobe_periodic_model import PeriodicQwenProbe, restore_engine
        saved = torch.load(self.reader.file(path), map_location='cpu', weights_only=not raw)
        if raw:
            engine = restore_engine(saved['head'], device='cpu', epochs=20)
            phases, completed, version = [], 0, None
        else:
            head = PeriodicQwenProbe.load(path, expected_checkpoint_sha=self.reader.digest(path), device='cpu')
            engine, phases, completed, version = head._engine, head.phases, head.completed_outer, head.version
        state = engine.state_dict()
        learned = {k:state[k] for k in ('model', 'optimizer', 'private_rng', 'update_count')}
        result = dict(adam_steps=engine._adam_steps(), fingerprint=tensor_tree_sha(learned), phases=phases,
            completed_outer=completed, version=version, feature_spec_sha=saved.get('feature_spec_sha'),
            history_sha256=document_sha(saved.get('history', {})),
            prior_head_sha256=saved.get('prior_head_sha256'),
            trainable_tensors=len(list(engine.model.parameters())),
            trainable_parameters=sum(p.numel() for p in engine.model.parameters()),
            private_rng_present=state.get('private_rng') is not None,
            initial_block_proof=state.get('initial_block_proof'),
            initial_readout_proof=state.get('initial_readout_proof'))
        # History contains scalar summaries only; no rollout/tensor is returned.
        if raw:result['history'] = saved.get('history', {})
        del engine, state, learned, saved
        if not raw:del head
        gc.collect();self.reader.tick()
        return result

    def phase_state(self, path):
        self.require_allocation()
        import torch
        from search_g1.qprobe_periodic_model import restore_engine
        saved = torch.load(self.reader.file(path), map_location='cpu', weights_only=False)
        engine = restore_engine(saved['head'], device='cpu', epochs=20)
        state = engine.state_dict()
        result = {k:saved[k] for k in ('outer', 'pre_receipt_sha256', 'phase', 'indices', 'labels',
                  'frozen_before', 'feature_hashes', 'worker_rng', 'reason')}
        require(all(k in saved for k in ('driver_rng', 'refit_rng')), 'REFIT_DRIVER_OR_PRIVATE_RNG_MISSING', path)
        check_rng_schema(saved['driver_rng'], worker=False, tensor_check=self.rng_tensor)
        import numpy as np
        try:np.random.default_rng().bit_generator.state = saved['refit_rng']
        except (ValueError, TypeError):raise Incomplete('REFIT_PRIVATE_POOL_RNG_INVALID', path) from None
        result.update(head_adam_steps=engine._adam_steps(),
            head_fingerprint=tensor_tree_sha({k:state[k] for k in ('model', 'optimizer', 'private_rng', 'update_count')}),
            dynamic_tensor_fields_present=any(k in saved for k in ('features', 'feature_tensors', 'rollouts', 'raw_rollout')))
        phase = result['phase']
        if phase.get('order') is not None:phase['order'] = phase['order'].tolist()
        require(sorted(r['rank'] for r in result['worker_rng']) == list(range(4)), 'REFIT_SAMPLING_RNG_RANKS_MISSING', path)
        for row in result['worker_rng']:
            require(self.reader.digest(row['path']) == row['sha256'], 'REFIT_WORKER_RNG_FULL_SHA_CHANGED', row['path'])
            worker = torch.load(self.reader.file(row['path']), map_location='cpu', weights_only=False)
            require(worker['rank'] == row['rank'] and worker['world_size'] == 4 and
                    worker['completed_outer'] == result['outer'], 'REFIT_WORKER_RNG_BOUNDARY_CHANGED', row['path'])
            check_rng_schema(worker['rng'], worker=True, tensor_check=self.rng_tensor)
            del worker
        del saved, engine, state;gc.collect();self.reader.tick()
        return result

    def refit_pool(self, base, run, state):
        """Bind saved indices/labels and generation parents to canonical train rows."""
        import pyarrow.parquet as pq
        from search_g1.qprobe_features import original_question_and_id
        path = self.reader.shared_input(base / 'data/train.parquet', CANONICAL_DATA / 'train.parquet')
        require(self.reader.digest(path) == run['input_sha256']['train.parquet'],
                'REFIT_CANONICAL_TRAIN_SHA_CHANGED', base)
        frame = pq.read_table(path).to_pandas()
        require(len(frame) == 169028, 'REFIT_CANONICAL_TRAIN_ROW_COUNT_CHANGED')
        parents, contents = [], set()
        from search_g1.mixed_selection import question_identity
        for ordinal, (index, label) in enumerate(zip(state['indices'], state['labels'])):
            original = frame.iloc[index]
            question, qid = original_question_and_id(original, 'prompt')
            require(label['qid'] == qid and label['question_hash'] == hashlib.sha256(question.encode()).hexdigest(),
                    'REFIT_LABEL_CANONICAL_QUESTION_CHANGED')
            contents.add(question_identity(question))
            group = original.get('extra_info', {}).get('index', 0)
            if hasattr(group, 'item'):group = group.item()
            parents.append(str(ordinal) + '-' + document_sha(group))
        del frame;gc.collect()
        require(len(contents) == 5120, 'REFIT_CANONICAL_CONTENT_DUPLICATED')
        state['canonical_parent_group_ids'] = parents

    def frozen_refit_workers(self, directory, run, outer, writes, before_rows, allowed_roots):
        import torch
        candidates = sorted(directory.glob('frozen-after-fit-*'))
        require(candidates and len(candidates) <= 256, 'REFIT_POST_FIT_FROZEN_WORKER_BOUNDARY_MISSING', directory)
        before_paths = []
        for row in before_rows:
            proof = row['rng_receipt'];path = Path(proof['path'])
            require(path.parent in [root / 'frozen-before' for root in allowed_roots] and
                    self.reader.digest(path) == proof['sha256'], 'REFIT_INHERITED_FROZEN_BOUNDARY_CHANGED', path)
            before_paths.append(path)
        for paths in [before_paths] + [sorted(boundary.glob('rank-*.pt')) for boundary in candidates]:
            boundary = paths[0].parent if paths else directory
            require(len(paths) == 4, 'REFIT_FROZEN_BOUNDARY_FOUR_RANK_RNG_MISSING', boundary)
            ranks = []
            for path in paths:
                self.reader.digest(path)
                saved = torch.load(self.reader.file(path), map_location='cpu', weights_only=False)
                rank = saved['rank'];ranks.append(rank)
                require(saved['run_id'] == run['run_id'] and saved['world_size'] == 4 and
                        saved['completed_outer'] == outer and saved['actual_optimizer_writes'] == writes and
                        'rng' in saved and 'optimizer' not in saved,
                        'REFIT_FROZEN_STUDENT_COUNTER_OR_RNG_CHANGED', path)
                check_rng_schema(saved['rng'], worker=True, tensor_check=self.rng_tensor)
                del saved
            require(sorted(ranks) == list(range(4)), 'REFIT_FROZEN_BOUNDARY_RANKS_CHANGED', boundary)

    def segment(self, base, run, first, last, rows):
        from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
        from reproduction.common8.runtime import verify_scheduler_receipts, budget_rows
        from reproduction.common8.worker_evidence import read_console_form
        from reproduction.common8.formal_attempt_audit import verify_attempt_history
        from reproduction.common8.formal_inline_safety import verify_plan
        gates = {str(row['step']):read_gate_receipt(base, run, row['step'], update_record=row) for row in rows}
        verify_write_receipt_links(base, gates, first, last)
        receipts = verify_scheduler_receipts(base, run, first, last, gates)
        budget_rows(read_console_form(base, run['run_id']), first, last, complete_records=rows, run=run)
        artifact = self.reader.json(base / 'logs/policy-safety-calibration.json')
        _, plan_sha = verify_plan(base, artifact, run)
        verify_attempt_history(base, run, rows, receipts, last=last, plan_sha256=plan_sha)
        if run.get('full_state_restore'):
            from search_g1.qprobe_checkpoint import verified_runtime_prefix
            verified_runtime_prefix(base, run)
        return {int(rank):dict(cumulative_writes=v['cumulative_actual_optimizer_writes'],
            complete_outers=[r['logical_update'] for r in v['selected_rows'] if r['kind'] == 'COMPLETE'],
            counter_history=v['selected_rows'])
            for rank, v in receipts.items()}

    def teacher(self, root, identity, complete):
        from search_g1.qprobe_warmup import TeacherCache, STATUS
        cache = TeacherCache.__new__(TeacherCache);cache.root, cache.identity = root, identity
        require(complete.get('status') == STATUS and complete.get('questions') == 5120 and
                complete.get('trajectories') == 25600 and
                complete.get('identity_sha256') == self.reader.digest(root / 'identity.json'),
                'INITIAL_TEACHER_COMPLETE_MISSING', root)
        groups, trajectories = [], set()
        require(len(complete.get('chunks', [])) == 10, 'INITIAL_TEACHER_CHUNKS_MISSING', root)
        for i, entry in enumerate(complete['chunks']):
            name = 'chunk-%02d.json' % i
            require(entry['relative_path'] == name and self.reader.digest(root / name) == entry['sha256'],
                    'INITIAL_TEACHER_CHUNK_SHA_CHANGED', root / name)
            chunk = self.reader.json(root / name);cache.validate_chunk(i, chunk)
            require({k:chunk['contracts'][k] for k in complete['protocol']} == complete['protocol'],
                    'INITIAL_TEACHER_CHUNK_PROTOCOL_CHANGED', root / name)
            for group in chunk['groups']:
                require(not trajectories.intersection(group['trajectory_ids']), 'INITIAL_TEACHER_DUPLICATE_TRAJECTORY')
                trajectories.update(group['trajectory_ids']);groups.append(group)
        require(len(groups) == 5120 and len(trajectories) == 25600, 'INITIAL_TEACHER_POOL_INCOMPLETE')

    def terminal_job(self, base, run, submission, *, require_success=True):
        import pwd
        job = submission.get('job')
        require(isinstance(job, str) and job.isdecimal() and submission.get('returncode') == 0 and
                submission.get('commit') == run['source_commit'], 'FORMAL_JOB_IDENTITY_MISSING', base)
        account = pwd.getpwuid(os.getuid()).pw_name
        result = subprocess.run(['sacct', '-nP', '-X', '-j', job,
            '--format=JobID,State,ExitCode,User'], text=True, capture_output=True, timeout=30, check=True)
        rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
        require(len(rows) == 1 and rows[0][0] == job and rows[0][3] == account,
                'FORMAL_JOB_ACCOUNTING_IDENTITY_CHANGED', base, result.stdout.strip())
        state = rows[0][1].split()[0]
        terminal = ('COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL', 'PREEMPTED')
        require((state == 'COMPLETED' and rows[0][2] == '0:0') if require_success else state in terminal,
                'FORMAL_JOB_NOT_SUCCESSFULLY_TERMINAL' if require_success else 'ANCESTOR_JOB_NOT_TERMINAL',
                base, result.stdout.strip())
        active = subprocess.run(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i'],
            text=True, capture_output=True, timeout=30, check=True).stdout.split()
        require(job not in active, 'FORMAL_JOB_STILL_ACTIVE', base)
        return dict(job=job, state=state, exit_code=rows[0][2], account=account, freshly_queried=True)


class Auditor:
    def __init__(self, base, *, reader=None, backend=None):
        self.reader = reader or Reader()
        self.base = self.reader.namespace(base)
        self.backend = backend or NativeBackend(self.reader)
        self.chain, self.pairs, self.heads = [], {}, {}
        self.report = dict(schema=SCHEMA, accepted=False, structural_accepted=False,
            base=str(self.base), checked_unix=time.time(), requirements={}, incomplete=[],
            semantics='Fresh independent evidence audit; no learner update, rollback, fabricated raw row, or acceptance by legacy green')

    def stage(self, name, work):
        try:
            self.reader.tick()
            self.report['requirements'][name] = dict(accepted=True, evidence=work())
        except Incomplete as error:
            self.report['requirements'][name] = dict(accepted=False, incomplete=error.record())
            self.report['incomplete'].append(dict(requirement=name, **error.record()))
        except Exception as error:
            failure = Incomplete('VERIFIER_FAILED_OR_EVIDENCE_UNAVAILABLE', detail=type(error).__name__ + ': ' + str(error))
            self.report['requirements'][name] = dict(accepted=False, incomplete=failure.record())
            self.report['incomplete'].append(dict(requirement=name, **failure.record()))

    def read_pair(self, path):
        path = self.reader.file(path)
        key = str(path)
        if key not in self.pairs:
            # Check external-CP dependency pointers before the older recursive
            # read_source parser; a cyclic pointer graph must not recurse.
            dependency, seen = path, set()
            for _ in range(32):
                require(str(dependency) not in seen, 'PARENT_RECEIPT_CYCLE', dependency)
                seen.add(str(dependency));item = self.reader.json(dependency)
                self.reader.namespace(item['source_base'])
                inherited = item.get('inherited_checkpoint_receipt')
                if inherited is None:break
                dependency = Path(inherited)
            else:raise Incomplete('PARENT_RECEIPT_CHAIN_DEPTH_EXCEEDED', path)
            raw = self.reader.json(path)
            source = self.reader.namespace(raw['source_base'])
            require(raw.get('status') == PAIR_STATUS and raw.get('world_size') == 4 and
                    raw.get('source_run_manifest', {}).get('path') == str(source / 'logs/run-manifest.json'),
                    'PAIR_SOURCE_NAMESPACE_OR_KIND_CHANGED', path)
            self.reader.record(raw['source_run_manifest'])
            verified = self.backend.pair(path)
            require(raw == verified, 'PAIR_VERIFIER_READBACK_CHANGED', path)
            self.pairs[key] = raw
        return self.pairs[key]

    def head(self, path, raw=False):
        key = (str(path), raw)
        if key not in self.heads:self.heads[key] = self.backend.head(path, raw=raw)
        return self.heads[key]

    def ancestors(self):
        base, limit, seen, seen_receipts = self.base, 200, set(), set()
        identity = None
        for _ in range(32):
            require(str(base) not in seen, 'PARENT_NAMESPACE_CYCLE', base)
            seen.add(str(base));run = self.reader.json(base / 'logs/run-manifest.json')
            require(run.get('execution_site') == 'hpc3' and run.get('training_world_size') == 4 and
                    run.get('target_updates') == 200 and run.get('model_revision') == REVISION and
                    run.get('question_probe', {}).get('head_kind') == HEAD_KIND and
                    run.get('question_probe_refit_outers', list(REFIT_OUTERS)) == list(REFIT_OUTERS) and
                    run.get('extra_checkpoint_steps', list(SAVE_OUTERS[1:])) == list(SAVE_OUTERS[1:]),
                    'PERIODIC_RUN_IDENTITY_OR_BUDGET_CHANGED', base)
            current = {k:run.get(k) for k in ('run_id', 'budget', 'input_sha256', 'initial_model_sha256',
                       'model_revision', 'retriever_version', 'training_world_size', 'precision_profile')}
            if identity is None:identity = current;self.run_id = run['run_id']
            require(current == identity, 'PARENT_SCIENTIFIC_IDENTITY_CHANGED', base)
            self.backend.run(base, run)
            restore = run.get('full_state_restore')
            offset = restore.get('completed_outer', 0) if restore else 0
            require(type(offset) is int and 0 <= offset <= limit and limit <= 200,
                    'PARENT_OUTER_BOUNDARY_INVALID', base)
            self.chain.append(dict(base=base, run=run, first=offset + 1, last=limit))
            if not restore:
                self.chain.reverse()
                return dict(namespaces=[str(s['base']) for s in self.chain],
                    immutable_training_commits=[s['run']['source_commit'] for s in self.chain],
                    parent_sources_fully_hashed=True)
            require(restore.get('kind') == 'question_probe_pair', 'UNSUPPORTED_OR_UNAUTHENTICATED_PARENT_KIND', base)
            path = Path(restore['source_receipt'])
            require(str(path) not in seen_receipts, 'PARENT_RECEIPT_CYCLE', path)
            seen_receipts.add(str(path))
            require(self.reader.digest(path) == restore['source_receipt_sha256'], 'PARENT_RECEIPT_SHA_CHANGED', path)
            receipt = self.read_pair(path)
            parent = self.reader.namespace(restore['source_base'])
            require(parent != base and receipt['source_base'] == str(parent) and receipt['run_id'] == run['run_id'] and
                    receipt['step'] == offset and receipt['checkpoint'] == restore['source_checkpoint'] and
                    receipt['source_run_manifest']['full_read_sha256'] == restore['source_run_manifest_sha256'],
                    'PARENT_RECEIPT_BOUNDARY_OR_NAMESPACE_CHANGED', path)
            base, limit = parent, offset
        raise Incomplete('PARENT_CHAIN_DEPTH_EXCEEDED')

    def train_segments(self):
        require(self.chain, 'AUTHENTICATED_CHAIN_UNAVAILABLE')
        collected, result, previous_writes = [], [], 0
        for segment in self.chain:
            base, run, first, last = (segment[k] for k in ('base', 'run', 'first', 'last'))
            rows = self.reader.jsonl(base / 'logs/actual-updates.jsonl')
            expected = list(range(first, last + 1))
            observed = [r.get('step') for r in rows]
            if observed == list(range(first, 200)) and last == 200:
                raise Incomplete('RAW_OUTER200_ROW_MISSING_CP200_IS_NOT_A_REPLACEMENT', base,
                    'Authenticate CP200 and all four rank states, then run missing evaluation separately; a supplemental recovery proof must not invent the missing original row')
            require(observed == expected, 'RAW_OUTER_HISTORY_INCOMPLETE_OR_UNACCEPTED_ROLLBACK', base,
                    dict(expected_first=first, expected_last=last, observed_first=observed[:1], observed_last=observed[-1:]))
            if not rows:continue
            require(all(r.get('run_id') == self.run_id and r.get('opportunities_processed') == 20 and
                        type(r.get('actual_optimizer_writes')) is int and 1 <= r['actual_optimizer_writes'] <= 20 and
                        r['optimizer_writes_after'] - r['optimizer_writes_before'] == r['actual_optimizer_writes'] and
                        all(type(v) in (int, float) and math.isfinite(v) for v in r['metrics'].values()) for r in rows),
                    'ACTUAL_OUTER_COUNTERS_OR_FINITE_METRICS_CHANGED', base)
            proof = self.backend.segment(base, run, first, last, rows)
            require(sorted(proof) == list(range(4)) and all(v['complete_outers'] == expected and
                    v['cumulative_writes'] == rows[-1]['optimizer_writes_after'] for v in proof.values()),
                    'FOUR_RANK_SCHEDULER_COMPLETION_MISSING', base)
            for rank in range(4):check_counter_history(rows, proof[rank], previous_writes)
            for row in rows:row['_base'] = base
            segment['rows'] = rows;collected.extend(rows);previous_writes = rows[-1]['optimizer_writes_after']
            result.append(dict(base=str(base), first=first, last=last, actual_outers=len(rows),
                four_rank_cumulative_writes=previous_writes, raw_history_sha256=self.reader.digest(base / 'logs/actual-updates.jsonl')))
        require([r['step'] for r in collected] == list(range(1, 201)), 'FULL200_RAW_OUTER_HISTORY_MISSING')
        self.rows = collected
        return dict(actual_outers=200, four_ranks=True, segments=result,
            optimizer_counting='Every complete outer processes20 opportunities and retains1..20 actual writes, preserving the bound formal contract')

    def find(self, relative, outer):
        candidates = [s['base'] / relative for s in reversed(self.chain)
            if s['first'] - 1 <= outer <= s['last'] and (s['base'] / relative).is_file()]
        require(candidates, 'REQUIRED_ORIGINAL_NAMESPACE_ARTIFACT_MISSING', relative, dict(outer=outer))
        return candidates[0]

    def checkpoint_pairs(self):
        require(self.chain, 'AUTHENTICATED_CHAIN_UNAVAILABLE')
        result, last_head = [], None
        self.boundaries = {}
        for outer in SAVE_OUTERS:
            root = Path('artifacts/question-probe')
            pending_path = self.find(root / 'pending-probe-pairs' / ('global_step_' + str(outer)) / 'source-receipt.json', outer)
            completed_path = self.find(root / 'completed-pairs' / ('global_step_' + str(outer)) / 'source-receipt.json', outer)
            pre, post = self.read_pair(pending_path), self.read_pair(completed_path)
            expected_phase = 'PROBE_PENDING' if outer in REFIT_OUTERS else ('FINAL_EVALUATION_PENDING' if outer == 200 else 'STUDENT_READY')
            require(pre['step'] == post['step'] == outer and pre['run_id'] == post['run_id'] == self.run_id and
                    pre['checkpoint'] == post['checkpoint'] and pre['execution_phase'] == expected_phase and
                    post['execution_phase'] == ('FINAL_EVALUATION_PENDING' if outer == 200 else 'STUDENT_READY'),
                    'CHECKPOINT_PENDING_COMPLETED_PHASE_PAIR_CHANGED', completed_path)
            before, after = self.head(pre['head_path']), self.head(post['head_path'])
            require(before['completed_outer'] == after['completed_outer'] == outer and
                    before['version'] == pre['head_version'] and after['version'] == post['head_version'],
                    'PAIRED_PROBE_READBACK_CHANGED', completed_path)
            before_expected = exact_phases(before['phases'], outer, pending=outer in REFIT_OUTERS)
            after_expected = exact_phases(after['phases'], outer)
            require(before['adam_steps'] == before_expected and after['adam_steps'] == after_expected and
                    before['private_rng_present'] and after['private_rng_present'], 'PAIRED_PROBE_ADAM_OR_RNG_MISSING', completed_path)
            require(before['trainable_tensors'] == after['trainable_tensors'] == 14 and
                    before['trainable_parameters'] == after['trainable_parameters'] == 78076418,
                    'NATIVE_LAST_BLOCK_READOUT_SHAPE_CHANGED', completed_path)
            if last_head is not None:
                require(before['fingerprint'] == last_head['fingerprint'], 'PROBE_CHANGED_BETWEEN_REFITS', pending_path)
            if outer not in REFIT_OUTERS:
                require(before['fingerprint'] == after['fingerprint'], 'UNSCHEDULED_PROBE_UPDATE', completed_path)
            self.boundaries[outer] = dict(pre=pre, post=post, pre_path=pending_path, post_path=completed_path,
                before=before, after=after)
            last_head = after
            result.append(dict(outer=outer, checkpoint=post['checkpoint'], pending_pair=str(pending_path),
                pending_pair_sha256=self.reader.digest(pending_path), completed_pair=str(completed_path),
                completed_pair_sha256=self.reader.digest(completed_path), head_adam_steps=after['adam_steps'],
                execution_phase=post['execution_phase'], full_files_fresh_sha_and_cpu_state_readback=True))
        require(last_head['adam_steps'] == 3200 and [p['outer'] for p in last_head['phases']] == list(REFIT_OUTERS),
                'FINAL_ALL_EIGHT_PHASES_AND_ADAM3200_NOT_PROVEN')
        return result

    def initial_teacher_and_warmup(self):
        require(self.chain and hasattr(self, 'boundaries'), 'CHECKPOINT_CHAIN_UNAVAILABLE')
        first = self.chain[0];base, run = first['base'], first['run']
        official = self.reader.json(base / 'preparation/official-model-manifest.json')
        require(official.get('repo') == 'Qwen/Qwen2.5-3B-Instruct' and official.get('revision') == REVISION,
                'OFFICIAL_INITIAL_MODEL_IDENTITY_CHANGED')
        from reproduction.hpc3_qprobe.model_manifest import FILES
        records = {r['file']:r for r in official.get('files', [])}
        require(len(records) == len(official.get('files', [])) == 9 and
                {k:v['sha256'] for k, v in records.items()} == FILES == run.get('initial_model_sha256'),
                'OFFICIAL_INITIAL_NINE_FILE_MANIFEST_CHANGED')
        for name, row in records.items():
            path = Path(run['initial_model']) / name
            require(path.stat().st_size == row['bytes'] and self.reader.digest(path) == row['sha256'],
                    'OFFICIAL_INITIAL_MODEL_FRESH_FULL_SHA_CHANGED', path)
        root = Path(run['question_probe']['teacher_warmup_cache'])
        require(root == TEACHER, 'INITIAL_TEACHER_NAMESPACE_CHANGED', root)
        identity = self.reader.json(root / 'identity.json');complete = self.reader.json(root / 'complete.json')
        check_teacher_identity(identity, FILES, run)
        self.backend.teacher(root, identity, complete)
        ready = self.reader.json(base / 'warmup/ready.json');raw_path = base / 'warmup/continued-head.pt'
        require(ready.get('head_adam_steps') == 400 and ready.get('fit_epochs') == 20 and
                ready.get('inherited_adam_steps') == 200 and ready.get('new_adam_updates') == 200 and
                ready.get('actual_saved_model_adam_rng_readback') is True and
                ready.get('student_parameters_copied') is False and ready.get('optimizer_reset') is False and
                self.reader.digest(raw_path) == ready.get('head_sha256'), 'INITIAL_WARMUP400_INHERITANCE_NOT_PROVEN', raw_path)
        provenance = ready.get('label_producer_provenance', {})
        require(provenance.get('teacher_identity_sha256') == self.reader.digest(root / 'identity.json') and
                provenance.get('teacher_complete_sha256') == self.reader.digest(root / 'complete.json') and
                provenance.get('questions') == 5120 and provenance.get('trajectories_reused') == 25600,
                'INITIAL_WARMUP_REAL_LABEL_PROVENANCE_CHANGED')
        raw = self.head(raw_path, raw=True)
        previous_ready = self.reader.json(INITIAL_PREPARATION / 'warmup/ready.json')
        previous = INITIAL_PREPARATION / 'warmup/fresh-head.pt'
        require(previous_ready['head_adam_steps'] == 200 and
                ready['inherited_head_sha256'] == previous_ready['head_sha256'] == self.reader.digest(previous) and
                raw['prior_head_sha256'] == previous_ready['head_sha256'], 'ORIGINAL_TEN_EPOCH_STATE_NOT_RETAINED', previous)
        old = self.head(previous, raw=True)
        require(old['adam_steps'] == 200 and raw['adam_steps'] == 400 and
                raw['history'].get('epoch_history', [])[:11] == old['history'].get('epoch_history') and
                raw['history'].get('minibatch_history', [])[:200] == old['history'].get('minibatch_history') and
                len(raw['history'].get('epoch_history', [])) == 21 and
                len(raw['history'].get('minibatch_history', [])) == 400,
                'INITIAL_TEN_EPOCH_HISTORY_DISCARDED')
        adopted_path = Path(run['question_probe']['initial_head_checkpoint'])
        require(self.reader.digest(adopted_path) == run['question_probe']['initial_head_checkpoint_sha256'],
                'INITIAL_ADOPTED_HEAD_SHA_CHANGED', adopted_path)
        adopted = self.head(adopted_path)
        require(adopted['adam_steps'] == 400 and adopted['fingerprint'] == raw['fingerprint'] ==
                self.boundaries[5]['before']['fingerprint'], 'INITIAL_ADAM_MODEL_RNG_RESET_OR_EARLY_HEAD_UPDATE')
        exact_phases(adopted['phases'], 0)
        return dict(official_revision=REVISION, official_nine_files_fresh_sha=True,
            teacher_full_model_files_sha256_equal=True, real_initial_questions=5120, real_initial_trajectories=25600,
            inherited_adam_steps=200, preserved_initial_adam_steps=400, inherited_history_retained=True,
            adopted_head=str(adopted_path), adopted_head_sha256=self.reader.digest(adopted_path))

    def periodic_refits(self):
        require(hasattr(self, 'boundaries') and len(self.boundaries) == len(SAVE_OUTERS), 'ALL_CHECKPOINT_BOUNDARIES_UNAVAILABLE')
        result = []
        for outer in REFIT_OUTERS[1:]:
            boundary = self.boundaries[outer];pre, post = boundary['pre'], boundary['post']
            directory = Path(post['source_base']) / 'artifacts/question-probe/periodic-refits' / str(outer)
            complete_path = directory / 'complete.json';complete = self.reader.json(complete_path)
            require(self.reader.digest(complete['pre_checkpoint_receipt']) == complete['pre_checkpoint_sha256'] and
                    self.read_pair(complete['pre_checkpoint_receipt']) == pre,
                    'REFIT_PENDING_CHECKPOINT_RECEIPT_CHANGED', complete_path)
            post_state = directory / 'post-probe-state.pt'
            require(str(post_state) == post['question_probe_state_path'] and
                    self.reader.digest(post_state) == complete['post_state_sha256'], 'REFIT_POST_STATE_SHA_CHANGED', post_state)
            pointer = self.reader.json(directory / 'latest-state.json')
            state_path = directory / 'latest-state.pt'
            require(self.reader.digest(state_path) == pointer['sha256'] and pointer.get('outer') == outer and
                    pointer.get('epoch') == 20 and pointer.get('batch') == 0,
                    'REFIT_EPOCH_RECOVERY_STATE_SHA_OR_POSITION_CHANGED', state_path)
            state = self.backend.phase_state(state_path)
            require(state['pre_receipt_sha256'] == complete['pre_checkpoint_sha256'], 'REFIT_LIVE_STATE_PRECHECKPOINT_CHANGED')
            ledger = self.reader.json(directory / 'update-ledger.json')
            require(ledger.get('outer') == outer and ledger.get('possible_adam_steps') == complete['head_adam_steps_after'] and
                    ledger.get('stage') == 'complete', 'REFIT_UNSAVED_OR_AMBIGUOUS_ADAM_WRITES', directory)
            chunks, chunk_files = resolve_refit_chunks(self.reader, self.backend, self.chain,
                directory, outer, state, complete['pre_checkpoint_sha256'])
            segment = next(s for s in self.chain if s['base'] == Path(post['source_base']))
            authenticated_runs = {str(s['base']):s['run'] for s in self.chain}
            chunk_runs = [authenticated_runs[item['source_base']] for item in chunk_files]
            self.backend.refit_pool(segment['base'], segment['run'], state)
            self.backend.frozen_refit_workers(directory, segment['run'], outer,
                pre['ranks'][0]['actual_optimizer_writes'], state['frozen_before'],
                [s['base'] / 'artifacts/question-probe/periodic-refits' / str(outer) for s in self.chain])
            check_refit_documents(complete, state, chunks, pre, post, segment['run'], outer,
                                  boundary['before'], boundary['after'], chunk_runs=chunk_runs)
            # Phase tensors are explicitly hash-only. Extra named tensor banks
            # in this transaction are not silently treated as acceptable cache.
            require(not any(p.is_file() for pattern in ('*features*.pt', '*features*.bin', '*features*.npy',
                    '*rollout*.pt', '*rollout*.parquet') for p in directory.rglob(pattern)),
                    'DYNAMIC_FEATURE_OR_RAW_ROLLOUT_TENSOR_PERSISTED', directory)
            result.append(dict(outer=outer, complete=str(complete_path), complete_sha256=self.reader.digest(complete_path),
                pending_checkpoint_before_collection=True, questions=5120, trajectories=25600, epochs=20,
                adam_steps_before=complete['head_adam_steps_before'], adam_steps_after=complete['head_adam_steps_after'],
                latest_epoch_state_full_sha=True, actual_minibatch_history=400,
                authenticated_chunk_files=chunk_files,
                feature_scope='Saved hash-only live features and frozen current-student rank counters; layer27/no-grad/RAM execution is additionally witnessed by every actual selection receipt and immutable encoder source'))
        return result

    def selection_intervals(self):
        require(hasattr(self, 'rows') and hasattr(self, 'boundaries'), 'RAW_HISTORY_AND_HEAD_BOUNDARIES_UNAVAILABLE')
        initial = self.boundaries[5]['before'];versions = {0:initial['version']}
        for outer in REFIT_OUTERS[1:]:versions[outer] = self.boundaries[outer]['after']['version']
        files = []
        for row in self.rows:
            step = row['step'];phase = ((step - 1) // 25) * 25
            path = row['_base'] / 'artifacts/question-probe/attempts' / row['attempt_id'] / 'selection.json'
            selected = self.reader.json(path)
            require(selected.get('attempt_id') == row['attempt_id'], 'SELECTION_ATTEMPT_ID_CHANGED', path)
            check_online_features(selected, step, row['optimizer_writes_before'], versions[phase])
            event = self.reader.json(row['_base'] / 'artifacts/question-probe/completed' / (row['attempt_id'] + '.json'))
            require(event.get('outer') == step and event.get('attempt_id') == row['attempt_id'] and
                    event.get('head_version') == versions[phase] and event.get('head_update', {}).get('head_optimizer_updates') == 0 and
                    event['head_update'].get('fit_epochs') == 0 and event['head_update'].get('replay_groups') == 0,
                    'PROBE_UPDATED_DURING_ORDINARY_OUTER', path)
            files.append(dict(outer=step, sha256=self.reader.digest(path)))
        return dict(actual_online_selection_receipts=200, feature_layer=27, detached=True,
            remote_ram_only=True, probe_frozen_between_refits=True, receipt_map_sha256=document_sha(files))

    def evaluations(self):
        require(self.chain, 'AUTHENTICATED_CHAIN_UNAVAILABLE')
        canonical = self.backend.canonical(self.chain[0]['base'], 'validation')
        final_canonical = self.backend.canonical(self.chain[0]['base'], 'final')
        result = {}
        for outer in DEV_OUTERS:
            path = self.find(Path('logs') / ('validation-step' + str(outer) + '.json'), outer)
            value = self.reader.json(path)
            proof = check_evaluation(value, canonical, run_id=self.run_id, step=outer)
            result[str(outer)] = dict(path=str(path), sha256=self.reader.digest(path), **proof)
        path = self.base / 'logs/final-identity.json'
        proof = check_evaluation(self.reader.json(path), final_canonical, run_id=self.run_id, step=200, final=True)
        result['final'] = dict(path=str(path), sha256=self.reader.digest(path), **proof)
        return result

    def cleanup(self):
        require(self.chain, 'AUTHENTICATED_CHAIN_UNAVAILABLE')
        result = []
        for segment in self.chain:
            base, run = segment['base'], segment['run']
            cleanup = self.reader.json(base / 'logs/cleanup-status.json')
            ledger = self.reader.json(base / 'logs/process-ledger.json')
            require(cleanup.get('run_id') == ledger.get('run_id') == self.run_id and cleanup.get('cleaned') is True and
                    all(job.get('state') == 'STOPPED' and not job.get('remaining') for job in ledger.get('jobs', {}).values()),
                    'OWNED_RESOURCE_CLEANUP_NOT_VERIFIED', base)
            require(ledger.get('jobs'), 'OWNED_PROCESS_LEDGER_EMPTY', base)
            if base == self.base:
                submission = self.reader.json(base / 'preparation/formal-submission.json')
                job = self.backend.terminal_job(base, run, submission)
                status = self.reader.json(base / 'logs/pipeline-status.json')
                require(status.get('run_id') == self.run_id and
                        status.get('stage') == 'FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP',
                        'TRAINING_PIPELINE_NOT_FINISHED', base)
            else:
                # Ancestors may be failed jobs, but all their owned processes
                # must have been reconciled before the authenticated restart.
                submission = self.reader.json(base / 'preparation/formal-submission.json')
                job = self.backend.terminal_job(base, run, submission, require_success=False)
                job['scope'] = 'Fresh terminal ancestor accounting and owned cleanup; failed ancestor need not have completed the target'
            result.append(dict(base=str(base), cleanup_sha256=self.reader.digest(base / 'logs/cleanup-status.json'), **job))
        return result

    def execute(self):
        started = time.monotonic()
        for name, method in (
                ('authenticated_parent_chain_and_sources', self.ancestors),
                ('full200_four_rank_actual_history', self.train_segments),
                ('all_checkpoint_and_pair_readbacks', self.checkpoint_pairs),
                ('initial_official_teacher_and_continuous_warmup', self.initial_teacher_and_warmup),
                ('all_seven_current_student_refits', self.periodic_refits),
                ('all_current_student_selection_intervals', self.selection_intervals),
                ('fixed512_and_final51713_evaluations', self.evaluations),
                ('owned_cleanup_and_successful_terminal_job', self.cleanup)):
            self.stage(name, method)
        self.report['accepted'] = self.report['structural_accepted'] = not self.report['incomplete']
        self.report.update(elapsed_seconds=time.monotonic() - started, full_sha_bytes=self.reader.hashed_bytes,
            source_files_read=self.reader.files_read, training_source_mutated=False, learner_updates_performed=0,
            final_accuracy_is_measured_not_a_success_threshold=True,
            audit_cpu_allocation=getattr(self.backend, 'allocation', None))
        return self.report


@contextlib.contextmanager
def deadline_alarm(seconds):
    """Bound third-party verifier calls too; only used by this standalone CLI."""
    def timed_out(_signum, _frame):raise Incomplete('AUDIT_DEADLINE_EXCEEDED')
    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(max(1, math.ceil(seconds)))
    try:yield
    finally:signal.alarm(0);signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Small JSON only, outside the immutable source tree')
    parser.add_argument('--max-seconds', type=int, default=7200)
    parser.add_argument('--max-hash-bytes', type=int, default=2 * 1024**4)
    args = parser.parse_args()
    require(0 < args.max_seconds <= 86400 and 0 < args.max_hash_bytes <= 4 * 1024**4, 'INVALID_AUDIT_BUDGET')
    reader = Reader(max_seconds=args.max_seconds, max_hash_bytes=args.max_hash_bytes)
    base = reader.namespace(args.base)
    require(str(Path(__file__).resolve()).startswith('/data/'), 'TENSOR_AUDIT_MUST_EXECUTE_ON_REMOTE_SHARED_STORAGE')
    require(args.output.is_absolute() and args.output.resolve() == args.output and
            args.output.is_relative_to(base / 'logs') and not args.output.exists(), 'AUDIT_OUTPUT_MUST_BE_NEW_OWNED_SMALL_RECEIPT')
    with deadline_alarm(args.max_seconds):result = Auditor(base, reader=reader).execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(accepted=result['accepted'], report=str(args.output),
        incomplete=result['incomplete'], elapsed_seconds=result['elapsed_seconds'], full_sha_bytes=result['full_sha_bytes']),
        ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result['accepted'] else 2)


if __name__ == '__main__':main()
