"""Authenticated CP200 evaluation recovery; never enter a training loop.

Deploy this module in a NEW numbered namespace. ``prepare`` authenticates the
complete terminal learner on CPU and binds this immutable execution. ``run``
requires a real four-GPU Slurm allocation, starts its own canonical retriever,
and evaluates only missing canonical dev512/final51713 products. GPU workers
use the upstream ``rollout`` role: no actor optimizer or scheduler is created.
CPU checkpoint qualification may instantiate the existing probe verification
loader, but that loader is never fitted or used by the inference workers.

Committed 64-occurrence scalar chunks can be imported into another new recovery
namespace with --previous-base. A failed in-flight chunk is recomputed; accepted
chunks are immutable. This is evaluation recovery, not a reconstruction of a
missing training row200 or a claim of bitwise training/RNG continuation.
"""
import argparse
from collections import Counter
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys
import time
import uuid

PREFIX = '/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016'
HEAD_KIND = 'qwen_native_periodic_lastblock_binary_v1'
ROLE = 'dynrpo_cp200_inference_only_recovery'
PAIR_STATUS = 'QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS'
COUNTS = {'validation': 512, 'final': 51713}
CHUNK_SIZE = 64
STATE_AUTH_MIN_MEMORY_BYTES = 64 * 1024**3
ROW_FIELDS = frozenset(('question_id', 'data_source', 'em', 'f1', 'actual_calls',
    'valid_action', 'turns', 'answer_sha256', 'answer_present',
    'response_token_count', 'model_token_count'))
TERMINAL = frozenset(('COMPLETED', 'FAILED', 'CANCELLED', 'OUT_OF_MEMORY',
                     'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'))


def need(condition, message):
    if not condition:
        raise ValueError('FINAL_EVAL_' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path, *, limit=32 * 1024**2):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
         'MISSING_UNSAFE_OR_OVERSIZED_DOCUMENT_' + str(path))
    return json.loads(path.read_text())


def owned(base):
    base = Path(base)
    need(re.fullmatch(re.escape(PREFIX) + r'-r[1-9][0-9]*', str(base)) is not None
         and base.is_dir() and base.resolve() == base and not base.is_symlink()
         and base.stat().st_uid == os.getuid(), 'NEW_OWNED_NUMBERED_NAMESPACE_REQUIRED')
    return base


def source_namespace(base):
    """The terminal learner must belong to this independent experiment."""
    base=Path(base)
    need(re.fullmatch(re.escape(PREFIX)+r'(?:-r[1-9][0-9]*)?',str(base)) is not None,
         'SOURCE_MUST_BELONG_TO_THIS_EXPERIMENT')
    return base


def output(base, path):
    base, path = owned(base), Path(path)
    need(path.is_relative_to(base) and path.resolve() == path and not path.is_symlink(),
         'OUTPUT_ESCAPES_OWN_NAMESPACE')
    if path.exists():
        need(path.stat().st_uid == os.getuid(), 'OUTPUT_BELONGS_TO_ANOTHER_OWNER')
    return path


def write_once(base, path, value):
    path = output(base, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b'\n'
    if path.exists():
        need(path.read_bytes() == raw, 'IMMUTABLE_OUTPUT_DIFFERS_' + str(path))
        return
    # An interrupted temporary remains visible; no existing result is replaced.
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def execution_guard(base):
    path = output(base, Path(base)/'preparation/final-eval.lock')
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('FINAL_EVAL_ALREADY_ACTIVE') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def qualify_boundary(receipt, parent, checkpoint):
    """Additional terminal-only checks after the existing authenticated reader."""
    need(receipt.get('status') == PAIR_STATUS and type(receipt.get('step')) is int
         and receipt['step'] == 200 and receipt.get('world_size') == 4,
         'ACTUAL_COMPLETE_FOUR_RANK_CP200_REQUIRED')
    need(parent.get('question_probe', {}).get('head_kind') == HEAD_KIND
         and parent.get('training_world_size') == 4 and parent.get('execution_site') == 'hpc3',
         'WRONG_METHOD_OR_TOPOLOGY')
    need(receipt.get('execution_phase') == 'FINAL_EVALUATION_PENDING'
         and receipt['actor_snapshot_id'] == parent['run_id'] + ':200'
         and receipt['run_id'] == parent['run_id'], 'WRONG_TERMINAL_PHASE_OR_ACTOR')
    need(Path(checkpoint).name == 'global_step_200', 'WRONG_CHECKPOINT_DIRECTORY')
    progress = receipt['attempt_progress']
    need(progress.get('completed_outer') == 200 and
         type(progress.get('cumulative_writes')) is int and progress['cumulative_writes'] >= 200,
         'ACTUAL_OPTIMIZER_PROGRESS_MISSING')
    need(sorted(row['rank'] for row in receipt['ranks']) == list(range(4))
         and all(row['completed_outer'] == row['scheduler_last_epoch'] == 200
                 and row['scheduler_step_count'] == 201
                 and row['actual_optimizer_writes'] == progress['cumulative_writes']
                 for row in receipt['ranks']), 'FOUR_RANK_TERMINAL_COUNTERS_DIVERGE')
    return dict(completed_student_outers=200,
        source_optimizer_writes_per_rank=progress['cumulative_writes'],
        reconstruction_of_original_update_row200=False, training_resume=False)


def validate_state_auth_resources(job, nodes, *, job_id, uid, username, hostname,
                                  cgroup_memory_limit_bytes=None):
    """Pure resource check: actual allocation, never requested TRES/host RAM."""
    need(isinstance(job_id, str) and re.fullmatch('[1-9][0-9]*', job_id) is not None
         and job.get('JobId') == job_id and job.get('JobState') == 'RUNNING',
         'STATE_AUTH_REQUIRES_ACTUAL_RUNNING_SLURM_JOB')
    user = re.fullmatch(r'([^()]+)\(([0-9]+)\)', job.get('UserId', ''))
    need(user is not None and user[1] == username and int(user[2]) == uid,
         'STATE_AUTH_SLURM_USER_IDENTITY_CHANGED')
    need(isinstance(nodes, (list, tuple)) and bool(nodes)
         and hostname.split('.')[0] in {node.split('.')[0] for node in nodes},
         'STATE_AUTH_MUST_RUN_INSIDE_ALLOCATED_NODELIST')
    # Slurm's AllocTRES is authoritative. ReqTRES/MinMemoryNode or total host
    # memory cannot prove memory actually assigned to this process's job.
    tres = job.get('AllocTRES', '')
    match = re.search(r'(?:^|,)mem=([0-9]+)([KMGTPE]?)(?:,|$)', tres)
    need(match is not None, 'STATE_AUTH_ACTUAL_ALLOCTRES_MEMORY_MISSING')
    units = {'': 2**20, 'K': 2**10, 'M': 2**20, 'G': 2**30, 'T': 2**40, 'P': 2**50, 'E': 2**60}
    actual_memory = int(match[1]) * units[match[2]]
    need(actual_memory >= STATE_AUTH_MIN_MEMORY_BYTES, 'STATE_AUTH_REQUIRES_AT_LEAST_64_GIB_ALLOCATED_MEMORY')
    if cgroup_memory_limit_bytes is not None:
        need(type(cgroup_memory_limit_bytes) is int
             and cgroup_memory_limit_bytes >= STATE_AUTH_MIN_MEMORY_BYTES,
             'STATE_AUTH_ACTUAL_CGROUP_MEMORY_LIMIT_BELOW_64_GIB')
    return dict(slurm_job_id=job_id, actual_job_state=job['JobState'], uid=uid,
        username=username, hostname=hostname, allocated_nodes=list(nodes),
        actual_alloc_tres=tres, actual_allocated_memory_bytes=actual_memory,
        actual_cgroup_memory_limit_bytes=cgroup_memory_limit_bytes,
        minimum_required_memory_bytes=STATE_AUTH_MIN_MEMORY_BYTES,
        gpu_allocation_required=False, host_total_memory_used_as_evidence=False)


def cgroup_memory_limits():
    """Read current and visible ancestor hard limits on Linux cgroup v1/v2."""
    memberships = [line.split(':', 2) for line in Path('/proc/self/cgroup').read_text().splitlines()]
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        columns, filesystem = left.split(), right.split()
        if filesystem[0] not in ('cgroup', 'cgroup2'):
            continue
        mounts.append((filesystem[0], columns[3], columns[4], filesystem[-1].split(',')))
    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), str(value))
    observed = []
    for _, controllers, member in memberships:
        for kind, root, mount, options in mounts:
            unified = kind == 'cgroup2' and controllers == ''
            if not unified and not (kind == 'cgroup' and 'memory' in controllers.split(',') and 'memory' in options):
                continue
            root, mount, member = Path(unescape(root)), Path(unescape(mount)), Path(unescape(member))
            need('..' not in member.parts and root.is_absolute() and mount.is_absolute() and member.is_absolute(),
                 'STATE_AUTH_UNSAFE_CGROUP_PATH')
            if member.is_relative_to(root):
                relative = member.relative_to(root)
            else:
                # A private cgroup namespace can report its own visible root.
                need(member == Path('/'), 'STATE_AUTH_CGROUP_MEMBERSHIP_NOT_MAPPED')
                relative = Path('.')
            current = mount/relative
            need(current.is_dir() and current.resolve() == current, 'STATE_AUTH_CGROUP_DIRECTORY_UNAVAILABLE')
            filename = 'memory.max' if unified else 'memory.limit_in_bytes'
            checked = 0
            while current.is_relative_to(mount):
                path = current/filename
                if path.is_file():
                    raw = path.read_text().strip()
                    need(raw == 'max' or raw.isdecimal(), 'STATE_AUTH_INVALID_CGROUP_MEMORY_LIMIT')
                    observed.append(dict(path=str(path), bytes=None if raw == 'max' else int(raw)))
                    checked += 1
                if current == mount:
                    break
                current = current.parent
            # The root of an unbounded unified controller has no memory.max.
            need(checked > 0 or (unified and relative == Path('.')), 'STATE_AUTH_CGROUP_MEMORY_CONTROLLER_UNREADABLE')
            if checked == 0:
                observed.append(dict(path=str(mount), bytes=None, unified_root_unbounded=True))
    need(bool(observed), 'STATE_AUTH_ACTUAL_CGROUP_MEMORY_LIMITS_UNAVAILABLE')
    finite = [row['bytes'] for row in observed if row['bytes'] is not None]
    return (min(finite) if finite else None), observed


def state_auth_resource_preflight():
    """Must run before importing Torch or parsing large learner tensors."""
    job_id = os.environ.get('SLURM_JOB_ID')
    need(isinstance(job_id, str) and re.fullmatch('[1-9][0-9]*', job_id) is not None,
         'STATE_AUTH_REQUIRES_ALLOCATED_SLURM_CPU_OR_GPU_JOB_NOT_LOGIN_HOST')
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', job_id], text=True, timeout=30)
    job = dict(re.findall(r'(\w+)=(\S+)', raw))
    node_list = job.get('NodeList', '')
    need(node_list and node_list not in ('(null)', 'None', 'N/A'), 'STATE_AUTH_ALLOCATED_NODELIST_MISSING')
    nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list], text=True, timeout=30).splitlines()
    limit, cgroups = cgroup_memory_limits()
    value = validate_state_auth_resources(job, nodes, job_id=job_id, uid=os.getuid(),
        username=pwd.getpwuid(os.getuid()).pw_name, hostname=socket.gethostname(),
        cgroup_memory_limit_bytes=limit)
    value.update(actual_cgroup_limits=cgroups, checked_unix=time.time())
    return value


def authenticate(path):
    """Fresh full SHA, actual CPU Adam/scheduler/RNG/cursor and native probe."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('CP200 receipt does not yet exist: ' + str(path))
    need(not path.is_symlink() and path.resolve() == path, 'UNSAFE_SOURCE_RECEIPT')
    candidate=next((p for p in path.parents if re.fullmatch(
        re.escape(PREFIX)+r'(?:-r[1-9][0-9]*)?',str(p))),None)
    need(candidate is not None, 'SOURCE_MUST_BELONG_TO_THIS_EXPERIMENT')
    resources = state_auth_resource_preflight()
    import torch
    from search_g1.qprobe_checkpoint import read_source, validate_rank_state, boundary_worker_rng
    from search_g1.qprobe_periodic_model import PeriodicQwenProbe, REFIT_OUTERS
    from search_g1.full_state_restore import equal_tree
    p, receipt, checkpoint, origin, parent = read_source(path)
    origin=source_namespace(origin)
    need(origin==candidate and origin.stat().st_uid==os.getuid(),
         'SOURCE_NAMESPACE_OR_OWNER_CHANGED')
    proof = qualify_boundary(receipt, parent, checkpoint)
    records = receipt['files'] + receipt['boundary_worker_files'] + [receipt['boundary_driver'],
        receipt['source_run_manifest'], receipt['reward_probe_manifest']] + receipt['reward_files']
    records += receipt.get('question_probe_files', [])
    seen = set()
    for row in records:
        current = Path(row['path'])
        if str(current) in seen:
            continue
        seen.add(str(current))
        before = current.stat()
        need(not current.is_symlink() and digest(current) == row['full_read_sha256'],
             'FRESH_SOURCE_FULL_SHA_CHANGED_' + str(current))
        after = current.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), 'SOURCE_CHANGED_DURING_SHA')
    by_name = {row['relative_path']: row for row in receipt['files']}
    for rank in range(4):
        saved = torch.load(by_name[f'training_state/rank-{rank}.pt']['path'],
                           map_location='cpu', weights_only=False)
        actual = validate_rank_state(saved, rank, 4, 200,
            expected_writes=receipt['attempt_progress']['cumulative_writes'])
        expected = next(row for row in receipt['ranks'] if row['rank'] == rank)
        equal_tree(actual, expected)
        rng = boundary_worker_rng(receipt, rank, saved)
        need(set(('torch_cpu', 'torch_cuda', 'numpy', 'python', 'actor_cuda_saved',
                  'rollout_cuda_saved', 'request_generators')).issubset(rng), 'WORKER_RNG_MISSING')
        need(torch.is_tensor(rng['torch_cpu']) and rng['torch_cpu'].dtype == torch.uint8,
             'INVALID_WORKER_RNG')
        del saved, rng
    driver = torch.load(receipt['boundary_driver']['path'], map_location='cpu', weights_only=False)
    need(driver['completed_outer'] == 200 and set(('torch_cpu', 'numpy', 'python')).issubset(driver['rng']),
         'DRIVER_RNG_OR_TERMINAL_STATE_MISSING')
    cursor = driver['candidate_stream']
    need(type(cursor['consumed_source_rows']) is int
         and cursor['consumed_source_rows'] == receipt['driver_state']['source_consumed']
         and len(cursor['candidate_order']) == 169028, 'CANDIDATE_CURSOR_OR_ORDER_MISSING')
    head = PeriodicQwenProbe.load(receipt['head_path'],
        expected_actor_snapshot_id=receipt['actor_snapshot_id'],
        expected_feature_spec_sha=receipt['feature_spec_sha'],
        expected_checkpoint_sha=digest(receipt['head_path']), device='cpu')
    need(head.completed_outer == 200 and [row['outer'] for row in head.phases] == list(REFIT_OUTERS)
         and head._adam_steps() == 3200 and head.version == receipt['head_version'],
         'ALL_EIGHT_CONTINUOUS_PROBE_PHASES_REQUIRED')
    head.validate_state()
    proof.update(source_receipt=str(p), source_receipt_sha256=digest(p),
        source_checkpoint=str(checkpoint), source_base=str(origin), run_id=parent['run_id'],
        source_head_version=head.version, source_probe_adam_steps=head._adam_steps(),
        actual_cpu_model_adam_scheduler_rng_cursor_authentication=True,
        state_auth_resource_preflight=resources,
        fresh_source_files_full_sha_count=len(seen), actual_gpu_inference_verified=False)
    del head, driver
    return proof, receipt, parent


def canonical_product(base, split):
    need(split in COUNTS, 'ONLY_CANONICAL_DEV512_AND_FINAL51713_ALLOWED')
    from reproduction.common8.validation_identity import expected
    record, frame, ids = expected(Path(base), split)
    sources = frame['data_source'].tolist()
    need(len(frame) == len(ids), 'CANONICAL_FRAME_OR_IDS_MISALIGNED')
    validate_product_binding(record, ids, sources, split)
    return record, frame, ids, sources


def validate_product_binding(record, ids, sources, split):
    need(split in COUNTS, 'ONLY_CANONICAL_DEV512_AND_FINAL51713_ALLOWED')
    need(len(ids) == len(sources) == record['rows'] == COUNTS[split], 'CANONICAL_CARDINALITY_CHANGED')
    need(all(isinstance(value, str) and value for value in ids), 'CANONICAL_IDS_MISSING')
    need(dict(Counter(sources)) == record['source_counts'], 'CANONICAL_SOURCE_COUNTS_CHANGED')


def measured_rows(rows, ids, sources, *, start=0):
    need(len(rows) > 0 and start >= 0 and start + len(rows) <= len(ids), 'ROW_INTERVAL_INVALID')
    for ordinal, row in enumerate(rows, start):
        need(isinstance(row, dict) and set(row) == ROW_FIELDS and row['question_id'] == ids[ordinal]
             and row['data_source'] == sources[ordinal], 'CANONICAL_OCCURRENCE_ORDER_CHANGED')
        need(type(row['em']) is int and row['em'] in (0, 1)
             and type(row['f1']) in (int, float) and math.isfinite(row['f1'])
             and row['em'] <= row['f1'] <= 1, 'INVALID_STRICT_EM_OR_F1')
        need(type(row['turns']) is int and 1 <= row['turns'] <= 5
             and type(row['actual_calls']) is int and type(row['valid_action']) is int
             and 0 <= row['actual_calls'] <= 4
             and row['actual_calls'] <= row['valid_action'] <= row['turns'], 'INVALID_EXECUTOR_COUNTS')
        need(type(row['response_token_count']) is int and type(row['model_token_count']) is int
             and 0 <= row['model_token_count'] <= row['response_token_count'], 'INVALID_MODEL_TOKEN_COUNTS')
        present, sha = row['answer_present'], row['answer_sha256']
        need(type(present) is bool and ((present and isinstance(sha, str) and
             re.fullmatch('[0-9a-f]{64}', sha) is not None) or
             (not present and sha is None and row['em'] == row['f1'] == 0)), 'INVALID_ANSWER_PROVENANCE')
    return rows


def accepted_source_score(origin, proof, split):
    """Reuse only an actually complete canonical source score; never a stage64."""
    need(split in COUNTS, 'ONLY_CANONICAL_DEV512_AND_FINAL51713_ALLOWED')
    path = Path(origin)/'logs'/('validation-step200.json' if split == 'validation' else 'final-identity.json')
    if not path.exists():
        return None
    record, _, ids, sources = canonical_product(origin, split)
    value = read(path)
    need(value.get('run_id') == proof['run_id'] and type(value.get('step')) is int and value['step'] == 200
         and value.get('split') == split and value.get('final') is (split == 'final')
         and value.get('validation_sha256') == record['sha256'] and value.get('question_ids') == ids
         and value.get('data_sources') == sources and len(value.get('em', [])) == COUNTS[split]
         and len(value.get('f1', [])) == COUNTS[split], 'SOURCE_SCORE_IDENTITY_OR_CARDINALITY_CHANGED')
    need(all(type(em) in (int, float) and em in (0, 1) and type(f1) in (int, float)
             and math.isfinite(f1) and em <= f1 <= 1 for em, f1 in zip(value['em'], value['f1'])),
         'SOURCE_SCORE_NOT_FINITE_STRICT_EM_F1')
    return dict(path=str(path), sha256=digest(path), questions=COUNTS[split],
        weighted_em=sum(value['em'])/COUNTS[split], weighted_f1=math.fsum(value['f1'])/COUNTS[split],
        source_counts=record['source_counts'], actual_source_evaluation_reused=True)


def context(run, proof):
    """Bind every generation/retrieval/scoring source, plus the evaluator itself."""
    prefixes = ('search_r1/llm_agent/', 'search_r1/qa_protocol', 'search_r1/search/',
                'verl/workers/rollout/', 'verl/third_party/vllm/')
    names = ('reproduction/common8/calibration_evaluation.py', 'reproduction/common8/behavior_runtime.py',
        'reproduction/common8/context_budget.py', 'reproduction/common8/generation_behavior.py',
        'reproduction/common8/sampling_capture.py', 'reproduction/common8/http_retrieval.py',
        'reproduction/common8/retriever.py', 'reproduction/common8/validation_identity.py',
        'reproduction/common8/verify_data.py', 'reproduction/hpc3_dynrpo_periodic/final_eval.py',
        'search_g1/async_rollout_runtime.py', 'search_g1/rollout_phase_contract.py',
        'search_g1/signals.py', 'verl/utils/reward_score/qa_em.py', 'verl/workers/fsdp_workers.py')
    sources = {name: sha for name, sha in run['source_sha256'].items()
               if name.startswith(prefixes) or name in names}
    need(all(name in sources for name in names), 'SCORING_OR_GENERATION_SOURCE_MISSING')
    value = dict(source_receipt_sha256=proof['source_receipt_sha256'],
        actor_snapshot_id=proof['run_id']+':200', data_manifest_sha256=run['budget']['common_manifest_sha256'],
        canonical_products={split: run['input_sha256'][split+'.parquet'] for split in COUNTS},
        source_files_sha256=sources, infra_rollout=run['infra_rollout'], retriever_topk=3,
        validation_sampling='canonical_greedy_one_response', model_compute_dtype='bfloat16',
        max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500, scoring='actual_generated_model_spans_strict_EM_alias_F1',
        retain_all_ordered_occurrences=True, student_updates_forbidden=True, probe_updates_forbidden=True)
    return value, hashlib.sha256(canonical(value)).hexdigest()


def verify_deployment(base, commit):
    base = owned(base)
    publication = read(base/'preparation/source-publication.json')
    need(publication['commit'] == commit and publication['pushed_and_fetched'] is True,
         'PUBLISHED_IMMUTABLE_CODE_REQUIRED')
    root = base/'source'/commit
    need(root.resolve() == root and not root.is_symlink() and (base/'repo').resolve() == root
         and Path(__file__).resolve() == root/'reproduction/hpc3_dynrpo_periodic/final_eval.py',
         'EVALUATOR_SOURCE_IDENTITY_CHANGED')
    for name, expected in publication['files'].items():
        need(digest(root/name) == expected, 'DEPLOYED_SOURCE_SHA_CHANGED_' + name)
    from reproduction.common8.source_files import runtime_files
    return {str(path.relative_to(root)): digest(path) for path in runtime_files(root)}


def validate_generated(generated, identity):
    """Canonical greedy evaluation has N occurrences, never the G5 count2560."""
    import torch
    from search_g1.mixed_runtime import ROW_META
    from search_g1.qprobe_runtime import _derived_policy_snapshot_id
    ids = identity['trajectory_ids']
    need(identity.get('validation') is True and len(ids) == len(set(ids)) == len(generated),
         'EVALUATION_TRAJECTORY_CARDINALITY_CHANGED')
    actual = generated.non_tensor_batch.get('behavior_trajectory_id')
    turns = generated.non_tensor_batch.get('behavior_turns')
    need(actual is not None and list(actual) == ids and turns is not None
         and len(turns) == len(ids) and all(turns), 'EVALUATION_TRAJECTORY_IDENTITY_CHANGED')
    need(_derived_policy_snapshot_id(generated.meta_info, turns, identity) == identity['policy_snapshot_id'],
         'EVALUATION_POLICY_SNAPSHOT_CHANGED')
    need(all(key in generated.meta_info and len(generated.meta_info[key]) == len(ids) for key in ROW_META),
         'EVALUATION_EXECUTOR_METADATA_MISSING')
    required = ('responses', 'prompts', 'attention_mask', 'info_mask', 'sampling_mask', 'sampling_log_probs')
    need(all(key in generated.batch for key in required), 'EVALUATION_GENERATED_TENSORS_MISSING')
    response, mask, probabilities = (generated.batch[key] for key in ('responses', 'sampling_mask', 'sampling_log_probs'))
    need(response.shape == mask.shape == probabilities.shape
         and torch.isfinite(probabilities[mask.bool()]).all()
         and generated.batch['attention_mask'].shape[-1] == generated.batch['prompts'].shape[-1]+response.shape[-1]
         and torch.equal(mask.bool(), generated.batch['info_mask'][:, -response.shape[-1]:].bool()),
         'EVALUATION_MODEL_TOKEN_CONTEXT_OR_PROBABILITIES_CHANGED')


def source_terminal(origin):
    admission = read(Path(origin)/'logs/native-allocation-admission.json')
    job = admission['slurm_job_id']
    need(isinstance(job, str) and job.isdecimal(), 'SOURCE_JOB_ID_MISSING')
    text = subprocess.check_output(['sacct', '-nPX', '-j', job, '--format=JobID,User,State'], text=True)
    rows = [line.split('|') for line in text.splitlines() if line.strip()]
    need(len(rows) == 1 and rows[0][0] == job and rows[0][1] == pwd.getpwuid(os.getuid()).pw_name
         and rows[0][2].split()[0].split('+')[0] in TERMINAL,
         'SOURCE_JOB_MUST_BE_AUTHENTICATED_TERMINAL')
    return dict(job=job, accounting=rows[0])


def prepare(base, source_receipt, commit, previous_base=None):
    base = owned(base)
    need(not (base/'logs/run-manifest.json').exists() and not (base/'preparation/final-eval-binding.json').exists(),
         'EXISTING_EXECUTION_MUST_NOT_BE_REBOUND')
    source_map = verify_deployment(base, commit)
    proof, receipt, parent = authenticate(source_receipt)
    need(base != Path(proof['source_base']), 'SOURCE_NAMESPACE_IS_READ_ONLY')
    terminal = source_terminal(proof['source_base'])
    accepted = {split: accepted_source_score(proof['source_base'], proof, split) for split in COUNTS}
    missing = [split for split, value in accepted.items() if value is None]
    if not missing:
        return dict(status='SOURCE_DEV512_AND_FINAL51713_ALREADY_COMPLETE', accepted=accepted,
                    new_gpu_execution_needed=False)
    run = copy.deepcopy(parent)
    for key in ('slurm_job_id', 'ray_tmp_dir', 'ray_tmp_dir_identity', 'ray_tmp_retention', 'execution_transition'):
        run.pop(key, None)
    # completed_updates=0 remains the original fresh scientific contract field;
    # full_state_restore/evaluation_only explicitly identify actual CP200. No
    # trainer.fit or learner restoration entry is invoked by this module.
    run.update(source_commit=commit, source_sha256=source_map, native_ray_execution_id=uuid.uuid4().hex,
        checkpoint_dir=str(base/'checkpoints/search-g1-common8'),
        train_log=str(base/'logs/FORBIDDEN-training.log'), evaluation_only=True,
        optimizer_semantics='no inference optimizer exists; authentic terminal learner remains immutable',
        full_state_restore=dict(kind='question_probe_pair', evaluation_only=True,
            source_receipt=proof['source_receipt'], source_receipt_sha256=proof['source_receipt_sha256'],
            source_checkpoint=proof['source_checkpoint'], source_base=proof['source_base'],
            completed_outer=200, actual_optimizer_writes_per_rank=proof['source_optimizer_writes_per_rank']))
    run['question_probe'].update(output_dir=str(base/'artifacts/UNUSED-question-probe'),
        snapshot_path=receipt['question_probe_state_path'])
    from reproduction.common8.native_ray import prepare as prepare_ray
    prepare_ray(run)
    ctx, ctx_sha = context(run, proof)
    for name, value in ctx['source_files_sha256'].items():
        if name != 'reproduction/hpc3_dynrpo_periodic/final_eval.py':
            need(parent['source_sha256'].get(name) == value, 'SOURCE_GENERATION_OR_SCORER_CHANGED_'+name)
    for split in COUNTS:
        record, _, _, _ = canonical_product(base, split)
        need(record['sha256'] == run['input_sha256'][split+'.parquet'], 'RECOVERY_DATA_CHANGED')
    binding = dict(schema_version=1, role=ROLE, source=proof, source_terminal=terminal,
        source_context=ctx, source_context_sha256=ctx_sha, missing_splits=missing,
        accepted_source_scores=accepted, previous_base=str(previous_base) if previous_base else None,
        checked_unix=time.time(), actual_gpu_inference_verified=False, original_training_rows_modified=False)
    write_once(base, base/'logs/run-manifest.json', run)
    write_once(base, base/'preparation/final-eval-binding.json', binding)
    if previous_base is not None:
        import_chunks(base, Path(previous_base), binding)
    return dict(status='AUTHENTICATED_CP200_EVALUATION_ONLY_PREPARED',
                completed_student_outers=200, missing_splits=missing,
                actual_gpu_inference_verified=False, original_training_rows_modified=False)


def validate_chunk(value, binding, split, start, ids, sources):
    need(split in COUNTS, 'ONLY_CANONICAL_DEV512_AND_FINAL51713_ALLOWED')
    need(value.get('role') == ROLE and value.get('source_context_sha256') == binding['source_context_sha256']
         and value.get('split') == split and value.get('start') == start
         and value.get('end') == min(start+CHUNK_SIZE, COUNTS[split])
         and value.get('actual_gpu_inference_verified') is True,
         'CHUNK_SOURCE_OR_INTERVAL_CHANGED')
    need(len(value.get('rows', [])) == value['end']-start, 'INCOMPLETE_CHUNK')
    measured_rows(value['rows'], ids, sources, start=start)
    need(value.get('rows_sha256') == hashlib.sha256(canonical(value['rows'])).hexdigest(),
         'CHUNK_ROWS_SHA_CHANGED')
    proofs = value.get('gpu_readback', [])
    need(sorted(row.get('rank') for row in proofs) == list(range(4))
         and len({row['actual_cuda_uuid'] for row in proofs}) == 4
         and all(row['no_optimizer_constructed'] is True and row['vllm_readback']['accepted'] is True
                 for row in proofs), 'CHUNK_ACTUAL_FOUR_GPU_WEIGHT_READBACK_MISSING')
    return value


def import_chunks(base, previous, binding):
    previous = owned(previous)
    need(previous != base, 'RECOVERY_SOURCE_EQUALS_TARGET')
    old = read(previous/'preparation/final-eval-binding.json')
    need(old['role'] == ROLE and old['source_context_sha256'] == binding['source_context_sha256'],
         'PARTIAL_EVALUATION_CONTEXT_CHANGED')
    imported = []
    for split in binding['missing_splits']:
        _, _, ids, sources = canonical_product(base, split)
        root = previous/'evaluations'/split
        for start in range(0, COUNTS[split], CHUNK_SIZE):
            path = root/('chunk-%06d.json' % start)
            if not path.exists():
                break
            value = validate_chunk(read(path, limit=256*1024), binding, split, start, ids, sources)
            write_once(base, base/'evaluations'/split/path.name, value)
            imported.append(dict(split=split, start=start, source_path=str(path), sha256=digest(path)))
        # A gap is not a license to reorder or stitch an unauthenticated suffix.
        expected_names = {'chunk-%06d.json' % row['start'] for row in imported if row['split'] == split}
        existing = {p.name for p in root.glob('chunk-*.json')} if root.exists() else set()
        need(existing == expected_names, 'PARTIAL_EVALUATION_HAS_NONCONTIGUOUS_CHUNKS')
    write_once(base, base/'preparation/imported-evaluation-chunks.json', dict(source=str(previous), chunks=imported))


class NoUpdates:
    """Pure CPU-testable guard inherited by every inference worker."""
    def reject_update(self, *args, **kwargs):
        raise RuntimeError('FINAL_EVAL_NO_STUDENT_OR_PROBE_OPTIMIZER_UPDATES')

    def assert_no_optimizer(self):
        need(not self._is_actor and not self._is_ref and self._is_rollout
             and self.actor_optimizer is None and self.actor_lr_scheduler is None,
             'INFERENCE_MUST_CONSTRUCT_NO_OPTIMIZER_OR_SCHEDULER')


def hf_model_mapping(actual_names, weight_map, *, tie_word_embeddings):
    """HF safetensors may omit the tied lm_head alias, never another tensor."""
    actual, declared = set(actual_names), set(weight_map)
    need(declared <= actual, 'CHECKPOINT_CONTAINS_UNLOADED_MODEL_TENSORS')
    aliases = actual - declared
    need(not aliases or (tie_word_embeddings is True and aliases == {'lm_head.weight'}
         and 'model.embed_tokens.weight' in declared), 'UNVERIFIED_LOADED_MODEL_TENSOR')
    return {name: ('model.embed_tokens.weight' if name in aliases else name) for name in actual}


def worker_class():
    """Subclass the existing generation worker, without modifying runtime code."""
    import torch
    from verl.single_controller.base.decorator import register, Dispatch
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    class EvaluationOnlyWorker(NoUpdates, ActorRolloutRefWorker):
        def __init__(self, config, role='rollout'):
            need(role == 'rollout', 'ONLY_OPTIMIZER_FREE_ROLLOUT_ROLE_ALLOWED')
            super().__init__(config, role)
            self._eval_vllm_readback = None

        def _build_model_optimizer(self, *args, **kwargs):
            need(kwargs.get('optim_config') is None and not self._is_actor,
                 'OPTIMIZER_CONSTRUCTION_FORBIDDEN')
            result = super()._build_model_optimizer(*args, **kwargs)
            need(result[1] is None and result[2] is None, 'INFERENCE_OPTIMIZER_WAS_CONSTRUCTED')
            return result

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model()
            self.assert_no_optimizer()
            self.actor_module_fsdp.eval()
            self.actor_module_fsdp.requires_grad_(False)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def update_actor(self, *args, **kwargs):
            self.reject_update()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def save_checkpoint(self, *args, **kwargs):
            self.reject_update()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def restore_mixed_full_state(self, *args, **kwargs):
            self.reject_update()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def sampling_readback(self):
            from reproduction.common8.native_allocation import cuda_uuids
            self.assert_no_optimizer()
            need(self._eval_vllm_readback is not None and self._eval_vllm_readback['accepted'],
                 'ACTUAL_SAMPLING_WEIGHT_READBACK_MISSING')
            return dict(rank=self.rank, no_optimizer_constructed=True,
                actual_cuda_uuid=cuda_uuids(torch.cuda.device_count())[torch.cuda.current_device()],
                vllm_readback=self._eval_vllm_readback)

        def _parameter_sha(self):
            result = hashlib.sha256()
            for name, parameter in self.actor_module_fsdp.named_parameters():
                value = parameter.detach().cpu().contiguous()
                need(not parameter.requires_grad and torch.isfinite(value).all(), 'INFERENCE_PARAMETERS_NOT_FROZEN_FINITE')
                result.update(name.encode()); result.update(str(value.dtype).encode())
                result.update(str(tuple(value.shape)).encode())
                result.update(value.view(torch.uint8).numpy().tobytes())
            return result.hexdigest()

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inference_readback(self, final=False):
            from safetensors import safe_open
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            from reproduction.common8.native_allocation import cuda_uuids
            self.assert_no_optimizer()
            run = read(Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json')
            cp = Path(run['full_state_restore']['source_checkpoint'])
            checked, error = 0, None
            if not final:
                with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    state = self.actor_module_fsdp.state_dict()
                if self.rank == 0:
                    try:
                        weights = read(cp/'model.safetensors.index.json')['weight_map']
                        mapping = hf_model_mapping(state, weights,
                            tie_word_embeddings=self.actor_model_config.tie_word_embeddings)
                        if 'lm_head.weight' in state and self.actor_model_config.tie_word_embeddings:
                            need(torch.equal(state['lm_head.weight'], state['model.embed_tokens.weight']),
                                 'LOADED_TIED_OUTPUT_EMBEDDING_CHANGED')
                        for shard in sorted(set(weights.values())):
                            with safe_open(str(cp/shard), framework='pt', device='cpu') as tensors:
                                for name in tensors.keys():
                                    value = tensors.get_tensor(name).to(dtype=state[name].dtype)
                                    need(torch.equal(state[name], value), 'LOADED_CP200_MODEL_VALUE_CHANGED_'+name)
                                    for alias, original in mapping.items():
                                        if alias != name and original == name:
                                            need(torch.equal(state[alias], value), 'LOADED_CP200_TIED_ALIAS_CHANGED')
                                    checked += 1
                    except Exception as failure:
                        error = repr(failure)
                del state
                errors = [None]*4
                torch.distributed.all_gather_object(errors, error)
                need(not any(errors), 'ACTUAL_CP200_GPU_LOAD_FAILED_' + repr(errors))
                self._eval_parameter_sha = self._parameter_sha()
            else:
                need(self._parameter_sha() == self._eval_parameter_sha and self._eval_vllm_readback is not None,
                     'INFERENCE_CHANGED_MODEL_OR_NEVER_GENERATED')
            return dict(rank=self.rank, model_parameter_sha256=self._eval_parameter_sha,
                actual_cuda_uuid=cuda_uuids(torch.cuda.device_count())[torch.cuda.current_device()],
                checkpoint=str(cp), checkpoint_tensors_checked_rank0=checked,
                no_optimizer_constructed=True, no_scheduler_constructed=True, model_gradients_disabled=True,
                vllm_readback=self._eval_vllm_readback, final=final)

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def begin_generation_phase(self, snapshot_meta):
            result = super().begin_generation_phase(snapshot_meta)
            self.assert_no_optimizer()
            if self._eval_vllm_readback is None:
                from safetensors import safe_open
                from reproduction.hpc3_dynrpo_periodic.smoke_identity import qwen_weight_readback
                cp = Path(self.config.model.actor_checkpoint_path)
                names = read(cp/'model.safetensors.index.json')['weight_map']
                weights = {}
                for shard in sorted(set(names.values())):
                    with safe_open(str(cp/shard), framework='pt', device='cpu') as tensors:
                        weights.update({name: tensors.get_tensor(name) for name in tensors.keys()})
                engine = self.rollout.inference_engine.llm_engine
                actual = engine.model_executor.worker.model_runner.model
                self._eval_vllm_readback = qwen_weight_readback(weights, actual, self.actor_model_config)
                del weights
                proofs = [None]*4
                torch.distributed.all_gather_object(proofs, self._eval_vllm_readback)
                need(all(row['accepted'] for row in proofs), 'ACTUAL_VLLM_CP200_WEIGHTS_CHANGED')
            return result
    return EvaluationOnlyWorker


def evaluate_split(base, binding, split, config, tokenizer, group):
    import torch
    from torch.utils.data import DataLoader, Subset
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_g1.rollout_phase_contract import configure_generation, scalar_metrics
    from reproduction.common8.behavior_runtime import prepare_identity, attach_generation_identity
    from reproduction.common8.calibration_evaluation import rows_from_batch
    from reproduction.common8.validation_identity import bind_validation_dataframe
    record, frame, ids, sources = canonical_product(base, split)
    dataset = RLHFDataset(str(base/'data'/record['path']), tokenizer,
        prompt_key=config.data.prompt_key, max_prompt_length=config.data.max_prompt_length,
        filter_prompts=False, cache_dir=str(base/'cache/evaluation-dataset'), truncation='error')
    dataset.dataframe = bind_validation_dataframe(frame, base, split)
    gen = configure_generation(GenerationConfig(max_turns=config.max_turns,
        max_start_length=config.data.max_start_length, max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length, max_obs_length=config.data.max_obs_length,
        num_gpus=4, no_think_rl=config.algorithm.no_think_rl,
        search_url=config.retriever.url, topk=config.retriever.topk), config.actor_rollout_ref.rollout)
    manager = LLMGenerationManager(tokenizer, group, gen, is_validation=True)
    rows, generation_seconds = [], 0.
    for start in range(0, COUNTS[split], CHUNK_SIZE):
        end = min(start+CHUNK_SIZE, COUNTS[split])
        target = base/'evaluations'/split/('chunk-%06d.json' % start)
        if target.exists():
            value = validate_chunk(read(target, limit=256*1024), binding, split, start, ids, sources)
        else:
            batch_dict = next(iter(DataLoader(Subset(dataset, range(start, end)),
                batch_size=end-start, shuffle=False, drop_last=False, collate_fn=collate_fn)))
            batch = DataProto.from_single_dict(batch_dict)
            identity = prepare_identity(batch, 200, repeats=1, validation=True, candidate_offset=start)
            request = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            request.meta_info = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                     recompute_log_prob=False, do_sample=False, validate=True)
            attach_generation_identity(request, identity, do_sample=False)
            started = time.monotonic(); manager.timing_raw = {}
            generated = manager.run_llm_loop(request, request.batch['input_ids'][:, -gen.max_start_length:].clone())
            validate_generated(generated, identity)
            batch = batch.union(generated)
            measured = rows_from_batch(batch, tokenizer, allowed_sources=record['source_counts'])
            measured_rows(measured, ids, sources, start=start)
            elapsed = time.monotonic()-started
            value = dict(role=ROLE, split=split, start=start, end=end, rows=measured,
                rows_sha256=hashlib.sha256(canonical(measured)).hexdigest(),
                source_context_sha256=binding['source_context_sha256'],
                source_receipt_sha256=binding['source']['source_receipt_sha256'],
                actual_gpu_inference_verified=True, producer_slurm_job_id=os.environ['SLURM_JOB_ID'],
                generation_identity={key:identity[key] for key in ('run_id', 'slurm_job_id',
                    'policy_snapshot_id', 'parent_group_ids', 'trajectory_ids')},
                gpu_readback=group.sampling_readback(), generation_seconds=elapsed,
                actual_infrastructure_metrics=scalar_metrics(generated), checked_unix=time.time())
            write_once(base, target, value)
            validate_chunk(read(target, limit=256*1024), binding, split, start, ids, sources)
            del batch, generated, request
        rows.extend(value['rows']); generation_seconds += value['generation_seconds']
        print(json.dumps(dict(stage='CP200_EVALUATION_CHUNK_ACCEPTED', split=split,
            accepted_questions=len(rows), total_questions=COUNTS[split], checked_unix=time.time())), flush=True)
    need(len(rows) == COUNTS[split], 'COMPLETE_CANONICAL_EVALUATION_REQUIRED')
    measured_rows(rows, ids, sources)
    source_scores = {source: dict(questions=sources.count(source),
        em=sum(row['em'] for row in rows if row['data_source']==source)/sources.count(source),
        f1=math.fsum(row['f1'] for row in rows if row['data_source']==source)/sources.count(source))
        for source in record['source_counts']}
    result = dict(role=ROLE, split=split, checkpoint_outer=200, questions=len(rows),
        canonical_parquet_sha256=record['sha256'], source_context_sha256=binding['source_context_sha256'],
        source_receipt_sha256=binding['source']['source_receipt_sha256'], rows=rows,
        weighted_em=sum(row['em'] for row in rows)/len(rows),
        weighted_f1=math.fsum(row['f1'] for row in rows)/len(rows), source_scores=source_scores,
        actual_generation_seconds=generation_seconds, student_optimizer_updates=0, probe_optimizer_updates=0,
        original_training_row200_reconstructed=False, inference_only=True)
    write_once(base, base/'evaluations'/split/'complete.json', result)
    return {key: value for key, value in result.items() if key != 'rows'}


def ray_evaluate(base):
    import ray
    from reproduction.common8.configuration import resolved
    from reproduction.common8.native_ray import verify as verify_ray
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from reproduction.a800.owned_processes import identity
    binding = read(base/'preparation/final-eval-binding.json')
    run, config, _ = resolved(base)
    admission = read(base/'logs/native-allocation-admission.json')
    claim = read(base/'preparation/final-eval-start-claim.json')
    validate_worker_execution(claim, admission, os.environ.get('SLURM_JOB_ID'),
                              identity(claim['controller']['pid']))
    need(run['evaluation_only'] is True and config.trainer.n_gpus_per_node == 4,
         'INFERENCE_ONLY_FOUR_GPU_CONFIG_REQUIRED')
    need(config.actor_rollout_ref.model.actor_checkpoint_path == binding['source']['source_checkpoint']
         and config.max_turns == 4 and config.retriever.topk == 3, 'INFERENCE_CONFIG_CHANGED')
    tokenizer = hf_tokenizer(binding['source']['source_checkpoint'], trust_remote_code=False)
    need(not ray.is_initialized(), 'PRIVATE_FRESH_RAY_REQUIRED')
    ray.init(address='local', num_gpus=4, num_cpus=16,
        _temp_dir=verify_ray(run, run['ray_tmp_dir']),
        runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    try:
        pool = RayResourcePool([4], use_gpu=True, max_colocate_count=1,
                               name_prefix='cp200-eval-'+run['native_ray_execution_id'][:12])
        wrapped = create_colocated_worker_cls({'actor_rollout': RayClassWithInitArgs(
            cls=ray.remote(worker_class()), config=config.actor_rollout_ref, role='rollout')})
        groups = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=wrapped)
        group = groups.spawn(prefix_set={'actor_rollout'})['actor_rollout']
        group.init_model()
        actual = group.inference_readback(final=False)
        need(sorted(row['rank'] for row in actual) == list(range(4))
             and {row['actual_cuda_uuid'] for row in actual} == set(admission['actual_cuda_runtime_uuids'])
             and actual[0]['checkpoint_tensors_checked_rank0'] > 0, 'ACTUAL_FOUR_GPU_CP200_LOAD_NOT_VERIFIED')
        write_once(base, base/'logs/final-eval-gpu-loaded.json', dict(ranks=actual,
            source_receipt_sha256=binding['source']['source_receipt_sha256']))
        # Verify real synchronized vLLM weights even when every requested scalar
        # chunk was accepted by an earlier allocation. This phase submits zero
        # trajectories and cannot fabricate or replace an evaluation answer.
        snapshot_id = run['run_id']+':200:weight-readback-'+uuid.uuid4().hex
        identity = dict(run_id=run['run_id'], source_sha256=run['source_sha256'],
            slurm_job_id=os.environ['SLURM_JOB_ID'], logical_update=200,
            policy_snapshot_id=snapshot_id, parent_group_ids=[], trajectory_ids=[], validation=True)
        handles = group.begin_generation_phase(dict(phase_id=uuid.uuid4().hex,
            policy_snapshot_id=snapshot_id, behavior_identity=identity, expected_trajectory_count=0))
        need(len(handles) == 4 and all(handle == handles[0] for handle in handles), 'WEIGHT_READBACK_PHASE_DIVERGED')
        closed = group.end_generation_phase(handles[0], abort=False)
        need(len(closed) == 4 and all(row['completed_requests'] == 0 and row['weight_syncs'] == 1
             for row in closed), 'WEIGHT_READBACK_PHASE_GENERATED_UNREQUESTED_TRAJECTORIES')
        write_once(base, base/'logs/final-eval-vllm-loaded.json',
            dict(ranks=group.sampling_readback(), zero_request_phase=closed))
        results = {split: evaluate_split(base, binding, split, config, tokenizer, group)
                   for split in binding['missing_splits']}
        final = group.inference_readback(final=True)
        need(all(row['vllm_readback']['accepted'] and row['no_optimizer_constructed'] for row in final),
             'FINAL_INFERENCE_WEIGHTS_OR_NO_OPTIMIZER_GUARD_FAILED')
        write_once(base, base/'logs/final-eval-gpu-frozen.json', dict(ranks=final))
        write_once(base, base/'evaluations/inference-complete.json', dict(role=ROLE, results=results,
            inherited_source_scores=binding['accepted_source_scores'], actual_four_gpu_inference_verified=True,
            student_optimizer_updates=0, probe_optimizer_updates=0, original_training_rows_modified=False))
    finally:
        ray.shutdown()


def run(base):
    base = owned(base)
    binding = read(base/'preparation/final-eval-binding.json')
    run_manifest = read(base/'logs/run-manifest.json')
    need(binding['role'] == ROLE and run_manifest.get('evaluation_only') is True, 'WRONG_EXECUTION_ROLE')
    verify_deployment(base, run_manifest['source_commit'])
    need(os.environ.get('SLURM_JOB_ID', '').isdecimal(), 'ACTUAL_SLURM_FOUR_GPU_ALLOCATION_REQUIRED')
    os.environ.update(SEARCH_G1_BASE=str(base), SEARCH_G1_PROFILE='common8', SEARCH_G1_TRAINING_WORLD_SIZE='4',
        SEARCH_G1_EXECUTION_PROFILE='native4_bf16_varlen_fp32_master',
        SEARCH_G1_SOURCE_COMMIT=run_manifest['source_commit'], QPROBE_SOURCE=str(base/'source'/run_manifest['source_commit']),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TMPDIR=str(base/'tmp'), HF_HOME=str(base/'cache/hf'), XDG_CACHE_HOME=str(base/'cache'))
    with execution_guard(base):
        need(not (base/'preparation/final-eval-start-claim.json').exists(),
             'EXISTING_SUBMISSION_MUST_USE_NEW_RECOVERY_NAMESPACE')
        source_terminal(binding['source']['source_base'])
        actual, _, _ = authenticate(binding['source']['source_receipt'])
        need(actual['source_receipt_sha256'] == binding['source']['source_receipt_sha256'], 'CP200_CHANGED_AFTER_QUEUE')
        ctx, ctx_sha = context(run_manifest, actual)
        need(ctx_sha == binding['source_context_sha256'] and ctx == binding['source_context'], 'EVALUATION_CONTEXT_CHANGED')
        from reproduction.a800.owned_processes import identity
        write_once(base, base/'preparation/final-eval-start-claim.json', dict(role=ROLE,
            job=os.environ['SLURM_JOB_ID'], controller=identity(os.getpid()), checked_unix=time.time()))
        from reproduction.common8.native_allocation import admit
        admit(base)
        from reproduction.common8.run_pipeline import Pipeline
        from reproduction.common8.http_retrieval import admit_http
        pipeline = Pipeline(base, run_manifest)
        success = False
        try:
            p = pipeline.start('retrieval-stage', [pipeline.python, '-m', 'reproduction.common8.stage_retrieval',
                '--base', str(base), '--port', '18016'], [])
            pipeline.wait('retrieval-stage', p)
            libs = list((base/'env/lib/python3.10/site-packages/nvidia').glob('*/lib'))
            libs += [base/'env/lib/python3.10/site-packages/torch/lib']
            p = pipeline.start('retriever', [pipeline.python, '-m', 'reproduction.common8.retriever'],
                pipeline.retriever_gpus, {'PYTHONPATH': str(base/'retriever-gpu-lib')+':'+str(base/'repo'),
                                         'LD_LIBRARY_PATH': ':'.join(map(str, libs))})
            health = pipeline.endpoint('retriever', p, '18010/health')
            need(health['ok'] is True and health['run_id'] == run_manifest['run_id']
                 and health['slurm_job_id'] == os.environ['SLURM_JOB_ID'] and health['shard_count'] == 4,
                 'ACTUAL_CANONICAL_RETRIEVER_NOT_READY')
            admit_http(base)
            p = pipeline.start('cp200-evaluator', [pipeline.python, '-m',
                'reproduction.hpc3_dynrpo_periodic.final_eval', 'worker', '--base', str(base)], range(4))
            pipeline.wait('cp200-evaluator', p)
            proof = read(base/'evaluations/inference-complete.json')
            need(proof['actual_four_gpu_inference_verified'] is True, 'INFERENCE_DID_NOT_FINISH')
            success = True
        finally:
            pipeline.owned.stop_all()
            from reproduction.common8.temporary_cache import cleanup as clean_cache
            from reproduction.common8.stage_retrieval import cleanup as clean_stage
            if (base/'logs/corpus-cache-claim.json').exists():
                clean_cache(base)
            if (base/'logs/retrieval-staging-claim.json').exists():
                clean_stage(base)
            write_once(base, base/'logs/final-eval-cleanup.json', dict(owned_processes_cleaned=True,
                inference_completed=success, job=os.environ['SLURM_JOB_ID'], checked_unix=time.time()))
        write_once(base, base/'evaluations/complete.json', dict(role=ROLE,
            status='AUTHENTICATED_CP200_DEV512_FINAL51713_INFERENCE_AND_OWNED_CLEANUP_PASS',
            source_receipt_sha256=binding['source']['source_receipt_sha256'],
            inference_proof_sha256=digest(base/'evaluations/inference-complete.json'),
            cleanup_sha256=digest(base/'logs/final-eval-cleanup.json'),
            original_training_row200_reconstructed=False, student_optimizer_updates=0, probe_optimizer_updates=0))


def validate_worker_execution(claim, admission, job, controller):
    need(isinstance(job, str) and job.isdecimal() and claim.get('role') == ROLE
         and claim.get('job') == admission.get('slurm_job_id') == job
         and admission.get('accepted') is True and admission.get('training_world_size') == 4,
         'WORKER_REQUIRES_THIS_ACTIVE_FOUR_GPU_CONTROLLER')
    need(controller['state'] != 'Z' and controller['uid'] == os.getuid()
         and all(controller[key] == claim['controller'][key] for key in ('pid', 'start_ticks', 'sid', 'uid')),
         'WORKER_CONTROLLER_PROCESS_IDENTITY_CHANGED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'run', 'worker'))
    parser.add_argument('--source-receipt', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--previous-base', type=Path)
    args = parser.parse_args()
    if args.action == 'check':
        need(args.source_receipt is not None, 'SOURCE_RECEIPT_REQUIRED')
        result = authenticate(args.source_receipt)[0]
    elif args.action == 'prepare':
        need(args.source_receipt is not None and args.base is not None and args.commit is not None,
             'BASE_PUBLISHED_COMMIT_AND_SOURCE_RECEIPT_REQUIRED')
        with execution_guard(owned(args.base)):
            result = prepare(args.base, args.source_receipt, args.commit, args.previous_base)
    elif args.action == 'run':
        need(args.base is not None, 'OWN_BASE_REQUIRED')
        run(args.base); result = dict(status='EVALUATION_RECOVERY_COMPLETE', base=str(args.base))
    else:
        need(args.base is not None, 'OWN_BASE_REQUIRED')
        ray_evaluate(owned(args.base)); result = dict(status='INFERENCE_COMPLETE')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
