"""Once-only final200 native4 restore control; CPU orchestration, never training.

Archive transport is external. This module consumes its completed source/archive
receipts and never manufactures archive or CUDA success. All mutations are
exclusive; a partial copy, rejected submission, or unknown sbatch outcome must
be inspected, never retried here. No credentials are read or written.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time

from reproduction.prefix20_joint4 import checkpoint_archive as archive
from reproduction.prefix20_joint4.restore_only_assets import ROLE, publish, sha

PERSONAL_ROOT = Path('/data/user/jxu957/experiments')
BUDGET = 2 * 10**12
MARGIN = 8 * 2**30
MAX_AGE = 21600
REGISTRATION = 'preparation/final200-restore-target-registration.json'
COPY_PERMIT = 'preparation/restore-copy-permit.json'
PREPARED = 'preparation/restore-controller-prepared.json'
SUBMIT_CLAIM = 'preparation/restore-submit-claim.json'
SUBMISSION = 'preparation/restore-submission.json'
EXECUTE_PERMIT = 'preparation/restore-execution-permit.json'
CONSUMER_RECEIPT = 'logs/actual-recovery-for-cpu-consumer.json'


def need(value, reason):
    if not value:
        raise ValueError('RESTORE_CONTROL_' + reason)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('RESTORE_CONTROL_NONFINITE_JSON')))


def document(path, expected_sha=None):
    path = archive.safe_components(path)
    need(path.is_file() and path.stat().st_size < 512 * 2**20, 'SMALL_REGULAR_DOCUMENT_REQUIRED')
    before = archive.regular(path)
    raw = path.read_bytes()
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    need(expected_sha is None or digest == expected_sha, 'DOCUMENT_SHA_DIFFERS')
    need(archive.regular(path) == before, 'DOCUMENT_CHANGED_DURING_READ')
    return strict(raw), dict(path=str(path), sha256=digest, **before)


def storage_observation_document(source):
    """Read one complete canonical version despite an atomic refresh rename.

    All other document inputs remain immutable. The selected descriptor must
    retain its content identity; the canonical path may move to the next valid
    proof while this descriptor still reads the preceding complete proof.
    """
    path = archive.safe_components(Path(source)/'preparation/personal-storage-observation.json')
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size < 512 * 2**20, 'SMALL_REGULAR_DOCUMENT_REQUIRED')
        raw = stream.read()
        after = os.fstat(stream.fileno())
    identity = lambda value:(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    need(identity(before) == identity(after) and len(raw) == before.st_size, 'DOCUMENT_CHANGED_DURING_READ')
    import hashlib
    record = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), device=before.st_dev,
                  inode=before.st_ino, bytes=before.st_size, mtime_ns=before.st_mtime_ns)
    return strict(raw), record


def namespace(source, target=None):
    source = archive.own_base(source)
    need(source.parent == PERSONAL_ROOT, 'HPC3_PERSONAL_ROOT_REQUIRED')
    if target is not None:
        target = archive.own_base(target)
        need(target.parent == PERSONAL_ROOT and target != source and
             target.name.startswith('qprobe-teacher-warmup200-prefix20-restore200-'), 'DISTINCT_RESTORE_TARGET_REQUIRED')
    return source, target


def query(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    need(result.returncode == 0, 'ACTUAL_QUERY_FAILED')
    return dict(command=command, returncode=result.returncode, stdout=result.stdout,
                stderr=result.stderr, checked_unix=time.time())


def terminal_job(job, expected_name):
    need(isinstance(job, str) and re.fullmatch('[1-9][0-9]*', job), 'ACTUAL_JOB_ID_REQUIRED')
    account = query(['sacct', '-nP', '-j', job, '--format=JobID,State,ExitCode,User,UID,Account,QOS,JobName,Start,End,NodeList'])
    rows = [row.split('|') for row in account['stdout'].splitlines() if row]
    rows = [row for row in rows if row[0] == job]
    need(len(rows) == 1 and len(rows[0]) >= 11, 'EXACT_MAIN_ACCOUNTING_ROW_REQUIRED')
    row = rows[0]
    need(row[1:7] == ['COMPLETED', '0:0', 'jxu957', '14090', 'jxu957', 'formal-user'] and
         row[7] == expected_name and row[8] not in ('', 'Unknown', 'N/A') and row[9] not in ('', 'Unknown', 'N/A') and row[10],
         'OWN_SUCCESSFUL_TERMINAL_REQUIRED')
    queue = query(['squeue', '-h', '-j', job, '-o', '%i'])
    need(not queue['stdout'].strip(), 'JOB_STILL_QUEUED')
    return dict(job=job, accounting=account, queue=queue, actual_terminal=True)


def history_gate(run, updates, cpu, *, source, commit, run_id):
    need(run.get('run_id') == run_id and run.get('source_commit') == commit and
         run.get('training_world_size') == 4 and run.get('target_updates') == 200 and
         run.get('data_selection_mode') == 'prefix20_probe' and not run.get('full_state_restore'), 'SOURCE_RUN_DIFFERS')
    need([row.get('step') for row in updates] == list(range(1, 201)) and
         all(type(row.get('step')) is int and row.get('run_id') == run_id for row in updates), 'ACTUAL_COMPLETE200_OR_NO201_MISSING')
    need(cpu.get('kind') == 'prefix20_joint4_independent_cpu_recorded_evidence_audit' and
         cpu.get('accepted') is True and cpu.get('status') == 'RECORDED_EVIDENCE_PASS' and
         cpu.get('source_base') == str(source) and cpu.get('source_commit') == commit and cpu.get('run_id') == run_id and
         cpu.get('torch_imported') is False and cpu.get('model_reloads') == 0 and cpu.get('gpu_allocations') == 0 and
         cpu.get('training_updates') == 0, 'INDEPENDENT_CPU_FINAL_ACCEPTANCE_REQUIRED')
    h = cpu['actual_history']
    need(h.get('completed_outers') == 200 and h.get('no_update201') is True and
         integer(h.get('actual_actor_adam_writes_per_rank'), 200), 'CPU_ACTUAL_HISTORY_DIFFERS')
    evaluations = cpu['evaluations']
    need([row['step'] for row in evaluations['fixed_boundaries']] == list(archive.STEPS), 'CPU_FIXED_BOUNDARIES_MISSING')
    final = evaluations['final_native51713']
    need(final.get('rows') == 51713 and len(final.get('per_source', {})) == 7, 'ACTUAL_NATIVE51713_REQUIRED')
    need([row.get('outer') for row in cpu['saved_checkpoints']] == list(archive.STEPS) and
         all(row.get('full_recorded_checkpoint_sha_checked') is True for row in cpu['saved_checkpoints']), 'CPU_ALL_CHECKPOINTS_REQUIRED')
    return h


def archive_gate(source, commit, run_id):
    """Read external archive acceptance; no transport or synthetic re-labeling."""
    result, final = [], None
    for step in archive.STEPS:
        root = source/'preparation/checkpoint-archives'/('global_step_' + str(step))
        complete, record = document(root/'complete.json')
        need(complete.get('status') == archive.STATUS and complete.get('run_id') == run_id and
             complete.get('training_source_commit') == commit and complete.get('source_base') == str(source) and
             complete.get('step') == step and complete.get('world_size') == 4 and
             complete.get('original_sources_retained') is True and complete.get('source_retention_recheck_pending') is False and
             complete.get('all_archive_files_fresh_sha_pass') is True and complete.get('all_archive_files_fsync_completed') is True and
             complete.get('actual_cuda_restore_accepted') is False, 'ACTUAL_SOURCE_ARCHIVE_FULL_SHA_PASS_REQUIRED')
        snapshot = archive.load_snapshot(complete['source_snapshot']['path'], complete['source_snapshot']['sha256'],
            base=source, commit=commit, run_id=run_id, step=step)
        # accept_archive updates checked_unix after the immutable archive receipt.
        # The original signed-by-hash document is copied as a dependency by the
        # transport controller; require it explicitly rather than reconstruct its timestamp.
        imported = root/'archive-receipt-import.json'
        wrapper, imported_record = document(imported)
        checked = archive.validate_archive_proof(wrapper, snapshot, complete['source_snapshot']['sha256'])
        for key in checked:
            if key not in ('checked_unix', 'source_retention_recheck_pending'):
                need(complete[key] == checked[key], 'ARCHIVE_ACCEPTANCE_ORIGINAL_PROOF_DIFFERS')
        need(complete['archive_receipt'] == dict(path=wrapper['archive_receipt_file'], sha256=wrapper['archive_receipt_sha256']),
             'ARCHIVE_IMPORTED_PROOF_BINDING_DIFFERS')
        readback = complete['source_readback']
        need(readback.get('all_source_files_fresh_sha_pass') is True and readback.get('run_id') == run_id and
             readback.get('step') == step and readback.get('snapshot_sha256') == complete['source_snapshot']['sha256'] and
             readback.get('files') == len(snapshot['files']) and readback.get('bytes') == snapshot['bytes'], 'SOURCE_RETENTION_READBACK_MISSING')
        result.append(dict(step=step, complete=record, imported_archive_proof=imported_record,
                           snapshot=complete['source_snapshot'], files=len(snapshot['files']), bytes=snapshot['bytes']))
        if step == 200:
            final = snapshot
    return result, final


def source_gate(source, commit, run_id, cpu_path, cpu_sha, *, check_terminal=True):
    source, _ = namespace(source)
    archive.binding(source, commit, run_id, 200)
    run, run_record = document(source/'logs/run-manifest.json')
    raw_path = source/'logs/actual-updates.jsonl'
    archive.safe_components(raw_path)
    raw = raw_path.read_bytes()
    need(raw.endswith(b'\n'), 'TRUNCATED_ACTUAL_HISTORY')
    updates = [strict(row) for row in raw.splitlines()]
    cpu, cpu_record = document(cpu_path, cpu_sha)
    history = history_gate(run, updates, cpu, source=source, commit=commit, run_id=run_id)
    proofs, snapshot = archive_gate(source, commit, run_id)
    need(snapshot['pair_receipt']['attempt_progress']['cumulative_writes'] == history['actual_actor_adam_writes_per_rank'] and
         snapshot['pair_receipt']['driver_state']['source_consumed'] == history['candidate_questions_consumed'],
         'CPU_ARCHIVE_ACTOR_COUNTER_DIFFERS')
    need(not (source/'checkpoints/search-g1-common8/actor/global_step_201').exists(), 'UPDATE201_CHECKPOINT_PRESENT')
    terminal = None
    if check_terminal:
        job = (source/'job.id').read_text().strip()
        submitted, _ = document(source/'preparation/training-submission.json')
        need(submitted.get('job') == job and submitted.get('source_commit') == commit and submitted.get('returncode') == 0,
             'SOURCE_SUBMISSION_BINDING_DIFFERS')
        terminal = terminal_job(job, 'qprobe-prefix20-joint200')
        cleanup, _ = document(source/'logs/cleanup-status.json')
        need(cleanup.get('run_id') == run_id and cleanup.get('cleaned') is True, 'SOURCE_OWNED_CLEANUP_REQUIRED')
    return dict(source_base=str(source), source_commit=commit, source_run_id=run_id, completed_outer=200,
                run_manifest=run_record, cpu_final_audit=cpu_record, archives=proofs, terminal=terminal,
                actual_actor_writes=history['actual_actor_adam_writes_per_rank'],
                source_attempt_progress=snapshot['pair_receipt']['attempt_progress'],
                source_hf_saved_tensor_count=snapshot['pair_receipt']['hf_saved_tensor_count'],
                candidate_questions_consumed=history['candidate_questions_consumed'],
                source_pair_receipt=snapshot['source_receipt'], source_pair_receipt_sha256=snapshot['source_receipt_sha256']), snapshot


def budget_projection(observation, snapshot_bytes, source_bytes, *, now=None):
    now = time.time() if now is None else now
    need(observation.get('measurement_complete') is True and observation.get('account_experiment_root') == str(PERSONAL_ROOT) and
         type(observation.get('checked_unix')) in (int, float) and 0 <= now-observation['checked_unix'] <= MAX_AGE,
         'FRESH_COMPLETE_PERSONAL_STORAGE_REQUIRED')
    measured = observation.get('allocated_bytes_upper_bound')
    need(integer(measured) and integer(snapshot_bytes, 1) and integer(source_bytes, 1), 'ACTUAL_STORAGE_VALUES_REQUIRED')
    # Snapshot deduplicates official/reward dependencies. The target may copy
    # these separately: reserve twice the complete graph, two source copies,
    # filesystem/metadata overhead, and a disclosed concurrent-growth margin.
    allowance = 2*snapshot_bytes + 2*source_bytes + 512*2**20
    projected = measured + allowance + MARGIN
    need(projected <= BUDGET, 'EXPLICIT_2TB_BUDGET_EXCEEDED')
    return dict(personal_budget_bytes=BUDGET, actual_measured_allocated_bytes=measured,
        target_allowance_bytes=allowance, concurrent_growth_margin_bytes=MARGIN,
        projected_bytes=projected, measurement_checked_unix=observation['checked_unix'],
        unique_hpc3_restore_target_copies=1, hard_quota_verified=False)


def budget_gate(source, snapshot, manifest):
    from reproduction.prefix20_joint4.storage_complete import validate_upper_bound
    observation, record = storage_observation_document(source)
    need(observation.get('measurement_kind') == 'permission_partition_allocated_byte_upper_bound', 'ACTUAL_PERMISSION_PARTITION_MEASUREMENT_REQUIRED')
    validate_upper_bound(observation, source)
    # Validate its raw complete partition and producer identities; never relabel
    # old timestamps or reuse the launch-time future checkpoint reserve.
    from reproduction.prefix20_joint4.assets import sha as file_sha
    from reproduction.prefix20_joint4 import storage_observe
    need(observation.get('observer_source_sha256') == file_sha(Path(storage_observe.__file__)) and
         (observation.get('observer_uid'), observation.get('observing_user')) in ((14090, 'jxu957'), (21930, 'mou176')),
         'ACTUAL_STORAGE_OBSERVER_DIFFERS')
    projection = budget_projection(observation, snapshot['bytes'], sum(row['bytes'] for row in manifest['files'].values()))
    return dict(projection, actual_observation=record)


@contextmanager
def registry_lock():
    need(os.getuid() == 14090, 'JXU_OWNER_REQUIRED')
    archive.safe_components(PERSONAL_ROOT)
    path = PERSONAL_ROOT/'.prefix20-final200-restore-controller.lock'
    need(not path.is_symlink(), 'REGISTRY_LOCK_SYMLINK')
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def registered(base):
    permit, _ = document(base/COPY_PERMIT)
    source, target = namespace(permit['source_base'], base)
    registration, record = document(source/REGISTRATION, permit['registration_sha256'])
    need(registration.get('target_base') == str(target) and registration.get('restore_source_commit') == permit['restore_source_commit'] and
         registration.get('source_gate') == permit['source_gate'] and registration.get('budget') == permit['budget'] and
         permit.get('role') == ROLE, 'UNIQUE_TARGET_REGISTRATION_DIFFERS')
    matches = sorted(PERSONAL_ROOT.glob('qprobe-teacher-warmup200-prefix20-restore200-*'))
    need(matches == [target] and target.resolve() == target and target.is_dir(), 'MULTIPLE_OR_REPLACED_HPC3_RESTORE_TARGETS')
    return permit, registration, record


def validate_copy_permit(base, source_receipt, commit):
    permit, _, _ = registered(Path(base))
    need(permit['restore_source_commit'] == commit and permit['source_gate']['source_pair_receipt'] == str(source_receipt) and
         sha(source_receipt) == permit['source_gate']['source_pair_receipt_sha256'], 'COPY_PERMIT_SOURCE_OR_COMMIT_DIFFERS')
    need(permit['budget']['personal_budget_bytes'] == BUDGET and permit['budget']['projected_bytes'] <= BUDGET,
         'COPY_PERMIT_BUDGET_DIFFERS')
    return permit


def prepare_target(source, commit, run_id, target, restore_commit, cpu_path, cpu_sha):
    from reproduction.prefix20_joint4.control import verify_source, bootstrap
    from reproduction.prefix20_joint4.restore_only_assets import prepare as copy_target
    source, target = namespace(source, target)
    with registry_lock():
        need(not target.exists() and not (source/REGISTRATION).exists(), 'RESTORE_TARGET_ALREADY_ATTEMPTED_INSPECT')
        need(not list(PERSONAL_ROOT.glob('qprobe-teacher-warmup200-prefix20-restore200-*')), 'ANOTHER_HPC3_RESTORE_TARGET_EXISTS')
        gate, snapshot = source_gate(source, commit, run_id, cpu_path, cpu_sha)
        runtime, manifest = verify_source(source, restore_commit)
        budget = budget_gate(source, snapshot, manifest)
        registration = dict(role=ROLE, target_base=str(target), restore_source_commit=restore_commit,
                            source_gate=gate, budget=budget, created_unix=time.time())
        publish(source/REGISTRATION, registration)  # First irreversible claim; never renewed.
        target.mkdir()
        (target/'preparation').mkdir()
        shutil.copytree(runtime, target/'source'/restore_commit)
        bootstrap(target, restore_commit)
        permit = dict(registration, source_base=str(source), registration_sha256=sha(source/REGISTRATION))
        publish(target/COPY_PERMIT, permit)
        copy_path = copy_target(target, gate['source_pair_receipt'], restore_commit)
        copied, copy_record = document(copy_path)
        need(copied['copied_bytes'] <= budget['target_allowance_bytes'], 'ACTUAL_COPY_EXCEEDS_RESERVED_ALLOWANCE')
        measured = query(['du', '-s', '-B1', str(target)])
        row = measured['stdout'].strip().split('\t')
        need(len(row) == 2 and row[1] == str(target) and row[0].isdecimal() and
             int(row[0]) <= budget['target_allowance_bytes'], 'ACTUAL_TARGET_ALLOCATION_EXCEEDS_ALLOWANCE')
        value = dict(role=ROLE, target_base=str(target), restore_source_commit=restore_commit,
                     copy_receipt=copy_record, registered_target=sha(source/REGISTRATION),
                     source_gate=gate, budget=budget, actual_target_du=measured,
                     training_submitted=False, actual_cuda_restore_accepted=False, checked_unix=time.time())
        publish(target/PREPARED, value)
        return value


def command_for(base, commit):
    return ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user', '--account=jxu957',
            '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--cpus-per-task=48', '--mem=512G', '--time=04:00:00',
            '--job-name=prefix20-restore200-only', '--export=NONE', '--chdir='+str(base/'repo'),
            '--output='+str(base/'logs/restore-%j.log'), '--error='+str(base/'logs/restore-%j.log'),
            str(base/'source'/commit/'reproduction/prefix20_joint4/restore_only.sh'), str(base)]


def submit(base):
    from reproduction.prefix20_joint4.restore_only_assets import verify_target_copy
    base = Path(base)
    with registry_lock():
        need(not any((base/name).exists() for name in (SUBMIT_CLAIM, SUBMISSION, 'restore-job.id', EXECUTE_PERMIT,
             'preparation/restore-execution-claim.json')), 'SUBMISSION_ALREADY_ATTEMPTED_INSPECT_NO_RETRY')
        permit, _, _ = registered(base)
        prepared, prepared_record = document(base/PREPARED)
        gate = permit['source_gate']
        fresh, snapshot = source_gate(Path(gate['source_base']), gate['source_commit'], gate['source_run_id'],
                                     gate['cpu_final_audit']['path'], gate['cpu_final_audit']['sha256'])
        from reproduction.prefix20_joint4.control import verify_source, bind_repo
        _, manifest = verify_source(base, permit['restore_source_commit'])
        bind_repo(base, permit['restore_source_commit'])
        budget = budget_gate(Path(gate['source_base']), snapshot, manifest)
        verify_target_copy(base, base/'preparation/restore-copy-complete.json')
        need(prepared['copy_receipt']['sha256'] == sha(base/'preparation/restore-copy-complete.json'), 'PREPARED_COPY_RECEIPT_CHANGED')
        cmd = command_for(base, permit['restore_source_commit'])
        claim = dict(role=ROLE, command=cmd, prepared_receipt=prepared_record, source_gate=fresh,
                     budget=budget, created_unix=time.time(), retry_allowed=False)
        publish(base/SUBMIT_CLAIM, claim)
        publish(base/EXECUTE_PERMIT, dict(role=ROLE, submit_claim_sha256=sha(base/SUBMIT_CLAIM),
            copy_receipt_sha256=prepared['copy_receipt']['sha256'], registration_sha256=permit['registration_sha256'],
            restore_source_commit=permit['restore_source_commit'], completed_source_outer=200,
            world_size=4, restore_only=True, training_permitted=False, issued_unix=time.time()))
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        except BaseException as error:
            publish(base/SUBMISSION, dict(role=ROLE, outcome='UNKNOWN_INSPECT_SLURM_NO_RETRY',
                error=type(error).__name__, submit_claim_sha256=sha(base/SUBMIT_CLAIM), checked_unix=time.time()))
            raise
        output = result.stdout.strip()
        valid = result.returncode == 0 and re.fullmatch('[1-9][0-9]*(;[A-Za-z0-9._-]+)?', output)
        value = dict(role=ROLE, outcome='SUBMITTED' if valid else 'REJECTED_OR_UNKNOWN_INSPECT_NO_RETRY',
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, command=cmd,
            submit_claim_sha256=sha(base/SUBMIT_CLAIM), checked_unix=time.time())
        if valid:
            value['job'] = output.split(';')[0]
        publish(base/SUBMISSION, value)
        need(valid, 'SUBMISSION_NOT_UNAMBIGUOUS_NO_RETRY')
        publish(base/'restore-job.id', dict(job=value['job']))
        return value


def validate_execute_permit(base, commit):
    """Called by actual job before native admission or any model construction."""
    base = Path(base)
    permit, _, _ = registered(base)
    execute, _ = document(base/EXECUTE_PERMIT)
    need(execute.get('role') == ROLE and execute.get('restore_source_commit') == commit and
         execute.get('completed_source_outer') == 200 and execute.get('world_size') == 4 and
         execute.get('restore_only') is True and execute.get('training_permitted') is False and
         execute.get('registration_sha256') == permit['registration_sha256'] and
         execute.get('submit_claim_sha256') == sha(base/SUBMIT_CLAIM) and
         execute.get('copy_receipt_sha256') == sha(base/'preparation/restore-copy-complete.json'), 'EXECUTION_PERMIT_DIFFERS')
    deadline = time.monotonic()+30
    while not (base/SUBMISSION).exists() and time.monotonic() < deadline:
        time.sleep(.2)  # sbatch can start the allocation before returning its ID.
    submitted, _ = document(base/SUBMISSION)
    need(submitted.get('outcome') == 'SUBMITTED' and submitted.get('returncode') == 0 and
         submitted.get('job') == os.environ.get('SLURM_JOB_ID') and
         submitted.get('command') == command_for(base, commit) and
         submitted.get('submit_claim_sha256') == sha(base/SUBMIT_CLAIM), 'ACTUAL_SUBMITTED_RESTORE_JOB_DIFFERS')
    need(os.getuid() == 14090, 'ACTUAL_EXECUTION_OWNER_DIFFERS')
    gate = permit['source_gate']
    fresh, snapshot = source_gate(Path(gate['source_base']), gate['source_commit'], gate['source_run_id'],
                                 gate['cpu_final_audit']['path'], gate['cpu_final_audit']['sha256'])
    from reproduction.prefix20_joint4.control import verify_source
    _, manifest = verify_source(base, commit)
    budget_gate(Path(gate['source_base']), snapshot, manifest)
    return execute


def observe(base):
    base = Path(base)
    registered(base)
    submitted, _ = document(base/SUBMISSION)
    need(submitted.get('outcome') == 'SUBMITTED', 'UNKNOWN_SUBMISSION_MUST_BE_INSPECTED')
    job = submitted['job']
    docs = {}
    for relative in ('preparation/restore-execution-claim.json', 'logs/native-allocation-admission.json',
                     'logs/restore-only-readback-A.json', 'logs/restore-only-readback-B.json',
                     'logs/restore-only-owned-release.json', 'logs/restore-only-complete.json'):
        if (base/relative).exists():
            value, record = document(base/relative)
            docs[relative] = dict(document=value, file=record)
    return dict(role=ROLE, job=job, queue=query(['squeue', '-h', '-j', job, '-o', '%i|%T|%R']),
                accounting=query(['sacct', '-nP', '-j', job, '--format=JobID,State,ExitCode,Start,End,NodeList']),
                actual_readback_phases=[phase for phase in ('A', 'B') if 'logs/restore-only-readback-'+phase+'.json' in docs],
                evidence=docs, accepted=False, semantics='Read-only measured progress; RUNNING or receipt presence alone is not recovery acceptance',
                checked_unix=time.time())


def adapt_actual_readbacks(a, b, complete, released, allocation, gate, *, target, restore_commit, job, now=None):
    """Pure CPU receipt consumer: no model import/reload or fabricated GPU test."""
    from reproduction.prefix20_joint4.restore_only import invariant_actor, validate_actor_readbacks
    need(complete.get('role') == ROLE and complete.get('accepted') is True and complete.get('completed_outer') == 200 and
         complete.get('actual_native4_cuda_restore') is True and complete.get('actual_two_independent_readbacks') is True and
         complete.get('no_update201') is True and type(complete.get('all_update_consumption_increments')) is int and
         complete['all_update_consumption_increments'] == 0 and complete.get('actual_slurm_job_id') == job, 'ACTUAL_RESTORE_COMPLETE_REQUIRED')
    need(allocation.get('accepted') is True and allocation.get('training_world_size') == 4 and
         allocation.get('slurm_job_id') == job and allocation.get('run_id') == gate['source_run_id'] and len(set(allocation['actual_cuda_runtime_uuids'])) == 4, 'ACTUAL_NATIVE4_ALLOCATION_REQUIRED')
    need(released.get('role') == ROLE and released.get('all_recorded_processes_terminal') is True and
         released.get('assigned_gpu_contexts_released') is True, 'ACTUAL_OWNED_GPU_RELEASE_REQUIRED')
    increments = ('optimizer_write_increment', 'scheduler_advance_increment', 'probe_update_increment',
                  'candidate_consumption_increment', 'generation_calls', 'training_checkpoint_save_calls')
    actor_invariants = []
    for phase, row in (('A', a), ('B', b)):
        need(row.get('role') == ROLE and row.get('phase') == phase and row.get('source_step') == 200 and
             row.get('source_run_id') == gate['source_run_id'] and
             all(type(row.get(key)) is int and row[key] == 0 for key in increments), 'ACTUAL_READBACK_PHASE_OR_INCREMENT_DIFFERS')
        need(all(row.get(key) is True for key in ('candidate_dataequal', 'selector_rng_equal', 'driver_rng_equal', 'eligible_mapping_equal')),
             'ACTUAL_DRIVER_SELECTOR_ELIGIBLE_DATAEQUAL_MISSING')
        membership = row['partial_pending_membership']
        need(membership.get('consumed_source_rows') == gate['candidate_questions_consumed'] and
             re.fullmatch('[0-9a-f]{64}', membership.get('ordered_membership_sha256', '')) and
             membership == row['verified_saved_candidate_membership'], 'ACTUAL_PENDING_MEMBERSHIP_NOT_SOURCE_EQUAL')
        need(row['source_attempt_progress'] == gate['source_attempt_progress'] and
             row['source_attempt_progress']['cumulative_writes'] == gate['actual_actor_writes'], 'ACTUAL_SOURCE_COUNTER_DIFFERS')
        actors = row['actual_actor_readbacks']
        validate_actor_readbacks(actors, allocation, gate['actual_actor_writes'])
        need(all(actor.get('phase') == phase and actor['process_identity']['uid'] == 14090 for actor in actors), 'ACTUAL_ACTOR_PHASE_OR_OWNER_DIFFERS')
        need(actors[0]['model_hf_exact_tensors_checked_on_rank0'] == gate['source_hf_saved_tensor_count'] > 0, 'ACTUAL_MODEL_TENSOR_COMPARISON_MISSING')
        actor_invariants.append([invariant_actor(value) for value in actors])
        probes = row['actual_probe_readbacks']
        need([probe.get('rank') for probe in probes] == list(range(4)) and all(
            probe.get('completed_outer') == 200 and probe.get('actual_steps') == 2000 and
            probe.get('model_optimizer_rng_exact_equal') is True and probe.get('private_rng_activated_and_equal') is True and
            probe.get('actor_rng_unchanged') is True and probe.get('forbidden_call_counts') == {} for probe in probes),
            'ACTUAL_FOUR_PROBE_READBACKS_REQUIRED')
    stable_birth = lambda value:{key:value[key] for key in ('pid', 'start_ticks', 'uid', 'sid')}
    need(actor_invariants[0] == actor_invariants[1] and a['actual_probe_readbacks'] == b['actual_probe_readbacks'] and
         a['partial_pending_membership'] == b['partial_pending_membership'] and a['source_receipt_sha256'] == b['source_receipt_sha256'] and
         stable_birth(a['driver_process_identity']) == stable_birth(b['driver_process_identity']) and b.get('independent_second_readback_equal') is True and
         a.get('independent_second_readback_equal') is False and b['checked_unix'] >= a['checked_unix'], 'ACTUAL_SECOND_READBACK_DIFFERENT_OR_MISSING')
    births = [row['process_identity'] for row in b['actual_actor_readbacks']] + [b['driver_process_identity']]
    need(all(p.get('uid') == 14090 and all(integer(p.get(k), 1) for k in ('pid', 'start_ticks', 'sid')) for p in births), 'ACTUAL_PROCESS_BIRTH_INVALID')
    need(released.get('recorded_process_births') == births and len({(p['pid'], p['start_ticks']) for p in births}) == 5,
         'ACTUAL_RELEASE_BIRTHS_DIFFER')
    return dict(schema_version=1, kind='prefix20_final200_native4_actual_restore', source_base=gate['source_base'],
        source_run_id=gate['source_run_id'], source_commit=gate['source_commit'], target_base=str(target),
        restore_source_commit=restore_commit, completed_outer=200, world_size=4, accepted=True, restore_only=True,
        optimizer_step_increments=0, scheduler_step_increments=0, probe_adam_step_increments=0,
        generation_calls=0, training_updates=0, dataequal=True, selector_rng_equal=True, driver_rng_equal=True,
        eligible_mapping_equal=True, candidate_pending_cursor_equal=True,
        actual_rank_readbacks=[dict(rank=rank, actor_model_optimizer_scheduler_rng_equal=True,
            probe_model_optimizer_private_rng_equal=True, probe_actual_adam_steps=2000, readback_count=2) for rank in range(4)],
        actual_slurm_job_id=job, no_update201=True, checked_unix=time.time() if now is None else now,
        semantics='Consumes two actual native4 state readbacks and owned release; this CPU adapter performs no model/GPU operation')


def accept(base):
    from reproduction.prefix20_joint4.restore_only_assets import verify_target_copy
    base = Path(base)
    permit, _, _ = registered(base)
    submitted, _ = document(base/SUBMISSION)
    need(submitted.get('outcome') == 'SUBMITTED', 'SUCCESSFUL_SUBMISSION_BINDING_REQUIRED')
    job = submitted['job']
    terminal = terminal_job(job, 'prefix20-restore200-only')
    need('prefix20-restore200-only' in terminal['accounting']['stdout'], 'RESTORE_JOB_NAME_DIFFERS')
    docs, records = {}, {}
    for key, name in (('a', 'readback-A'), ('b', 'readback-B'), ('complete', 'complete'), ('released', 'owned-release')):
        docs[key], records[key] = document(base/'logs'/('restore-only-'+name+'.json'))
    allocation, records['allocation'] = document(base/'logs/native-allocation-admission.json')
    copied = verify_target_copy(base, base/'preparation/restore-copy-complete.json')
    target_pair = copied['target_pair_receipt']
    need(docs['a']['source_receipt_sha256'] == sha(target_pair), 'ACTUAL_RESTORE_LOADED_OTHER_TARGET_RECEIPT')
    need(copied['original_receipt_sha256'] == permit['source_gate']['source_pair_receipt_sha256'], 'COPIED_ORIGINAL200_RECEIPT_DIFFERS')
    result = adapt_actual_readbacks(**docs, allocation=allocation, gate=permit['source_gate'], target=base,
        restore_commit=permit['restore_source_commit'], job=job)
    need(not any((base/'logs'/name).exists() for name in ('training-launch-claim.json', 'forbidden-training.log')) and
         not list((base/'logs').glob('optimizer-rank-*.jsonl')) and not list((base/'logs').glob('scheduler-rank-*.jsonl')) and
         not list((base/'checkpoints').rglob('global_step_201')), 'RESTORE_CREATED_TRAINING_OR201_ARTIFACT')
    result.update(evidence=records, actual_terminal=terminal,
                  copy_receipt_sha256=sha(base/'preparation/restore-copy-complete.json'),
                  unique_target_registration_sha256=permit['registration_sha256'],
                  original_source_archive_acceptance=permit['source_gate']['archives'])
    path = base/CONSUMER_RECEIPT
    publish(path, result)
    return dict(receipt=str(path), sha256=sha(path), accepted=True, actual_native4_restore_consumed=True,
                cpu_consumer_arguments=['--recovery-receipt', str(path), '--recovery-sha256', sha(path)], goal_complete=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'submit', 'observe', 'accept', 'check-execute'))
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--source-base', type=Path)
    parser.add_argument('--source-commit')
    parser.add_argument('--run-id')
    parser.add_argument('--restore-commit')
    parser.add_argument('--cpu-final-audit', type=Path)
    parser.add_argument('--cpu-final-audit-sha')
    args = parser.parse_args()
    if args.mode == 'prepare':
        required = ('source_base', 'source_commit', 'run_id', 'restore_commit', 'cpu_final_audit', 'cpu_final_audit_sha')
        if any(getattr(args, key) is None for key in required):
            parser.error('prepare requires explicit source identity, published restore commit and CPU final audit/SHA')
        value = prepare_target(args.source_base, args.source_commit, args.run_id, args.base, args.restore_commit,
                               args.cpu_final_audit, args.cpu_final_audit_sha)
    elif args.mode == 'check-execute':
        run, _ = document(args.base/'logs/run-manifest.json')
        validate_execute_permit(args.base, run['source_commit'])
        value = dict(role=ROLE, controller_permit_validated=True, actual_cuda_restore_accepted=False)
    else:
        value = {'submit':submit, 'observe':observe, 'accept':accept}[args.mode](args.base)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
