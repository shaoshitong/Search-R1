"""Validate a small explicit restart request before loading saved learner state.

Tensor/SHA/state readback stays in the existing resume entry, inside the actual
four-GPU 512-GiB Slurm allocation. A present invalid request never falls back to
fresh student/probe initialization.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys

from .common import DEFAULT_BASE, sha

PAIR_STATUS = 'QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS'
SOURCE_LOCKS = ('preparation/formal.lock', 'preparation/preparation.lock', 'gpu-job.lock')


def need(condition, message):
    if not condition:
        raise ValueError('RESTORE_REQUEST_' + message)


def read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid(),
         'MISSING_OR_FOREIGN_METADATA_' + str(path))
    return json.loads(path.read_text())


def namespace(path, *, target=False):
    path = Path(path)
    suffix = r'-r[1-9][0-9]*' if target else r'(?:-r[1-9][0-9]*)?'
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and
         path.is_dir() and path.stat().st_uid == os.getuid() and
         re.fullmatch(re.escape(str(DEFAULT_BASE)) + suffix, str(path)) is not None,
         'UNSAFE_NAMESPACE')
    return path


def validate_unsaved_updates(source, request, receipt):
    """A complete older pair is not authority to discard later real Adam writes."""
    outer = receipt['step']
    checkpoint_writes = receipt.get('attempt_progress', {}).get('cumulative_writes')
    need(type(checkpoint_writes) is int and checkpoint_writes >= 0,
         'CHECKPOINT_UPDATE_COUNTER_MISSING')
    latest_outer = outer
    updates = source/'logs/actual-updates.jsonl'
    need(updates.is_file() and not updates.is_symlink() and updates.stat().st_uid == os.getuid(),
         'SOURCE_ACTUAL_UPDATES_MISSING')
    for line in updates.read_text().splitlines():
        row = json.loads(line)
        if row.get('update_status') == 'complete' and row.get('full_budget_complete') is True:
            latest_outer = max(latest_outer, row['step'])
    counts = []
    for rank in range(4):
        path = source/'logs'/('optimizer-rank-'+str(rank)+'.jsonl')
        need(path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid(),
             'SOURCE_RANK_UPDATES_MISSING')
        writes = [row for row in map(json.loads, path.read_text().splitlines())
                  if row.get('kind') == 'OPTIMIZER_WRITE']
        need(writes and writes[-1].get('rank') == rank and
             writes[-1].get('run_id') == request['source_run_id'] and
             type(writes[-1].get('write_count')) is int and
             writes[-1]['write_count'] >= checkpoint_writes, 'SOURCE_RANK_UPDATE_BINDING')
        counts.append(writes[-1]['write_count'])
    if latest_outer == outer and counts == [checkpoint_writes]*4:
        return
    approval = request.get('unsaved_suffix_authorization') or {}
    need(approval.get('approved_by') == 'user' and
         type(approval.get('approved_unix')) in (int, float) and approval['approved_unix'] > 0 and
         isinstance(approval.get('human_instruction'), str) and bool(approval['human_instruction'].strip()) and
         approval.get('source_job') == request['source_job'] and
         approval.get('checkpoint_outer') == outer and
         approval.get('source_completed_outer') == latest_outer and
         approval.get('source_optimizer_writes_per_rank') == counts and
         approval.get('recompute_complete_outers') == list(range(outer+1, latest_outer+1)) and
         approval.get('lossless_resume') is False, 'UNSAVED_UPDATES_REQUIRE_EXPLICIT_USER_APPROVAL')


def validate_request(base, commit):
    """Metadata only: deliberately never import torch or load Adam tensors."""
    base = namespace(base, target=True)
    need(re.fullmatch(r'[0-9a-f]{40}', commit) is not None, 'INVALID_COMMIT')
    request = read(base/'preparation/restore-request.json')
    need(request.get('schema_version') == 1 and request.get('target_base') == str(base) and
         request.get('target_commit') == commit, 'TARGET_BINDING_MISMATCH')
    publication = read(base/'preparation/source-publication.json')
    need(publication.get('commit') == commit and publication.get('pushed_and_fetched') is True,
         'SOURCE_NOT_PUBLISHED')
    source = namespace(request['source_base'])
    need(source != base and request.get('source_job', '').isdecimal() and
         isinstance(request.get('source_run_id'), str) and request['source_run_id'],
         'INVALID_SOURCE_BINDING')
    receipt_path = Path(request['source_receipt'])
    need(receipt_path.is_absolute() and receipt_path.resolve() == receipt_path and
         receipt_path.is_relative_to(source/'artifacts/question-probe'), 'RECEIPT_OUTSIDE_SOURCE')
    receipt = read(receipt_path)
    need(sha(receipt_path) == request.get('source_receipt_sha256'), 'RECEIPT_SHA_CHANGED')
    outer = receipt.get('step')
    need(receipt.get('status') == PAIR_STATUS and receipt.get('kind') == 'question_probe_pair' and
         type(outer) is int and 1 <= outer < 200 and receipt.get('world_size') == 4 and
         receipt.get('run_id') == request['source_run_id'] and receipt.get('source_base') == str(source),
         'SOURCE_PAIR_BINDING_MISMATCH')
    need(receipt_path in [source/'artifacts/question-probe'/kind/('global_step_'+str(outer))/'source-receipt.json'
                         for kind in ('pending-probe-pairs', 'completed-pairs')], 'INVALID_PAIR_PATH')
    source_run = read(source/'logs/run-manifest.json')
    need(source_run.get('run_id') == request['source_run_id'] and
         source_run.get('question_probe', {}).get('head_kind') == 'qwen_native_periodic_lastblock_binary_v1',
         'SOURCE_RUN_MISMATCH')
    need(read(source/'preparation/formal-submission.json').get('job') == request['source_job'],
         'SOURCE_SUBMISSION_MISMATCH')
    admission = read(source/'logs/native-allocation-admission.json')
    need(admission.get('run_id') == request['source_run_id'] and
         admission.get('slurm_job_id') == request['source_job'] and admission.get('accepted') is True,
         'SOURCE_ALLOCATION_MISMATCH')
    cleanup = read(source/'logs/cleanup-status.json')
    need(cleanup.get('run_id') == request['source_run_id'] and cleanup.get('cleaned') is True,
         'SOURCE_OWNED_CLEANUP_INCOMPLETE')
    status = read(source/'logs/pipeline-status.json')
    need(status.get('run_id') == request['source_run_id'] and
         status.get('stage') in ('FAILED', 'OWNED_CLEANUP_FAILED'),
         'SOURCE_NOT_FAILED')
    if status['stage'] == 'OWNED_CLEANUP_FAILED':
        # A terminal Slurm job may have killed the remaining descendants after
        # the old kernel refused pidfds. Only the original-node reconciler can
        # prove this; never reinterpret a login-node PID or a terminal flag as
        # evidence that the owned compute sessions and CUDA contexts are gone.
        need(cleanup.get('failed_job') == request['source_job'] and
             cleanup.get('original_allocation_hostname_verified') is True and
             cleanup.get('all_owned_sessions_absent') is True and
             cleanup.get('recorded_cuda_contexts_absent') is True and
             cleanup.get('execution_hostname') == admission.get('node') and
             str(cleanup.get('cleanup_job_id', '')).isdecimal() and
             cleanup['cleanup_job_id'] != request['source_job'],
             'SOURCE_ORIGINAL_NODE_CLEANUP_UNPROVEN')
    rows = subprocess.check_output(['sacct', '-nP', '-X', '-j', request['source_job'],
        '--format=JobID,State,ExitCode,User'], text=True).strip().splitlines()
    need(len(rows) == 1, 'SOURCE_TERMINAL_UNAVAILABLE')
    fields = rows[0].split('|')
    need(len(fields) >= 4 and fields[0] == request['source_job'] and
         fields[1].split()[0] in ('FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED') and
         fields[3] == pwd.getpwuid(os.getuid()).pw_name, 'SOURCE_NOT_OWNED_TERMINAL')
    active = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i'], text=True).split()
    need(request['source_job'] not in active, 'SOURCE_STILL_ACTIVE')
    validate_unsaved_updates(source, request, receipt)
    return request, receipt


def validate_existing(base, commit, request, receipt):
    run = read(base/'logs/run-manifest.json')
    restore = run.get('full_state_restore') or {}
    need(run.get('source_commit') == commit and run.get('run_id') == request['source_run_id'] and
         restore.get('source_receipt') == request['source_receipt'] and
         restore.get('source_receipt_sha256') == request['source_receipt_sha256'] and
         restore.get('source_base') == request['source_base'] and
         restore.get('completed_outer') == receipt['step'], 'EXISTING_RESTORE_BINDING_MISMATCH')
    return run


def prepare(base, commit):
    request, receipt = validate_request(base, commit)
    base = Path(base)
    if (base/'logs/run-manifest.json').exists():
        validate_existing(base, commit, request, receipt)
        return 'EXISTING_FULL_STATE_RESTORE_BINDING'
    # The heavy existing reader must never execute under the login-node 8-GiB
    # cgroup. run.sh is submitted with precisely this Slurm memory/GPU request.
    need(os.environ.get('SLURM_JOB_ID', '').isdecimal() and
         os.environ.get('SLURM_JOB_ID') != request['source_job'] and
         int(os.environ.get('SLURM_MEM_PER_NODE', '0')) >= 512*1024,
         'REQUIRES_NEW_512G_SLURM_ALLOCATION')
    with contextlib.ExitStack() as stack:
        for name in SOURCE_LOCKS:
            path = Path(request['source_base'])/name
            need(not path.is_symlink(), 'UNSAFE_SOURCE_LOCK')
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([sys.executable, '-u', '-m', 'reproduction.hpc3_dynrpo_periodic.resume',
            '--base', str(base), '--source-receipt', request['source_receipt'], '--commit', commit], check=True)
    validate_existing(base, commit, request, receipt)
    return 'FULL_STATE_RESTORE_PREPARED_IN_SLURM'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(dict(stage=prepare(args.base, args.commit), optimizer_reset=False,
                          student_prefix_recomputed=False)))


if __name__ == '__main__':
    main()
