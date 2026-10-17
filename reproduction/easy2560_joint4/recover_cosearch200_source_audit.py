"""Preserve one known failed CPU SHA attempt; no checkpoint or training edits.

Default inspection is read-only. Explicit --apply moves only the six exact
source phase audit files into an immutable failed-attempt directory. It never
launches a replacement worker, retries a partial move, or modifies plan.json.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

if __package__:
    from . import retire_old_cosearch200 as op
else:
    import retire_old_cosearch200 as op


OPERATION_ID = '19ec0e1bc6224d8e85bbcba0b8ba9eb8'
FAILED_BIRTH = dict(host='ACD-Login-1', pid=110273, start_ticks=96708929, uid=21930,
    boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a', pid_namespace='pid:[4026531836]')
ATTEMPT_NAME = 'failed-source-attempt-d45-pid110273-start96708929'
MOVE_NAMES = ('source-launch-claim.json', 'source-process.json', 'source-worker.json',
              'source-exit.json', 'source.log', 'source.lock')
FORBIDDEN = ('deletion-claim.json', 'actions.jsonl', 'deletion-result.json',
             'deletion-failure.json', 'complete.json', 'source-validation.json', 'progress.json')
FAILED_ERROR = 'COSEARCH200_RETIRE_UNEXPECTED_LARGE_NONTARGET_PRESERVED'


def original_process_exited(proc_root=Path('/proc')):
    """A reused PID cannot be mistaken for the failed attempt's birth."""
    proc = proc_root/str(FAILED_BIRTH['pid'])
    if not proc.exists(): return dict(original_birth_exited=True, reason='PID_ABSENT')
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    observed = dict(host=op.socket.gethostname(), pid=FAILED_BIRTH['pid'], start_ticks=int(fields[19]),
        uid=proc.stat().st_uid, boot_id=(proc_root/'sys/kernel/random/boot_id').read_text().strip(),
        pid_namespace=os.readlink(proc/'ns/pid'))
    exited = observed != FAILED_BIRTH or fields[0] in ('Z', 'X')
    op.need(exited, 'FAILED_SOURCE_ORIGINAL_BIRTH_STILL_LIVE')
    return dict(original_birth_exited=True, reason='BIRTH_REUSED' if observed != FAILED_BIRTH else 'EXITED_STATE',
                observed_birth=observed, process_state=fields[0])


def inspect(root, proc_root=Path('/proc')):
    op.context('hpc'); root = op.no_links(root)
    op.need(root == op.HPC_AUDIT/OPERATION_ID and root.stat().st_uid == op.SOURCE_UID,
            'EXACT_FAILED_AUDIT_ROOT_REQUIRED')
    op.need(not (root/ATTEMPT_NAME).exists() and not (root/'source-recovery-claim.json').exists(),
            'SOURCE_AUDIT_RECOVERY_ALREADY_CLAIMED')
    for name in FORBIDDEN:
        op.need(not os.path.lexists(root/name), 'FAILED_SOURCE_AUDIT_HAS_'+name)
    rows = {}
    for name in MOVE_NAMES+('plan.json',):
        path = root/name; record = op.identity(path)
        op.need(record['uid'] == op.SOURCE_UID and record['links'] == 1 and record['bytes'] <= 2*1024**2,
                'FAILED_SOURCE_AUDIT_FILE_IDENTITY_DIFFERS')
        record['sha256'] = op.small_raw(path)[1]; rows[name] = record
    claim = op.small(root/'source-launch-claim.json')[0]
    process = op.small(root/'source-process.json')[0]
    worker = op.small(root/'source-worker.json')[0]
    exit_record = op.small(root/'source-exit.json')[0]
    plan = op.small(root/'plan.json')[0]
    op.need(claim['operation_id'] == process['operation_id'] == OPERATION_ID and
        claim['candidate_sha256'] == plan['candidate_sha256'] == op.CANDIDATE_SHA and
        claim['phase'] == process['phase'] == 'source' and
        claim['operator_source_sha256'] == op.LEGACY_HK_OPERATOR_SHA and
        process['child'] == worker == FAILED_BIRTH and
        exit_record['returncode'] == 1 and exit_record['error'] == FAILED_ERROR,
        'EXACT_FAILED_SOURCE_ATTEMPT_REQUIRED')
    exited = original_process_exited(proc_root)
    return dict(status='EXACT_FAILED_SOURCE_AUDIT_READY_TO_PRESERVE', operation_id=OPERATION_ID,
        checked_unix=time.time(), process_identity=op.birth(), original_birth=FAILED_BIRTH,
        exited_observation=exited, files=rows, move_names=list(MOVE_NAMES),
        retained_plan_sha256=rows['plan.json']['sha256'], destination=str(root/ATTEMPT_NAME),
        checkpoint_files_read=False, checkpoint_files_changed=False)


def preserve(root, proc_root=Path('/proc')):
    root = Path(root); prepared = inspect(root, proc_root)
    claim_path = root/'source-recovery-claim.json'
    op.write(claim_path, prepared)
    destination = root/ATTEMPT_NAME; destination.mkdir(mode=0o700); op.fsync_dir(root)
    moved = []
    try:
        for name in MOVE_NAMES:
            path = root/name; record = op.identity(path)
            op.need(record == {k:v for k,v in prepared['files'][name].items() if k != 'sha256'} and
                op.small_raw(path)[1] == prepared['files'][name]['sha256'], 'FAILED_AUDIT_CHANGED_BEFORE_RENAME')
            op.need(not os.path.lexists(destination/name), 'FAILED_ATTEMPT_DESTINATION_EXISTS')
            op.journal(destination, dict(stage='PREPARED_RENAME', name=name, identity=prepared['files'][name], checked_unix=time.time()))
            os.rename(path, destination/name); op.fsync_dir(root); op.fsync_dir(destination)
            op.need(op.identity(destination/name) == record and
                op.small_raw(destination/name)[1] == prepared['files'][name]['sha256'], 'FAILED_AUDIT_CHANGED_AFTER_RENAME')
            moved.append(name)
            op.journal(destination, dict(stage='RENAMED', name=name, checked_unix=time.time()))
        op.need(op.identity(root/'plan.json') == {k:v for k,v in prepared['files']['plan.json'].items() if k != 'sha256'} and
            op.small_raw(root/'plan.json')[1] == prepared['retained_plan_sha256'], 'RETAINED_PLAN_CHANGED')
        result = dict(prepared, status='FAILED_SOURCE_AUDIT_PRESERVED_NEW_MANUAL_LAUNCH_ALLOWED',
            confirmed_moved_names=moved, checked_unix=time.time(), worker_launched=False)
        op.write(destination/'preserved.json', result)
        return result
    except BaseException as error:
        op.write(destination/'preserve-failure.json', dict(confirmed_moved_names=moved,
            error_type=type(error).__name__, error=str(error), checked_unix=time.time(), auto_retry_allowed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(); root = Path(args.root)
    if args.apply:
        # Original phase lock was released by worker exit; hold its inode while
        # moving it, so another old work invocation cannot race this preservation.
        lock = os.open(op.no_links(root/'source.lock'), os.O_WRONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); result = preserve(root)
        finally: os.close(lock)
    else: result = inspect(root)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__': main()
