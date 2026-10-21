"""Persistent HKUST3 CPU controller; credentials enter stdin and stay in RAM."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

from . import checkpoint_archive as core
from .control import own_base, publish
from reproduction.prefix20_joint4.archive_supervisor import ssh_argv

HK_ROOT = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')


def controller_root(base, commit, job, controller_suffix=None):
    """A failed controller keeps its claim and logs; recovery uses a new child."""
    root = HK_ROOT / 'archive-controller' / ('job-' + job)
    if controller_suffix is None:
        return root
    if (not isinstance(controller_suffix, str) or
            re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', controller_suffix) is None):
        raise ValueError('EASY2560_ARCHIVE_RECOVERY_SUFFIX_INVALID')
    claim = json.loads((root/'claim.json').read_text())
    failure = json.loads((root/'failure.json').read_text())
    if (claim.get('base') != str(base) or claim.get('source_commit') != commit or
            claim.get('source_job') != job or not failure.get('error_type')):
        raise ValueError('EASY2560_ARCHIVE_RECOVERY_PARENT_IDENTITY_DIFFERS')
    birth = claim['birth']
    try:
        process = Path('/proc')/str(birth['pid'])
        fields = (process/'stat').read_text().rsplit(')', 1)[1].split()
        same = (int(fields[19]) == birth['start_ticks'] and process.stat().st_uid == birth['uid']
                and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == birth['boot_id'])
        if same and fields[0] != 'Z':
            raise ValueError('EASY2560_ARCHIVE_RECOVERY_PARENT_STILL_LIVE')
    except (FileNotFoundError, ProcessLookupError):
        pass
    return root / ('recovery-' + controller_suffix)


def source_command(base, *arguments):
    base = own_base(base)
    return ('QPROBE_SOURCE=' + shlex.quote(str(base / 'repo')) + ' ' +
            shlex.join([str(base / 'env/bin/python'), '-B', '-m',
                        'reproduction.easy2560_joint4.checkpoint_archive', *map(str, arguments)]))


def checkpoint_status_code(base, step, job):
    """Generated remote Python uses an unambiguous newline byte, not escapes."""
    # A complete checkpoint is durable evidence after Slurm has purged its job.
    # Queue lookup is relevant only while waiting for an unfinished checkpoint.
    return '\n'.join((
        'import json,subprocess', 'from pathlib import Path',
        'b=Path(' + repr(str(base)) + ')',
        "p=b/'checkpoints/search-g1-common8/actor/global_step_%d/complete.json'" % step,
        'v=json.loads(p.read_text()) if p.is_file() else {}',
        "u=b/'logs/actual-updates.jsonl';raw=u.read_bytes() if u.is_file() else b''",
        'rows=[json.loads(x) for x in raw.split(bytes((10,)))[:-1] if x]',
        "a=b/'logs/native-allocation-admission.json'",
        'allocation=json.loads(a.read_text()) if a.is_file() else {}',
        "ready=all('training_state/rank-%d.pt'%r in v.get('files_sha256',{}) for r in range(4)) and " +
        'any(x.get("step")==%d and x.get("run_id")==v.get("run_id") for x in rows)' % step,
        'assert not ready or allocation.get("slurm_job_id")=='+repr(job),
        "j='';queue_rc=None;queue_error=''",
        'if not ready:',
        '    q=subprocess.run(["squeue","-h","-j",'+repr(job)+',"--format=%T"],capture_output=True,text=True)',
        '    j=q.stdout.strip();queue_rc=q.returncode;queue_error=q.stderr[-1000:]',
        'print(json.dumps(dict(ready=ready,job_state=j,queue_returncode=queue_rc,queue_error=queue_error)))',
    ))


def event(root, **values):
    value = dict(values, checked_unix=time.time())
    temporary = root / 'progress.json.partial'
    with temporary.open('w') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(root / 'progress.json')
    print(json.dumps(value), flush=True)


def observe_checkpoint(base, step, job, env, root, *, attempts=3, retry_seconds=15,
                       completed_steps=()):
    """Retry only read-only transport failures; never replay archive writers."""
    if type(attempts) is not int or attempts < 1 or retry_seconds < 0:
        raise ValueError('EASY2560_ARCHIVE_OBSERVATION_RETRY_POLICY_INVALID')
    code = checkpoint_status_code(base, step, job)
    command = shlex.join([str(base / 'env/bin/python'), '-B', '-c', code])
    for attempt in range(1, attempts + 1):
        try:
            checked = subprocess.run(ssh_argv(command), env=env, capture_output=True,
                                     text=True, timeout=90)
        except subprocess.TimeoutExpired:
            detail = 'SSH read-only observation timed out'
        else:
            if checked.returncode == 0:
                return json.loads(checked.stdout)
            detail = 'rc=%d: %s' % (checked.returncode, checked.stderr[-1000:])
            # sshpass returns5 for authentication failure, SSH255 for transport.
            # Host-key failures6/7 and bound-host mismatch86 are never retried.
            if checked.returncode not in (5, 255):
                raise RuntimeError('EASY2560_ARCHIVE_SOURCE_OBSERVATION_UNKNOWN: ' + detail)
        if attempt == attempts:
            raise RuntimeError('EASY2560_ARCHIVE_SOURCE_OBSERVATION_UNKNOWN_AFTER_RETRIES: ' + detail)
        event(root, phase='SOURCE_OBSERVATION_RETRY', source_job=job, step=step,
              attempt=attempt, attempts=attempts, observation_error=detail,
              source_job_state_unknown=True, completed_steps=list(completed_steps))
        time.sleep(retry_seconds)


def run(base, commit, job, password, poll_seconds=60, first_step=5, controller_suffix=None):
    if socket.gethostname() != 'xielab3' or os.getuid() != 1002:
        raise ValueError('EASY2560_ARCHIVE_CONTROLLER_REQUIRES_HKUST3')
    base = own_base(base)
    if first_step not in core.STEPS: raise ValueError('EASY2560_ARCHIVE_FIXED_FIRST_MILESTONE_REQUIRED')
    if not str(job).isdecimal(): raise ValueError('EASY2560_ARCHIVE_SOURCE_JOB_REQUIRED')
    job_root = HK_ROOT / 'archive-controller' / ('job-' + job)
    job_root.mkdir(parents=True, exist_ok=True)
    family_lock = (job_root/'controller-family.lock').open('a')
    lock = None
    try:
        fcntl.flock(family_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = controller_root(base, commit, job, controller_suffix)
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / 'controller.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim = root / 'claim.json'
        if claim.exists(): raise ValueError('EASY2560_ARCHIVE_CONTROLLER_ALREADY_ATTEMPTED_INSPECT_BIRTH_AND_RESULT')
        birth = dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                     boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), uid=os.getuid())
        publish(claim, dict(base=str(base), source_commit=commit, source_job=job, birth=birth,
                            requested_steps=[step for step in core.STEPS if step>=first_step],
                            controller_suffix=controller_suffix,
                            recovery_parent=str(job_root) if controller_suffix else None,
                            credentials_persisted=False, checked_unix=time.time()))
    except BaseException:
        if lock is not None: lock.close()
        family_lock.close()
        raise
    env = dict(os.environ, SSHPASS=password); completed = []; password = None
    try:
        for step in (value for value in core.STEPS if value>=first_step):
            source = base / 'preparation/checkpoint-archives' / ('global_step_' + str(step)) / 'source-snapshot.json'
            while True:
                # Observe actual job and a fully produced native4 checkpoint.
                state = observe_checkpoint(base, step, job, env, root, completed_steps=completed)
                if state['ready']: break
                if state['queue_returncode'] != 0:
                    raise RuntimeError('EASY2560_ARCHIVE_SOURCE_QUEUE_OBSERVATION_UNKNOWN: ' + state['queue_error'])
                event(root, phase='WAITING_CHECKPOINT', source_job=job, step=step,
                      actual_job_state=state['job_state'], completed_steps=completed)
                if not state['job_state']:
                    raise RuntimeError('EASY2560_ARCHIVE_SOURCE_JOB_NOT_LIVE_AND_CHECKPOINT_MISSING')
                time.sleep(poll_seconds)
            event(root, phase='SOURCE_SNAPSHOT', step=step, completed_steps=completed)
            captured = subprocess.run(ssh_argv(source_command(base, 'snapshot', '--base', base,
                                        '--commit', commit, '--step', step)), env=env,
                                      capture_output=True, text=True, timeout=7200)
            if captured.returncode: raise RuntimeError('EASY2560_ARCHIVE_SNAPSHOT_FAILED: ' + captured.stderr[-2000:])
            snapshot = json.loads(captured.stdout)
            if snapshot['source_commit'] != commit: raise ValueError('EASY2560_ARCHIVE_SNAPSHOT_COMMIT_DIFFERS')
            directory = HK_ROOT / 'checkpoints' / snapshot['run_id'] / ('global_step_' + str(step))
            event(root, phase='STREAMING_SOURCE_TO_HKUST3', step=step, total_bytes=snapshot['total_bytes'], completed_steps=completed)
            with (root / ('stream-step-%d.stderr' % step)).open('xb') as error:
                child = subprocess.Popen(ssh_argv(source_command(base, 'stream', '--snapshot', source)),
                                         env=env, stdout=subprocess.PIPE, stderr=error)
                try:
                    receipt = core.receive(directory, child.stdout)
                    child.stdout.close(); result = child.wait(timeout=120)
                    if result: raise RuntimeError('EASY2560_ARCHIVE_SOURCE_STREAM_EXIT_NONZERO: ' + str(result))
                except BaseException:
                    child.terminate()
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    raise
            accepted = subprocess.run(ssh_argv(source_command(base, 'accept', '--snapshot', source)),
                                      input=json.dumps(receipt), env=env, capture_output=True, text=True, timeout=7200)
            if accepted.returncode: raise RuntimeError('EASY2560_ARCHIVE_SOURCE_READBACK_FAILED: ' + accepted.stderr[-2000:])
            proof = json.loads(accepted.stdout)
            if proof['status'] != core.STATUS: raise ValueError('EASY2560_ARCHIVE_SOURCE_ACCEPTANCE_DIFFERS')
            publish(directory / 'source-acceptance.json', proof); completed.append(step)
            event(root, phase='CHECKPOINT_SOURCE_ARCHIVE_ACCEPTED', step=step, completed_steps=completed,
                  total_bytes=snapshot['total_bytes'], actual_cuda_restore_accepted=False)
        publish(root / 'result.json', dict(status=core.STATUS, source_job=job, source_commit=commit,
                                           base=str(base), completed_steps=completed, checked_unix=time.time()))
    except BaseException as error:
        publish(root / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                            completed_steps=completed, checked_unix=time.time()))
        raise
    finally:
        env.pop('SSHPASS', None)
        lock.close()
        family_lock.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--commit', required=True)
    p.add_argument('--job', required=True); p.add_argument('--poll-seconds', type=int, default=60)
    p.add_argument('--first-step', type=int, choices=core.STEPS, default=5)
    p.add_argument('--controller-suffix', help='New child directory for a verified dead failed controller')
    a = p.parse_args()
    secret = json.load(sys.stdin); password = secret.pop('password'); secret = None
    if not isinstance(password, str) or not password or '\n' in password:
        raise ValueError('EASY2560_RAM_PASSWORD_STDIN_REQUIRED')
    run(a.base, a.commit, a.job, password, a.poll_seconds, a.first_step, a.controller_suffix)


if __name__ == '__main__': main()
