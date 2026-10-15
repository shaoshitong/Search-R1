"""Once-only CPU allocated-byte walk of JXU experiments; never training control.

The detached process survives a short SSH observer timeout. Existing claims are
inspected, never replayed. Only small observation files are written. A completed
du walk is non-atomic with live training and is not a filesystem quota proof.
Python3.6-compatible stdlib only: no model, tensor, CUDA or numerical imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
MEASURE = ROOT.parent
CONTROL = ROOT / 'observations' / 'storage-sigma3-after-exact78-permission-repaired-20261016'
UID = 14090
SHARED_UID = 21930  # Actual mou176 owner of the already-authorized shared parents.
SHARED_GID = 21919
TIMEOUT = 1800
RELATIVE_SOURCE = 'reproduction/hpc3_qprobe/storage_observer_sigma3_after_exact78_permission_repaired.py'


def owned(path, directory=False):
    s = Path(path).lstat()
    if s.st_uid != UID or not (stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode)):
        raise ValueError('Wrong owner or unsafe observation path: ' + str(path))
    return s


def shared_directory(path, write=False):
    s = Path(path).lstat()
    access = os.R_OK | os.X_OK | (os.W_OK if write else 0)
    if (not stat.S_ISDIR(s.st_mode) or s.st_uid not in (UID, SHARED_UID) or
            s.st_gid != SHARED_GID or SHARED_GID not in os.getgroups() or
            not os.access(str(path), access)):
        raise ValueError('Actual authorized shared-parent access differs: ' + str(path))
    return s


def read_json(path, with_sha=False):
    owned(path)
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != UID or before.st_size > 2**21:
            raise ValueError('Unsafe observation metadata')
        with os.fdopen(fd, 'rb') as f:
            fd = None
            payload = f.read(2**21 + 1)
            after = os.fstat(f.fileno())
        value = json.loads(payload.decode('utf-8'))
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Observation metadata changed during read')
        if not isinstance(value, dict):
            raise ValueError('Observation metadata is not an object')
        return (value, hashlib.sha256(payload).hexdigest()) if with_sha else value
    finally:
        if fd is not None:
            os.close(fd)


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.link(str(temporary), str(path))
    temporary.unlink()
    directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def birth(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=int(pid), start_ticks=int(fields[19]), sid=int(fields[3]),
                    uid=directory.stat().st_uid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def source():
    file = Path(__file__).absolute()
    root = file.parents[2]
    if root.parent != ROOT / 'source' or not re.fullmatch('[0-9a-f]{40}', root.name):
        raise ValueError('Only a separately published immutable source may run this observer')
    for directory in (ROOT, ROOT / 'source', ROOT / 'preparation'):
        shared_directory(directory)
    for directory in (root, file.parent.parent, file.parent):
        owned(directory, directory=True)
    owned(file)
    receipt = read_json(ROOT / 'preparation' / ('source-' + root.name + '-verification.json'))
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if (receipt.get('status') != 'PUBLISHED_SOURCE_FULL_SHA_PASS' or
            receipt.get('commit') != root.name or receipt.get('published_remote_sha') != root.name or
            receipt.get('source') != str(root) or receipt.get('files', {}).get(RELATIVE_SOURCE) != digest):
        raise ValueError('Published source identity differs')
    return dict(path=str(file), commit=root.name, sha256=digest)


def validate():
    if os.getuid() != UID:
        raise ValueError('Actual JXU owner required')
    shared_directory(MEASURE)
    shared_directory(ROOT, write=True)
    observations = CONTROL.parent
    if observations.exists() or observations.is_symlink():
        owned(observations, directory=True)
    return source()


def allocated_usage(stdout, stderr, returncode):
    match = re.fullmatch(r'([0-9]+)\t' + re.escape(str(MEASURE)) + r'\n?', stdout)
    if returncode != 0 or stderr != '' or match is None:
        return None
    return int(match.group(1))


def bound_outputs(value):
    for name in ('stdout', 'stderr'):
        output = value.get(name)
        if isinstance(output, str):
            raw = output.encode('utf-8')
            value[name + '_sha256'] = hashlib.sha256(raw).hexdigest()
            value[name + '_bytes'] = len(raw)
            value[name + '_truncated'] = len(raw) > 16384
            value[name] = raw[:16384].decode('utf-8', errors='replace')
    return value


def inspect():
    current = validate()
    if not CONTROL.exists() and not CONTROL.is_symlink():
        return dict(status='NOT_STARTED', control=str(CONTROL), checked_unix=time.time())
    owned(CONTROL, directory=True)
    claim, claim_sha = read_json(CONTROL / 'claim.json', with_sha=True)
    if claim.get('source') != current or claim.get('measure') != str(MEASURE) or claim.get('uid') != UID:
        raise ValueError('Observation claim identity differs')
    process = read_json(CONTROL / 'process.json') if (CONTROL / 'process.json').exists() else None
    child = read_json(CONTROL / 'du-process.json') if (CONTROL / 'du-process.json').exists() else None
    result = read_json(CONTROL / 'result.json') if (CONTROL / 'result.json').exists() else None
    actual = birth(process['pid']) if process else None
    child_live = birth(child['pid']) == child if child else False
    live = actual == process and actual is not None
    if live:
        command = (Path('/proc') / str(process['pid']) / 'cmdline').read_bytes().split(b'\0')
        live = (claim['source']['path'].encode() in command and b'--worker' in command)
    if result is not None:
        if result.get('source') != claim['source'] or result.get('measure') != claim['measure']:
            raise ValueError('Observation result source/scope differs')
        status = result['status']
        if status == 'ACTUAL_DU_COMPLETED0_USAGE_MEASURED':
            if (process is None or child is None or result.get('process') != process or
                    result.get('du_process') != child or result.get('uid') != UID or
                    result.get('claim_sha256') != claim_sha or
                    result.get('actual_du_child_still_live') is not False or
                    result.get('du_cleanup_pending', False) or child_live or
                    result.get('hard_quota_verified') is not False or
                    result.get('training_control_actions') != 0 or
                    result.get('completed_unix', 0) < result.get('started_unix', float('inf')) or
                    result.get('started_unix', 0) < claim.get('started_unix', float('inf'))):
                raise ValueError('Actual completed observation binding differs')
            stdout, stderr = result.get('stdout'), result.get('stderr')
            allocated = allocated_usage(stdout, stderr, result.get('actual_du_exit_code'))
            if (allocated is None or allocated != result.get('allocated_bytes') or
                    result.get('stdout_truncated') is not False or result.get('stderr_truncated') is not False or
                    result.get('stdout_bytes') != len(stdout.encode()) or result.get('stderr_bytes') != 0 or
                    hashlib.sha256(stdout.encode()).hexdigest() != result.get('stdout_sha256') or
                    hashlib.sha256(stderr.encode()).hexdigest() != result.get('stderr_sha256')):
                raise ValueError('Actual completed du output/exit/bytes differ')
    else:
        status = 'ACTUAL_OBSERVER_LIVE' if live else 'CLAIM_WITHOUT_LIVE_HANDLE_OR_RESULT_NO_REPLAY'
    return dict(status=status, actual_live_birth=live, actual_process=actual, process=process,
                actual_du_live_birth=child_live,
                du_process=child, claim=claim, result=result,
                control=str(CONTROL), checked_unix=time.time())


def worker():
    current = validate()
    owned(CONTROL, directory=True)
    claim, claim_sha = read_json(CONTROL / 'claim.json', with_sha=True)
    if claim.get('source') != current or claim.get('measure') != str(MEASURE) or claim.get('uid') != UID:
        raise ValueError('Worker source/claim differs')
    process = birth(os.getpid())
    if process is None or process['uid'] != UID:
        raise ValueError('Actual observer process unavailable')
    publish(CONTROL / 'process.json', process)
    started = time.time()
    value = dict(source=current, measure=str(MEASURE), uid=UID, process=process, claim_sha256=claim_sha,
                 started_unix=started, scope='allocated bytes under experiments; non-atomic du walk',
                 hard_quota_verified=False, training_control_actions=0)
    child = None
    actual_child = None
    try:
        # No shell, no link following, one metadata walk. Timeout concerns this
        # owned measurement child only and cannot terminate any training process.
        env = dict(os.environ, LC_ALL='C')
        child = subprocess.Popen(['nice', '-n', '19', 'du', '-s', '-B1', str(MEASURE)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, env=env)
        actual_child = birth(child.pid)
        fields = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        if (actual_child is None or actual_child['uid'] != UID or
                actual_child['sid'] != process['sid'] or int(fields[1]) != os.getpid()):
            raise ValueError('Actual direct du child binding differs')
        publish(CONTROL / 'du-process.json', actual_child)
        stdout, stderr = child.communicate(timeout=TIMEOUT)
        value.update(actual_du_exit_code=child.returncode, stdout=stdout, stderr=stderr,
                     du_process=actual_child,
                     stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                     stderr_sha256=hashlib.sha256(stderr.encode()).hexdigest())
        allocated = allocated_usage(stdout, stderr, child.returncode)
        if allocated is None:
            value['status'] = 'ACTUAL_DU_FAILED_NO_USAGE_ACCEPTED'
        else:
            value.update(status='ACTUAL_DU_COMPLETED0_USAGE_MEASURED',
                         allocated_bytes=allocated)
    except subprocess.TimeoutExpired as exc:
        value.update(status='ACTUAL_DU_TIMEOUT_NO_USAGE_ACCEPTED', timeout_seconds=TIMEOUT,
                     stdout=(exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout, bytes) else exc.stdout,
                     stderr=(exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr, bytes) else exc.stderr)
    except Exception as exc:
        value.update(status='ACTUAL_OBSERVER_FAILED_NO_USAGE_ACCEPTED',
                     error_type=type(exc).__name__, error=str(exc))
    finally:
        # Popen retains the direct child's identity; never look up or signal a
        # training PID. D-state cleanup can remain pending and is reported.
        if child is not None and child.poll() is None:
            child.kill()
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                value['du_cleanup_pending'] = True
        if child is not None:
            value.update(du_process=actual_child, actual_du_exit_code=child.poll(),
                         actual_du_child_still_live=child.returncode is None)
    value.update(completed_unix=time.time(), duration_seconds=time.time() - started)
    bound_outputs(value)
    publish(CONTROL / 'result.json', value)
    return value


def start():
    current = validate()
    if CONTROL.exists() or CONTROL.is_symlink():
        return inspect()
    os.umask(0o007)
    CONTROL.parent.mkdir(mode=0o770, exist_ok=True)
    owned(CONTROL.parent, directory=True)
    CONTROL.mkdir(mode=0o770)  # Exclusive claim directory; never replace it.
    publish(CONTROL / 'claim.json', dict(source=current, measure=str(MEASURE), uid=UID,
                                        parent_process=birth(os.getpid()), started_unix=time.time()))
    logfile_fd = os.open(str(CONTROL / 'worker.log'), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               BLIS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with os.fdopen(logfile_fd, 'wb') as logfile:
        child = subprocess.Popen([sys.executable, current['path'], '--worker'],
                                 stdin=subprocess.DEVNULL, stdout=logfile, stderr=logfile,
                                 start_new_session=True, close_fds=True, env=env)
    # This is observation only. A slow receipt or missing child never authorizes
    # another launch, and the detached child is not killed by this deadline.
    for _ in range(50):
        if (CONTROL / 'process.json').exists() or (CONTROL / 'result.json').exists() or child.poll() is not None:
            break
        time.sleep(0.2)
    return inspect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    os.umask(0o007)
    print(json.dumps(worker() if args.worker else start() if args.start else inspect()), flush=True)
