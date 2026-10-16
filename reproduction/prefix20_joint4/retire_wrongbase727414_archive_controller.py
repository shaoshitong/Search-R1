"""Preserve one actually failed HKUST3 CPU controller under a distinct name.

No SSH, credentials, signals, training operation, or controller launch occurs.
An exclusive outside claim makes an interrupted rename UNKNOWN, never retryable.
The parent may launch a correctly configured CPU controller only after PASS.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import time


PARENT = Path('/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/archive-controller')
ORIGINAL = PARENT/'job-727414'
ROOT = ORIGINAL
DESTINATION = PARENT/'job-727414-failed-wrong-base-70b5d83'
CLAIM = PARENT/'job-727414-wrong-base-retirement-claim.json'
RESULT = PARENT/'job-727414-wrong-base-retirement-result.json'
COMMIT = '70b5d833012572ba670b6891a0ca1e4c3c3ab4f2'
WRONG_BASE = '/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017'
CORRECT_BASE = '/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727251-20261017'
HOST = dict(host='xielab3', boot_id='34b77aa6-ff04-435f-88be-ff4b0cc3d0ae',
            pid_namespace='pid:[4026531836]', uid=1002)
MEMBERS = {
    'config.json', 'controller-birth.json', 'controller-exit.json',
    'controller-launch-claim.json', 'controller-result.json',
    'controller-supervisor-birth.json', 'controller-supervisor.lock',
    'controller-supervisor.log', 'controller.lock', 'controller.log',
    'events.jsonl', 'launcher-observed-birth.json', 'progress.json',
    'source-sidecar-deployment.json', 'supervisor-observed-controller-birth.json',
}
EXPECTED_SHA = {
    'config.json': 'a4543c1879dd2c2324a379ea572168242a7c92f347727f6249980eea5e6139e1',
    'source-sidecar-deployment.json': '5172bb2a48e2bd405baf6f382fd3d384f1c64f496d193c19fa493137aec1af2b',
    'controller-exit.json': '2430aff21836f0e65bc857c653ba4f5c0b6120813285f6fdf0e1befe1c0eaeba',
    'controller.log': 'b9726d946f7aa7deea5b413642f7faf12d292d411aa262b20d713538714ada71',
}


def need(value, message):
    if not value:
        raise ValueError('PREFIX20_WRONGBASE727414_' + message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(path):
    path = Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK_OR_RELATIVE_PATH')


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, value):
    """Exclusive final path, including after uncertain execution outcomes."""
    safe(path)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)


def proc(pid):
    path = Path('/proc')/str(pid)
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    pid_namespace=os.readlink('/proc/self/ns/pid'), uid=path.stat().st_uid,
                    pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_birth(old, current):
    fields = ('host', 'boot_id', 'pid_namespace', 'uid', 'pid', 'start_ticks', 'sid')
    return current is not None and current['state'] != 'Z' and all(current[k] == old[k] for k in fields)


def read_tree(root):
    safe(root)
    need(root.is_dir() and {p.name for p in root.iterdir()} == MEMBERS, 'EXACT_FAILED_ROOT_FILE_SET_REQUIRED')
    documents = {}; manifest = {}
    for name in sorted(MEMBERS):
        path = root/name; safe(path); before = path.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_size <= 2**20, 'SMALL_REGULAR_EVIDENCE_REQUIRED')
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            need((opened.st_dev, opened.st_ino, opened.st_size) ==
                 (before.st_dev, before.st_ino, before.st_size), 'EVIDENCE_CHANGED_BEFORE_OPEN')
            raw = stream.read(2**20 + 1); after = os.fstat(stream.fileno())
        need(len(raw) == before.st_size and
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
             (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns), 'EVIDENCE_CHANGED_DURING_READ')
        manifest[name] = dict(bytes=len(raw), sha256=digest(raw))
        documents[name] = raw
    return documents, manifest


def validate(root):
    actual = proc(os.getpid())
    need(actual is not None and all(actual[k] == v for k, v in HOST.items()), 'PINNED_HKUST3_OWNER_KERNEL_REQUIRED')
    raw, manifest = read_tree(root)
    for name, expected in EXPECTED_SHA.items():
        need(manifest[name]['sha256'] == expected, 'FIXED_ACTUAL_RECEIPT_SHA_DIFFERS_' + name)
    docs = {name: json.loads(value) for name, value in raw.items() if name.endswith('.json')}
    config = docs['config.json']; config_sha = manifest['config.json']['sha256']
    need(config['base'] == WRONG_BASE and config['source_job'] == '727414' and
         config['training_commit'] == config['code_commit'] == COMMIT and
         config['controller_root'] == str(ORIGINAL) and 'recovery_of' not in config, 'FIXED_WRONG_BASE_CONFIG_REQUIRED')
    exited = docs['controller-exit.json']; deployment = docs['source-sidecar-deployment.json']
    need(exited['returncode'] == 1 and exited['config_sha256'] == config_sha and
         exited['stdout_sha256'] == manifest['controller.log']['sha256'], 'ACTUAL_BOUND_EXIT1_REQUIRED')
    need(deployment['returncode'] == 127 and deployment['stdout'] == '' and
         deployment['stderr'] == 'bash: ' + WRONG_BASE + '/env/bin/python: No such file or directory\n',
         'ACTUAL_PRE_INTERPRETER_RC127_REQUIRED')
    need(docs['controller-launch-claim.json']['config_sha256'] == config_sha, 'OLD_LAUNCH_CLAIM_DIFFERS')
    failure = 'PREFIX20_ARCHIVE_SIDECAR_INSTALL_NOT_ACCEPTED_INSPECT_EXISTING_DIRECTORY'
    for name in ('controller-result.json', 'progress.json'):
        row = docs[name]
        need(row['source_job'] == '727414' and row['run_id'] is None and row['completed_steps'] == [] and
             row['error'] == failure and row['error_type'] == 'ValueError', 'FAILED_BEFORE_RUN_OR_OPERATIONS_REQUIRED')
    result = docs['controller-result.json']
    need(result['status'] == 'ARCHIVE_CONTROLLER_INCOMPLETE' and result['training_job_changed'] is False and
         result['unknown_ssh_observations'] == 0, 'KNOWN_FAILURE_WITHOUT_TRAINING_ACTION_REQUIRED')
    events = [json.loads(line) for line in raw['events.jsonl'].splitlines()]
    need(len(events) == 2 and [row['stage'] for row in events] ==
         ['CONTROLLER_STARTED', 'ARCHIVE_CONTROLLER_INCOMPLETE'] and
         all(row['source_job'] == '727414' and row['run_id'] is None and row['completed_steps'] == [] for row in events),
         'NO_SOURCE_OPERATION_EVENTS_REQUIRED')
    fields = ('host', 'boot_id', 'pid_namespace', 'uid', 'pid', 'start_ticks', 'sid')
    born = docs['controller-birth.json']; supervisor = docs['controller-supervisor-birth.json']
    need(all(born[k] == exited['controller'][k] == docs['supervisor-observed-controller-birth.json'][k] for k in fields) and
         all(supervisor[k] == docs['launcher-observed-birth.json'][k] for k in fields), 'OLD_BIRTH_RECEIPTS_DIFFER')
    need((born['pid'], born['start_ticks'], born['sid']) == (1235612, 996042196, 1235611) and
         (supervisor['pid'], supervisor['start_ticks'], supervisor['sid']) == (1235611, 996042184, 1235611) and
         all(row[k] == value for row in (born, supervisor) for k, value in HOST.items()), 'FIXED_OWN_BIRTH_REQUIRED')
    births = [dict(recorded=row, current=proc(row['pid'])) for row in (born, supervisor)]
    need(all(not same_birth(row['recorded'], row['current']) for row in births), 'OLD_CONTROLLER_OR_SUPERVISOR_STILL_ALIVE')
    return dict(files=manifest, old_births=births, observer=actual, old_config_sha256=config_sha,
                old_actual_exit=exited, no_source_operation_launched_by_failed_controller=True,
                no_operations_scope='exact empty operation evidence and rc127 before sidecar Python executed; no source filesystem scan')


def rename_no_replace(source, destination):
    """One Linux kernel rename; no overwrite or fallback after ENOSYS."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    need(rename is not None, 'ATOMIC_NOREPLACE_RENAME_UNAVAILABLE')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def retire():
    for path in (ROOT, DESTINATION, CLAIM, RESULT):
        safe(path)
    need(not CLAIM.exists() and not RESULT.exists(), 'PRIOR_RETIREMENT_ATTEMPT_INSPECT_NO_RETRY')
    need(not DESTINATION.exists(), 'PRESERVED_DESTINATION_ALREADY_EXISTS')
    proof = validate(ROOT)
    intent = dict(operation='fixed_failed_wrong_base_cpu_controller_directory_retirement',
        original_root=str(ORIGINAL), preserved_root=str(DESTINATION), wrong_base=WRONG_BASE,
        intended_correct_training_base=CORRECT_BASE, source_job='727414', training_commit=COMMIT,
        training_job_changed=False, controller_relaunch_performed=False, credentials_used=False,
        proof=proof, created_unix=time.time())
    publish(CLAIM, intent)  # O_EXCL: even an uncertain previous rename may never be retried.
    try:
        latest = validate(ROOT)
        need(latest['files'] == proof['files'], 'EVIDENCE_CHANGED_AFTER_CLAIM')
        rename_no_replace(ROOT, DESTINATION)
        fsync_directory(ROOT.parent)
        _, readback = read_tree(DESTINATION)
        need(readback == proof['files'] and not ROOT.exists(), 'RENAMED_EVIDENCE_READBACK_DIFFERS')
        value = dict(status='FAILED_WRONG_BASE_CONTROLLER_DIRECTORY_PRESERVED_FULL_SHA_PASS',
            original_root=str(ORIGINAL), preserved_root=str(DESTINATION), files=readback,
            claim_sha256=digest(CLAIM.read_bytes()), original_default_root_absent=True,
            old_births_dead=True, training_job_changed=False, controller_relaunch_performed=False,
            fresh_correctly_bound_controller_launch_permitted=True, checked_unix=time.time())
        publish(RESULT, value)
        return value
    except Exception as error:
        publish(RESULT, dict(status='RETIREMENT_INCOMPLETE_INSPECT_NO_RETRY', error_type=type(error).__name__,
            error=str(error), original_root=str(ORIGINAL), preserved_root=str(DESTINATION),
            training_job_changed=False, fresh_correctly_bound_controller_launch_permitted=False,
            checked_unix=time.time()))
        raise


if __name__ == '__main__':
    print(json.dumps(retire(), sort_keys=True))
