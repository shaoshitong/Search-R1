#!/usr/bin/env python3
"""Retire exactly the 45 authorized HK3 reserve checkpoint files, once.

No remote transport, model loading, directory removal, wildcard deletion, or
resume exists here. An existing claim always requires human inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import time

ROOT = Path('/backup/shaoshitong/qprobe-old-original-reserve-20261016')
PAYLOAD = ROOT / 'payload'
CONTROL = ROOT / 'control' / 'checkpoint-retire45-abb9f5806865'
PLAN_SHA = 'abb9f58068655dcc11f44607ea8c5379cefdbd97e0c6e4dc306c94da9f19834c'
NONREFERENCE_SHA = '11af0feaf969edbe780cac398e1ddb841b26ec9e0cc48a6df8951cae9c2dc607'
EXPECTED_UID = 1002
EXPECTED_HOST = 'xielab3'
EXPECTED_CANDIDATES = 45
EXPECTED_PROTECTED = 73
EXPECTED_BYTES = 115049176725
EXPECTED_ALLOCATED = 115049357312
PROCS = Path('/proc')
RECORDED_BIRTHS = ((3806263, 987513616), (3806268, 987513683),
                  (3808257, 987529760), (3806265, 987513621),
                  (3823765, 987634658), (3806261, 987513611),
                  (3830464, 987683519))
BIRTH_RECORDS = tuple(ROOT / 'control' / name for name in (
    'process.json', 'result.json', 'rsync-copy.child.json', 'rsync-copy.wait.json',
    'source-after.child.json', 'source-after.wait.json', 'source-before.child.json',
    'source-before.wait.json', 'source-source-manifest.child.json',
    'source-source-manifest.wait.json', 'worker-process.json', 'worker-result.json')) + (
    Path('/backup/shaoshitong/qprobe-old-original-reserve-durability-20261016/final/claim.json'),)
# SSH login and user-session services are not checkpoint consumers. Their proc
# details can be hidden even from the same UID. All other unreadable consumers
# fail closed; every inaccessible service is recorded in the receipt.
PROC_SERVICE_EXCEPTIONS = frozenset(('sshd', 'systemd', '(sd-pam)'))
FIELDS = ('device', 'inode', 'nlink', 'size', 'allocated_bytes', 'mtime_ns', 'uid', 'gid', 'mode')


class Refuse(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise Refuse(reason)


def read_small(path, limit=1048576):
    require(not path.is_symlink(), 'SYMLINK_INPUT:' + str(path))
    require(path.stat().st_size <= limit, 'OVERSIZE_INPUT:' + str(path))
    return path.read_bytes()


def inspect_plain_path(path, directory=False):
    require(path.is_absolute(), 'ABSOLUTE_PATH_REQUIRED')
    for member in (path,) + tuple(path.parents):
        s = member.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'SYMLINK_PATH:' + str(member))
        if member != path or directory:
            require(stat.S_ISDIR(s.st_mode), 'NON_DIRECTORY_PATH:' + str(member))
    require(path.resolve() == path, 'NONCANONICAL_PATH:' + str(path))


def load_plan(directory):
    path = directory / 'retirement-plan.json'
    raw = read_small(path)
    require(hashlib.sha256(raw).hexdigest() == PLAN_SHA, 'EXACT_PLAN_SHA_REQUIRED')
    plan = json.loads(raw)
    require(plan.get('root') == str(ROOT), 'FIXED_RESERVE_ROOT_REQUIRED')
    candidates = plan.get('candidates', [])
    protected = plan.get('protected_files', [])
    require(len(candidates) == EXPECTED_CANDIDATES, 'EXACT_45_CANDIDATES_REQUIRED')
    require(len(protected) == EXPECTED_PROTECTED, 'EXACT_73_PROTECTED_REQUIRED')
    require(sum(x['size'] for x in candidates) == EXPECTED_BYTES, 'EXACT_BYTE_TOTAL_REQUIRED')
    require(sum(x['allocated_bytes'] for x in candidates) == EXPECTED_ALLOCATED,
            'EXACT_ALLOCATED_TOTAL_REQUIRED')
    paths = [x['path'] for x in candidates + protected]
    require(len(paths) == len(set(paths)), 'DUPLICATE_PLAN_MEMBER')
    require(len({(x['device'], x['inode']) for x in candidates}) == EXPECTED_CANDIDATES,
            'INDEPENDENT_CANDIDATE_INODES_REQUIRED')
    for entry in candidates + protected:
        path = Path(entry['path'])
        require(path.is_absolute() and '..' not in path.parts, 'UNSAFE_PLAN_PATH')
        require(PAYLOAD in path.parents, 'MEMBER_OUTSIDE_FIXED_PAYLOAD')
        require(entry['uid'] == EXPECTED_UID, 'PLAN_OWNER_DIFFERS')
    for entry in candidates:
        source = entry['source_path']
        require(entry['nlink'] == 1, 'PLAN_CANDIDATE_IS_HARDLINKED')
        require(str(PAYLOAD / source.lstrip('/')) == entry['path'], 'SOURCE_MAPPING_DIFFERS')
        require(source.startswith('/data/user/jxu957/experiments/qprobe-teacher-warmup200-'),
                'ONLY_RETIRED_TEACHER_RUNS_ALLOWED')
        require('/checkpoints/search-g1-common8/actor/global_step_' in source
                or '/artifacts/question-probe/completed-pairs/global_step_' in source,
                'ONLY_CHECKPOINT_OR_PAIR_STATE_ALLOWED')
        require(Path(source).suffix in ('.pt', '.safetensors'), 'ONLY_PINNED_STATE_TENSORS_ALLOWED')
        require(not any(x in source for x in ('/topd-', '/assets/', '/logs/', '/evals/',
                                             '/initial-probe/', '/initialprobe/', '/model/')),
                'PROTECTED_ASSET_SCOPE')
    evidence_raw = read_small(directory / 'hpc3-nonreference-evidence.json', 20000)
    require(hashlib.sha256(evidence_raw).hexdigest() == NONREFERENCE_SHA,
            'EXACT_HPC3_NONREFERENCE_EVIDENCE_REQUIRED')
    envelope = json.loads(evidence_raw)
    evidence = json.loads(envelope['stdout'])
    require(envelope['returncode'] == 0 and evidence['old_hk3_reserve_references'] == [],
            'HPC3_EVIDENCE_CONTAINS_RESERVE_REFERENCE')
    return plan


def observed_stat(path, directory_fd=None):
    s = os.stat(str(path) if directory_fd is None else path.name,
                dir_fd=directory_fd, follow_symlinks=False)
    require(stat.S_ISREG(s.st_mode), 'NOT_REGULAR_FILE:' + str(path))
    return dict(device=s.st_dev, inode=s.st_ino, nlink=s.st_nlink, size=s.st_size,
                allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns,
                uid=s.st_uid, gid=s.st_gid, mode=oct(stat.S_IMODE(s.st_mode)))


def same_stat(entry, directory_fd=None):
    path = Path(entry['path'])
    if directory_fd is None:
        inspect_plain_path(path)
    actual = observed_stat(path, directory_fd)
    require(all(actual[key] == entry[key] for key in FIELDS), 'FRESH_STAT_DIFFERS:' + str(path))
    return actual


def inventory():
    members = set()
    for base, dirs, names in os.walk(str(PAYLOAD), followlinks=False):
        for name in dirs:
            require(not (Path(base) / name).is_symlink(), 'PAYLOAD_DIRECTORY_SYMLINK')
        for name in names:
            path = Path(base) / name
            require(stat.S_ISREG(path.lstat().st_mode), 'PAYLOAD_NONREGULAR_MEMBER')
            members.add(str(path))
            require(len(members) <= 200, 'PAYLOAD_BOUND_EXCEEDED')
    return members


def process_birth(pid):
    path = PROCS / str(pid) / 'stat'
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return dict(pid=pid, state='ABSENT', start_ticks=None)
    parts = raw[raw.rfind(')') + 2:].split()
    require(len(parts) > 19, 'MALFORMED_PROC_STAT')
    return dict(pid=pid, state=parts[0], start_ticks=int(parts[19]))


def dead_recorded_processes():
    recorded = set()

    def births(value):
        if isinstance(value, dict):
            if 'pid' in value and ('start_ticks' in value or 'starttime' in value):
                recorded.add((int(value['pid']), int(value.get('start_ticks', value.get('starttime')))))
            for child in value.values():
                births(child)
        elif isinstance(value, list):
            for child in value:
                births(child)

    for path in BIRTH_RECORDS:
        inspect_plain_path(path)
        require(path.stat().st_uid == EXPECTED_UID, 'PROCESS_RECORD_OWNER_DIFFERS')
        births(json.loads(read_small(path, 20000)))
    require(recorded == set(RECORDED_BIRTHS), 'RECORDED_PROCESS_BINDING_CHANGED')
    result = []
    for pid, ticks in RECORDED_BIRTHS:
        actual = process_birth(pid)
        actual['expected_ticks'] = ticks
        require(actual['start_ticks'] != ticks or actual['state'] in ('Z', 'X'),
                'RECORDED_RESERVE_PROCESS_STILL_ACTIVE:' + str(pid))
        result.append(actual)
    return result


def no_candidate_locks(candidates):
    keys = {(os.major(x['device']), os.minor(x['device']), x['inode']) for x in candidates}
    for line in (PROCS / 'locks').read_text().splitlines():
        for field in line.split():
            if field.count(':') != 2:
                continue
            try:
                major, minor, inode = field.split(':')
                key = (int(major, 16), int(minor, 16), int(inode))
            except ValueError:
                continue
            require(key not in keys, 'CANDIDATE_HAS_PROC_LOCK')


def no_live_references(candidates):
    names = {x['path'] for x in candidates}
    keys = {(x['device'], x['inode']) for x in candidates}
    limitations = []
    scanned = 0
    fd_count = 0
    deadline = time.monotonic() + 20
    for process in PROCS.iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        require(time.monotonic() < deadline, 'PROC_SCAN_TIME_BOUND_EXCEEDED')
        try:
            if process.stat().st_uid != EXPECTED_UID:
                continue
            comm = (process / 'comm').read_text().strip()
            scanned += 1
            require(scanned <= 2000, 'PROC_SCAN_COUNT_BOUND_EXCEEDED')
        except FileNotFoundError:
            continue

        def permission(stage):
            require(comm in PROC_SERVICE_EXCEPTIONS, 'UNREADABLE_POSSIBLE_CONSUMER:' + process.name)
            limitations.append(dict(pid=int(process.name), comm=comm, stage=stage))

        try:
            cwd = Path(os.readlink(str(process / 'cwd')))
            require(cwd != PAYLOAD and PAYLOAD not in cwd.parents, 'LIVE_PROCESS_CWD_IN_PAYLOAD')
        except FileNotFoundError:
            pass
        except PermissionError:
            permission('cwd')
        try:
            descriptors = list((process / 'fd').iterdir())
        except FileNotFoundError:
            descriptors = []
        except PermissionError:
            permission('fd')
            descriptors = []
        for descriptor in descriptors:
            fd_count += 1
            require(fd_count <= 100000, 'PROC_FD_BOUND_EXCEEDED')
            try:
                link = os.readlink(str(descriptor))
                clean = link[:-10] if link.endswith(' (deleted)') else link
                require(clean not in names, 'LIVE_CANDIDATE_FILE_DESCRIPTOR')
                s = descriptor.stat()
                require((s.st_dev, s.st_ino) not in keys, 'LIVE_CANDIDATE_INODE_DESCRIPTOR')
            except FileNotFoundError:
                continue
            except PermissionError:
                permission('fd-member')
        try:
            with (process / 'maps').open() as stream:
                size = 0
                for line in stream:
                    size += len(line)
                    require(size <= 2000000, 'PROC_MAPS_BOUND_EXCEEDED')
                    fields = line.split(None, 5)
                    if len(fields) == 6:
                        link = fields[5].strip()
                        clean = link[:-10] if link.endswith(' (deleted)') else link
                        require(clean not in names, 'LIVE_CANDIDATE_MMAP')
        except FileNotFoundError:
            continue
        except PermissionError:
            permission('maps')
    return dict(own_uid_processes=scanned, fd_count=fd_count,
                inaccessible_service_details=limitations)


def preflight(plan):
    require(socket.gethostname().split('.')[0] == EXPECTED_HOST, 'FIXED_HK3_HOST_REQUIRED')
    require(os.getuid() == EXPECTED_UID, 'FIXED_HK3_UID_REQUIRED')
    for path in (ROOT, PAYLOAD, ROOT / 'control'):
        inspect_plain_path(path, directory=True)
        require(path.stat().st_uid == EXPECTED_UID, 'ROOT_OWNER_DIFFERS')
    require(not CONTROL.exists() and not CONTROL.is_symlink(), 'ONE_SHOT_CONTROL_ALREADY_EXISTS_INSPECT')
    for entry in plan['candidates'] + plan['protected_files']:
        same_stat(entry)
    require(inventory() == {x['path'] for x in plan['candidates'] + plan['protected_files']},
            'PAYLOAD_MEMBER_SET_DIFFERS')
    births = dead_recorded_processes()
    no_candidate_locks(plan['candidates'])
    consumers = no_live_references(plan['candidates'])
    return dict(recorded_processes=births, consumers=consumers,
                actual_identity=dict(host=socket.gethostname(), uid=os.getuid(),
                                     pid=os.getpid(), checked_unix=time.time()))


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_json(path, value):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def journal(stream, value):
    stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def absent_pass(candidates):
    for entry in candidates:
        try:
            Path(entry['path']).lstat()
        except FileNotFoundError:
            continue
        raise Refuse('RETIRED_MEMBER_STILL_EXISTS:' + entry['path'])


def execute(plan, proof):
    # The directory and claim are immutable one-shot admission. A crash at any
    # point leaves a visible refused retry, including a crash before claim fsync.
    CONTROL.mkdir(mode=0o700)
    sync_directory(CONTROL.parent)
    claim = dict(schema_version=1, plan_sha256=PLAN_SHA, root=str(ROOT),
                 candidate_files=EXPECTED_CANDIDATES, protected_files=EXPECTED_PROTECTED,
                 hpc3_nonreference_evidence_sha256=NONREFERENCE_SHA,
                 admitted_unix=time.time(), preflight=proof,
                 automatic_resume_permitted=False)
    exclusive_json(CONTROL / 'claim.json', claim)
    removed = 0
    try:
        fd = os.open(str(CONTROL / 'journal.jsonl'),
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        sync_directory(CONTROL)
        with os.fdopen(fd, 'w') as stream:
            dead_recorded_processes()
            no_candidate_locks(plan['candidates'])
            no_live_references(plan['candidates'])
            for index, entry in enumerate(plan['candidates']):
                path = Path(entry['path'])
                inspect_plain_path(path)
                parent_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    before = same_stat(entry, parent_fd)
                    no_candidate_locks([entry])
                    journal(stream, dict(event='UNLINK_INTENT', index=index, path=str(path),
                                         stat=before, plan_sha256=PLAN_SHA, checked_unix=time.time()))
                    # Recheck after durable intent so a changed member never
                    # becomes an implicit replacement deletion.
                    same_stat(entry, parent_fd)
                    os.unlink(path.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    removed += 1
                    journal(stream, dict(event='UNLINK_FSYNC_DONE', index=index, path=str(path),
                                         removed_count=removed, checked_unix=time.time()))
                finally:
                    os.close(parent_fd)
            absent_pass(plan['candidates'])
            journal(stream, dict(event='FIRST_ABSENT_PASS', files=removed, checked_unix=time.time()))
            for entry in plan['protected_files']:
                same_stat(entry)
            require(inventory() == {x['path'] for x in plan['protected_files']},
                    'PROTECTED_PAYLOAD_MEMBER_SET_DIFFERS')
            absent_pass(plan['candidates'])
            journal(stream, dict(event='SECOND_ABSENT_PASS', files=removed, checked_unix=time.time()))
        result = dict(schema_version=1, status='EXACT_45_CHECKPOINTS_RETIRED_TWO_ABSENT_PASSES',
                      plan_sha256=PLAN_SHA, candidate_files=removed,
                      retired_logical_bytes=EXPECTED_BYTES,
                      retired_allocated_bytes=EXPECTED_ALLOCATED,
                      protected_files=EXPECTED_PROTECTED, protected_fresh_stat_equal=True,
                      first_absent_pass=True, second_absent_pass=True,
                      no_directories_deleted=True, completed_unix=time.time())
        exclusive_json(CONTROL / 'complete.json', result)
        return result
    except BaseException as exc:
        failure = dict(schema_version=1, status='STOPPED_INSPECT_NO_AUTOMATIC_REPLAY',
                       plan_sha256=PLAN_SHA, removed_count_observed=removed,
                       error_type=type(exc).__name__, error=str(exc), failed_unix=time.time())
        try:
            exclusive_json(CONTROL / 'failure.json', failure)
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='perform the already authorized exact retirement')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    plan = load_plan(directory)
    proof = preflight(plan)
    if args.execute:
        result = execute(plan, proof)
    else:
        result = dict(status='READ_ONLY_EXACT_PLAN_PREFLIGHT_PASS', plan_sha256=PLAN_SHA,
                      candidate_files=EXPECTED_CANDIDATES, protected_files=EXPECTED_PROTECTED,
                      allocated_bytes=EXPECTED_ALLOCATED, proof=proof, deletion_performed=False)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
