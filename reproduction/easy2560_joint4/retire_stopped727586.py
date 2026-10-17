"""Authorized exact39 tensor retirement; standalone CPU-only operator.

The human explicitly approved these stopped727586 HPC3 copies while retaining
the full HKUST3 archives. This never edits an active runtime or shared asset.
Long SHA passes run under a persistent claim/process birth and byte progress.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time

BASE = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017')
RUN = 'e4cf46d47ae24434a177931df5530751'
SOURCE_COMMIT = '478f46315147ebb0d4d1c556fcdd356c658e117e'
PLAN_SHA = '4cd6dd921bd3c0f73b3a2f6c326a09a1dc9f3b91bb63ea002407a6eab0d7574e'
ARCHIVES = Path('/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/checkpoints') / RUN
NAMES = ['model-%05d-of-00003.safetensors' % n for n in (1, 2, 3)] + [
    'training_state/' + name for name in (
        *('rank-%d.pt' % r for r in range(4)), *('prefix20-rank-%d.pt' % r for r in range(4)),
        'driver.pt', 'question-probe-state.pt')]
STEPS = (5, 50, 100)


def raw_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def under(path, parent):
    try: Path(path).relative_to(parent); return True
    except ValueError: return False


def write(path, value, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial-' + str(os.getpid()))
    with temporary.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    if replace: os.replace(temporary, path)
    else:
        os.link(temporary, path); temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def plan(path):
    if raw_sha(path) != PLAN_SHA: raise ValueError('EXACT_APPROVED_PLAN_SHA_REQUIRED')
    value = read(path)
    expected = {str(BASE / 'checkpoints/search-g1-common8/actor' / ('global_step_' + str(s)) / n)
                for s in STEPS for n in NAMES}
    if (value['base'] != str(BASE) or value['old_job'] != '727586' or
            value['new_active_job'] != '730080' or len(value['files']) != 39 or
            {x['path'] for x in value['files']} != expected or
            sum(x['allocated_bytes'] for x in value['files']) != 125939511296):
        raise ValueError('EXACT_APPROVED39_SCOPE_DIFFERS')
    return value


def identity(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)): raise ValueError('SYMLINK_FORBIDDEN')
    s = path.stat()
    if not stat.S_ISREG(s.st_mode): raise ValueError('REGULAR_FILE_REQUIRED')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                allocated_bytes=s.st_blocks * 512, links=s.st_nlink, uid=s.st_uid)


def birth(pid=None):
    pid = os.getpid() if pid is None else pid
    fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    return dict(host=socket.gethostname(), pid=pid, start_ticks=int(fields[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                uid=Path('/proc/%d' % pid).stat().st_uid, state=fields[0])


def progress(root, **value): write(Path(root) / 'progress.json', dict(value, checked_unix=time.time()), replace=True)


def measured_sha(path, root, *, phase, done, total, completed):
    h = hashlib.sha256(); count = 0; last = time.monotonic()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b); count += len(b)
            if time.monotonic() - last >= 2:
                progress(root, phase=phase, hashed_bytes=done + count, total_bytes=total,
                         completed_files=completed, current_file=str(path), current_file_bytes=count)
                last = time.monotonic()
    return h.hexdigest()


def validate_archive(value, root):
    if socket.gethostname() != 'xielab3' or os.getuid() != 1002: raise ValueError('HKUST3_OWNER_REQUIRED')
    mapped = {}; metadata = {}; done = 0; total = sum(x['bytes'] for x in value['files'])
    for step in STEPS:
        directory = ARCHIVES / ('global_step_' + str(step)); receipt = read(directory / 'archive-receipt.json')
        ack = read(directory / 'source-acceptance.json'); snapshot = read(directory / 'source-snapshot.json')
        if (receipt['status'] != 'SOURCE_ARCHIVE_FULL_SHA_PASS' or receipt['run_id'] != RUN or
                receipt['step'] != step or receipt['training_source_commit'] != SOURCE_COMMIT or
                not receipt['all_archive_files_fresh_sha_pass'] or not receipt['all_archive_files_fsync_completed'] or
                ack['status'] != 'SOURCE_ARCHIVE_FULL_SHA_PASS' or
                receipt['source_snapshot_sha256'] != raw_sha(directory / 'source-snapshot.json')):
            raise ValueError('ACCEPTED_FULL_ARCHIVE_BINDING_DIFFERS')
        rows = {x['source_path']: x for x in receipt['files']}
        metadata[str(step)] = dict(archive_root=str(directory), archive_receipt_sha256=raw_sha(directory / 'archive-receipt.json'),
                                   source_acceptance=ack, source_snapshot_sha256=raw_sha(directory / 'source-snapshot.json'))
        for approved in value['files']:
            if '/global_step_%d/' % step not in approved['path']: continue
            row = rows[approved['path']]; path = Path(row['archive_path'])
            if not under(path, directory) or row['bytes'] != approved['bytes']:
                raise ValueError('ARCHIVE_FILE_PATH_OR_SIZE_DIFFERS')
            before = identity(path)
            if any(before[k] != row['archive_identity'][k] for k in ('device', 'inode', 'bytes', 'mtime_ns')):
                raise ValueError('ARCHIVE_IDENTITY_CHANGED')
            if measured_sha(path, root, phase='HK_FRESH_SHA', done=done, total=total, completed=len(mapped)) != row['sha256']:
                raise ValueError('ARCHIVE_FRESH_SHA_DIFFERS')
            if identity(path) != before: raise ValueError('ARCHIVE_CHANGED_DURING_SHA')
            mapped[approved['path']] = row; done += row['bytes']
            progress(root, phase='HK_FRESH_SHA', hashed_bytes=done, total_bytes=total, completed_files=len(mapped))
    if len(mapped) != 39: raise ValueError('ALL39_ARCHIVE_MAPPING_REQUIRED')
    return dict(status='ALL39_HK_ARCHIVE_FRESH_SHA_PASS', plan_sha256=PLAN_SHA, files=mapped,
                metadata=metadata, checked_unix=time.time(), hashed_bytes=done, all_hk_archives_retained=True)


def terminal():
    result = {}
    for name, args in (
            ('queue', ['squeue', '-h', '-j', '727586', '-o', '%i|%T|%N']),
            ('accounting', ['sacct', '-n', '-P', '-j', '727586', '--format=JobIDRaw,State,ExitCode,NodeList']),
            ('node_pids', ['scontrol', 'listpids', '727586', 'ACD1-24'])):
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        result[name] = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    if result['queue']['stdout'].strip() or result['queue']['returncode'] not in (0, 1): raise ValueError('OLD_JOB_NOT_TERMINAL')
    lines = result['accounting']['stdout'].splitlines()
    if not any(x.startswith('727586|CANCELLED') for x in lines): raise ValueError('OLD_CANCELLED_ACCOUNTING_REQUIRED')
    if result['node_pids']['returncode'] != 1 or 'does not exist on node ACD1-24' not in result['node_pids']['stderr']:
        raise ValueError('OLD_COMPUTE_JOB_PIDS_NOT_CONFIRMED_ABSENT')
    if not any(x.startswith('727586.extern|COMPLETED|0:0|ACD1-24') for x in lines):
        raise ValueError('OLD_SLURM_EXTERN_NOT_CLOSED')
    path = BASE / 'logs/cleanup-status.json'
    result['cleanup'] = read(path) if path.is_file() else None
    result['original_compute_closed_by_slurm'] = True
    return result


def readers(value):
    targets = {x['path'] for x in value['files']}; found = []; scanned = 0
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid() or int(proc.name) == os.getpid(): continue
            scanned += 1; matches = []
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.endswith(' (deleted)'): target = target[:-10]
                    if target in targets: matches.append(target)
                except FileNotFoundError: pass
            try:
                for line in (proc / 'maps').read_text().splitlines():
                    if any(x in line for x in targets): matches.append(line)
            except FileNotFoundError: pass
            if matches: found.append(dict(pid=int(proc.name), matches=sorted(set(matches))))
        except (FileNotFoundError, ProcessLookupError): pass
    return dict(host=socket.gethostname(), uid=os.getuid(), checked_unix=time.time(),
                scanned_own_processes=scanned, readers=found)


def validate_source(value, archive, root):
    if os.getuid() != 14090 or socket.gethostname() != 'ACD-Login-1': raise ValueError('PINNED_HPC_JXU_OWNER_REQUIRED')
    if archive['status'] != 'ALL39_HK_ARCHIVE_FRESH_SHA_PASS' or archive['plan_sha256'] != PLAN_SHA:
        raise ValueError('FRESH_HK_PROOF_REQUIRED')
    before_terminal = terminal(); checked = {}; done = 0; total = sum(x['bytes'] for x in value['files'])
    metadata = {}
    for step in STEPS:
        cp = BASE / 'checkpoints/search-g1-common8/actor' / ('global_step_' + str(step))
        complete = read(cp / 'complete.json')
        source_complete = BASE / 'preparation/checkpoint-archives' / ('global_step_' + str(step)) / 'complete.json'
        accepted = archive['metadata'][str(step)]['source_acceptance']
        if (complete['run_id'] != RUN or complete['step'] != step or
                accepted['complete_file'] != str(source_complete) or accepted['complete_sha256'] != raw_sha(source_complete)):
            raise ValueError('SOURCE_CHECKPOINT_COMPLETE_OR_ARCHIVE_ACCEPTANCE_DIFFERS')
        metadata[str(step)] = dict(checkpoint_complete_sha256=raw_sha(cp / 'complete.json'),
                                   state_manifest_sha256=raw_sha(cp / 'training_state/manifest.json'),
                                   source_archive_complete_sha256=raw_sha(source_complete))
    for approved in value['files']:
        path = Path(approved['path']); row = archive['files'][str(path)]; before = identity(path)
        if any(before[k] != approved[k] for k in ('bytes', 'allocated_bytes', 'mtime_ns')):
            raise ValueError('SOURCE_APPROVED_IDENTITY_CHANGED')
        if any(before[k] != row['source_identity'][k] for k in ('device', 'inode', 'bytes', 'mtime_ns')):
            raise ValueError('SOURCE_ORIGINAL_ARCHIVE_IDENTITY_CHANGED')
        cp = next(BASE / 'checkpoints/search-g1-common8/actor' / ('global_step_' + str(s))
                  for s in STEPS if '/global_step_%d/' % s in str(path))
        if read(cp / 'complete.json')['files_sha256'][str(path.relative_to(cp))] != row['sha256']:
            raise ValueError('SOURCE_PRODUCER_SHA_MAPPING_DIFFERS')
        if measured_sha(path, root, phase='HPC_FRESH_SHA', done=done, total=total, completed=len(checked)) != row['sha256']:
            raise ValueError('SOURCE_FRESH_SHA_DIFFERS')
        if identity(path) != before: raise ValueError('SOURCE_CHANGED_DURING_SHA')
        checked[str(path)] = dict(identity=before, sha256=row['sha256']); done += approved['bytes']
        progress(root, phase='HPC_FRESH_SHA', hashed_bytes=done, total_bytes=total, completed_files=len(checked))
    observed_readers = readers(value)
    if observed_readers['readers']: raise ValueError('ACTIVE_CHECKPOINT_READER')
    return dict(status='ALL39_SOURCE_FRESH_SHA_PASS', plan_sha256=PLAN_SHA, files=checked,
                metadata=metadata, terminal_before=before_terminal, terminal_after=terminal(),
                reader_observation=observed_readers, checked_unix=time.time(), hashed_bytes=done)


def execute(value, source, archive, guards, root):
    if os.getuid() != 14090 or socket.gethostname() != 'ACD-Login-1': raise ValueError('PINNED_HPC_JXU_OWNER_REQUIRED')
    if (source['status'] != 'ALL39_SOURCE_FRESH_SHA_PASS' or archive['status'] != 'ALL39_HK_ARCHIVE_FRESH_SHA_PASS' or
            source['plan_sha256'] != PLAN_SHA or archive['plan_sha256'] != PLAN_SHA or
            set(source['files']) != {x['path'] for x in value['files']}): raise ValueError('ALL39_VERIFIED_PROOFS_REQUIRED')
    for uid in (14090, 21930):
        g = guards[str(uid)]
        if (g['uid'] != uid or g['host'] != 'ACD-Login-1' or g['readers'] or
                time.time() - g['checked_unix'] > 300): raise ValueError('FRESH_BOTH_OWNER_READER_PROOFS_REQUIRED')
    current_terminal = terminal(); current_readers = readers(value)
    if current_readers['readers']: raise ValueError('ACTIVE_READER_AT_RETIREMENT')
    claim = Path(root) / 'deletion-claim.json'
    write(claim, dict(plan_sha256=PLAN_SHA, authorization='删除这39个HPC3文件，保留完整远端归档',
                      terminal=current_terminal, readers=guards, birth=birth(), created_unix=time.time()))
    # Validate every identity before the first unlink; deletion itself is exact-file only.
    for row in value['files']:
        if identity(row['path']) != source['files'][row['path']]['identity']:
            raise ValueError('SOURCE_IDENTITY_CHANGED_BEFORE_FIRST_UNLINK')
    deleted = []; freed = 0
    for row in value['files']:
        path = Path(row['path']); before = identity(path)
        if before != source['files'][str(path)]['identity']: raise ValueError('SOURCE_IDENTITY_CHANGED_BEFORE_UNLINK')
        path.unlink(); deleted.append(str(path))
        if before['links'] == 1: freed += before['allocated_bytes']
        progress(root, phase='EXACT39_UNLINK', deleted_files=len(deleted), unlinked_allocated_bytes=freed)
    for directory in {Path(x).parent for x in deleted}:
        fd = os.open(directory, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    if any(Path(x).exists() for x in deleted): raise ValueError('SOURCE_FILE_REAPPEARED')
    return dict(status='AUTHORIZED_EXACT39_HPC_COPIES_RETIRED', plan_sha256=PLAN_SHA,
                deleted_files=deleted, deleted_count=len(deleted), unlinked_allocated_bytes=freed,
                all_hk_archives_retained=True, archive_roots=[str(ARCHIVES / ('global_step_' + str(s))) for s in STEPS],
                checked_unix=time.time())


def launch(args):
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    claim = root / (args.phase + '-launch-claim.json')
    write(claim, dict(parent=birth(), phase=args.phase, operator_sha256=raw_sha(__file__), plan_sha256=PLAN_SHA))
    log = (root / (args.phase + '.log')).open('xb')
    argv = [sys.executable, '-B', __file__, 'work', '--phase', args.phase, '--root', str(root), '--plan', args.plan]
    if args.archive: argv += ['--archive', args.archive]
    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    actual = birth(child.pid); write(root / (args.phase + '-process.json'), actual)
    return dict(pid=child.pid, birth=actual, phase=args.phase, launched=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('launch', 'work', 'readers', 'execute'))
    p.add_argument('--phase', choices=('hk', 'source')); p.add_argument('--root', required=True)
    p.add_argument('--plan', required=True); p.add_argument('--archive'); p.add_argument('--source'); p.add_argument('--guards')
    a = p.parse_args(); value = plan(a.plan); root = Path(a.root)
    if a.mode == 'launch': result = launch(a)
    elif a.mode == 'readers': result = readers(value)
    elif a.mode == 'execute':
        result = execute(value, read(a.source), read(a.archive), read(a.guards), root); write(root / 'deletion-result.json', result)
    else:
        lock = (root / (a.phase + '.lock')).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / (a.phase + '-worker.json'), birth())
        try:
            result = validate_archive(value, root) if a.phase == 'hk' else validate_source(value, read(a.archive), root)
            write(root / (a.phase + '-validation.json'), result)
            write(root / (a.phase + '-exit.json'), dict(returncode=0, checked_unix=time.time()))
        except BaseException as error:
            write(root / (a.phase + '-exit.json'), dict(returncode=1, error_type=type(error).__name__, error=str(error), checked_unix=time.time()))
            raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
