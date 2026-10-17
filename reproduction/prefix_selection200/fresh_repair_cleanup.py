"""Delete only human-rejected run weights/logs, preserving shared inputs/code.

Explicitly invoked after actual native cancellation and owned-process release.
No wildcard run deletion, symlink traversal, model loading, or remote download.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

REJECTED_RUNS = {'a008d7ce983347bbb6e81051b25a7c4e',
                 '5cba22ca409a4571a1e6b87d89c86b09'}
PARENT = Path('/mnt/data/shaoshitong')
BACKUPS = PARENT / 'search-g1-prefix-selection200-independent-backup-20261016'
WEIGHTS = {'.pt', '.safetensors', '.pkl', '.ckpt', '.pth'}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_bases():
    result = []
    for base in sorted(PARENT.glob('search-g1-prefix-selection200-*')):
        path = base / 'preparation/launch-config.json'
        if not path.is_file() or base.is_symlink():
            continue
        config = json.loads(path.read_text())
        if config.get('run_id') in REJECTED_RUNS:
            assert base.resolve() == base and base.parent == PARENT
            result.append((base, config['run_id']))
    return result


def backup_bases():
    result = []
    if not BACKUPS.is_dir():
        return result
    for base in sorted(BACKUPS.iterdir()):
        # Fixed recovery assets include valid shared official/grounding data.
        # Never touch them, even though they use the old logical run prefix.
        if base.is_symlink() or '-outer-' not in base.name:
            continue
        for run in REJECTED_RUNS:
            if base.name.startswith(run[:12] + '-outer-'):
                assert base.resolve() == base and base.parent == BACKUPS
                result.append((base, run))
    return result


def inventory(target):
    assert target in ('source', 'backup')
    assert socket.gethostname() == ('h100-8' if target == 'source' else 'h100-10')
    bases = source_bases() if target == 'source' else backup_bases()
    if target == 'source':
        queue = subprocess.run(['squeue', '-h', '-o', '%i|%Z'], text=True,
                               capture_output=True, timeout=30)
        assert queue.returncode == 0, queue.stderr
        for line in queue.stdout.splitlines():
            workdir = line.partition('|')[2].strip()
            assert not any(workdir == str(base) or workdir.startswith(str(base) + '/')
                           for base, _ in bases), 'A rejected namespace remains active'
    rows = []
    for base, run in bases:
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in
                       ('repo', '.git', 'cache', 'retrieval-cache', '__pycache__')
                       and not (Path(directory) / d).is_symlink()]
            for name in names:
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    continue
                relative = path.relative_to(base)
                weight = path.suffix in WEIGHTS
                log = ('logs' in relative.parts or path.suffix == '.log'
                       or (path.suffix == '.jsonl' and any(
                           x in ('actor', 'probe', 'grounding') for x in relative.parts)))
                if not (weight or log):
                    continue
                stat = path.stat()
                rows.append(dict(path=str(path), base=str(base), run_id=run,
                                 role='weight' if weight else 'log', bytes=stat.st_size,
                                 dev=stat.st_dev, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns))
    return dict(target=target, utc=utc(), hostname=socket.gethostname(),
                bases=[dict(base=str(b), run_id=r) for b, r in bases], files=rows,
                count=len(rows), logical_bytes=sum(x['bytes'] for x in rows),
                preserves=['warmup checkpoints', 'official weights', 'shared round0 grounding',
                           'fixed recovery assets', 'datasets and FAISS', 'source code',
                           'small audit metadata', 'original trajectory gzip files'])


def execute(plan, proof_path):
    proof_path = Path(proof_path)
    assert proof_path.parent == PARENT and proof_path.name.startswith('search-g1-rejected-cleanup-')
    intent = proof_path.with_suffix('.intent.json')
    receipt = proof_path.with_suffix('.receipt.json')
    if receipt.exists():
        return dict(status='EXISTING_CLEANUP_RECEIPT_NO_REPLAY', receipt=str(receipt))
    assert not intent.exists(), 'Reconcile interrupted cleanup before retrying'
    assert plan['target'] in ('source', 'backup') and plan['hostname'] == socket.gethostname()
    before = os.statvfs(str(PARENT))
    with proof_path.open('x') as handle:
        json.dump(plan, handle, sort_keys=True)
    digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    with intent.open('x') as handle:
        json.dump(dict(utc=utc(), plan_sha256=digest, human_authorization=
                       '2026-10-17 latest explicit delete wrong weights/logs and corrected fresh restart'), handle)
    deleted = 0
    for row in plan['files']:
        path = Path(row['path'])
        assert row['run_id'] in REJECTED_RUNS and Path(row['base']) in path.parents
        assert not path.is_symlink()
        st = path.stat()
        assert (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) == (
            row['dev'], row['inode'], row['bytes'], row['mtime_ns']), str(path)
        path.unlink()
        deleted += 1
    assert all(not Path(row['path']).exists() for row in plan['files'])
    after = os.statvfs(str(PARENT))
    result = dict(status='REJECTED_RUN_WEIGHTS_AND_LOGS_DELETED', target=plan['target'],
                  utc=utc(), deleted_files=deleted, deleted_logical_bytes=plan['logical_bytes'],
                  actual_physical_free_delta_bytes=(after.f_bfree-before.f_bfree)*after.f_frsize,
                  ordinary_available_bytes=after.f_bavail*after.f_frsize,
                  plan=str(proof_path), plan_sha256=digest, preserves=plan['preserves'])
    with receipt.open('x') as handle:
        json.dump(result, handle, sort_keys=True)
    return result
