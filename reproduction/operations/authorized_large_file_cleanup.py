"""Unlink inventoried, unchanged large binary files from explicit retired roots.

The plan must include protected dependency paths and stat identities. Text logs,
source, Git history, symlinks and files smaller than 1 MiB are always retained.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import stat
import time

TEXT_SUFFIXES = ('.log', '.txt', '.md', '.rst', '.csv', '.tsv', '.jsonl', '.yaml',
                 '.yml', '.toml', '.ini', '.cfg', '.py', '.sh', '.c', '.cc',
                 '.cpp', '.h', '.hpp', '.cu', '.cuh', '.ipynb', '.patch', '.diff')
BINARY_SUFFIXES = ('.pt', '.pth', '.safetensors', '.bin', '.npy', '.npz', '.pkl',
                   '.pickle', '.arrow', '.parquet', '.gz', '.xz', '.zst', '.zip',
                   '.tar', '.whl', '.a', '.o', '.so', '.index', '.ckpt', '.pyc')


def within(path, root):
    return path == root or path.startswith(root.rstrip('/') + '/')


def eligible(path, size):
    name = Path(path).name.lower()
    if size < 1048576 or name.endswith(TEXT_SUFFIXES):
        return False
    if any(x in name for x in ('credential', 'private-key', 'id_rsa', 'id_ed25519')):
        return False
    if '.git' in Path(path).parts or '.ssh' in Path(path).parts:
        return False
    if name.endswith(BINARY_SUFFIXES) or '.so.' in name:
        return True
    # Unknown extensions: retain text (including logs without a suffix).
    with open(path, 'rb') as stream:
        prefix = stream.read(8192)
    try:
        prefix.decode('utf-8')
        return b'\0' in prefix
    except UnicodeDecodeError:
        return True


def run(plan, output, execute=False):
    roots = plan['allowed_roots']
    protected = plan['protected_paths']
    assert all(os.path.isabs(p) and os.path.realpath(p) == p for p in roots + protected)
    files = plan['files']
    assert len({x['path'] for x in files}) == len(files)
    # Validate the entire plan before any unlink, including lexical AND resolved paths.
    for item in files:
        path = item['path']
        assert os.path.isabs(path) and os.path.realpath(path) == path, path
        assert any(within(path, r) and path != r for r in roots), path
        assert not any(within(path, p) for p in protected), path
    started = time.time()

    def worker(item):
        path = item['path']
        try:
            if os.path.realpath(path) != path:
                return dict(path=path, status='changed_path')
            s = os.lstat(path)
            if not stat.S_ISREG(s.st_mode) or [s.st_dev, s.st_ino] != item['inode'] or s.st_size != item['size'] or s.st_mtime != item['mtime']:
                return dict(path=path, status='changed_identity')
            if not eligible(path, s.st_size):
                return dict(path=path, status='retained_policy')
            if execute:
                os.unlink(path)
            return dict(path=path, status='deleted' if execute else 'eligible',
                        bytes=s.st_size, freed_blocks_lower_bound=s.st_blocks*512 if execute and s.st_nlink == 1 else 0)
        except FileNotFoundError:
            return dict(path=path, status='already_absent')
        except OSError as exc:
            return dict(path=path, status='error', error=str(exc))

    rows = []
    def save(finished):
        result = dict(started_unix=started, checked_unix=time.time(), finished=finished,
                      execute=execute, workers=8, planned_files=len(files), processed_files=len(rows),
                      files_removed=sum(x['status']=='deleted' for x in rows),
                      freed_blocks_lower_bound=sum(x.get('freed_blocks_lower_bound',0) for x in rows),
                      rows=rows, protected_paths=protected)
        tmp = output + '.tmp'
        Path(tmp).write_text(json.dumps(result))
        os.replace(tmp, output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(worker, item) for item in files}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
            rows.extend(f.result() for f in done)
            save(not pending)
    if not files:
        save(True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    with open(a.output+'.lock','w') as claim:
        fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(json.loads(Path(a.plan).read_text()),a.output,a.execute)
