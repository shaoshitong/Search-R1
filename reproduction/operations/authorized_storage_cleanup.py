"""Bounded parallel removal of explicitly inventoried generated payloads.

Requires an explicit JSON plan. Never follows symlinks, removes source/config
files, or traverses a protected path. Records small progress and error receipts.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import stat
import threading
import time

PAYLOAD_SUFFIXES = ('.pt', '.pth', '.safetensors', '.bin', '.npy', '.npz',
                    '.pkl', '.pickle', '.arrow', '.parquet', '.jsonl',
                    '.jsonl.gz', '.tar', '.tar.gz', '.tgz', '.zip',
                    '.h5', '.hdf5', '.tensor', '.fvecs', '.ivf', '.index')
SKIP_DIRS = {'.git', 'code', 'repo', 'env', 'environment', 'node_modules',
             'site-packages', '.ssh'}


def within(path, parent):
    return path == parent or path.startswith(parent.rstrip('/') + '/')


def payload(path, size):
    name = path.name.lower()
    if any(x in name for x in ('credential', 'private-key', 'id_rsa', 'id_ed25519')):
        return False
    return name.endswith(PAYLOAD_SUFFIXES) or (name.endswith('.json') and size > 1048576)


def run(plan, output, execute=False):
    roots = [os.path.realpath(x) for x in plan['allowed_roots']]
    protected = [os.path.realpath(x) for x in plan['protected_paths']]
    targets = plan['targets']
    assert len(targets) == len(set(targets))
    for target in targets:
        assert os.path.isabs(target) and os.path.realpath(target) == target
        assert any(within(target, root) and target != root for root in roots)
        assert not any(within(target, keep) for keep in protected), target
        assert not any(within(target, other) for other in targets if other != target)
    lock = threading.Lock()
    progress, errors = {}, []
    started = time.time()

    def keep(path):
        # Targets are canonical and traversal never follows symlinks.
        real = os.path.abspath(path)
        return any(within(real, p) for p in protected)

    def worker(target):
        row = dict(files_removed=0, eligible_files=0, removed_file_bytes=0,
                   freed_blocks_lower_bound=0, kept_files=0, errors=0, done=False)
        local_errors = []

        def error(exc):
            local_errors.append({'path':getattr(exc, 'filename', target), 'error':str(exc)})
            row['errors'] += 1

        for directory, dirs, files in os.walk(target, topdown=True, followlinks=False, onerror=error):
            if keep(directory):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                       and not os.path.islink(os.path.join(directory, d))
                       and not keep(os.path.join(directory, d))]
            for name in files:
                path = Path(directory) / name
                try:
                    s = path.lstat()
                    if not stat.S_ISREG(s.st_mode) or keep(str(path)) or not payload(path, s.st_size):
                        row['kept_files'] += 1
                        continue
                    row['eligible_files'] += 1
                    if execute:
                        path.unlink()
                        row['files_removed'] += 1
                        row['removed_file_bytes'] += s.st_size
                        if s.st_nlink == 1:
                            row['freed_blocks_lower_bound'] += s.st_blocks * 512
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    error(exc)
            with lock:
                progress[target] = dict(row)
        row['done'] = True
        with lock:
            progress[target] = dict(row)
            errors.extend(local_errors)

    def save(finished):
        with lock:
            rows = dict(progress)
            failed = list(errors)
        result = dict(started_unix=started, checked_unix=time.time(),
                      execute=execute, finished=finished, workers=8,
                      targets_total=len(targets), targets_completed=sum(r['done'] for r in rows.values()),
                      files_removed=sum(r['files_removed'] for r in rows.values()),
                      freed_blocks_lower_bound=sum(r['freed_blocks_lower_bound'] for r in rows.values()),
                      removed_file_bytes=sum(r['removed_file_bytes'] for r in rows.values()),
                      errors=failed, targets=rows, protected_paths=plan['protected_paths'])
        tmp = output + '.tmp'
        Path(tmp).write_text(json.dumps(result, ensure_ascii=False))
        os.replace(tmp, output)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(worker, target) for target in targets}
        while pending:
            completed, pending = concurrent.futures.wait(pending, timeout=5,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                future.result()
            save(not pending)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    with open(args.output + '.lock', 'w') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(json.loads(Path(args.plan).read_text()), args.output, args.execute)
