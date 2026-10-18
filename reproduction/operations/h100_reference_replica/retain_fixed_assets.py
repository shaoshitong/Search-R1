"""Retain read-only fixed inputs on NAS18; never alter a live mount or trainer.

Run once with a sealed JSON spec. Existing intent prevents duplicate copying.
Old immutable full-SHA backup payloads may be hardlinked with explicit provenance;
new copies are streamed and independently read back in full. No tensors on Mac.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def identity(p):
    st = p.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def run(spec_path):
    raw = Path(spec_path).read_bytes()
    spec = json.loads(raw)
    root = Path(spec['destination'])
    root.mkdir(parents=True, exist_ok=True)
    prefix = spec.get('record_prefix', '')
    assert not prefix or prefix.replace('-', '').isalnum()
    def record(name):
        return root / (prefix + name)
    if record('RESULT.json').exists():
        print(record('RESULT.json').read_text(), flush=True)
        return
    with record('INTENT.json').open('x') as f:
        json.dump({'spec_sha256': hashlib.sha256(raw).hexdigest(),
                   'started_unix': time.time(), 'pid': os.getpid()}, f)
        f.flush()
        os.fsync(f.fileno())
    rows = []
    if spec.get('prior_completed_manifest'):
        prior = spec['prior_completed_manifest']
        data = Path(prior['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == prior['sha256']
        rows = json.loads(data)
        for row in rows:
            dest = Path(row['target'])
            assert dest.is_file() and not dest.is_symlink()
            if row['kind'] == 'NEW_COPY_SOURCE_TARGET_FULL_SHA_PASS':
                assert dest.stat().st_size == row['bytes']
            else:
                assert identity(dest) == row['identity']
    copied = 0
    rate = spec['bytes_per_second']
    start = time.monotonic()

    def progress(stage, row=None, done=0):
        atomic(record('PROGRESS.json'), dict(stage=stage, current=row,
            current_bytes=done, copied_bytes=copied, complete_files=len(rows),
            checked_unix=time.time(), pid=os.getpid()))

    for r in spec['reused_backup_payloads']:
        proof_path = Path(r['proof_path'])
        proof_raw = proof_path.read_bytes()
        assert hashlib.sha256(proof_raw).hexdigest() == r['proof_sha256']
        proof = json.loads(proof_raw)
        matches = [x for x in proof['files'] if x['relative_path'] == r['relative_path']]
        assert len(matches) == 1
        proven = matches[0]
        assert proven['source_full_sha256'] == proven['destination_full_sha256']
        p = proof_path.parent / 'checkpoint' / r['relative_path']
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == proven['bytes']
        dest = root / r['target_relative']
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, dest)
        assert identity(p) == identity(dest)
        rows.append(dict(kind='HARDLINK_TO_EXISTING_IMMUTABLE_FULL_SHA_BACKUP',
            target=str(dest), retained_payload=str(p), identity=identity(dest),
            proven_sha256=proven['destination_full_sha256'],
            full_sha_proof=str(proof_path), full_sha_proof_sha256=r['proof_sha256'],
            new_full_hash_performed=False))
        progress('REUSED_SEALED_PAYLOAD', r['target_relative'])

    for r in spec['files']:
        source = Path(r['source'])
        assert source.is_file() and not source.is_symlink()
        assert not source.name.endswith('.lock'), 'Process lock files are not retained data'
        before = identity(source)
        assert before == r['identity'], (str(source), before, r['identity'])
        dest = root / r['target_relative']
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + '.copy-partial')
        h = hashlib.sha256()
        done = 0
        with source.open('rb') as src, temp.open('xb') as out:
            while True:
                chunk = src.read(4 * 1024**2)
                if not chunk:
                    break
                st = os.statvfs(root)
                if st.f_bavail * st.f_frsize < 128 * 1024**3 + len(chunk):
                    raise OSError('Retained target ordinary free space below 128GiB')
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                copied += len(chunk)
                progress('COPYING', r['target_relative'], done)
                delay = copied / rate - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
            out.flush()
            os.fsync(out.fileno())
        assert identity(source) == before
        if r.get('expected_sha256'):
            assert h.hexdigest() == r['expected_sha256']
        progress('INDEPENDENT_DESTINATION_FULL_READ', r['target_relative'], 0)
        target = hashlib.sha256()
        read_bytes = 0
        with temp.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024**2), b''):
                target.update(chunk)
                read_bytes += len(chunk)
                progress('INDEPENDENT_DESTINATION_FULL_READ', r['target_relative'], read_bytes)
        assert target.hexdigest() == h.hexdigest() and read_bytes == done
        os.replace(temp, dest)
        os.utime(dest, ns=(source.stat().st_atime_ns, before[-1]))
        rows.append(dict(kind='NEW_COPY_SOURCE_TARGET_FULL_SHA_PASS',
            source=str(source), target=str(dest), bytes=done,
            source_identity=before, source_sha256=h.hexdigest(),
            destination_sha256=target.hexdigest()))
        atomic(record('completed-files.json'), rows)
        progress('FILE_COMPLETE', r['target_relative'], done)
    atomic(record('RESULT.json'), dict(status='FIXED_INPUT_RETENTION_COMPLETE',
        finished_unix=time.time(), files=rows, source_files=len(spec['files']),
        copied_bytes=copied, reused_payloads=len(spec['reused_backup_payloads']),
        live_training_changed=False, live_mounts_changed=False,
        bootstrap_independent_of_source8=False, model_bytes_via_Mac=False))
    progress('COMPLETE')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    a = p.parse_args()
    run(a.spec)
