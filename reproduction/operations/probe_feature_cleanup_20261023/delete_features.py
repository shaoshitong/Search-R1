"""Delete explicitly authorized retired feature payloads, preserving all weights.

Run remotely after inventory.py completes. No model/tensor deserialization.
Full per-file audit stays remote; stdout contains only a small summary.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

AUDIT = Path('/data/user/jxu957/experiments/probe-feature-cleanup-20261023')
ROOT = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
NAMES = {'question.bin', 'delta.bin', 'policy.bin', 'prototype.bin', 'index.jsonl'}


def unchanged(path, record):
    stat = path.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (
        record['inode'], record['bytes'], record['mtime_ns'])


def main():
    lock = (AUDIT / 'delete.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (AUDIT / 'deleted.jsonl').exists():
        raise RuntimeError('Prior deletion audit exists; inspect it before any retry')
    raw = (AUDIT / 'inventory.json').read_bytes()
    inventory = json.loads(raw)
    assert inventory['status'] == 'FEATURE_PAYLOAD_INVENTORY_COMPLETE'
    assert inventory['root'] == str(ROOT)
    queue = subprocess.check_output(
        ['squeue', '-u', 'mou176,jxu957', '-h', '-o', '%i'], universal_newlines=True)
    assert set(queue.split()) <= {'751011', '751884'}, 'Inspect new jobs before deleting shared features'
    for record in inventory['protected_weights_and_pt']:
        assert unchanged(Path(record['path']), record), 'A protected weight changed'
    files = inventory['files']
    assert len({r['path'] for r in files}) == len(files)
    manifests = {}
    for record in files:
        path = Path(record['path'])
        assert path.name in NAMES and not path.is_symlink()
        assert os.path.commonpath([str(ROOT), str(path.resolve())]) == str(ROOT)
        assert unchanged(path, record), 'Feature changed since inventory'
        manifest = Path(record['manifest'])
        assert manifest.parent == path.parent
        if str(manifest) not in manifests:
            value = json.loads(manifest.read_text())
            assert value['status'] in ('LOSSLESS_NATIVE_FEATURES_COMPLETE',
                                       'LOSSLESS_HF_REPLAY_FEATURES_COMPLETE')
            manifests[str(manifest)] = value
    summary = dict(status='DELETING_AUTHORIZED_RETIRED_FEATURE_PAYLOADS',
                   inventory_sha256=hashlib.sha256(raw).hexdigest(),
                   started=time.time(), deleted_files=0, deleted_logical_bytes=0,
                   protected_weights_and_pt=len(inventory['protected_weights_and_pt']),
                   expected_files=len(files), source_manifests_preserved=True)
    (AUDIT / 'delete-controller.json').write_text(json.dumps(dict(pid=os.getpid(), **summary)))
    last = 0
    with (AUDIT / 'deleted.jsonl').open('x') as audit:
        for record in files:
            path = Path(record['path'])
            assert unchanged(path, record) and not path.is_symlink()
            stat = path.stat()
            event = dict(record, device=stat.st_dev, nlink_before=stat.st_nlink)
            path.unlink()
            audit.write(json.dumps(event) + '\n')
            summary['deleted_files'] += 1
            summary['deleted_logical_bytes'] += record['bytes']
            if time.time() - last > 5:
                audit.flush()
                (AUDIT / 'delete-progress.json').write_text(json.dumps(dict(time=time.time(), **summary)))
                last = time.time()
        audit.flush()
        os.fsync(audit.fileno())
    for record in files:
        assert not Path(record['path']).exists()
    for record in inventory['protected_weights_and_pt']:
        assert unchanged(Path(record['path']), record)
    for manifest in manifests:
        assert Path(manifest).exists()
    summary.update(status='AUTHORIZED_FEATURE_PAYLOAD_CLEANUP_VERIFIED',
                   finished=time.time(), all_listed_payloads_absent=True,
                   all_protected_weights_metadata_unchanged=True,
                   full_weights_rehashed=False,
                   logical_bytes_not_claimed_as_unique_physical_space=True)
    (AUDIT / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
