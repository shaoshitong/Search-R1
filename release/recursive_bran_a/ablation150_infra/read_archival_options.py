"""Small metadata-only inventory of existing mounts and retained A state pairs."""
import json
import os
from pathlib import Path
import time

BASE = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
A = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')


def members(manifest, owner_root):
    raw = manifest.read_bytes()
    value = json.loads(raw)
    records = value.get('worker_states', []) + [value.get('driver', {})]
    rows = []
    for record in records:
        path = Path(record['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(owner_root):
            raise ValueError('Retained state payload escapes its own A namespace')
        try:
            stat = path.stat()
            rows.append(dict(path=str(path), bytes=stat.st_size, recorded_sha256=record['sha256']))
        except FileNotFoundError:
            rows.append(dict(path=str(path), missing=True))
    return dict(path=str(manifest), status=value.get('status'),
        identity=value.get('identity'), phase=value.get('boundary_phase'),
        snapshot_kind=value.get('snapshot_kind'), hf_checkpoint=value.get('hf_checkpoint'),
        members=rows, actual_payload_bytes=sum(r.get('bytes', 0) for r in rows),
        payload_SHA_or_tensor_read=False)


def main():
    study = json.loads((BASE/'study-progress.json').read_text())
    assert study['source_commit'] == '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        target = Path(left.split()[4])
        filesystem = right.split()
        if filesystem[0] not in ('nfs', 'nfs4') or not str(target).startswith('/mnt/shared/workspace/'):
            continue
        fs = os.statvfs(target)
        mounts.append(dict(path=str(target), source=filesystem[1], fstype=filesystem[0],
                           free_bytes=fs.f_bavail*fs.f_frsize))
    roots = [A, A/'recovery-from100-20261013']
    groups = []
    for root in roots:
        canonical = root.resolve()
        rows = []
        for directory in ('checkpoints', 'native-rolling'):
            base = root/directory
            if not base.exists():
                continue
            for manifest in sorted(base.glob('**/manifest.json')):
                # Only native checkpoint metadata, never model shards or tensors.
                if len(manifest.relative_to(root).parts) > 7 or manifest.is_symlink():
                    continue
                try:
                    rows.append(members(manifest, canonical))
                except (ValueError, KeyError, OSError) as error:
                    rows.append(dict(path=str(manifest), error=str(error)))
        groups.append(dict(root=str(root), native_states=rows))
    print(json.dumps(dict(time=time.time(), current_group=study.get('current'),
        existing_mounted_same_cluster_storage=mounts, retained_A_metadata=groups,
        deletion_executed=False, transfer_executed=False, training_action='NONE',
        scope='Capacity discovery only; retained A artifacts are not cleanup authorized by this tool'),
        allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
