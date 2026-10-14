"""Create one tiny probe in an existing same-cluster archival mount; no copies."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

STUDY = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
MOUNT = Path('/mnt/shared/workspace/training_data_node_10')
ARCHIVE = MOUNT/'shaoshitong/recursive-ablation150-archive-20261014'


def main():
    study = json.loads((STUDY/'study-progress.json').read_text())
    assert study['source_commit'] == '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
    fs = os.statvfs(MOUNT)
    candidates = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, filesystem = left.split(), right.split()
        if fields[4] == str(MOUNT) and filesystem[0] == 'nfs4':
            candidates.append(dict(mountpoint=str(MOUNT), fstype=filesystem[0], source=filesystem[1]))
    assert len(candidates) == 1 and candidates[0]['source'] == 'h100-10:/mnt/data'
    assert fs.f_bavail * fs.f_frsize > 2_100_000_000_000
    for path in (ARCHIVE, *ARCHIVE.parents):
        assert not path.is_symlink()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    marker = ARCHIVE/'study-namespace.json'
    identity = dict(study_root=str(STUDY), scientific_source_commit=study['source_commit'],
                    scope='Own closed ablation150 checkpoint archive; no GPU resource on storage server')
    if marker.exists():
        assert json.loads(marker.read_text()) == identity
    else:
        with marker.open('x') as stream:
            stream.write(json.dumps(identity, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
    probe = ARCHIVE/('.readiness-probe-'+uuid.uuid4().hex)
    data = os.urandom(64)
    with probe.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    assert probe.read_bytes() == data
    sha = hashlib.sha256(data).hexdigest()
    record = dict(protocol='ablation150_existing_same_cluster_storage_pending_server_v1',
        ready=False, study_root=str(STUDY), archive_root=str(ARCHIVE), mount=candidates[0],
        probe=dict(path=str(probe), server_path=str(Path('/mnt/data')/probe.relative_to(MOUNT)),
                   bytes=64, sha256=sha, local_sha256=sha, write_rc=0),
        retention_until=None, verified_time=time.time(), actual_free_bytes=fs.f_bavail*fs.f_frsize,
        heavy_asset_transfer_executed=False, training_action='NONE')
    path = STUDY/'analysis/archive-storage-pending-server.json'
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
