"""Seal the actual pre-model directory failure; never stop the productive V6."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

b = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
root = b / 'preparation/relay-consumer10-control-v1/v1-terminal'
root.mkdir(exist_ok=True)
complete = root / 'complete.json'
if complete.exists():
    print(json.dumps(dict(status='EXISTING_TERMINAL_PROOF_READ_NO_REPLAY', proof=str(complete),
                          sha256=hashlib.sha256(complete.read_bytes()).hexdigest())))
else:
    unit = 'sst-q7b-g5-consumer10-v1-70188b8e50fd.service'
    attempts = list((b / 'pool/jobs/job-relay10-2316020').glob('attempt-*'))
    assert len(attempts) == 1
    job = attempts[0]
    failure = json.loads((job / 'failure.json').read_text())
    assert failure['error_type'] == 'FileNotFoundError' and 'sampler-launches/launch-00' in failure['error']
    assert not (job / 'progress.json').exists() and not (job / 'sampler-launches').exists()
    child = json.loads((job / 'children.json').read_text())
    pids = [2316020, *child['own_process_groups']]
    release = []
    for ordinal in range(2):
        text = subprocess.check_output(['systemctl','show',unit,'-p','MainPID','-p','ControlGroup',
                    '-p','ActiveState','-p','Result','-p','ExecMainStatus'], text=True)
        props = dict(x.split('=',1) for x in text.splitlines() if '=' in x)
        assert props['MainPID'] == '0' and props['ActiveState'] == 'failed' and props['ExecMainStatus'] == '1'
        groups = []
        for p in Path('/sys/fs/cgroup').glob('system.slice/' + unit + '/cgroup.procs'):
            groups += [int(x) for x in p.read_text().split()]
        remaining = [p for p in pids if Path('/proc/%d' % p).exists()]
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
        gpu = sorted(set(pids) & {int(x.strip()) for x in apps.splitlines() if x.strip().isdigit()})
        assert not groups and not remaining and not gpu
        release.append(dict(epoch=time.time(), unit=text, remaining_known_pids=remaining,
                            remaining_cgroup_pids=groups, own_gpu_pids=gpu))
        if ordinal == 0:
            time.sleep(3)
    paths = [p for p in job.rglob('*') if p.is_file()]
    paths += [p for p in (b / 'relay-consumer10/launch/native-2316020').rglob('*') if p.is_file()]
    records = []
    for p in paths:
        before = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''):
                h.update(chunk)
        after = p.stat()
        assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        records.append(dict(path=str(p), dev=before.st_dev, inode=before.st_ino,
                            bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=h.hexdigest()))
    value = dict(status='OWN_CONSUMER_V1_TERMINAL_FULL_SHA_RELEASE_PASS', unit=unit, controller_pid=2316020,
        failure=failure, original_files=records, original_files_count=len(records),
        original_bytes=sum(x['bytes'] for x in records), actual_release=release,
        real_gpu_model_initialized=False, new_complete_batches=0, original_upstream_v6_not_stopped=True,
        foreign_processes_signalled=False, epoch=time.time())
    with complete.open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=value['status'],proof=str(complete),sha256=hashlib.sha256(complete.read_bytes()).hexdigest(),
        original_files=len(records),original_bytes=value['original_bytes'],release_observations=2,new_complete_batches=0)))
