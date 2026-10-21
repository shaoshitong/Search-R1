"""Preserve terminal own CP1 restore failure; never cancel, train, or delete."""
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-20261021')
JOB = '745320'
assert os.getuid() == 14090
queue = subprocess.run(['squeue','-j',JOB,'-h','-o','%i|%T'],capture_output=True,text=True,timeout=12)
accounting = subprocess.run(['sacct','-j',JOB,'-n','-P','-o','JobIDRaw,State,ExitCode'],capture_output=True,text=True,timeout=12)
assert queue.returncode == accounting.returncode == 0 and not queue.stdout.strip()
assert JOB+'|FAILED|1:0' in accounting.stdout
run = json.loads((BASE/'logs/run-manifest.json').read_text())
assert run['source_commit'] == '0eed1015b0696101227fc8deccc9b132d9e3a207'
assert (BASE/'job.id').read_text().strip() == JOB
log = (BASE/'logs/search-g1-common8-training.log').read_text()
assert 'assert len(layers)==36' in log and 'varlen_execution.py' in log
assert not (BASE/'logs/actual-updates.jsonl').exists()
assert not (BASE/'artifacts/mixed-candidate-pools').exists()
assert not list((BASE/'logs').glob('optimizer-rank-*.jsonl'))
assert not (BASE/'logs/full-state-restored-workers.json').exists()
assert json.loads((BASE/'logs/cleanup-status.json').read_text())['cleaned'] is True
root = BASE/'preparation/failed-job-745320';root.mkdir(exist_ok=True)
receipt = root/'failure-evidence.json'
if receipt.exists():
    print(receipt.read_text())
else:
    archive = root/'logs-source-snapshot.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'x:gz') as bundle:
        bundle.add(BASE/'logs',arcname='logs')
        for name in ['search_g1/varlen_execution.py','search_g1/full_state_restore.py']:
            bundle.add(BASE/'repo'/name,arcname='runtime/'+name)
    value = dict(status='FAILED_OWN_JOB_PRESERVED_NO_NEW_UPDATES',job=JOB,checked_unix=time.time(),
        accounting=accounting.stdout,source_commit=run['source_commit'],run_id=run['run_id'],
        failure='Legacy36layer verification rejected actual28layer Qwen2.5-7B before Adam/scheduler/RNG restore',
        actual_new_optimizer_writes=0,actual_completed_new_outers=0,actual_full_cuda_restore_accepted=False,
        source_cp1_preserved=True,source_completed_outer=1,source_optimizer_writes_per_rank=20,
        archive_path=str(archive),archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        manual_deletions=0,healthy_external_service_job='745198',service_not_modified=True)
    with receipt.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(value))
