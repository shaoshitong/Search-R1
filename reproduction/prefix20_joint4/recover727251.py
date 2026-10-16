"""Fresh zero-outer recovery after the preserved Ray startup failure727251."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha
from reproduction.prefix20_joint4.control import ASSETS,own_base,publish,verify_source,bootstrap,storage_admission
from reproduction.prefix20_joint4.recover726927 import relocate_global_storage
from reproduction.prefix20_joint4.retire_stalled727251 import BASE as FAILED,JOB,RUN,COMMIT

TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727251-20261017')


def terminal_failure():
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    assert os.getuid()==14090 and (FAILED/'job.id').read_text().strip()==JOB
    assert run['run_id']==RUN and run['source_commit']==COMMIT and run['completed_updates']==0
    result=subprocess.run(['sacct','-nP','-j',JOB,'--format=JobID,State,ExitCode,User,UID,Account,NodeList'],
                          capture_output=True,text=True,timeout=25)
    rows=[line.split('|') for line in result.stdout.splitlines() if line.strip()]
    main=next(row for row in rows if row[0]==JOB)
    assert result.returncode==0 and main[1:6]==['FAILED','1:0','jxu957','14090','jxu957']
    assert any(row[:3]==[JOB+'.extern','COMPLETED','0:0'] for row in rows)
    cleanup=json.loads((FAILED/'logs/cleanup-status.json').read_text())
    assert cleanup['run_id']==RUN and cleanup['cleaned'] is True
    ledger=json.loads((FAILED/'logs/process-ledger.json').read_text())
    assert ledger['run_id']==RUN and all(row['state']=='STOPPED' for row in ledger['jobs'].values())
    for name in ('actual-updates.jsonl','actual-attempts.jsonl'):
        path=FAILED/'logs'/name
        assert not path.exists() or path.stat().st_size==0
    assert not (FAILED/'artifacts/question-probe').exists()
    for rank in range(4):
        for kind in ('optimizer','scheduler'):
            path=FAILED/'logs'/(kind+'-rank-'+str(rank)+'.jsonl')
            assert not path.exists() or path.stat().st_size==0
    with (FAILED/'gpu-job.lock').open('r+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot=FAILED/'preparation/failed-ray-startup727251/snapshot.json'
    assert json.loads(snapshot.read_text())['fault_recovery'] is True
    return dict(job=JOB,run_id=RUN,source_commit=COMMIT,completed_outers=0,
        actor_optimizer_writes=0,question_probe_optimizer_writes=0,cleanup_actual=True,
        terminal_accounting=result.stdout,fault_snapshot_sha256=sha(snapshot),checked_unix=time.time())


def prepare(commit):
    os.umask(0o007);own_base(TARGET)
    claim=FAILED/'preparation/recovery727251-target-claim.json'
    assert not TARGET.exists() and not claim.exists(), 'Inspect existing recovery; never repeat'
    actual=terminal_failure();source,manifest=verify_source(FAILED,commit)
    # The global observation retains its original time; only these own trees
    # are measured here, within the already reserved cumulative4GiB allowance.
    bases=[Path('/data/user/jxu957/experiments')/name for name in (
        'qprobe-teacher-warmup200-prefix20-joint4-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery726927-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727000-20261016',
        'qprobe-teacher-warmup200-prefix20-joint4-recovery727050-20261016',
        FAILED.name)]
    measured=subprocess.run(['du','-s','-B1',*map(str,bases)],capture_output=True,text=True,timeout=60)
    assert measured.returncode==0 and not measured.stderr
    rows=[line.split('\t') for line in measured.stdout.splitlines()]
    assert [row[1] for row in rows]==list(map(str,bases))
    projected=sum(int(row[0]) for row in rows)+sum(size for size,_ in ASSETS.values())+2*sum(row['bytes'] for row in manifest['files'].values())+64*2**20
    assert projected<=4*2**30, 'Cumulative recovery reserve exceeded'
    publish(claim,dict(target=str(TARGET),source_commit=commit,original=actual,created_unix=time.time()))
    TARGET.mkdir();(TARGET/'preparation').mkdir()
    shutil.copytree(source,TARGET/'source'/commit);bootstrap(TARGET,commit)
    root=TARGET/'assets/prefix20';root.mkdir()
    for name,(size,digest) in ASSETS.items():
        old=FAILED/'assets/prefix20'/name;new=root/name
        assert not old.is_symlink() and old.stat().st_size==size and sha(old)==digest
        shutil.copyfile(old,new)
        with new.open('rb') as stream:os.fsync(stream.fileno())
        assert new.stat().st_size==size and sha(new)==digest
    storage=relocate_global_storage(FAILED,TARGET,additional_prelaunch_reserve_bytes=4*2**30)
    value=dict(status='PRESERVED_ZERO_OUTER_RAY_FAILURE_FRESH_TARGET_PREPARED',target=str(TARGET),
        source_commit=commit,recovery_of=JOB,original=actual,projected_cumulative_prelaunch_bytes=projected,
        storage_origin=storage,storage=storage_admission(TARGET),source_files=len(manifest['files']),
        training_submitted=False,old_runtime_untouched=True,checked_unix=time.time())
    publish(TARGET/'preparation/recovery-from727251.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.commit)))
