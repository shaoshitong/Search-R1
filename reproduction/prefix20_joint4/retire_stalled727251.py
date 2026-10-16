"""Preserve the observed zero-outer Ray failure, then interrupt only its batch.

This is fault recovery for one bound job, never a user pause or a general kill.
The batch pipeline owns its children and performs their existing cleanup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from reproduction.prefix20_joint4.control import publish

BASE=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727148-20261016')
JOB='727251'
RUN='a79333e9574742a38f922974e30203e9'
COMMIT='78074b669a0d47648ad7c5b2a687159af00cd090'
BOOT='2ea81049-42ca-4fa4-bdd9-b142704d81e0'
ROOT=BASE/'preparation/failed-ray-startup727251'


def digest(raw):return hashlib.sha256(raw).hexdigest()


def bound_run():
    run=json.loads((BASE/'logs/run-manifest.json').read_text())
    assert os.getuid()==14090 and (BASE/'job.id').read_text().strip()==JOB
    assert run['run_id']==RUN and run['source_commit']==COMMIT and run['completed_updates']==0
    assert run['training_world_size']==4
    for name in ('actual-updates.jsonl','actual-attempts.jsonl'):
        path=BASE/'logs'/name
        assert not path.exists() or path.stat().st_size==0
    assert not (BASE/'artifacts/question-probe').exists()
    return run


def snapshot():
    bound_run()
    assert socket.gethostname()=='ACD1-25' and os.environ['SLURM_JOB_ID']==JOB
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
    identities=[]
    for pid,birth in ((1000216,797489458),(1177308,797588309)):
        p=Path('/proc')/str(pid);stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        assert p.stat().st_uid==14090 and int(stat[19])==birth and stat[0]!='Z'
        assert '/job_727251/step_batch' in p.joinpath('cgroup').read_text()
        command=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert str(BASE) in command or pid==1177308
        identities.append(dict(pid=pid,ppid=int(stat[1]),start_ticks=birth,sid=int(stat[3]),command=command,
                               cgroup=p.joinpath('cgroup').read_text()))
    batch=subprocess.run(['scontrol','listpids',JOB+'.batch'],capture_output=True,text=True,timeout=20)
    ray=Path('/tmp/g1m-a79333e95747/session_2026-10-16_23-36-38_732633_1177308/logs')
    core=next(ray.glob('python-core-driver-*1177308.log'))
    assert core.stat().st_size==228 and core.stat().st_mtime_ns==1792165001946224936
    training=BASE/'logs/search-g1-common8-training.log'
    assert training.stat().st_size==77
    ROOT.mkdir(exist_ok=False);files={}
    paths=[core,ray/'runtime_env_agent.err',ray/'raylet.err',ray/'raylet.out',ray/'gcs_server.out',
           training,BASE/'logs/process-ledger.json',BASE/'logs/run-manifest.json',
           BASE/'logs/pipeline-status.json']
    for index,path in enumerate(paths):
        assert path.is_file() and not path.is_symlink()
        st=path.stat();assert st.st_size<32*2**20
        with path.open('rb') as stream:raw=stream.read(st.st_size)
        name=str(index)+'-'+path.name;target=ROOT/name
        with target.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
        files[name]=dict(source=str(path),bytes=len(raw),sha256=digest(raw),
                        source_size_at_open=st.st_size,source_mtime_ns=st.st_mtime_ns,
                        snapshot_semantics='bounded initial opened-size prefix; later append permitted')
    value=dict(job=JOB,run_id=RUN,training_commit=COMMIT,node='ACD1-25',boot_id=BOOT,
        identities=identities,batch_pid_listing=dict(returncode=batch.returncode,stdout=batch.stdout,stderr=batch.stderr),
        files=files,completed_outers=0,training_progress_accepted=False,
        reason='Observed Ray thread exhaustion; soft8192 repair passed but two subsequent driver/outer observations remained unchanged',
        permanent_deadlock_proven=False,fault_recovery=True,user_pause=False,checked_unix=time.time())
    publish(ROOT/'snapshot.json',value)
    return value


def retire():
    bound_run();snapshot=json.loads((ROOT/'snapshot.json').read_text())
    assert snapshot['job']==JOB and snapshot['run_id']==RUN and time.time()-snapshot['checked_unix']<300
    for name,row in snapshot['files'].items():assert digest((ROOT/name).read_bytes())==row['sha256']
    from reproduction.common8.native_allocation import fields
    result=subprocess.run(['scontrol','show','job','-d',JOB],capture_output=True,text=True,timeout=25)
    actual=fields(result.stdout)
    assert result.returncode==0 and actual['JobState']=='RUNNING' and actual['UserId']=='jxu957(14090)'
    assert actual['BatchHost']=='ACD1-25' and actual['WorkDir']==str(BASE/'repo')
    assert actual['Command']==str(BASE/'source'/COMMIT/'reproduction/prefix20_joint4/train.sh')
    command=['scancel','--signal=TERM','--batch','--user=jxu957','--account=jxu957',
             '--state=RUNNING','--nodelist=ACD1-25',JOB]
    publish(ROOT/'fault-interrupt-claim.json',dict(command=command,actual_job=actual,
        snapshot_sha256=digest((ROOT/'snapshot.json').read_bytes()),fault_recovery=True,user_pause=False,
        goal_stays_active=True,source_backup_required_before_execution=True,created_unix=time.time()))
    env={key:value for key,value in os.environ.items() if not key.startswith('SCANCEL_')}
    try:
        done=subprocess.run(command,capture_output=True,text=True,env=env,timeout=25)
        value=dict(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,
            disposition='SIGNAL_REQUEST_RETURNED' if done.returncode==0 else 'REJECTED_INSPECT_NO_RETRY')
    except subprocess.TimeoutExpired:
        value=dict(returncode=124,stdout='',stderr='',disposition='UNKNOWN_INSPECT_EXISTING_CLAIM_NO_RETRY')
    value.update(command=command,actual_terminal_accepted=False,checked_unix=time.time())
    publish(ROOT/'fault-interrupt-result.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('snapshot','retire'))
    args=parser.parse_args();value=snapshot() if args.mode=='snapshot' else retire();print(json.dumps(value))
    raise SystemExit(value.get('returncode',0))
