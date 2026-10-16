"""Once local driver for explicit JXU grants and MOU-owned DU confirmations.

Uses the existing RAM credential bridge on the pinned login kernel. It never
submits or controls training/archive jobs; a completed driver is not final
goal acceptance. Unknown owner-start outcomes never authorize a replay.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from reproduction.hpc3_qprobe import local_account_bridge as bridge

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015')
RUN='70827e85fe584f0c84453aae907918a4'
HOST='10.120.48.31'
IDENTITY=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]')
MODULE='reproduction.hpc3_qprobe.owner_partition_storage_v2'
PROTOCOL='actual-full-owner-partition-du-v2'
FILES=('reproduction/hpc3_qprobe/local_owner_partition_driver.py','reproduction/hpc3_qprobe/local_account_bridge.py')


def need(ok,message):
    if not ok:raise ValueError('LOCAL_OWNER_DRIVER: '+message)


def publish(path,value):
    with Path(path).open('x') as stream:
        os.fchmod(stream.fileno(),0o600);json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def source_guard(commit):
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None,'exact immutable commit required')
    repository=Path(__file__).resolve().parents[2]
    for name in FILES:
        raw=subprocess.check_output(['git','show',commit+':'+name],cwd=repository)
        need((repository/name).read_bytes()==raw,'local driver/bridge differs from pinned published source')
    return repository


def command(account,mode,commit,*,step=None,request=None):
    need(account in ('jxu957','mou176'),'only the authorized owner pair')
    need((account,mode) in (('jxu957','inspect-storage'),('jxu957','authorize-owner'),
                          ('mou176','owner-start'),('mou176','owner-confirm')),'account/action role differs')
    repo=ROOT/'source'/commit
    interpreter=str(BASE/'env/bin/python') if account=='jxu957' else '/usr/bin/python3'
    args=['env','QPROBE_SOURCE='+str(repo),'PYTHONPATH='+str(repo),'CUDA_VISIBLE_DEVICES=',
          'PYTHONDONTWRITEBYTECODE=1']+[key+'=1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
          'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')]
    args += [interpreter,'-m',MODULE,mode,'--commit',commit]
    if mode=='inspect-storage':
        need(step in (150,200) and request is None,'explicit inspection step required');args+=['--step',str(step)]
    else:
        path=Path(request);expected=ROOT/'final-control'/RUN
        need(path.is_absolute() and path.parent.parent.parent==expected and
             path.parent.parent.name in ('archive-150','archive-200') and
             re.fullmatch('owner-storage-attempt-(?:[1-9]|1[0-6])',path.parent.name) is not None and
             path.name=='request.json','request outside exact bounded scope')
        args+=['--request',str(path)]
    return ' '.join(shlex.quote(item) for item in args)


class Driver:
    def __init__(self,commit,control_work,directory):
        self.commit=commit;self.control_work=Path(control_work);self.directory=Path(directory)
        self.started=set();self.counter=0

    def call(self,account,mode,**kwargs):
        cmd=command(account,mode,self.commit,**kwargs);self.counter+=1
        result=bridge.run(account,cmd,'',str(self.control_work),login_host=HOST,
                          expected_host_identity=IDENTITY,observation_timeout=180)
        need(len(result.stdout.encode())+len(result.stderr.encode())<2**20,'unexpected unbounded metadata output')
        record=dict(account=account,mode=mode,command=cmd,returncode=result.returncode,
                    stdout=result.stdout,stderr=result.stderr,checked_unix=time.time(),**kwargs)
        publish(self.directory/('%06d-%s.json'%(self.counter,mode)),record)
        need(result.returncode==0,'bridge/action outcome unknown or failed; retained evidence, never replay owner-start')
        return json.loads(result.stdout)

    def act(self,step,state):
        need(state['source']['commit']==self.commit and state['protocol']==PROTOCOL and state['step']==step and
             state['training_actions']==state['permission_changes']==0 and state['goal_complete'] is False,
             'remote inspected action source/scope differs')
        action=state['action']
        if action=='ARCHIVE_COMPLETED':return True
        need(action not in ('ARCHIVE_FAILED','UNKNOWN_ARCHIVE_ADMISSION','UNKNOWN_OWNER_ADMISSION'),
             'actual failed or unknown original control; inspect retained evidence')
        if action in ('AUTHORIZE_OWNER','START_OWNER','CONFIRM_OWNER'):
            proof=state['request'];path=proof['path'];key=proof['sha256']
            if action=='AUTHORIZE_OWNER':self.call('jxu957','authorize-owner',request=path)
            elif action=='CONFIRM_OWNER':self.call('mou176','owner-confirm',request=path)
            else:
                need(key not in self.started,'owner-start was already attempted; unknown outcome forbids replay')
                intent=self.directory/('owner-start-'+key+'.json')
                publish(intent,dict(request=proof,source_commit=self.commit,created_unix=time.time(),
                                    unknown_outcome_must_not_replay=True))
                self.started.add(key);self.call('mou176','owner-start',request=path)
        else:need(action in ('WAIT_FOR_ARCHIVE_CLAIM','WAIT_FOR_STORAGE_REQUEST','WAIT_FOR_NEXT_REQUEST',
                            'WAIT_FOR_JXU_ONLY_SCAN','OWNER_CONFIRMED'),'unrecognized remote action')
        return False

    def run(self,poll_seconds=20):
        source_guard(self.commit);need(5<=poll_seconds<=60,'bounded polling interval required')
        need(self.directory.is_absolute() and self.control_work.is_dir(),'explicit local small-audit paths required')
        self.directory.mkdir(mode=0o700)
        with (self.directory/'driver.lock').open('x') as lock:
            os.fchmod(lock.fileno(),0o600);fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            publish(self.directory/'claim.json',dict(source_commit=self.commit,actual_local_pid=os.getpid(),
                command_role='local_dual_account_storage_only',host=HOST,expected_host_identity=IDENTITY,
                created_unix=time.time(),steps=[150,200],training_actions=0,permission_changes=0,goal_complete=False))
            last=None
            try:
                for step in (150,200):
                    while True:
                        state=self.call('jxu957','inspect-storage',step=step)
                        marker=(step,state['action'],state.get('request',{}).get('sha256'))
                        if marker!=last:
                            print(json.dumps(dict(step=step,action=state['action'],request=state.get('request'),
                                                  checked_unix=time.time())),flush=True);last=marker
                        if self.act(step,state):break
                        time.sleep(poll_seconds)
                publish(self.directory/'result.json',dict(status='ACTUAL_STORAGE_DRIVER_DUTIES_COMPLETED',
                    exit_code=0,completed_unix=time.time(),steps=[150,200],goal_complete=False));return 0
            except BaseException as error:
                publish(self.directory/'result.json',dict(status='LOCAL_DRIVER_STOPPED_PRESERVED_NO_REPLAY',
                    exit_code=1,error=repr(error),completed_unix=time.time(),goal_complete=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True)
    p.add_argument('--control-work',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=20);a=p.parse_args()
    raise SystemExit(Driver(a.commit,a.control_work,a.directory).run(a.poll_seconds))
