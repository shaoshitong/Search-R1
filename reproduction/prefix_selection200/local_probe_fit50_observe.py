"""Read-only progress of the short frozen-actor50 experiment, no state writer."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

from reproduction.h1008_qprobe.established_controller_transport import remote


REMOTE = r'''
import pathlib,json,subprocess,sys,time
b=pathlib.Path(BASE);sys.path.insert(0,str(b/'repo'))
from reproduction.common8.native_allocation import fields
def read(p):return json.loads(p.read_text())if p.exists()else None
r=subprocess.run(['scontrol','show','job',JOB,'-o'],text=True,capture_output=True,timeout=30)
root=b/'outers/outer-000051'
def log_info(p):
 with p.open('rb')as f:
  f.seek(max(0,p.stat().st_size-32000));tail=f.read().decode(errors='replace')
 return dict(bytes=p.stat().st_size,error_tail='\n'.join(line for line in tail.splitlines()[-50:]if any(w in line for w in ['Traceback','Error','Exception','CUDA','failed'])))
complete=read(b/'logs/pipeline-complete.json');ledger_counts=None
if complete:
 ledger_counts=[]
 for rank in range(8):
  p=b/'train'/('rank-%d-updates.jsonl'%rank);rows=[json.loads(line)for line in p.read_text().splitlines()]
  assert len(rows)==1000 and [v['actual_adam']for v in rows]==list(range(201,1201))
  for epoch in range(1,11):
   values=[v for v in rows if v['epoch']==epoch]
   assert [v['step']for v in values]==list(range(1,101)) and sum(v['local_trajectories']for v in values)==3200
   assert all(v['learning_rate']==1e-5 for v in values)
  ledger_counts.append(dict(rank=rank,new_actual_writes=1000,last_actual_adam=1200,ledger_sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest()))
value=dict(observed_unix=time.time(),base=str(b),native=fields(r.stdout)if r.returncode==0 else dict(unavailable=r.stderr),stage=read(b/'logs/pipeline-status.json'),train_progress=read(b/'train/progress.json'),latest=read(b/'train/latest.json'),complete=complete,cleanup=read(b/'logs/cleanup-status.json'),prefix=[read(root/'prefix'/('rank-%d-progress.json'%rank))for rank in range(8)],full=[read(root/'full'/('rank-%d-progress.json'%rank))for rank in range(8)],logs={p.name:log_info(p)for p in (b/'logs').glob('*.log')},epoch_metrics=[read(p)for p in sorted((b/'train').glob('epoch-*-metrics.json'))],rank_ledger_counts=ledger_counts,training_modified=False)
print(json.dumps(value))
'''


def once(state, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix=directory/('local-probe50-actual-'+tag)
    code=REMOTE.replace('BASE',repr(state['base'])).replace('JOB',repr(state['current_job_id']))
    if state.get('fixed_data_source_base'):
        code=code.replace("root=b/'outers/outer-000051'",
                          "root=pathlib.Path("+repr(state['fixed_data_source_base'])+")/'outers/outer-000051'")
    value=json.loads(remote(code,str(prefix)+'.transport.json'))
    path=Path(str(prefix)+'.json');path.write_text(json.dumps(value,indent=2)+'\n')
    return value, path


def watch(state_path, directory, interval=60):
    original=json.loads(Path(state_path).read_text())
    while True:
        state=json.loads(Path(state_path).read_text())
        if (any(state[k]!=original[k]for k in ('base','current_job_id','current_runtime_commit'))
                or state.get('execution_authorization_active')is not True):
            print(json.dumps(dict(status='SUPERSEDED_LOCAL_OBSERVER_NO_TRAINING_CHANGE')),flush=True);return
        try:
            value,path=once(state,directory)
        except Exception as error:
            print(json.dumps(dict(status='OBSERVATION_ERROR_NOT_NATIVE_EXIT',error=repr(error))),flush=True)
            time.sleep(interval);continue
        metrics=value['epoch_metrics'];native=value['native'].get('JobState','UNKNOWN')
        summary=dict(actual=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            native=native,stage=value['stage'],train_progress=value['train_progress'],
            complete_epoch=(value.get('latest')or{}).get('epoch',0),
            prefix_completed=sum((r or{}).get('completed_trajectories',0)for r in value['prefix']),
            full_completed=sum((r or{}).get('completed_trajectories',0)for r in value['full']),
            latest_metrics=metrics[-1]if metrics else None,
            complete=value['complete'],cleanup=value['cleanup'])
        Path(directory,'latest-small-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        if native in ('FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT'):
            return
        if native=='COMPLETED':
            return
        time.sleep(5 if value['complete'] or summary['complete_epoch']>=9 else interval)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--directory',required=True);p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    watch(a.state,a.directory,a.interval)
