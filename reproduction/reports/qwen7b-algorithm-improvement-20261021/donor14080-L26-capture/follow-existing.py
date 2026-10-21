"""Finite read-only observer for the single existing immutable donor14080 job."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[3];HELPER=R/'reproduction/operations/qwen7b_warmup/deploy_donor14080_l26.py'
def compact(s):
 out={'epoch':s['epoch'],'same_name_jobs':s['same_name_jobs'],'sacct':s.get('main_sacct'),'shards':{}}
 for i in range(4):
  x=s[str(i)];p=x.get('progress.json',{});d=x.get('complete.json',{});out['shards'][str(i)]={**{'chunks':x['chunks']},**{k:p[k]for k in('trajectories','expected_trajectories','completed_chunks')if k in p},'complete':d.get('ref'),'failure':x.get('failure.json')}
 for key in('bank-acceptance-progress.json','bank/complete.json'):
  if key in s:out[key]=s[key]
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--helper-commit');p.add_argument('--max-seconds',type=int,default=9000);a=p.parse_args();lock=(HERE/'follow-existing.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);end=time.monotonic()+a.max_seconds;last=None;seq=0
 (HERE/'observer-handle.json').write_text(json.dumps(dict(pid=os.getpid(),commit=a.commit,helper_commit=a.helper_commit or a.commit,start_epoch=time.time(),max_seconds=a.max_seconds,remote_actions='status only; never submit or mutate active source'))+'\n')
 while time.monotonic()<end:
  r=subprocess.run(['python3',str(HELPER),'--action','status','--commit',a.helper_commit or a.commit,'--runtime-commit',a.commit],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=75)
  if r.returncode:raise RuntimeError(r.stderr[-1800:]+r.stdout[-1800:])
  s=json.loads(r.stdout.splitlines()[-1]);v=compact(s);seq+=1;v['read_number']=seq;(HERE/'live-latest.json').write_text(json.dumps(v,indent=2)+'\n')
  with(HERE/'actual-progress.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
  state=json.dumps({k:v[k]for k in('same_name_jobs','sacct','shards')},sort_keys=True)
  if state!=last or 'bank/complete.json'in v:print(json.dumps(v),flush=True);last=state
  if any(v['shards'][str(i)]['failure']for i in range(4)):raise RuntimeError('Actual producer failure preserved; inspect immutable log, never restart blindly')
  sacct=s.get('main_sacct','');terminal=any('|'+x+'|'in sacct for x in('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'))
  if terminal:
   if 'bank/complete.json'not in s or '|COMPLETED|0:0|'not in sacct:raise RuntimeError('Terminal job without accepted bank/normal exit')
   print(json.dumps(dict(status='FULL14080_BANK_AND_NORMAL_JOB_EXIT_OBSERVED',bank=s['bank/complete.json']['ref'],epoch=time.time())),flush=True);break
  time.sleep(45)
 else:print(json.dumps(dict(status='FINITE_OBSERVER_LIMIT_REACHED_NO_REMOTE_ACTION',epoch=time.time())),flush=True)
