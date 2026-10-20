"""Read-only follow of the single admitted Slurm pool and actual sealed batches."""
import argparse,json,time
from pathlib import Path
from .pool_control import BASE,PHYSICAL,ROOT,RUN_ID,transport

REMOTE=r'''
import json,subprocess,time
from pathlib import Path
b=Path(PHYSICAL);rows=[json.loads(p.read_text())for p in(b/'sampling').glob('shard-*/batches/batch-*/complete.json')]
assert all(x['run_id']==RUN_ID for x in rows)
r=subprocess.run(['squeue','--array','--noheader','--jobs='+JOB,'--format=%i|%T|%N|%R'],text=True,capture_output=True,timeout=20)
assert r.returncode==0,r.stderr
jobs=[]
for line in r.stdout.splitlines():
 ident,state,node,reason=line.split('|',3);jobs.append(dict(id=ident,state=state,node=node,reason=reason))
attempts=[]
for p in sorted((b/'pool/jobs').glob('job-*/attempt-*')):
 def read(name):
  q=p/name
  return json.loads(q.read_text())if q.exists()else None
 failure=read('failure.json')or read('sampler-failure.json');complete=read('complete.json');progress=read('progress.json')
 row=dict(path=str(p),state='FAILED'if failure else'COMPLETE'if complete else'SAMPLING'if progress else'STARTING',progress=progress,complete=complete,failure=failure)
 if failure:
  row['tails']={}
  for name in ['sampler.log','retriever.log']:
   q=p/name
   if q.exists():
    with q.open('rb')as f:f.seek(max(0,q.stat().st_size-2000));row['tails'][name]=f.read().decode(errors='replace')
 attempts.append(row)
print(json.dumps(dict(epoch=time.time(),run_id=RUN_ID,questions=sum(x['questions']for x in rows),trajectories=sum(x['trajectories']for x in rows),sealed_batches=len(rows),expected_trajectories=845140,jobs=jobs,attempts=attempts[-32:],all_attempts_failed=sum(x['state']=='FAILED'for x in attempts),goal_complete=False)))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',default='3967');p.add_argument('--interval',type=int,default=300);p.add_argument('--hours',type=float,default=24);a=p.parse_args();assert a.job.isdigit()and a.interval>=30
 stamp=time.time_ns();receipt=ROOT/'work'/('qwen7b-pool-follow-%d.jsonl'%stamp);latest=ROOT/'work/qwen7b-pool-follow-latest-small.json';until=time.monotonic()+a.hours*3600;last=None
 print(json.dumps(dict(status='READONLY_POOL_FOLLOWER_STARTED',job=a.job,receipt=str(receipt))),flush=True)
 while time.monotonic()<until:
  script='PHYSICAL='+repr(PHYSICAL)+'\nRUN_ID='+repr(RUN_ID)+'\nJOB='+repr(a.job)+'\n'+REMOTE
  try:
   r=transport(script);assert r.returncode==0,r.stderr[-500:];value=json.loads(r.stdout)
   snapshot=ROOT/'work'/('qwen7b-pool-follow-actual-%d.json'%time.time_ns());snapshot.write_text(json.dumps(value,indent=2)+'\n')
   states={state:sum(x['state']==state for x in value['jobs'])for state in sorted({x['state']for x in value['jobs']})}
   summary=dict(epoch=value['epoch'],job=a.job,questions=value['questions'],trajectories=value['trajectories'],sealed_batches=value['sealed_batches'],job_states=states,failed_attempts=value['all_attempts_failed'],actual=str(snapshot),goal_complete=False)
   with receipt.open('a')as f:f.write(json.dumps(summary)+'\n')
   latest.write_text(json.dumps(summary,indent=2)+'\n');change=(value['trajectories'],states,value['all_attempts_failed'])
   if change!=last:print(json.dumps(summary),flush=True);last=change
   if value['all_attempts_failed']:print('ROOT_ATTENTION_OWN_POOL_ATTEMPT_FAILURE',flush=True);return
   if value['questions']==169028 and value['trajectories']==845140 and value['sealed_batches']==1322:print('ROOT_ATTENTION_ALL_BATCHES_SEALED_FINAL_AUDIT_AND_OWN_QUEUE_CLEANUP_REQUIRED',flush=True);return
   if not value['jobs']:print('ROOT_ATTENTION_POOL_NO_LONGER_QUEUED_OR_RUNNING_READ_TERMINAL_JOBS',flush=True);return
  except Exception as error:print(json.dumps(dict(status='OBSERVATION_CHANNEL_ERROR_NOT_NATIVE_EXIT',error=str(error)[-700:],epoch=time.time())),flush=True)
  time.sleep(a.interval)
 print('ROOT_ATTENTION_FOLLOW_DURATION_ENDED_GOAL_STILL_INCOMPLETE',flush=True)
if __name__=='__main__':main()
