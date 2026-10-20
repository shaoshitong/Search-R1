"""Read-only first-batch acceptance for one already-started native sampler."""
import argparse,json,time
from pathlib import Path
from .pool_control import PHYSICAL,ROOT,RUN_ID,transport

REMOTE=r'''
import json,time
from pathlib import Path
b=Path(PHYSICAL)
def read(p):return json.loads(p.read_text())if p.exists()else None
rows=[read(p)for p in(b/'sampling').glob('shard-*/batches/batch-*/complete.json')]
assert all(x['run_id']==RUN_ID for x in rows)
attempts=sorted((b/'pool/jobs'/('job-shared5-'+PID)).glob('attempt-*'))
assert len(attempts)<=1
p=attempts[-1]if attempts else None
value=dict(epoch=time.time(),controller_pid=PID,questions=sum(x['questions']for x in rows),
 trajectories=sum(x['trajectories']for x in rows),sealed_batches=len(rows),goal_complete=False)
if p:
 value.update(directory=str(p),progress=read(p/'progress.json'),failure=read(p/'failure.json'),complete=read(p/'complete.json'),
  index_progress=read(p/'retrieval/gpu-index-load-progress.json'))
 value['model_initialized']=[read(q)for q in sorted((p/'sampler-launches').glob('launch-*/model-initialized.json'))]
 if value['failure']:
  value['log_tails']={}
  for name in ['retriever.log','sampler.log']:
   q=p/name
   if q.exists():
    with q.open('rb')as f:f.seek(max(0,q.stat().st_size-5000));value['log_tails'][name]=f.read().decode(errors='replace')
print(json.dumps(value))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--controller-pid',required=True);p.add_argument('--own-batches',type=int,default=2);p.add_argument('--interval',type=int,default=60);p.add_argument('--hours',type=float,default=4)
 a=p.parse_args();assert a.controller_pid.isdigit()and a.own_batches>0 and a.interval>=30
 latest=ROOT/'work/qwen7b-native-first-batches-latest-small.json';until=time.monotonic()+a.hours*3600;last=None
 print(json.dumps(dict(status='READONLY_NATIVE_FIRST_BATCHES_FOLLOWER_STARTED',controller_pid=a.controller_pid,target_own_complete_batches=a.own_batches)),flush=True)
 while time.monotonic()<until:
  try:
   script='PHYSICAL='+repr(PHYSICAL)+'\nRUN_ID='+repr(RUN_ID)+'\nPID='+repr(a.controller_pid)+'\n'+REMOTE
   result=transport(script);assert result.returncode==0,result.stderr[-700:];v=json.loads(result.stdout)
   out=ROOT/'work'/('qwen7b-native-first-batches-actual-%d.json'%time.time_ns());out.write_text(json.dumps(v,indent=2)+'\n')
   own=(v.get('progress')or v.get('complete')or{}).get('completed_own_batches',0)
   small=dict(epoch=v['epoch'],controller_pid=a.controller_pid,own_complete_batches=own,sealed_batches=v['sealed_batches'],questions=v['questions'],trajectories=v['trajectories'],failure=v.get('failure'),native_complete=v.get('complete'),actual=str(out),goal_complete=False)
   latest.write_text(json.dumps(small,indent=2)+'\n');change=(own,v['trajectories'],bool(v.get('failure')))
   if change!=last:print(json.dumps(small),flush=True);last=change
   if v.get('failure'):print('ROOT_ATTENTION_ACTUAL_NATIVE_FAILURE',flush=True);return
   if own>=a.own_batches:print('ROOT_ATTENTION_FIRST_REAL_NATIVE_BATCHES_COMPLETE_AUDIT_NEW_BATCHES_AND_CONTINUE',flush=True);return
   if v.get('complete'):print('ROOT_ATTENTION_NATIVE_JOB_ENDED_BEFORE_TARGET_READ_ACTUAL_REASON',flush=True);return
  except Exception as error:print(json.dumps(dict(status='READ_CHANNEL_ERROR_NOT_NATIVE_EXIT',error=str(error)[-700:],epoch=time.time())),flush=True)
  time.sleep(a.interval)
 print('ROOT_ATTENTION_FIRST_BATCH_FOLLOW_DURATION_ENDED',flush=True)
if __name__=='__main__':main()
