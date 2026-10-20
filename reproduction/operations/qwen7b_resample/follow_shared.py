"""Read-only first-batch acceptance for one already-started native sampler."""
import argparse,json,time
from pathlib import Path
from .pool_control import PHYSICAL,ROOT,RUN_ID,transport

REMOTE=r'''
import json,os,time
from pathlib import Path
b=Path(PHYSICAL)
def read(p):return json.loads(p.read_text())if p.exists()else None
rows=[read(p)for p in(b/'sampling').glob('shard-*/batches/batch-*/complete.json')]
assert all(x['run_id']==RUN_ID for x in rows)
attempts=sorted((b/'pool/jobs'/(globals().get('JOB_PREFIX','job-shared5-')+PID)).glob('attempt-*'))
assert len(attempts)<=1
p=attempts[-1]if attempts else None
value=dict(epoch=time.time(),controller_pid=PID,questions=sum(x['questions']for x in rows),
 trajectories=sum(x['trajectories']for x in rows),sealed_batches=len(rows),goal_complete=False)
if p:
 value.update(directory=str(p),progress=read(p/'progress.json'),failure=read(p/'failure.json'),complete=read(p/'complete.json'),
  index_progress=read(p/'retrieval/gpu-index-load-progress.json'))
 value['model_initialized']=[read(q)for q in sorted((p/'sampler-launches').glob('launch-*/model-initialized.json'))]
 # The NAS10 consumer's own open batch-lock FD identifies its active input;
 # observe only metadata from that input, without scanning raw trajectories.
 value['active_phases']=[]
 allocation=read(p/'allocation.json')or{};children=read(p/'children.json')or{}
 cgroup=allocation.get('native_cgroup')
 for child in children.get('own_process_groups',[]):
  try:
   if not cgroup or cgroup not in Path('/proc/%d/cgroup'%child).read_text():continue
   for fd in Path('/proc/%d/fd'%child).iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    if '/pool/batch-locks/'not in target:continue
    prefix,relative=target.split('/pool/batch-locks/',1)
    if not prefix.endswith('/shaoshitong/'+b.name):continue
    tail=Path(relative)
    if len(tail.parts)!=2 or tail.suffix!='.lock':continue
    dest=b/'sampling'/tail.parent/'batches'/tail.stem
    for phase in('prefix','full'):
     source=dest/'outers/outer-000001'/phase/'rank-0-progress.json'
     detail=read(source)
     if detail is not None:
      value['active_phases'].append(dict(batch=tail.stem,phase=phase,source=str(source),
       metadata={key:detail.get(key)for key in('stage','completed_trajectories','expected_trajectories',
        'actual_sampled_tokens','generation_requests','elapsed_seconds','utc')}))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 engines=read(p/'engines.json')
 if engines:
  value['engine_model_initialized']={str(x['engine_index']):[read(q)for q in sorted((Path(x['directory'])/'sampler-launches').glob('launch-*/model-initialized.json'))]for x in engines['engines']}
 if value['failure']:
  value['log_tails']={}
  for name in ['retriever.log','sampler.log','relay.log']:
   q=p/name
   if q.exists():
    with q.open('rb')as f:f.seek(max(0,q.stat().st_size-5000));value['log_tails'][name]=f.read().decode(errors='replace')
print(json.dumps(value))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--controller-pid',required=True);p.add_argument('--job-prefix',choices=('job-shared5-','job-relay10-'),default='job-shared5-');p.add_argument('--own-batches',type=int,default=2);p.add_argument('--per-engine-batches',type=int,default=0);p.add_argument('--interval',type=int,default=60);p.add_argument('--hours',type=float,default=4)
 a=p.parse_args();assert a.controller_pid.isdigit()and a.own_batches>0 and a.per_engine_batches>=0 and a.interval>=30
 latest=ROOT/'work/qwen7b-native-first-batches-latest-small.json';until=time.monotonic()+a.hours*3600;last=None
 print(json.dumps(dict(status='READONLY_NATIVE_FIRST_BATCHES_FOLLOWER_STARTED',controller_pid=a.controller_pid,target_own_complete_batches=a.own_batches)),flush=True)
 while time.monotonic()<until:
  try:
   script='PHYSICAL='+repr(PHYSICAL)+'\nRUN_ID='+repr(RUN_ID)+'\nPID='+repr(a.controller_pid)+'\nJOB_PREFIX='+repr(a.job_prefix)+'\n'+REMOTE
   result=transport(script);assert result.returncode==0,result.stderr[-700:];v=json.loads(result.stdout)
   out=ROOT/'work'/('qwen7b-native-first-batches-actual-%d.json'%time.time_ns());out.write_text(json.dumps(v,indent=2)+'\n')
   own=(v.get('progress')or v.get('complete')or{}).get('completed_own_batches',0)
   engines=(v.get('progress')or v.get('complete')or{}).get('engines',[])
   counts={str(x['engine_index']):x['completed_own_batches']for x in engines}
   small=dict(epoch=v['epoch'],controller_pid=a.controller_pid,own_complete_batches=own,engine_complete_batches=counts,active_phases=v.get('active_phases',[]),sealed_batches=v['sealed_batches'],questions=v['questions'],trajectories=v['trajectories'],failure=v.get('failure'),native_complete=v.get('complete'),actual=str(out),goal_complete=False)
   latest.write_text(json.dumps(small,indent=2)+'\n');change=(own,tuple(sorted(counts.items())),v['trajectories'],bool(v.get('failure')),tuple((x['batch'],x['phase'],x['metadata'].get('completed_trajectories'))for x in v.get('active_phases',[])))
   if change!=last:print(json.dumps(small),flush=True);last=change
   if v.get('failure'):print('ROOT_ATTENTION_ACTUAL_NATIVE_FAILURE',flush=True);return
   accepted=own>=a.own_batches and(not a.per_engine_batches or bool(counts)and min(counts.values())>=a.per_engine_batches)
   if accepted:print('ROOT_ATTENTION_FIRST_REAL_NATIVE_BATCHES_COMPLETE_AUDIT_NEW_BATCHES_AND_CONTINUE',flush=True);return
   if v.get('complete'):print('ROOT_ATTENTION_NATIVE_JOB_ENDED_BEFORE_TARGET_READ_ACTUAL_REASON',flush=True);return
  except Exception as error:print(json.dumps(dict(status='READ_CHANNEL_ERROR_NOT_NATIVE_EXIT',error=str(error)[-700:],epoch=time.time())),flush=True)
  time.sleep(a.interval)
 print('ROOT_ATTENTION_FIRST_BATCH_FOLLOW_DURATION_ENDED',flush=True)
if __name__=='__main__':main()
