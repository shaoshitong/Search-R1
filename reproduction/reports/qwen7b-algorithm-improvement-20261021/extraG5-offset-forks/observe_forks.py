"""Read only the two submitted owned arrays and small progress/receipt metadata."""
import importlib.util,json,hashlib,subprocess,shlex,time
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
REPO=CONTROL/'work/qwen7b-probe-warmup-20261019'
REMOTE=r'''
import pathlib,json,time,hashlib,subprocess
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 assert p.returncode==0,p.stderr[-500:];return p.stdout.strip()
r=dict(observed_epoch=time.time(),phases={},squeue=command(['squeue','--array','-h','-j','743148,743149','-o','%i|%j|%T|%M|%N|%R']),sacct=command(['sacct','-X','-n','-P','-j','743148,743149','--format=JobID,State,ExitCode,Elapsed']))
for offset,job in ((10,'743148'),(15,'743149')):
 b=pathlib.Path(A+'/training-extraG5-matched6528-offset%d-20261021/materialized'%offset);c=read(b/'config.json')
 assert (b/'job.id').read_text().strip()==job and c['extra_response_offset']==offset
 row=dict(base=str(b),job_id=job,run_id=c['run_id'],sampling_attempt_id=c['sampling_attempt_id'],response_indices=c['extra_response_indices'],config_sha256=digest(b/'config.json'),shards={})
 for shard in range(4):
  d=b/'sampling'/('shard-%02d'%shard);own={}
  for name in ['extra-g5-adapter-controller.json','controller.json','progress.json','failure.json','complete.json','extra-g5-complete.json']:
   if (d/name).is_file():
    v=read(d/name);keys=['status','pid','job_id','run_id','sampling_attempt_id','source_commit','response_indices','questions','trajectories','completed_batches','total_batches','epoch','elapsed','error','error_type']
    own[name]={k:v[k] for k in keys if k in v}
  batches=[]
  if (d/'batches').exists():
   for batch in sorted((d/'batches').iterdir()):
    if not batch.is_dir():continue
    o=dict(batch=batch.name,phases={})
    for phase in ['prefix','full']:
     p=batch/'outers/outer-000001'/phase/'rank-0-progress.json'
     if p.is_file():
      v=read(p);keys=['stage','phase','utc','completed_trajectories','expected_trajectories','active_trajectories','pending_generation','pending_retrieval','generation_requests','actual_sampled_tokens','elapsed_seconds']
      o['phases'][phase]={k:v[k] for k in keys if k in v}
    if (batch/'complete.json').is_file():
     v=read(batch/'complete.json');assert v['run_id']==c['run_id'] and v['actor_optimizer_updates']==0
     assert v['input_sha256']==digest(b/'inputs'/('shard-%02d'%shard)/(batch.name+'.json'))
     o['sealed']=dict(questions=v['questions'],trajectories=v['trajectories'],epoch=v['epoch'],receipt_sha256=digest(batch/'complete.json'),same_run_and_input_verified=True)
    batches.append(o)
  own['batches']=batches
  p=b/'logs'/('sampling-%s_%d.log'%(job,shard))
  if p.exists():
   with p.open('rb') as f:
    f.seek(max(0,p.stat().st_size-3500));lines=f.read().decode(errors='replace').splitlines()
   own['log_tail']=[line[:300] for line in lines[-8:]];own['log_bytes']=p.stat().st_size
  row['shards'][str(shard)]=own
 r['phases'][str(offset)]=row
r['squeue_start']=command(['squeue','--start','-j','743148,743149','-o','%i|%j|%T|%S|%R'])
print(json.dumps(r))
'''

def main():
 spec=importlib.util.spec_from_file_location('fork_progress_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
 c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 command='python3 -c '+shlex.quote(REMOTE);assert len(command.encode())<16000
 result=c.run(command,account='mou176',source_address=address,timeout=55)
 lines=[json.loads(x) for x in result.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
 if result.returncode or not lines:raise RuntimeError(result.stderr.decode(errors='replace')[-1000:])
 row=lines[-1];row['operator_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();row['operator_command_bytes']=len(command.encode())
 path=Path(__file__).parent/'observations.jsonl'
 with path.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
 latest=CONTROL/'outputs/qwen7b-warmup-20261019/extraG5-fork-live-observation.json';latest.write_text(json.dumps(row,indent=2)+'\n')
 print(json.dumps(row),flush=True)
if __name__=='__main__':main()
