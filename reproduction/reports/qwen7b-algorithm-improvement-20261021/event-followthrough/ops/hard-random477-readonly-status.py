"""Small actual-state audit for the two authorized E4 continuations; no mutation."""
import json,pathlib,subprocess,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hard-vs-random-E4-477-v1')
def read(p):return json.loads(p.read_text())
def command(cmd):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-1000:])
out=dict(observed_epoch=time.time(),base=str(B),cpu={},variants={})
for name in ('cpu-launch.json','cpu-result.json','preparation.json'):
 p=B/name
 if p.exists():
  v=read(p)
  out['cpu'][name]={k:v[k] for k in ('status','time','pid','start_ticks','commit','source_adam_steps','fit_updates_per_variant','traceback') if k in v}
for name in ('cpu-driver.log','actual-cpu-tests.log','cpu-prepare.log'):
 p=B/name
 if p.exists():out['cpu'][name]=p.read_text(errors='replace')[-1800:]
ids=[]
for variant in ('targeted','random_control'):
 b=B/variant;row={};out['variants'][variant]=row
 if (b/'job.id').exists():
  job=(b/'job.id').read_text().strip();assert job.isdigit();ids.append(job);row['job_id']=job
 for name in ('controller.json','progress.json','failure.json','interrupted.json','latest.json','evaluation-progress.json','complete.json'):
  p=b/name
  if p.exists():
   v=read(p);row[name]={k:v[k] for k in ('status','pid','job_id','cursor','steps','budget','loss','epoch_time','message','traceback','path','bytes','sha256','completed_questions','completed_trajectories','actor_updates','source_steps','canonical_source_rng_rank','four_rank_bitexact_resume_claim') if k in v}
   if name=='latest.json' and 'path' in v:
    cp=pathlib.Path(v['path']);row['checkpoint_actual_exists']=cp.is_file();row['checkpoint_actual_bytes']=cp.stat().st_size if cp.is_file() else None
 row['replays']={}
 for p in b.glob('next-update-replay-cursor*.json'):
  v=read(p);row['replays'][p.name]={k:v[k] for k in ('passed','source_steps','accepted_steps','next_update_replayed_twice','four_rank_bitexact_resume_claim') if k in v}
  row['replays'][p.name]['all_hashes_equal']=v.get('expected')==v.get('actual')
 logs=sorted((b/'logs').glob('fit-*.log')) if (b/'logs').exists() else []
 if logs:row['log_tail']=logs[-1].read_text(errors='replace')[-2500:]
if ids:
 out['squeue']=command(['squeue','-h','-j',','.join(ids),'-o','%i|%j|%T|%P|%R'])
 out['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed'])
 out['start_estimates']=command(['squeue','--start','-h','-j',','.join(ids),'-o','%i|%S|%T|%R'])
print(json.dumps(out))
