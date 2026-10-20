import pathlib,json,subprocess,time,re
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
b=B/'algorithm-improvement-20261021/training-extraG5-matched6528-20261021/materialized'
def read(p):return json.loads(pathlib.Path(p).read_text())
def selected(v,keys):return {k:v[k] for k in keys if k in v}
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(code=p.returncode,stdout=p.stdout[:5000],stderr=p.stderr[:500])
def loginfo(p):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-7000));raw=f.read().decode(errors='replace')
 errors=[line[:250] for line in raw.splitlines() if re.search(r'Traceback|Error:|Exception:|AssertionError|Killed|CUDA out of memory|failed',line,re.I)]
 return dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,error_lines=errors[-6:])
config=read(b/'config.json');job=(b/'job.id').read_text().strip();assert job=='743115'
out=dict(observed_epoch=time.time(),base=str(b),job_id=job,config=selected(config,['source_commit','code_root','run_id','sampling_attempt_id','capture_prefix_tokens','feature_layer','extra_response_indices','model_revision','warmup_only','actor_updates_authorized']))
out['queue']=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%M|%N|%R'])
out['accounting']=command(['sacct','-X','-n','-P','-j',job,'-o','JobID,State,ExitCode,Elapsed,NodeList'])
out['estimate']=command(['squeue','--start','-h','-j',job,'-o','%i|%T|%P|%S|%R'])
out['shards']={}
for i in range(4):
 root=b/'sampling'/('shard-%02d'%i);own=dict(exists=root.exists())
 for name in ('extra-g5-adapter-controller.json','controller.json','progress.json','failure.json','complete.json','extra-g5-complete.json'):
  p=root/name
  if p.exists():own[name]=selected(read(p),['status','role','pid','start_ticks','job_id','source_commit','epoch','response_indices','original_actor_fresh_all_file_SHA','generation_devices','retrieval_devices','completed_batches','total_batches','questions','trajectories','elapsed','error_type','error'])
 phases=[]
 for batch in sorted((root/'batches').glob('*')) if (root/'batches').exists() else []:
  item=dict(batch=batch.name)
  for phase in ('prefix','full'):
   output=batch/'outers/outer-000001'/phase
   p=output/'rank-0-progress.json'
   if p.exists():item[phase]=selected(read(p),['stage','phase','utc','completed_trajectories','expected_trajectories','retained_prefixes','active_trajectories','pending_generation','pending_retrieval','generation_requests','actual_sampled_tokens','elapsed_seconds','initialization_oom_attempts'])
   for name in ('rank-0-complete.json','complete.json'):
    p=output/name
    if p.exists():item[phase+'-'+name]=selected(read(p),['status','questions','trajectories','labels','actor_optimizer_updates','feature_layer'])
  p=batch/'complete.json'
  if p.exists():item['complete.json']=selected(read(p),['questions','trajectories','feature_layer','actor_optimizer_updates','epoch'])
  phases.append(item)
 own['batches']=phases
 if (root/'retriever.log').exists():own['retriever.log']=loginfo(root/'retriever.log')
 for name in ('gpu-index-load-progress.json','gpu-index-load-verified.json','ready.json'):
  p=root/'retriever'/name
  if p.exists():own[name]=selected(read(p),['status','added','total','documents','physical_gpus','gpu_name','elapsed_seconds','initialization_seconds','ready','run_id'])
 out['shards'][str(i)]=own
out['logs']={p.name:loginfo(p) for p in sorted((b/'logs').glob('sampling-743115_*.log'))}
for own in out['shards'].values():
 batches=own.pop('batches',[]);own['fully_completed_batches']=sum('complete.json' in b for b in batches);own['last_batch']=batches[-1] if batches else None
v=B/'frozen-validation-ema3-20261021';ev={}
for n in ['controller.json','progress.json','complete.json','failure.json']:
 p=v/n
 if p.exists():ev[n]=read(p)
for p in sorted(v.glob('*/progress.json')):ev[str(p.relative_to(v))]=read(p)
out['frozen_validation_evaluation']=ev
print(json.dumps(out))
