# Bounded read-only queue and genuine batch progress; no Slurm changes.
import pathlib,json,subprocess,time,re
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
V=B/'algorithm-improvement-20261021/validation-g10-20261021/materialized'
read=lambda p:json.loads(pathlib.Path(p).read_text())
select=lambda v,keys:{k:v[k] for k in keys if k in v}
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(code=p.returncode,stdout=p.stdout[:6000],stderr=p.stderr[:700])
def loginfo(p):
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-7000));raw=f.read().decode(errors='replace')
 errors=[line[:250] for line in raw.splitlines() if re.search(r'Traceback|Error:|Exception:|AssertionError|Killed|CUDA out of memory|failed',line,re.I)]
 return dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,error_lines=errors[-6:])
out=dict(observed_epoch=time.time(),base=str(V),events={})
for event,expected in [('event-A','743024'),('event-B','743025')]:
 b=V/event;c=read(b/'config.json');job=(b/'job.id').read_text().strip();assert job==expected
 assert c['source_commit'].startswith('c8b02d3b22b6')
 own=dict(job_id=job,config=select(c,['source_commit','code_root','validation_role','evaluation_only','untouched_final','fitting_authorized','actor_updates_authorized','capture_prefix_tokens','feature_layer','run_id','sampling_attempt_id','event_index']))
 own['queue']=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%M|%N|%R'])
 own['accounting']=command(['sacct','-X','-n','-P','-j',job,'-o','JobID,State,ExitCode,Elapsed,NodeList'])
 own['estimate']=command(['squeue','--start','-h','-j',job,'-o','%i|%T|%P|%S|%R'])
 own['shards']={}
 for shard in range(4):
  root=b/'sampling'/('shard-%02d'%shard);item=dict(exists=root.exists())
  for name in ('controller.json','progress.json','failure.json','complete.json'):
   p=root/name
   if p.exists():item[name]=select(read(p),['status','pid','start_ticks','job_id','epoch','generation_devices','retrieval_devices','completed_batches','total_batches','questions','trajectories','elapsed','error_type','error'])
  batch_dirs=sorted((root/'batches').glob('*')) if (root/'batches').exists() else []
  visible=list({str(p):p for p in batch_dirs[:1]+batch_dirs[-1:]}.values());phases=[]
  for batch in visible:
   phase=dict(batch=batch.name)
   for mode in ('prefix','full'):
    base=batch/'outers/outer-000001'/mode;p=base/'rank-0-progress.json'
    if p.exists():phase[mode]=select(read(p),['stage','phase','utc','completed_trajectories','expected_trajectories','retained_prefixes','active_trajectories','pending_generation','pending_retrieval','generation_requests','actual_sampled_tokens','elapsed_seconds','initialization_oom_attempts'])
    p=base/'complete.json'
    if p.exists():phase[mode+'-complete']=select(read(p),['status','questions','trajectories','labels','actor_optimizer_updates','feature_layer'])
   p=batch/'complete.json'
   if p.exists():phase['batch_complete']=select(read(p),['questions','trajectories','feature_layer','actor_optimizer_updates','epoch'])
   phases.append(phase)
  item['batch_directory_count']=len(batch_dirs);item['first_and_latest_batches']=phases
  if (root/'retriever.log').exists():item['retriever_log']=loginfo(root/'retriever.log')
  own['shards'][str(shard)]=item
 own['logs']={p.name:loginfo(p) for p in sorted((b/'logs').glob('sampling-'+job+'_*.log'))}
 out['events'][event]=own
print(json.dumps(out))
