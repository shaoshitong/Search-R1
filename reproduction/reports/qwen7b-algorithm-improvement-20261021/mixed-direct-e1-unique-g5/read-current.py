import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-e1-unique-g5-20261022-v1')
x=dict(time=time.time(),base=str(b))
for name in ('controller.json','progress.json','failure.json','complete.json','source-restore-complete.json','phase-transition.json','fixed-training-evaluation-progress.json','evaluation-progress.json','update-intent.json'):
 p=b/name
 if p.exists():
  d=json.loads(p.read_bytes())
  if name=='source-restore-complete.json':d={k:d[k]for k in('status','source_cursor','raw_replaced_by_EMA','raw_EMA_twoAdam_global_RNG_preserved','inherited_sampler_RNG','previous_hard_pool_not_replayed','unique_sampling_seed','epoch_time')}
  x[name]=d
p=b/'latest.json'
if p.exists():
 d=json.loads(p.read_bytes());x['checkpoint']={k:d[k]for k in('cursor','path','bytes','sha256','native_steps','head_steps','training_phase','full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed')}
p=b/'submission-result.json'
if p.exists():
 job=json.loads(p.read_bytes())['job_id'];x['job_id']=job
 for command,key in ((['squeue','-h','-j',job,'-o','%i|%T|%q|%R'],'squeue'),(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],'sacct')):
  r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);x[key]=r.stdout.strip()
 for log in (b/'logs').glob('*'+job+'*.log'):
  with log.open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-1600));x['log_tail']=f.read().decode(errors='replace')
for epoch in (1,2,3):
 for folder,label in (('evaluations','independent'),('fixed-training-evaluations','training')):
  p=b/folder/('epoch-%02d'%epoch)/'complete.json'
  if p.exists():
   d=json.loads(p.read_bytes());x[label+str(epoch)]=dict(cursor=d['cursor'],metrics=d['metrics'])
print(json.dumps(x))
