import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');x=dict(time=time.time(),variants=[])
for n in ('fixture-complete.json','prepare-progress.json'):
 p=b/n
 if p.exists():
  v=json.loads(p.read_bytes());x[n]={k:v[k]for k in('status','stage','trajectories','expected','source_files','seconds','training_questions','validation_questions')if k in v}
for layer,prefix in ((27,20),(21,20),(27,80)):
 d=b/('layer%d-prefix%d'%(layer,prefix));v=dict(layer=layer,prefix=prefix)
 p=d/'root-submission.json'
 if p.exists():
  job=json.loads(p.read_bytes())['job_id'];v['job']=job
  r=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);v['sacct']=r.stdout.strip()
  p=d/('train-'+job+'.log')
  if p.exists():
   with p.open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-1400));v['log_tail']=f.read().decode(errors='replace')
 for pattern in ('controller.json','progress-rank-*.json','failure-rank-*.json','latest.json','complete.json','evaluation-epoch-*.json'):
  for p in d.glob(pattern):v[p.name]=json.loads(p.read_bytes())
 x['variants'].append(v)
jobs=','.join(v['job']for v in x['variants']if 'job'in v)
r=subprocess.run(['squeue','-h','-j',jobs or '0','-o','%i|%T|%r|%S'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);x['queue_reasons']=r.stdout.strip()
print(json.dumps(x))
