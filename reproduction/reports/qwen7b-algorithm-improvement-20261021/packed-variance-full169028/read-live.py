import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1')
assert b.parent.is_dir()
out={'time':time.time(),'experiments':{}}
for n in ['prepare-process.json','fixture/prepare-progress.json','fixture/fixture-complete.json','fixture/failure-rank-cpu.json']:
 p=b/n
 if p.exists():
  v=json.loads(p.read_bytes())
  if n.endswith('fixture-complete.json'):v={k:v[k]for k in ['status','training_questions','validation_questions','trajectories','seconds']if k in v}
  out[n]=v
for loss in ['mse','mae','z_mse_anchor']:
 base=b/loss;cell={}
 for n in ['root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','evaluation-progress-rank-0.json','evaluation-progress-rank-1.json','latest.json','complete.json']:
  p=base/n
  if p.exists():
   v=json.loads(p.read_bytes())
   if n=='root-submission.json':v={k:v[k]for k in ['job_id','name','epochs','EMA_decay','loss','publication_commit','time']}
   cell[n]=v
 if 'root-submission.json'in cell:
  job=cell['root-submission.json']['job_id']
  for k,cmd in [('accounting',['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList']),('queue',['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R'])]:
   p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);cell[k]={'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
  log=base/('train-'+job+'.log')
  if log.exists():
   with log.open('rb')as f:f.seek(max(0,log.stat().st_size-1400));cell['log_tail']=f.read().decode(errors='replace')
 for p in base.glob('failure-rank-*.json'):cell[p.name]=json.loads(p.read_bytes())
 for p in base.glob('coverage-epoch-*.json'):cell[p.name]=json.loads(p.read_bytes())
 for p in sorted(base.glob('evaluation-epoch-*.json')):
  e=json.loads(p.read_bytes());m=e['metrics'];cell[p.name]={'epoch':e['epoch'],'cursor':e['cursor'],'seconds':e['seconds'],'metrics':{}}
  for pool in ['fixed_train2048_G5','fixed_validation2048_G5','fixed_validation2048_M3','fixed_validation2048_M5','fixed_validation2048_M10']:
   cell[p.name]['metrics'][pool]={st:{k:v[k]for k in ['variance_MSE','variance_MAE','variance_Pearson','variance_Spearman','Top512','predeclared_G5_classification_diagnostic']if k in v}for st,v in m[pool].items()}
 out['experiments'][loss]=cell
print(json.dumps(out))
