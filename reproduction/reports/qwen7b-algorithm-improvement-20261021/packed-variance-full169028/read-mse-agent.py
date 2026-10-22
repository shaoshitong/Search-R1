"""Independent read-only observer for the existing full169028 MSE job749568."""
import pathlib,json,subprocess,time,hashlib
a=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1');b=a/'mse';assert a.parent.is_dir()and b.is_dir(),'Actual shared task mount must exist';x=dict(time=time.time(),job_id='749568',base=str(b),observer='independent_mse_only_no_mutations',final_read=False)
def read(p):return json.loads(p.read_bytes())
r=read(b/'root-submission.json');assert r['job_id']=='749568'and r['loss']=='mse'and r['epochs']==10 and r['EMA_decay']==.9995;x['submission']={k:r[k]for k in('job_id','name','loss','epochs','EMA_decay','publication_commit','time')}
for name in('controller.json','progress-rank-0.json','progress-rank-1.json','evaluation-progress-rank-0.json','evaluation-progress-rank-1.json','latest.json','complete.json'):
 p=b/name
 if p.exists():x[name]=read(p)
x['failures']={p.name:read(p)for p in b.glob('failure-rank-*.json')}
for name,args in(('queue',['squeue','--start','-h','-j','749568','-o','%i|%j|%P|%T|%S|%R']),('accounting',['sacct','-X','-n','-P','-j','749568','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);x[name]=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
for p in sorted(b.glob('coverage-epoch-*.json')):x[p.name]=read(p)
for p in sorted(b.glob('evaluation-epoch-*.json')):
 e=read(p);x[p.name]=dict(epoch=e['epoch'],cursor=e['cursor'],seconds=e['seconds'],final4096_scored=e['final4096_scored'],metrics={})
 for pool,values in e['metrics'].items():x[p.name]['metrics'][pool]={state:{k:v[k]for k in('variance_MSE','variance_MAE','variance_Pearson','variance_Spearman','Top512','predeclared_G5_classification_diagnostic')if k in v}for state,v in values.items()}
p=b/'curve.jsonl'
if p.exists():
 with p.open('rb')as f:f.seek(0,2);n=f.tell();f.seek(max(0,n-120000));raw=f.read().decode(errors='replace').splitlines()
 rows=[]
 for line in raw:
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 x['last_actual_update_rows']=rows[-8:]
p=b/'train-749568.log'
if p.exists():
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-1600));x['log_tail']=f.read().decode(errors='replace')
x['fixture']={k:v for k,v in read(a/'fixture/fixture-complete.json').items()if k in('status','training_questions','validation_questions','trajectories','seconds')}
print(json.dumps(x))
