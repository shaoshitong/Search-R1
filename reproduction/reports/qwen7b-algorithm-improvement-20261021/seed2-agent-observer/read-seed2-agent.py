"""Only small metadata for the two root-submitted second-seed replications."""
import pathlib,json,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
CELLS={'current64':('749965',A/'online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1/current64'),'Z':('749935',A/'online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1')}
x=dict(time=time.time(),cells={},no_Slurm_mutations=True,final4096_read=False)
def read(p):return json.loads(p.read_bytes())
for cell,(job,b)in CELLS.items():
 assert b.is_dir(),'Staged immutable second-seed root required';v=dict(job_id=job,base=str(b))
 for name in('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
  if(b/name).exists():v[name]=read(b/name)
 if v.get('root-submission.json',{}).get('job_id',job)!=job:raise ValueError('Root-submitted exact job identity differs')
 if'controller.json'in v:
  c=v['controller.json'];assert c['job_id']==job and c.get('init_seed')==19137 and c.get('train_plan_seed')==2026102302 and c.get('diagnostic_seed')==2026102228,'Actual explicit training/diagnostic seeds differ'
 v['failures']={p.name:read(p)for p in b.glob('failure-rank-*.json')};v['epoch_metrics']={p.name:read(p)for p in sorted(b.glob('evaluation-epoch-*.json'))}
 p=b/'curve.jsonl';rows=[]
 if p.exists():
  with p.open('rb')as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-160000));tail=f.read().decode(errors='replace').splitlines()
  for line in tail:
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 v['last_six_real_attempts']=[{k:r[k]for k in('epoch','cursor','actual_Adam_updates','EMA_updates','Adam_and_EMA_step_performed','current_logit_gradient_L2','native_gradient_L2','head_gradient_L2','time')if k in r}for r in rows[-6:]];x['cells'][cell]=v
for key,args in(('queue',['squeue','-h','-j','749965,749935','-o','%i|%j|%P|%T|%M|%R']),('accounting',['sacct','-X','-n','-P','-j','749965,749935','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);x[key]=dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
print(json.dumps(x))
