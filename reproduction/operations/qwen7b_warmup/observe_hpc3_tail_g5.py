"""Finite read-only observer for the two unique tail jobs; never resubmits."""
import argparse,fcntl,importlib.util,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail-newseed-preparation'
WORKER=r'''import pathlib,json,subprocess,time,hashlib,os
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def safe(p,keys):
 if not p.exists():return None
 try:v=read(p)
 except FileNotFoundError:return None
 return dict(ref=proof(p),fields={k:v[k]for k in keys if k in v})
rows=[];ids=[]
for w in (0,1):
 job=(b/('worker-%d.job.id'%w)).read_text().strip();assert job in ('745518','745519');ids.append(job);d=b/'workers'/('worker-%d'%w)
 row=dict(worker=w,job_id=job)
 row['controller']=safe(d/'controller.json',['pid','birth_ticks','host','Slurm_job_id','execution_run_id','sampling_protocol_source_run_id','execution_host_cluster','source_commit','epoch'])
 row['progress']=safe(d/'progress.json',['status','execution_run_id','worker','completed_batches','questions','trajectories','latest_batch','elapsed','epoch'])
 row['complete']=safe(d/'complete.json',['status','questions','trajectories','batches','actor_updates','epoch'])
 row['failure']=safe(d/'failure.json',['type','error','epoch'])
 for name in ['ready.json','gpu-index-load-progress.json','gpu-index-load-verified.json']:
  row['retriever_'+name]=safe(d/'retriever'/name,['status','run_id','pid','ready','gpu_shards','dimension','documents','encoder_dtype','index_dtype','added','total','elapsed_seconds','gpu_name','gpu_total_bytes','initialization_seconds'])
 log=b/'logs'/('worker-%d-%s.log'%(w,job));row['log_bytes']=log.stat().st_size if log.exists()else 0
 if log.exists():
  with log.open('rb')as f:f.seek(max(0,log.stat().st_size-20000));tail=f.read().decode(errors='replace')
  row['recent_error_lines']=[line[:350]for line in tail.splitlines()if any(s in line for s in ['Traceback','RuntimeError:','ValueError:','OutOfMemoryError','CUDA out of memory','Error:'])][-6:]
 rows.append(row)
sealed=[];phase=[]
for d in sorted((b/'sampling').glob('shard-*/batches/batch-*')):
 if(d/'complete.json').exists():
  v=read(d/'complete.json');assert v['execution_run_id']=='36b784a3bf964e69a6130bcc41d68809'and v['execution_job']in ids
  sealed.append(dict(worker=0 if v['execution_job']=='745518'else 1,ref=proof(d/'complete.json'),questions=v['questions'],trajectories=v['trajectories'],epoch=v['epoch']))
 else:
  for mode in ['prefix','full']:
   p=d/'outers/outer-000001'/mode/'rank-0-progress.json'
   if p.exists():phase.append(dict(batch=d.name,phase=mode,progress=safe(p,['stage','event','phase','completed','trajectories','retained','active','sampled_tokens','actual_sampled_tokens','epoch','elapsed_seconds'])))
for row in rows:
 own=[x for x in sealed if x['worker']==row['worker']];row['sealed_batch_count']=len(own);row['sealed_questions']=sum(x['questions']for x in own);row['sealed_trajectories']=sum(x['trajectories']for x in own);row['latest_seal']=max(own,key=lambda x:x['epoch'])if own else None
queue=subprocess.check_output(['squeue','-h','-j',','.join(ids),'-o','%i|%j|%T|%R|%N'],universal_newlines=True).strip()
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,JobName,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).strip()
print(json.dumps(dict(epoch=time.time(),status='READONLY_ACTUAL_HPC3_TAIL_PROGRESS',queue=queue,sacct=accounting,workers=rows,inflight_phases=phase[-4:],sealed_questions=sum(x['questions']for x in sealed),sealed_trajectories=sum(x['trajectories']for x in sealed),potentialOverlap=True,observer_restarts_or_submissions=0,model_or_tensor_loaded=False),sort_keys=True))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=0);p.add_argument('--interval',type=int,default=30);a=p.parse_args()
 if not 0<=a.seconds<=21600 or a.interval<20:p.error('Bounded observer, interval>=20s')
 compile(WORKER,'remote-tail-observer','exec')
 spec=importlib.util.spec_from_file_location('transport',Path(__file__).with_name('deploy_hpc3_tail_g5.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 lock=(REPORT/'finite-production-observer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 end=time.monotonic()+a.seconds;last=None
 while True:
  try:
   x=m.remote(WORKER,timeout=45);raw=json.dumps(x,sort_keys=True)
   with(REPORT/'actual-production-observations.jsonl').open('a')as f:f.write(raw+'\n')
   (REPORT/'actual-production-latest-good.json').write_text(json.dumps(x,indent=2)+'\n')
   key=(x['queue'],x['sealed_questions'],[(w['retriever_gpu-index-load-progress.json']or{}).get('fields',{}).get('added')for w in x['workers']],bool(x['inflight_phases']))
   if key!=last:print(raw,flush=True);last=key
   if any(w['failure']for w in x['workers']):raise RuntimeError('Actual owned worker failure; preserve data and inspect, no auto-restart')
   if all(w['sealed_batch_count']>=2 for w in x['workers']):print(json.dumps(dict(status='BOTH_OWN_WORKERS_TWO_REAL_G5_BATCHES_PRESENT_AUDIT_REQUIRED',epoch=x['epoch'])),flush=True);break
  except Exception as error:
   with(REPORT/'observer-errors.jsonl').open('a')as f:f.write(json.dumps(dict(epoch=time.time(),type=type(error).__name__,error=str(error)[-700:],producer_restarted=False))+'\n')
   raise
  if time.monotonic()>=end:break
  time.sleep(min(a.interval,max(0,end-time.monotonic())))
if __name__=='__main__':main()
