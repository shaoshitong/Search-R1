import json,time,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-20261021');s=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021');old=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');d=dict(epoch=time.time(),base=str(b))
newjob=(b/'job.id').read_text().strip()if(b/'job.id').exists()else None;d['newjob']=newjob
for key,cmd in [('queue',['squeue','-j',','.join(x for x in ['745022','745198',newjob]if x),'-h','-o','%i|%u|%T|%N|%R|%S']),('accounting',['sacct','-j',','.join(x for x in ['745022','745198',newjob]if x),'-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed,NodeList']),('service_steps',['sstat','-a','-j','745198','-n','-P','--format=JobID,MaxRSS,AveCPU,AveDiskRead,AveDiskWrite'])]:
 try:r=subprocess.run(cmd,capture_output=True,text=True,timeout=10);d[key]=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except subprocess.TimeoutExpired:d[key]=dict(timeout=True)
for name in ['preparation/cp1-source-audit/launch.json','preparation/cp1-source-audit/process.json','preparation/cp1-source-audit/exit.json','preparation/cp1-full-state-source-receipt.json','preparation/dense-exact-fallback-decision.json','logs/cp1-numerical-transition.json','diagnostics/cp1-sparse-numerical-qualification.json','logs/run-manifest.json','logs/pipeline-status.json','logs/cleanup-status.json','logs/easy2560-configuration-prepared.json','preparation/rolling-checkpoints/latest.json','logs/actual-updates.jsonl']:
 p=b/name
 if not p.exists():continue
 try:
  v=[json.loads(x)for x in p.read_text().splitlines()if x][-2:]if name.endswith('jsonl')else json.loads(p.read_text())
  if name.endswith('cp1-full-state-source-receipt.json'):v={key:v.get(key)for key in ['status','step','world_size','run_id','hf_saved_tensor_count','actual_optimizer_writes_per_rank','post_callback_rng_captured','actual_cuda_restore_accepted','ranks','driver_state']}
  if name.endswith('run-manifest.json'):v={key:v.get(key)for key in ['run_id','source_commit','initial_model','training_world_size','actor_micro_batch_per_rank','full_state_restore','infra_optimizations','external_retriever']};v['external_retriever']={key:v['external_retriever'].get(key)for key in ['service_job_id','endpoint','physical_gpu_count','logical_shards']}
  d[name]=v
 except Exception as e:d[name]=dict(error=str(e))
d['restore_receipts']={}
for p in (b/'logs').glob('*restor*.json'):
 try:d['restore_receipts'][p.name]=json.loads(p.read_text())
 except ValueError:pass
d['diagnostic_failures']={p.name:json.loads(p.read_text())for p in (b/'diagnostics').glob('*failure.json')}
logs=[b/'preparation/cp1-source-audit/stdout.log',b/'logs/cp1-sparse-numerical-qualification.log',b/'logs/search-g1-common8-training.log',*list((b/'logs').glob('training-*.log'))]
d['log_tails']=[]
for p in logs:
 if not p.is_file():continue
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-6000));tail=f.read().decode(errors='replace')
 d['log_tails'].append(dict(path=str(p.relative_to(b)),bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=tail))
for name in ['preparation/human-stop-after-cp1-for-optimized-external2-20261021.json','preparation/cp1-transition-scancel-result.json','logs/cleanup-status.json']:
 p=old/name
 if p.exists():d['old/'+name]=json.loads(p.read_text())
p=s/'logs/service-ready.json'
if p.exists():
 try:
  import requests
  with requests.Session()as session:session.trust_env=False;r=session.get(json.loads(p.read_text())['endpoint'].replace('/retrieve','/health'),timeout=8);d['service_health']=r.json()
 except Exception as e:d['service_health_error']=str(e)
r=subprocess.run(['du','-x','-s','-B1',str(b)],capture_output=True,text=True,timeout=20);d['current_base_allocated_bytes']=int(r.stdout.split()[0]) if r.returncode==0 else None
claim=b/'preparation/training-submit-claim.json'
if claim.exists():
 q=Path(json.loads(claim.read_text())['restore_receipt'])
 if q.is_file():
  receipt=json.loads(q.read_text());d['source_cp1_cpu_receipt']={key:receipt.get(key)for key in ['status','step','world_size','hf_saved_tensor_count','actual_optimizer_writes_per_rank','post_callback_rng_captured','actual_cuda_restore_accepted']}
print(json.dumps(d))
