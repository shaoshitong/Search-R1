import json,subprocess,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-batch-production-r1-20261023')
def read(p):
 return json.loads(p.read_text()) if p.is_file() else None
job=(base/'job.id').read_text().strip() if (base/'job.id').is_file() else None
status=read(base/'status.json');ready=read(base/'logs/service-ready.json');stage=read(base/'logs/stage64-admission.json')
logs=sorted((base/'logs').glob('service-*.log'),key=lambda p:p.stat().st_mtime,reverse=True)
tails=[]
for p in logs[:1]:
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-4500));tail=f.read().decode(errors='replace')
 tails.append(dict(name=p.name,bytes=p.stat().st_size,tail=tail[-3000:]))
rows={}
if job:
 for name,argv in (('queue',['squeue','-h','-j',job,'-o','%i|%u|%T|%N|%M|%R']),('accounting',['sacct','-n','-X','-j',job,'-P','--format=JobIDRaw,User,State,ExitCode,Elapsed,NodeList'])):
  p=subprocess.run(argv,capture_output=True,text=True,timeout=15);rows[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
keys=('phase','checked_unix','rss_bytes','ready','ready_unix','service_run_id','service_job_id','source_commit','endpoint','request_rebatching_enabled','singleton_e5','faiss_batch_size','faiss_max_wait_ms','physical_gpu_count','logical_shards','logical_to_physical_gpu','optional_stage64_proof')
small=lambda v:{k:v.get(k) for k in keys if k in v} if v else None
print(json.dumps(dict(epoch=time.time(),base=str(base),job=job,status=small(status),ready=small(ready),stage64=dict(accepted=stage.get('accepted'),queries=stage.get('query_count'),id_rows=len(stage.get('gpu_document_ids',[])),score_rows=len(stage.get('gpu_document_scores',[])),document_sha_rows=len(stage.get('gpu_document_sha256',[]))) if stage else None,slurm=rows,logs=tails),sort_keys=True))
