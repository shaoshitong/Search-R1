from pathlib import Path
import subprocess,json,time
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');v=dict(epoch=time.time(),jobs=[])
for d in sorted((b/'pool/jobs').glob('job-*/attempt-*')):
 row=dict(directory=str(d),json={})
 for p in [d/'allocation.json',d/'children.json',d/'retriever-ready.json',d/'progress.json',d/'failure.json',d/'complete.json',d/'retrieval/gpu-index-load-progress.json']:
  if p.exists():row['json'][str(p.relative_to(d))]=json.loads(p.read_text())
 row['tails']={}
 for name in ['retriever.log','sampler.log']:
  p=d/name
  if p.exists():
   with p.open('rb')as f:f.seek(max(0,p.stat().st_size-1800));row['tails'][name]=f.read().decode(errors='replace')
 v['jobs'].append(row)
for key,cmd in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),('apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])]:
 r=subprocess.run(cmd,text=True,capture_output=True,timeout=15);v[key]=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
rows=[json.loads(p.read_text())for p in(b/'sampling').glob('shard-*/batches/batch-*/complete.json')]
v['sealed']=dict(batches=len(rows),questions=sum(x['questions']for x in rows),trajectories=sum(x['trajectories']for x in rows))
print(json.dumps(v))
