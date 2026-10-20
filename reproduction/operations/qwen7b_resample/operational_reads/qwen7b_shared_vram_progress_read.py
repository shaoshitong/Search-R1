import json,time
from pathlib import Path
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
def read(p):return json.loads(p.read_text())if p.exists()else None
attempts=[]
for p in sorted((b/'pool/jobs').glob('job-shared5-*/attempt-*')):
 row=dict(directory=str(p))
 for name in ['allocation.json','children.json','progress.json','failure.json','complete.json','retriever-ready.json','retrieval/gpu-index-load-progress.json','retrieval/ready.json','typed-initialization-oom-history.json']:
  x=p/name
  if x.exists():row[name]=read(x)
 row['sampler_launches']=[]
 for launch in sorted((p/'sampler-launches').glob('launch-*')):
  z=dict(path=str(launch))
  for name in ['model-initialization.json','model-initialized.json','typed-model-initialization-oom.json','failure.json']:
   if(launch/name).exists():z[name]=read(launch/name)
  row['sampler_launches'].append(z)
 row['log_tails']={}
 for name in ['retriever.log','sampler.log']:
  q=p/name
  if q.exists():
   with q.open('rb')as f:f.seek(max(0,q.stat().st_size-5000));row['log_tails'][name]=f.read().decode(errors='replace')
 attempts.append(row)
seals=[read(p)for p in(b/'sampling').glob('shard-*/batches/batch-*/complete.json')]
print(json.dumps(dict(epoch=time.time(),attempts=attempts,sealed_batches=len(seals),questions=sum(x['questions']for x in seals),trajectories=sum(x['trajectories']for x in seals))))
