from pathlib import Path
import json,time
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');rows=[]
for p in sorted((b/'pool/jobs').glob('job-shared5-1437188*/attempt-*')):
 v={'path':str(p)}
 for q in list(p.glob('*failure*.json'))+list(p.glob('sampler-launches/launch-*/*failure*.json'))+list(p.glob('sampler-launches/launch-*/typed-model-initialization-oom.json')):
  v[str(q.relative_to(p))]=json.loads(q.read_text())
 for name in ['sampler.log','retriever.log']:
  q=p/name
  if q.exists():
   with q.open('rb')as f:f.seek(max(0,q.stat().st_size-12000));v[name]=f.read().decode(errors='replace')
 rows.append(v)
print(json.dumps({'epoch':time.time(),'attempts':rows}))
