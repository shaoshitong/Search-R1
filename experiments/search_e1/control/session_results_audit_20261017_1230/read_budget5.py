MODE='A'
import json,time,re,hashlib
from pathlib import Path
NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
def scan(p):
 out=[]
 if not p.is_file():return dict(path=str(p),rows=[])
 start=p.stat().st_size
 for no,line in enumerate(p.open(errors='replace'),1):
  if 'step:' not in line:continue
  m=re.search(r'\bstep:\s*(\d+)\s*-',line)
  if not m:continue
  v={k:float(x) for k,x in re.findall(r'([\w/.-]+):\s*('+NUM+')',line)}
  v={k:x for k,x in v.items() if k in ['critic/score/mean','critic/rewards/mean','training/actual_outer_updates','actor/optimizer_steps'] or k.startswith('val/')}
  if v:out.append(dict(iter=int(m[1]),metrics=v,line=no,sha256=hashlib.sha256(line.encode()).hexdigest()))
 return dict(path=str(p),bytes_before=start,bytes_after=p.stat().st_size,rows=out)
base=Path('/dataset/shaoshitong/recursive-ablation150-20261014/budget-5')
o=dict(epoch=time.time(),name='Recursive budget5 ablation150',exists=base.exists(),logs=[scan(p) for p in sorted((base/'logs').glob('training*.log'))],receipts=[])
for p in base.glob('output/*/progress.json'):
 raw=p.read_bytes();o['receipts'].append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),data=json.loads(raw)))
print(json.dumps(o))
