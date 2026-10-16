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
if MODE=='A':
 base=Path('/dataset/shaoshitong/recursive-ablation150-20261014/C2-random-final')
 o=dict(epoch=time.time(),name='C2 random-final ablation150',logs=[scan(p) for p in sorted((base/'logs').glob('training*.log'))],receipts=[])
 for p in base.glob('output/*/progress.json'):
  raw=p.read_bytes();o['receipts'].append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),data=json.loads(raw)))
else:
 base=Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
 out=base/'output-speed-resume3-2e3a9029051d4771bc61c95b28b60e21'
 log=Path('/home/shaoshitong/rbb5-context2-ray/ray/session_latest/logs/worker-11344cfc20371f65c7cab4989b78e523fe63b02ac0ee8493b64d8895-01000000-1111494.out')
 ms=sorted((out/'checkpoints/training_state').glob('global_step_*/manifest.json'),key=lambda p:int(p.parent.name.split('_')[-1]));p=ms[-1];raw=p.read_bytes();m=json.loads(raw)
 o=dict(epoch=time.time(),name='Recursive B four GPU',logs=[scan(log)],checkpoint=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),completed=m['identity']['completed_step'],next_step=m['driver']['next_step'],workers=[dict(rank=w['identity']['rank'],adam=w['actual_total_optimizer_calls'],finite=w['full_worker_state_finite']) for w in m['worker_states']]))
print(json.dumps(o))
