"""Read-only remote scalar/receipt collector for immutable A and C1/C2 ablations."""
import json,hashlib,re,time
from pathlib import Path
BASE=Path('/dataset/shaoshitong/recursive-ablation150-20261014')
A=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
PAT=re.compile(r'([\w/.-]+):\s*('+NUM+')')
def small(p):
 try:
  if p.stat().st_size>750000:return None
  raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),data=json.loads(raw))
 except (OSError,ValueError):return None
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,uid=p.stat().st_uid,tick=int(stat[19]),state=stat[0],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
 except FileNotFoundError:return None
def log(p):
 rows=[];events=[];tail=[]
 with p.open(errors='replace') as f:
  for i,l in enumerate(f):
   m=re.search(r'\bstep:\s*(\d+)\s*-',l) if 'step:' in l else None
   if m:
    vals={k:float(v) for k,v in PAT.findall(l)}
    vals={k:v for k,v in vals.items() if k in ['critic/score/mean','critic/rewards/mean','actor/optimizer_steps','training/actual_outer_updates','actor/total_optimizer_calls','training/attempt_id','training/refused_attempts'] or k.startswith('val/')}
    rows.append(dict(iter=int(m[1]),line=i+1,sha256=hashlib.sha256(l.encode()).hexdigest(),metrics=vals))
   if any(k in l for k in ['TREE_DRIVER_STAGE=','TREE_TIMING=','ABLATION_NATIVE','ABLATION_ROLLING','Traceback','RuntimeError','ValueError','Error','Exception']):
    events.append(l.strip()[:1800]);events=events[-5:]
   if len(l)<5000:tail.append(l.strip()[:1200]);tail=tail[-5:]
 return dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime,rows=rows,events=events,tail=tail)
out=dict(at=time.time(),study=small(BASE/'study-progress.json'),experiments=[])
for name,root,logs in [('Recursive A',A,[A/'logs/training.log',A/'recovery-from100-20261013/logs/training.log']),('C1 root-only ablation150',BASE/'C1-root-only',None),('C2 random-final ablation150',BASE/'C2-random-final',None)]:
 e=dict(name=name,root=str(root),exists=root.exists(),logs=[],receipts=[],inventory=[])
 if not root.exists():out['experiments'].append(e);continue
 if logs is None:logs=sorted((root/'logs').glob('training*.log'))
 for p in logs:
  if p.exists():e['logs'].append(log(p))
 pats=['receipts/current-entry.json','receipts/train-start.json','receipts/*exit*.json','output/*/progress.json','output/*/training-complete.json','output/*/entry-exit.json','checkpoints/training_state/global_step_150/manifest.json','checkpoints/training_state/global_step_200/manifest.json','reports/*summary*.json','evaluations/*.json','evaluation/*.json']
 if name=='Recursive A':
  pats+=['recovery-from100-20261013/checkpoints/training_state/global_step_200/manifest.json','recovery-from100-20261013/final51713/*/*summary*.json','recovery-from100-20261013/final51713/*/*metrics*.json','recovery-from100-20261013/final51713-batch1024-v1/*exit*.json']
 for pat in pats:
  for p in root.glob(pat):
   v=small(p)
   if v:e['receipts'].append(v)
 entry=small(root/'receipts/current-entry.json')
 if entry:e['process']=proc(entry['data']['pid'])
 for sub in ['evaluations','evaluation','reports','final51713','recovery-from100-20261013/final51713']:
  d=root/sub
  if d.exists():
   for p in d.glob('*/*.json'):
    if any(k in p.name.lower() for k in ['summary','metrics','manifest','complete']):e['inventory'].append(dict(path=str(p),bytes=p.stat().st_size))
 out['experiments'].append(e)
out['finished_at']=time.time()
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
