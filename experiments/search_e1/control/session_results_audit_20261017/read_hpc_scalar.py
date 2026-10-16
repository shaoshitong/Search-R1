import json,time,hashlib,re,subprocess
from pathlib import Path
roots=[('HPC3 Q-probe sigma3 new loss 721485','/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015'),('Periodic25 Q-probe 725286','/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016-r16'),('Prefix20 joint4 727414','/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727251-20261017')]
roots=[r for r in roots if (('periodic25' in r[1]) == (ACCOUNT=='mou176'))]
def small(p):
 try:
  raw=p.read_bytes()
  if len(raw)>1800000:return None
  return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),data=json.loads(raw))
 except (OSError,ValueError):return None
def lines(p):
 if p.is_file():
  for l in p.open():
   try:yield json.loads(l),hashlib.sha256(l.encode()).hexdigest()
   except ValueError:pass
O=dict(epoch=time.time(),runs=[])
for name,root in roots:
 b=Path(root);o=dict(name=name,root=root,train=[],fixed=[],files=[],native_updates=[],errors=[])
 if not b.exists():o['missing']=True;O['runs'].append(o);continue
 for p in b.glob('logs/*.json'):
  if any(x in p.name for x in ['pipeline','manifest','return','error','summary','final','development','stage512','validation']):
   v=small(p)
   if v:o['files'].append(v)
 for p in b.glob('logs/validation-identities.jsonl'):
  for d,h in lines(p):
   ids=d.get('question_ids',[]);labels=d.get('em',[]);sources=d.get('data_sources',[]);n=len(ids);counts=d.get('counts',{});
   if labels and n==len(labels):o['fixed'].append(dict(iter=d.get('step'),run_id=d.get('run_id'),split=d.get('split'),final=d.get('final'),checked_unix=d.get('checked_unix'),validation_sha256=d.get('validation_sha256'),n=n,em=d.get('weighted_em'),f1=d.get('weighted_f1'),correct=sum(labels),unique=len(set(ids)),unique_source_scoped=len(set(zip(sources,ids))),counts=counts,source_metrics={s:dict(count=c,em=sum(e for e,t in zip(labels,sources) if t==s)/c,f1=sum(e for e,t in zip(d.get('f1',[]),sources) if t==s)/c) for s,c in counts.items()},path=str(p),row_sha256=h))
 o['probe_commits']=[]
 for p in b.glob('artifacts/question-probe/completed/*.json'):
  v=small(p)
  if v:o['probe_commits'].append(dict(path=str(p),sha256=v['sha256'],outer=v['data'].get('outer'),stage=v['data'].get('stage')))
 for p in b.glob('logs/actual-updates.jsonl'):
  for d,h in lines(p):
   if d.get('full_budget_complete'):o['native_updates'].append(dict(iter=d.get('step'),writes=d.get('actual_optimizer_writes'),path=str(p),row_sha256=h))
 for p in b.glob('logs/*training*.log'):
  for no,l in enumerate(p.open(errors='replace'),1):
   if 'step:' not in l:continue
   m=re.search(r'\bstep:\s*(\d+)\s*-',l)
   if not m:continue
   vals={k:float(v) for k,v in re.findall(r'([\w/.-]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',l)}
   kept={k:v for k,v in vals.items() if k in ['critic/score/mean','critic/rewards/mean'] or k.startswith('val/')}
   if kept:o.setdefault('scalar_logs',[]).append(dict(iter=int(m[1]),metrics=kept,path=str(p),line=no,sha256=hashlib.sha256(l.encode()).hexdigest()))
 for p in b.glob('artifacts/question-probe/attempts/*/rollout-audit.json'):
  v=small(p)
  if not v:continue
  raw=p.read_bytes();d=v['data'];groups=d.get('groups',[])
  if len(groups)==512 and all(len(g.get('strict_correctness',[]))==5 for g in groups):
   cor=sum(sum(g['strict_correctness']) for g in groups);o['train'].append(dict(iter=d.get('step',d.get('outer')),em=cor/2560,correct=cor,n=2560,path=str(p),sha256=hashlib.sha256(raw).hexdigest(),metric='strict EM of selected512x5 generated trajectories'))
 for pat in ['final/**/*summary*.json','final/**/*metrics*.json','evaluation/**/*summary*.json','results/*summary*.json','*final*/*.json']:
  for p in b.glob(pat):
   v=small(p)
   if v:o['files'].append(v)
 for p in b.glob('logs/*.log'):
  if 'train' not in p.name and 'pipeline' not in p.name:continue
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-10000));tail=f.read().decode(errors='replace')
  ls=tail.splitlines()
  o['errors'] += [dict(path=str(p),context=ls[i:i+10]) for i,l in enumerate(ls) if any(w in l for w in ['Traceback (most recent','RuntimeError:','CUDA out of memory'])]
 O['runs'].append(o)
r=subprocess.run(['squeue','-h','-j','721485,725286,727414','-o','%i|%T|%N|%M'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);O['queue']=r.stdout
print(json.dumps(O))
