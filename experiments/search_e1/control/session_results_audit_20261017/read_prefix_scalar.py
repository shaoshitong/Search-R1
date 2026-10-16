import json,time,hashlib
from pathlib import Path

def doc(p):
 try:
  raw=p.read_bytes()
  if len(raw)>2000000:return None
  return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),data=json.loads(raw))
 except (OSError,ValueError):return None
O=dict(epoch=time.time(),runs=[])
for name,pattern in PATTERNS:
 roots=sorted(Path('/mnt/data/shaoshitong').glob(pattern));o=dict(name=name,roots=[str(p) for p in roots],points=[],qa=[],latest=[],errors=[])
 for b in roots:
  j=doc(b/'latest-joint.json')
  if j:o['latest'].append(j)
  for p in b.glob('outers/outer-*/complete.json'):
   v=doc(p)
   if not v:continue
   j=v['data'];outer=j.get('outer');hist=j.get('actor_selection',{}).get('actor512',{}).get('correct_count_histogram');roll=j.get('rollout',{});correct=None
   if hist:correct=sum(int(k)*v for k,v in hist.items());n=2560
   else:correct=roll.get('selected_strict_correct');n=roll.get('trajectories',roll.get('expected_trajectories',2560))
   if correct is not None:o['points'].append(dict(iter=outer,correct=correct,n=n,em=correct/2560,path=str(p),sha256=v['sha256'],run_id=j.get('run_id'),actor_adam=j.get('actor',{}).get('actual_adam_steps_per_rank'),probe_adam=j.get('probe',{}).get('actual_adam_steps_per_rank')))
  for pat in ['recovered-validation/complete.json','outers/outer-*/validation/complete.json','final-evaluation/complete.json']:
   for p in b.glob(pat):
    v=doc(p)
    if v:o['qa'].append(v)
  for n in ['pipeline-stage.json','logs/pipeline-stage.json','logs/pipeline-error.json','logs/pipeline-complete.json']:
   v=doc(b/n)
   if v:o.setdefault('status_files',[]).append(v)
 O['runs'].append(o)
print(json.dumps(O))
