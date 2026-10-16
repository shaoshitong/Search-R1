import json,time,hashlib
from pathlib import Path
base=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016-r16');out=dict(epoch=time.time(),ancestors=[],train=[],fixed=[]);cut=10;seen=set()
for depth in range(10):
 m=json.loads((base/'logs/run-manifest.json').read_text());rest=m.get('full_state_restore') or {};parent=rest.get('source_base')
 if not parent:break
 base=Path(parent)
 if str(base) in seen:raise ValueError('ancestor cycle')
 seen.add(str(base));raw=(base/'logs/run-manifest.json').read_bytes();m=json.loads(raw);out['ancestors'].append(dict(base=str(base),run_id=m['run_id'],manifest_sha256=hashlib.sha256(raw).hexdigest(),cutoff=cut,restore=m.get('full_state_restore')))
 for p in base.glob('artifacts/question-probe/attempts/*/rollout-audit.json'):
  raw=p.read_bytes();d=json.loads(raw);step=d.get('step',d.get('outer'));gs=d.get('groups',[])
  if step is not None and step<=cut and len(gs)==512 and all(len(g.get('strict_correctness',[]))==5 for g in gs):out['train'].append(dict(iter=step,em=sum(sum(g['strict_correctness']) for g in gs)/2560,n=2560,path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
 p=base/'logs/validation-identities.jsonl'
 if p.exists():
  for l in p.open():
   d=json.loads(l);n=len(d.get('question_ids',[]))
   if n==512 and d.get('step',999)<=cut:out['fixed'].append(dict(iter=d['step'],n=n,em=d['weighted_em'],f1=d['weighted_f1'],correct=d['correct'],counts=d['counts'],validation_sha256=d.get('validation_sha256'),path=str(p),row_sha256=hashlib.sha256(l.encode()).hexdigest()))
 cut=(m.get('full_state_restore')or{}).get('completed_outer',0)
print(json.dumps(out))
