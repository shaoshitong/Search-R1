from pathlib import Path
import json,datetime,hashlib,re,subprocess,sys,ast
roots=json.loads(sys.argv[1]);O={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':[]}
for root in roots:
 r=Path(root);o={'root':root,'train':[],'fixed512':[],'named':[],'recovery_files':[]}
 if not r.exists():O['roots'].append(o);continue
 p=r/'logs/actual-updates.jsonl';updates=[json.loads(x) for x in p.read_bytes().splitlines()] if p.exists() else []
 o['completed_iter']=max([x.get('step',0) for x in updates]+[0])
 for u in updates:
  if not u.get('attempt_id'):continue
  ap=r/'artifacts/question-probe/attempts'/u['attempt_id']/'rollout-audit.json'
  if ap.exists():
   d=json.loads(ap.read_bytes());m=d.get('prequential',{}).get('all',{});o['train'].append({'iter':u['step'],'em':m.get('actual_em_mean'),'questions':m.get('questions'),'reward':u.get('metrics',{}).get('critic/rewards/mean'),'path':str(ap)})
 paths=set(r.glob('**/*512*.json'))|set(r.glob('**/*development*.json'))
 for p in paths:
  if any(x in str(p) for x in ['/repo/','/source/','/ray/','/cache/','/env/']):continue
  if p.stat().st_size>2000000:continue
  raw=p.read_bytes()
  try:d=json.loads(raw)
  except:continue
  if isinstance(d,dict):
   m=d.get('metrics',d);n=m.get('val/qa_count/all',d.get('question_count',d.get('qa_count',d.get('rows'))))
   em=m.get('val/em/all',d.get('weighted_em',d.get('em')));f1=m.get('val/f1/all',d.get('weighted_f1',d.get('f1')))
   if n==512 and em is not None:o['fixed512'].append({'iter':d.get('step',d.get('outer_step',d.get('actual_completed_outer'))),'em':em,'f1':f1,'n':512,'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
 for p in r.glob('artifacts/question-probe/completed-pairs/*/source-receipt.json'):
  d=json.loads(p.read_bytes())
  for k,v in d.get('historical_artifacts',{}).get('development',{}).items():
   p1=Path(v['path']);raw=p1.read_bytes();v1=json.loads(raw);m=v1.get('metrics',v1);n=m.get('val/qa_count/all',v1.get('question_count',v1.get('qa_count')));em=m.get('val/em/all',v1.get('weighted_em',v1.get('em')))
   if n==512 and em is not None:o['fixed512'].append({'iter':v1.get('step'),'em':em,'f1':m.get('val/f1/all',v1.get('weighted_f1',v1.get('f1'))),'n':512,'path':str(p1),'sha256':hashlib.sha256(raw).hexdigest(),'binding_sha_match':hashlib.sha256(raw).hexdigest()==v['full_read_sha256']})
 for p in r.glob('*'):
  if p.is_file():o['named'].append({'path':str(p),'bytes':p.stat().st_size})
 for p in r.glob('logs/*'):
  if p.is_file() and ('restore' in p.name or 'pipeline' in p.name or p.suffix=='.log'):
   o['recovery_files'].append({'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime})
 for p in [r/'calibration.log']:
  if p.exists():o['final_log_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 O['roots'].append(o)
print(json.dumps(O))
