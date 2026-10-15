from pathlib import Path
import json,datetime,hashlib
r=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3347-1791494104474180576');o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'records':[],'names':[]}
for i in sorted(set([5,50,100,150,200]+list(range(191,201)))):
 p=r/('calibration-attempt'+str(i)+'.json')
 if p.exists():
  raw=p.read_bytes();d=json.loads(raw);o['records'].append({'iter':i,'complete':d.get('counted_complete'),'metrics':{k:v for k,v in d.get('metrics',{}).items() if isinstance(v,(int,float))},'sha256':hashlib.sha256(raw).hexdigest(),'path':str(p)})
o['names']=[p.name for p in r.iterdir() if any(k in p.name for k in ['valid','512','metric','final','ledger','status','complete','train']) and 'attempt' not in p.name and 'before-training' not in p.name and not p.suffix=='.pt']
print(json.dumps(o))
