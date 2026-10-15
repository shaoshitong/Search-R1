from pathlib import Path
import sys,json,datetime
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'g1-paper-comparison-20261014'));from h100_remote import call
c=r'''from pathlib import Path
import json,re,datetime,subprocess,hashlib,math
r=Path('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v2');o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'files':[],'progress':[]}
for rel in ['logs/final-evaluator.log','logs/final-pipeline-status.json','logs/final-identity.json','logs/final-acceptance.json','logs/cleanup-status.json','logs/final-only-acceptance.json']:
 p=r/rel
 if not p.exists():continue
 e={'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime}
 if p.suffix=='.log':
  for l in p.open(errors='replace'):
   if 'SEARCH_G1_FINAL_RESUME_PROGRESS=' in l:
    try:o['progress'].append(json.loads(l.split('SEARCH_G1_FINAL_RESUME_PROGRESS=',1)[1].strip()))
    except:pass
  o['progress']=o['progress'][-3:]
 else:
  raw=p.read_bytes();d=json.loads(raw);e['sha256']=hashlib.sha256(raw).hexdigest();e['scalars']={k:v for k,v in d.items() if isinstance(v,(str,int,float,bool,type(None))) and len(str(v))<1000}
  n=sum(d.get('counts',{}).values())
  if n==51713:
   e['counts']=d['counts'];em=d.get('em');f1=d.get('f1');e['recomputed_em']=math.fsum(em)/len(em) if isinstance(em,list) else None;e['recomputed_f1']=math.fsum(f1)/len(f1) if isinstance(f1,list) else None;e['array_count']=len(em) if isinstance(em,list) else None;e['binary_em_verified']=all(x in (0,1) for x in em) if isinstance(em,list) else None
 o['files'].append(e)
o['jobs']=subprocess.run(['squeue','-h','-j','3395','-o','%i|%T|%N'],text=True,capture_output=True).stdout
o['metadata_inventory']=[{'path':str(p.relative_to(r)),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime} for p in (r/'logs').glob('*.json')]
print(json.dumps(o))'''
v=call('h100-8',c);p=D/('g1-mixed-final-'+sys.argv[1]+'.json');p.write_text(json.dumps(v,ensure_ascii=False,indent=2));print(json.dumps(v,ensure_ascii=False)[:20000])
