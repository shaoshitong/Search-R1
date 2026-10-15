from pathlib import Path
import json,datetime,hashlib,re,subprocess,sys
roots=json.loads(sys.argv[1]);O={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':[]}
def scalar(x,dep=0):
 if dep>5:return None
 if isinstance(x,dict):return {k:scalar(v,dep+1) for k,v in x.items() if not any(s in k.lower() for s in ('question_ids','tokens','prompt','environment','tensor')) and (not isinstance(v,list) or len(v)<12)}
 if isinstance(x,list):return [scalar(v,dep+1) for v in x]
 return x
for root in roots:
 r=Path(root);o={'root':root,'files':[],'metrics':[],'rank':[]};paths=set()
 for pat in ['logs/*development*.json','logs/*stage512*.json','logs/*pipeline*.json','logs/cleanup*.json','logs/*exception*.json','preparation/submit-receipt.json','artifacts/question-probe/attempts/*/*audit*.json','artifacts/question-probe/attempts/*/selection.json','artifacts/question-probe/completed-pairs/*/*development*.json','artifacts/question-probe/completed-pairs/*/source-receipt.json']:
  paths.update(r.glob(pat))
 for p in paths:
  if p.stat().st_size>500000:continue
  raw=p.read_bytes()
  try:d=json.loads(raw)
  except:continue
  o['files'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'data':scalar(d)})
 for p in r.glob('logs/optimizer-rank-*.jsonl'):
  rows=[json.loads(l) for l in p.read_bytes().splitlines()];x=[v for v in rows if v.get('kind')=='OPTIMIZER_WRITE'];o['rank'].append({'path':str(p),'last':scalar(x[-1]) if x else None})
 for p in list(r.glob('*.log'))+list(r.glob('logs/*training*.log')):
  rows=[]
  for line in p.open(errors='replace'):
   if not any(x in line for x in ('train/perf/em','critic/score/mean','val/em/all')):continue
   m={k:float(v) for k,v in re.findall(r'([\w/.-]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',line)};s=re.search(r'\bstep:\s*(\d+)',line)
   if s:m['iteration']=int(s[1])
   if m:rows.append({k:v for k,v in m.items() if k in ['iteration','train/perf/em','critic/score/mean','val/em/all','val/qa_count/all','val/f1/all']})
  if rows:o['metrics'].append({'path':str(p),'bytes':p.stat().st_size,'rows':rows})
 O['roots'].append(o)
for cmd in [['squeue','-h','-j','3400,3399','-o','%i|%T|%N|%M'],['sacct','-nP','-X','-j','721485,719225','--format=JobID,State,ExitCode,NodeList']]:
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);O[str(cmd)]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(O))
