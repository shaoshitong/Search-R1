from pathlib import Path
import json,datetime,hashlib,re,subprocess
O={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':subprocess.run(['hostname'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True).stdout.strip(),'runs':[]}
def compact(x,depth=0):
 if depth>6:return None
 if isinstance(x,dict):return {k:compact(v,depth+1) for k,v in x.items() if not any(s in k.lower() for s in ['question_ids','prompt','tokens','tensor','environment','rank_receipts','responses']) and (not isinstance(v,list) or len(v)<30)}
 if isinstance(x,list):return [compact(v,depth+1) for v in x] if len(x)<30 else {'length':len(x)}
 return x
roots=['/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011','/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015','/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014','/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015']
for root in roots:
 r=Path(root);o={'root':root,'exists':r.exists(),'json':[],'training':[],'named':[]}
 if not r.exists():O['runs'].append(o);continue
 paths=set()
 for pat in ['controller-state.json','recovery-plan.json','logs/actual-updates.jsonl','logs/*development*.json','logs/pipeline-status.json','common-scoring-v4/common-validation/*.receipt.json','progress*.json','complete*.json','*acceptance*.json','*result*.json','logs/*progress*.json','logs/*summary*.json','logs/*complete*.json','final-control/*/*.json']:
  paths.update(r.glob(pat))
 for p in sorted(paths):
  if p.stat().st_size>1500000:continue
  raw=p.read_bytes()
  try:d=json.loads(raw) if p.suffix!='.jsonl' else [json.loads(l) for l in raw.splitlines()]
  except:continue
  if p.name=='actual-updates.jsonl':
   o['updates_count']=len(d);o['updates']=d
   d=[compact(x) for x in d[-12:]]
  else:d=compact(d)
  o['json'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':d})
 if 'cosearch-instruct' in root:
  job=json.loads((r/'controller-state.json').read_text())['job']
  for p in r.glob('common-scoring-v4/common-training/*/completed-outer-*.json'):
   d=json.loads(p.read_text());i=d['step']
   if i>100 and p.parent.name!=str(job):continue
   m=d['all_actual_generated_responses'];o['training'].append({'iteration':i,'em':m['em'],'correct':m['correct'],'path':str(p)})
 for pat in ['*','logs/*','results/*','evaluation/*','artifacts/*']:
  o['named'] += [{'path':str(p.relative_to(r)),'bytes':p.stat().st_size if p.is_file() else None} for p in r.glob(pat)][:110]
 O['runs'].append(o)
O['queue']=subprocess.run(['squeue','-h','-j','721485,721542,719225','-o','%i|%T|%N|%M'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True).stdout
print(json.dumps(O))
