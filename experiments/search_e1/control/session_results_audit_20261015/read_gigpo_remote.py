from pathlib import Path
import json,datetime,re,hashlib,subprocess,ast
O={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':subprocess.run(['hostname'],capture_output=True,text=True).stdout.strip(),'runs':[]}
def scalar(v):
 if isinstance(v,dict):return {k:scalar(x) for k,x in v.items() if isinstance(x,(str,int,float,bool,type(None),dict)) and not any(s in k.lower() for s in ('question_ids','token','prompt','tensor','environment'))}
 return v
for root in ['/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3347-1791494104474180576','/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2','/mnt/data/shaoshitong/search-g1-qprobe3-resume5-fb68e88-20261015']:
 r=Path(root);o={'root':root,'exists':r.exists(),'json':[],'metrics':[],'named':[]}
 if not r.exists():O['runs'].append(o);continue
 paths=set()
 for pat in ['logs/actual-updates.jsonl','logs/*development*.json','logs/*pipeline*.json','preparation/terminal3399-once-v1/RESULT.json','preparation/saved-pair-recovery-fb68e88-once-v2/*.json','preparation/*submit*.json','preparation/*result*.json','artifacts/question-probe/completed-pairs/global_step_5/source-receipt.json']:
  paths.update(r.glob(pat))
 for p in sorted(paths):
  if p.stat().st_size>1500000:continue
  raw=p.read_bytes()
  try:d=json.loads(raw) if p.suffix!='.jsonl' else [json.loads(l) for l in raw.splitlines()]
  except:continue
  if isinstance(d,list):d=[scalar(x) for x in d]
  else:d=scalar(d)
  o['json'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':d})
 for p in [r/'calibration.log',r/'logs/search-g1-common8-training.log',r/'logs/training.log']:
  if not p.exists():continue
  rows=[]
  for line in p.open(errors='replace'):
   line=re.sub(r'\x1b\[[0-9;]*m','',line)
   vals={k:float(v) for k,v in re.findall(r'([\w/.-]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',line)}
   if '{' in line and ('val/' in line):
    try:vals.update({k:v for k,v in ast.literal_eval(line[line.index('{'):line.rfind('}')+1]).items() if isinstance(v,(int,float))})
    except:pass
   step=re.search(r'\bstep:\s*(\d+)',line)
   if step:vals['iteration']=int(step[1])
   if 'critic/score/mean' in vals or 'val/em/all' in vals:rows.append(vals)
  o['metrics'].append({'path':str(p),'bytes':p.stat().st_size,'rows':rows})
 for pat in ['logs/*','preparation/*','artifacts/question-probe/completed-pairs/*']:
  o['named'] += [str(p.relative_to(r)) for p in r.glob(pat) if not p.is_dir()][:100]
 O['runs'].append(o)
O['queue']=subprocess.run(['squeue','-h','-j','3399','-o','%i|%T|%N'],capture_output=True,text=True).stdout
print(json.dumps(O))
