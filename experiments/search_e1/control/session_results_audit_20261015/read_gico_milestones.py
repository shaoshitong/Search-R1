from pathlib import Path
import json,datetime,sys,re,hashlib
O={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':[]}
for root in json.loads(sys.argv[1]):
 r=Path(root);o={'root':root,'keys':{},'found':[],'log_eval':[]}
 if not r.exists():O['roots'].append(o);continue
 if 'gigpo-e1-fresh' in root:
  paths=sorted(r.glob('calibration-attempt*.json'),key=lambda p:int(re.findall(r'attempt(\d+)',p.name)[0]));o['calibrations']=[p.name for p in paths[-3:]]
  paths=paths[-1:]+[r/'completed200-human-nonblocking-state-roundtrip.json',r/'final200-state-roundtrip.json']
  for p in paths:
   if not p.exists():continue
   d=json.loads(p.read_bytes());o['keys'][str(p)]=list(d) if isinstance(d,dict) else type(d).__name__
   def walk(v,key=''):
    if isinstance(v,dict):
     for k,x in v.items():
      nk=key+'/'+k
      if isinstance(x,(int,float,str,bool)) and any(a in k.lower() for a in ('step','iter','reward','em','count','accuracy','score','completed','update')):o['found'].append([nk,x])
      elif isinstance(x,(dict,list)):walk(x,nk)
    elif isinstance(v,list):
     for i,x in enumerate(v[-12:]):
      if isinstance(x,(dict,list)):walk(x,key+'/'+str(i-len(v)))
   walk(d,str(p))
 else:
  for p in r.glob('**/*.json'):
   s=str(p.relative_to(r))
   if any(x in s for x in ('repo/','source/','cache/','ray/','env/','question-features/')):continue
   if any(x in s.lower() for x in ['validation','evaluation','dev512','fixed512','stage512']):
    o['found'].append({'path':str(p),'bytes':p.stat().st_size})
  p=r/'logs/search-g1-common8-training.log'
  if p.exists():
   for line in p.open(errors='replace'):
    if 'val/' not in line or 'step:' not in line:continue
    m={k:float(v) for k,v in re.findall(r'([\w/.-]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',line)};s=re.search(r'\bstep:\s*(\d+)',line)
    if s:o['log_eval'].append({'iter':int(s[1]),'metrics':{k:v for k,v in m.items() if k.startswith('val/')}})
 O['roots'].append(o)
print(json.dumps(O))
