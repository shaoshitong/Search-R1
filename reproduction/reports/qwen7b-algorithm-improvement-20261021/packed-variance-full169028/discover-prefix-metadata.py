import pathlib,json,hashlib
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');todo=[A/'joint169028-native-training-bank-v1/complete.json',A/'hpc3-tail-full169028-qualified-raw-union-v1/complete.json'];seen=set();events={};prefixes={};unknown=[];metadata=[]
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def walk(v,key=''):
 if isinstance(v,dict):
  if {'path','bytes','sha256'}<=set(v):
   p=pathlib.Path(v['path'])
   if p.name in('source-events.json','source-refs.json')or(p.name=='complete.json'and key in('complete','raw_complete','full_raw_G10_acceptance')):
    if any(w in str(p)for w in('final4096','confirmation-materialization')):return
    if p not in seen:todo.append(p)
   if key=='prefix':prefixes[str(p)]=dict(v)
  for k,x in v.items():walk(x,k)
 elif isinstance(v,list):
  for x in v:walk(x,key)
while todo:
 p=todo.pop()
 if p in seen:continue
 seen.add(p)
 if len(seen)>150:raise RuntimeError('Metadata traversal too broad')
 if not p.exists():unknown.append(str(p));continue
 raw=p.read_bytes()
 if len(raw)>3000000:raise RuntimeError('Unexpected large metadata'+str(p))
 v=json.loads(raw);metadata.append(proof(p))
 if p.name=='source-events.json':
  events[str(p)]=dict(keys=list(v)if isinstance(v,dict)else None,length=len(v),sample=v['events'][0]if isinstance(v,dict)and'events'in v else None)
  # Event file already contains prefix refs; no need to follow per-batch complete receipts.
  def only(v,key=''):
   if isinstance(v,dict):
    if key=='prefix'and {'path','bytes','sha256'}<=set(v):prefixes[v['path']]=dict(v)
    for k,x in v.items():only(x,k)
   elif isinstance(v,list):
    for x in v:only(x,key)
  only(v)
 else:walk(v)
out=dict(metadata_files=len(metadata),source_event_files=len(events),prefix_files=len(prefixes),unknown=unknown,source_events=list(events),prefix_first=list(prefixes.values())[:3],prefix_missing=[r for r in list(prefixes.values())[:8]if not pathlib.Path(r['path']).exists()],metadata=metadata)
print(json.dumps(out))
