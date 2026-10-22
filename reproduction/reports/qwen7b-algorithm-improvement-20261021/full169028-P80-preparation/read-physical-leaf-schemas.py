import json,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(Path(p).read_bytes())
def refs(x):
 if isinstance(x,dict):
  if {'path','bytes','sha256'}<=x.keys():yield x;return
  for v in x.values():yield from refs(v)
 elif isinstance(x,list):
  for v in x:yield from refs(v)
out=[];seen=set()
def walk(ref):
 if ref['path']in seen:return
 seen.add(ref['path']);x=read(ref['path']);children=x.get('source_refs',[])
 if children:
  for e in children:walk(e['complete'])
 else:
  gr=x.get('groups')or x.get('outputs',{}).get('groups.json');s=dict(complete=ref,status=x.get('status'),complete_keys=sorted(x),groups_ref=gr)
  if gr:
   g=read(gr['path']);g=g.get('train',g);g=list(g.values())if isinstance(g,dict)else g;s.update(questions=len(g),first_keys=sorted(g[0]),first_provenance={k:v for k,v in g[0].items()if 'seal'in k or'execution'in k or 'source'in k})
  for k in('raw_complete','raw_groups','source_resolution_contract','source_events','preparation'):
   if k in x:s[k]=x[k]
  out.append(s)
p=A/'joint169028-native-training-bank-v1/complete.json';walk(dict(path=str(p)))
print(json.dumps(dict(time=time.time(),leaf_banks=out,final4096_read=False)))
