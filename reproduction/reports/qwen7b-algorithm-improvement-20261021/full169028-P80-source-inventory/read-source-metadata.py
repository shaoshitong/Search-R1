"""Read accepted training source metadata only; never final/CP/hidden tensors."""
from collections import Counter
import hashlib,json,os,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
F=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def allowed(path):
 return not any(x in str(path)for x in('final4096','confirmation-materialization','stability2048','confirmation2048'))
def refs(value,path=()):
 if isinstance(value,dict):
  if {'path','bytes','sha256'}<=value.keys():
   if allowed(value['path']):yield path,{k:value[k]for k in('path','bytes','sha256')}
   return
  for k,v in value.items():yield from refs(v,path+(str(k),))
 elif isinstance(value,list):
  for v in value:yield from refs(v,path+('[]',))
d=read(F/'full-training-descriptor.json');f=read(F/'fixture-complete.json');assert d['questions']==169028 and d['trajectories']==1690280
gref=d['groups'];assert proof(gref['path'])==gref;raw=read(gref['path']);groups=raw['train']if'train'in raw else raw
if isinstance(groups,list):groups={g['qid']:g for g in groups}
assert len(groups)==169028
ids=set();bad=0
for q,g in groups.items():
 bad+=not(q==g['qid']and len(g['trajectory_ids'])==len(g['labels'])==10 and len(set(g['trajectory_ids']))==10)
 ids.update(g['trajectory_ids'])
assert not bad and len(ids)==1690280
meta=[];allrefs={};prefixrefs={r['path']:r for r in d['prefix_sources']}
for ref in d['source_metadata']:
 assert allowed(ref['path'])and ref['bytes']<3000000 and proof(ref['path'])==ref
 x=read(ref['path']);simple={k:v for k,v in x.items()if isinstance(v,(str,int,float,bool))or v is None}if isinstance(x,dict)else{};roles=Counter()
 for path,r in refs(x):
  key='.'.join(path);roles[key]+=1;item=allrefs.setdefault(r['path'],dict(ref=r,roles=set(),metadata=set()));assert item['ref']==r;item['roles'].add(key);item['metadata'].add(ref['path'])
 meta.append(dict(ref=ref,scalar_fields=simple,reference_role_counts=dict(roles),keys=list(x)if isinstance(x,dict)else['LIST',len(x)]))
full={p:v for p,v in allrefs.items()if any(path.split('.')[-1]in('full','raw')for path in v['roles'])and(p.endswith('.gz')or p.endswith('.jsonl'))}
statuses={}
for name,rs in [('prefix',prefixrefs),('retained_full', {p:v['ref']for p,v in full.items()})]:
 missing=[];size_bad=[];present=0
 for path,ref in rs.items():
  p=Path(path)
  if not p.is_file():missing.append(path)
  elif p.stat().st_size!=ref['bytes']:size_bad.append(path)
  else:present+=1
 statuses[name]=dict(files=len(rs),compressed_bytes=sum(r['bytes']for r in rs.values()),fresh_present_with_exact_size=present,missing=missing[:20],size_mismatch=size_bad[:20],large_payloads_rehashed=False)
v=os.statvfs(A)
report=dict(status='TRAINING_SOURCE_METADATA_INVENTORY_NO_EXTRACTION',time=time.time(),descriptor=proof(F/'full-training-descriptor.json'),accepted_P20_fixture=proof(F/'fixture-complete.json'),full_G10_groups=gref,questions=len(groups),unique_own_trajectories=len(ids),all_groups_own_G10_valid=True,source_metadata=meta,source_payloads=statuses,retained_full_examples=[dict(ref=x['ref'],roles=sorted(x['roles']),metadata=sorted(x['metadata']))for x in list(full.values())[:6]],all_reference_role_counts=dict(Counter(role for x in allrefs.values()for role in x['roles'])),all_reference_file_count=len(allrefs),free_shared_bytes=v.f_bavail*v.f_frsize,filesystem_device=os.stat(A).st_dev,final4096_read=False,checkpoint_or_hidden_tensors_read=False,new_rollouts_or_tokens_generated=False,full_P80_coverage_not_yet_proven=True)
print(json.dumps(report))
