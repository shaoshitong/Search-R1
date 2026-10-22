"""Small retained-full source graph and per-origin schema samples, no extraction."""
from collections import Counter
import gzip,hashlib,json,os,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');F=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
def refs(v,keys=()):
 if isinstance(v,dict):
  if {'path','bytes','sha256'}<=v.keys():
   if not any(x in v['path']for x in('final4096','confirmation-materialization','confirmation2048','stability2048')):yield keys,{k:v[k]for k in('path','bytes','sha256')}
   return
  for k,x in v.items():yield from refs(x,keys+(str(k),))
 elif isinstance(v,list):
  for x in v:yield from refs(x,keys+('[]',))
d=read(F/'full-training-descriptor.json');found={};roles=Counter();metadata=[]
for ref in d['source_metadata']:
 assert proof(ref['path'])==ref;x=read(ref['path']);metadata.append(ref)
 for keys,r in refs(x):
  roles['.'.join(keys)]+=1
  if keys[-1:]in(('full',),('raw',))and r['path'].endswith(('.gz','.jsonl')):
   old=found.setdefault(r['path'],dict(ref=r,metadata=set(),roles=set()));assert old['ref']==r;old['metadata'].add(ref['path']);old['roles'].add('.'.join(keys))
origins={};missing=[];wrong=[]
for path,item in found.items():
 p=Path(path);kind='|'.join(sorted(item['metadata']))+'::'+'|'.join(sorted(item['roles']));origin=origins.setdefault(kind,dict(files=0,compressed_bytes=0,present_exact_size=0,sample_ref=None));origin['files']+=1;origin['compressed_bytes']+=item['ref']['bytes']
 if not p.is_file():missing.append(path)
 elif p.stat().st_size!=item['ref']['bytes']:wrong.append(path)
 else:
  origin['present_exact_size']+=1
  if origin['sample_ref']is None:origin['sample_ref']=item['ref']
samples=[]
for origin,value in origins.items():
 ref=value['sample_ref']
 if ref is None:continue
 with gzip.open(ref['path'],'rt')as stream:row=json.loads(next(stream))
 out=dict(origin=origin,ref=ref,row_keys=list(row),actor_version=row.get('actor_version'),policy_tokens=row.get('policy_tokens'),strict_em_type=type(row.get('strict_em')).__name__,strict_em_is_binary=row.get('strict_em')in(0,1),full_payload_rehashed=False)
 for key in('behavior_turns','segments','turns','requests'):
  if isinstance(row.get(key),list)and row[key]:
   seq=row[key];out[key]=dict(turns=len(seq),first_keys=list(seq[0])if isinstance(seq[0],dict)else None)
   if key=='behavior_turns':
    out['retained_policy_tokens_from_behavior']=sum(min(t.get('kept_token_count',0),len(t.get('raw_generated_token_ids',[])))for t in seq);out['all_behavior_turns_have_raw_and_kept']=all('raw_generated_token_ids'in t and'kept_token_count'in t for t in seq)
   if key=='segments':out['retained_policy_tokens_from_segments']=sum(t.get('policy_length',0)for t in seq)
 for key in('initial_prompt_token_ids','prompt','policy_token_ids','prefix_tokens','generated_token_ids'):
  if isinstance(row.get(key),list):out[key+'_length']=len(row[key])
 samples.append(out)
v=os.statvfs(A)
print(json.dumps(dict(status='RETAINED_FULL_REF_AND_SCHEMA_SAMPLE_INVENTORY_NOT_ALL_ROWS',time=time.time(),descriptor=proof(F/'full-training-descriptor.json'),declared_questions=d['questions'],declared_trajectories=d['trajectories'],retained_full_files=len(found),retained_full_compressed_bytes=sum(x['ref']['bytes']for x in found.values()),fresh_missing_full_refs=missing[:30],fresh_size_mismatch_full_refs=wrong[:30],origins=origins,schema_samples=samples,metadata_files_fresh_SHA=len(metadata),reference_roles=dict(roles),free_shared_bytes=v.f_bavail*v.f_frsize,full_payloads_rehashed=False,all169028_ownG10_P80_coverage_proven=False,final4096_read=False,checkpoint_or_hidden_tensors_read=False,new_generation=False)))
