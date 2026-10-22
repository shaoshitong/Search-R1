"""Independently bind viewed dev/fixedB labels to actual source seals and own EM.

Only development/fixed-train metadata and their existing selected source files.
No model, tensors, inference, final holdouts, label changes, or Slurm calls.
"""
import gzip,hashlib,json,pathlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
REVISION='official7b-a09a35458c702b33eeacc393d103063234e8bc28'
def safe(p):
 p=pathlib.Path(p);assert not any(s in str(p)for s in('final4096','confirmation','stability2048'));return p
def read(p):return json.loads(safe(p).read_bytes())
def proof(p):
 p=safe(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=safe(r['path']);assert proof(p)=={k:r[k]for k in('path','bytes','sha256')},'Source SHA differs '+str(p);return p
def refs(x):
 if isinstance(x,dict):
  if {'path','bytes','sha256'}<=set(x):yield{k:x[k]for k in('path','bytes','sha256')}
  else:
   for v in x.values():yield from refs(v)
 elif isinstance(x,list):
  for v in x:yield from refs(v)
def runids(x):
 out=[]
 if isinstance(x,dict):
  for k,v in x.items():
   if 'run_id'in k and isinstance(v,(str,int)):out.append((k,v))
   else:out+=runids(v)
 elif isinstance(x,list):
  for v in x:out+=runids(v)
 return sorted(set(out))
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
AUDIT_POOL='viewed_development2048_G5'
start=time.time();pf=A/'online-packed-matched-L15P80-8192-20261023-v1/fixture/fixture-complete.json';pilot=read(pf);causal=read(checked(pilot['source_fixture']));selected={(r['path'],r['bytes'],r['sha256'])for r in causal['source_full']}
validation=read(checked(pilot['validation_groups']));fixed=read(checked(pilot['fixed_train_G5']));train=read(checked(pilot['train_groups']));assert len(validation)==len(fixed)==2048
pools={'viewed_development2048_G5':list(validation.values()),'fixed_trainB2048_G5':fixed};expected={};observed={};summaries={}
for name,gs in pools.items():
 e={};qset=set()
 for g in gs:
  q=g['qid'];qset.add(q);assert len(g['trajectory_ids'])==len(g['labels'])==5
  for tid,y in zip(g['trajectory_ids'],g['labels']):assert tid not in e;e[tid]=(q,int(y))
 assert len(e)==10240 and len(qset)==2048;expected[name]=e;observed[name]={};summaries[name]=dict(questions=2048,trajectories=10240,source_events_read=0,full_files_fresh_SHA_verified=0,seal_bindings_verified=0,own_em_rows_read=0,missing=0,duplicate_source_identity=0,label_mismatches=0,wrong_question=0,wrong_actor=0,examples=[],execution_run_ids=[])
assert not(set(expected['viewed_development2048_G5'])&set(expected['fixed_trainB2048_G5']))
va_path=A/'validation-g10-20261021/materialized/bank/source-audit.json';va=read(va_path);ti_path=A/'peer-newseed8192-g10-train-v1/training-input.json';ti=read(ti_path);raw=read(checked(ti['raw_complete']));events_path=checked(raw['outputs']['source-events']);te=read(events_path);bindings=[];unselected=0
batches=[]
for event,bs in va['events'].items():
 for b in bs:batches.append(('viewed_development2048_G5',event,b['batch'],b['full_completion'],b['phase_payloads']['full'],b['phase_payloads']['own_em']))
for event in te['events']:
 for b in event['batches']:batches.append(('fixed_trainB2048_G5',event['event'],b['seal'],None,b['full'],b['own_em']))
for name,event,seal_ref,full_completion,full_ref,em_ref in batches:
 if name!=AUDIT_POOL:continue
 key=(full_ref['path'],full_ref['bytes'],full_ref['sha256'])
 if key not in selected:unselected+=1;continue
 s=summaries[name];s['source_events_read']+=1;seal=read(checked(seal_ref));sealed=list(refs(seal));completion=None
 if full_completion is not None:completion=read(checked(full_completion));sealed+=list(refs(completion))
 # Copied/materialized paths may differ; payload identity is the exact size+SHA.
 for r in(full_ref,em_ref):assert any(z['bytes']==r['bytes']and z['sha256']==r['sha256']for z in sealed),'Actual selected source not bound by its real seal: '+r['path']
 s['seal_bindings_verified']+=1;ids=runids(seal)+(runids(completion)if completion else[])
 for identity in ids:
  if identity not in s['execution_run_ids']:s['execution_run_ids'].append(identity)
 erows=[]
 with checked(em_ref).open()as f:
  for line in f:
   r=json.loads(line);s['own_em_rows_read']+=1;tid=r['trajectory_id']
   if tid in expected[name]:erows.append(r)
 if not erows:continue
 checked(full_ref);s['full_files_fresh_SHA_verified']+=1
 binding=dict(pool=name,event=event,seal={k:seal_ref[k]for k in('path','bytes','sha256')},full={k:full_ref[k]for k in('path','bytes','sha256')},own_em={k:em_ref[k]for k in('path','bytes','sha256')},execution_run_ids=ids);bindings.append(binding)
 for r in erows:
  tid=r['trajectory_id'];q,y=expected[name][tid];entry=(r['qid'],int(r['strict_em']),r.get('actor_version'),full_ref['sha256'],em_ref['sha256'],seal_ref['sha256'])
  if tid in observed[name]:s['duplicate_source_identity']+=1
  observed[name].setdefault(tid,[]).append(entry)
  bad=[]
  if r['qid']!=q:s['wrong_question']+=1;bad.append('wrong_qid')
  if r.get('actor_version')!=REVISION:s['wrong_actor']+=1;bad.append('wrong_actor')
  if int(r['strict_em'])!=y:s['label_mismatches']+=1;bad.append('label_mismatch')
  if bad and len(s['examples'])<8:s['examples'].append(dict(reasons=bad,qid=q,trajectory_id=tid,fixture_label=y,source_own_EM=int(r['strict_em']),source_binding=binding))
for name,s in summaries.items():
 if name!=AUDIT_POOL:s['status']='NOT_RUN_IN_THIS_BOUNDED_PASS';continue
 s['missing']=len(set(expected[name])-set(observed[name]));s['unique_selected_own_EM_rows']=len(observed[name]);s['all_selected_labels_equal_authoritative_own_EM']=not any(s[k]for k in('missing','duplicate_source_identity','label_mismatches','wrong_question','wrong_actor'));s['observed_selected_identity_and_labels_SHA256']=sha(sorted((tid,values)for tid,values in observed[name].items()));s['status']='ALL_10240_SELECTED_LABELS_OWN_EM_AND_ACTUAL_FULL_SEAL_VERIFIED'if s['all_selected_labels_equal_authoritative_own_EM']else'SOURCE_OR_LABEL_UNRESOLVED_DO_NOT_CLAIM_VALID'
out=dict(time=time.time(),seconds=time.time()-start,status='VIEWED_DEV_AND_FIXEDB_ACTUAL_SOURCE_LABELS_VERIFIED'if summaries[AUDIT_POOL]['all_selected_labels_equal_authoritative_own_EM']else'VIEWED_LABEL_SOURCE_AUDIT_REQUIRES_ACTION',pilot_fixture=proof(pf),causal_fixture=pilot['source_fixture'],validation_groups=pilot['validation_groups'],fixed_train_G5=pilot['fixed_train_G5'],pilot_train_groups=pilot['train_groups'],dev_source_audit=proof(va_path),train_source_events=raw['outputs']['source-events'],training_input=proof(ti_path),selected_actual_full_source_count=len(selected),ignored_unselected_source_events=unselected,source_binding_count=len(bindings),source_bindings_SHA256=sha(bindings),source_binding_examples=bindings[:2]+bindings[-2:],pools=summaries,scope='Exact viewed ownG5 labels and actual selected complete source payloads; not a claim about all full169028 labels or final holdouts',model_or_tensor_loaded=False,new_inference=False,Slurm_mutated=False,final4096_read=False)
print(json.dumps(out))
