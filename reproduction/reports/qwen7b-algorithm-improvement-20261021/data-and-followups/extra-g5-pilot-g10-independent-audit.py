# Independent CPU SHA / identity / K audit after the one-time immutable merge.
# No model/tensor load, sampling, submission, or raw output is performed.
import pathlib,json,hashlib,collections,time,os
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
b=B/'algorithm-improvement-20261021/training-extraG5-128-20261021/materialized'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(item):
 actual=fp(item['path']);assert actual['bytes']==item['bytes'] and actual['sha256']==item['sha256'],item['path'];return pathlib.Path(item['path'])
def canonical_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,v):
 t=p.with_suffix('.tmp')
 with t.open('w') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 t.replace(p)
p=b/'bank/complete.json'
if not p.exists():
 out=dict(status='G10_MERGE_NOT_YET_COMPLETE',observed_epoch=time.time())
 controller=b/'bank/merge-controller.json'
 if controller.exists():
  c=read(controller);pid=c['pid'];live=False
  try:live=pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]==str(c['start_ticks'])
  except OSError:pass
  out.update(merge_pid=pid,merge_identity_alive=live)
 log=b/'bank/extra-g5-merge.log'
 if log.exists():
  with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000));raw=f.read().decode(errors='replace')
  out['merge_log_bytes']=log.stat().st_size
  out['error_lines']=[line[:250] for line in raw.splitlines() if any(t in line for t in ('Error','Traceback','Exception'))][-6:]
 print(json.dumps(out));raise SystemExit(0)
c=read(b/'config.json');assert c['source_commit']=='c8b02d3b22b6a3267481e7c632b2f7113b018e93'
for relative,digest in c['code_files_sha256'].items():assert sha(pathlib.Path(c['code_root'])/relative)==digest
complete=read(p);assert complete['status']=='PILOT_ACTUAL_ORIGINAL5_PLUS_EXTRA5_G10_SHA_OWN_EM_VERIFIED'
assert complete['questions']==128 and complete['trajectories']==1280 and complete['extra_real_trajectories']==640
assert complete['original_fixed5_preserved'] is True and complete['synthetic_EM'] is False and complete['actor_updates']==0
plan=read(b/'inputs/questions.json');qids=[g['qid'] for g in plan['groups']]
assert len(qids)==len(set(qids))==128 and canonical_sha(qids)==plan['ordered_qid_sha256']==complete['ordered_qid_sha256']
original_bank=read(checked(complete['original_G5_bank']));old_by_qid={g['qid']:g for g in original_bank['train']}
original_features=checked(complete['original_feature_manifest'])
for item in plan['original_batches']+plan['original_input_batches']:checked(item)
original_native={}
for item in plan['original_batches']:
 if pathlib.Path(item['path']).name!='complete.json':continue
 batch=read(item['path']);source=batch['prefix']['actor_hidden'];checked(source)
 original_native[source['path']]=source
new_groups={};new_feature_sources=[];shards=[]
for descriptor in complete['extra_shards']:
 r=read(checked(descriptor));assert r['status']=='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED' and r['run_id']==c['run_id']
 assert r['actor_updates']==0 and r['original_G5_modified'] is False and r['synthetic_EM'] is False
 assert r['questions']==32 and r['trajectories']==160
 for group in r['groups']:
  assert group['qid'] not in new_groups;new_groups[group['qid']]=group
 new_feature_sources.extend(r['independent_G5_feature_sources']);shards.append(descriptor)
assert set(new_groups)==set(qids) and len(shards)==4
sources=read(checked(complete['extra_G5_feature_sources']));assert sources==new_feature_sources
payload_count=payload_bytes=0
for descriptor in sources:
 m=read(checked(descriptor));assert m['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE' and m['feature_layer']==26 and m['dimension']==3584 and m['question_max_tokens']==273 and m['prefix_tokens']==20
 for item in m['files'].values():checked(item);payload_count+=1;payload_bytes+=item['bytes']
groups=read(checked(complete['groups']))['train'];assert [g['qid'] for g in groups]==qids
oldK=collections.Counter();newK=collections.Counter();K=collections.Counter();old_ids=set();new_ids=set();transitions=collections.Counter()
for group,declared in zip(groups,plan['groups']):
 qid=group['qid'];old=old_by_qid[qid];new=new_groups[qid];assert old==declared
 assert group['question_hash']==old['question_hash']==new['question_hash']
 assert group['trajectory_ids']==old['trajectory_ids']+new['trajectory_ids']
 assert group['labels']==old['labels']+new['labels'] and all(type(y) is int and y in (0,1) for y in group['labels'])
 assert group['original_fixed5_trajectory_ids']==old['trajectory_ids'] and group['original_fixed5_labels']==old['labels']
 assert new['response_indices']==group['extra_response_indices']==list(range(5,10))
 assert group['original_generation_run_id']==c['original_run_id'] and group['extra_generation_run_id']==c['run_id'] and group['extra_sampling_attempt_id']==c['sampling_attempt_id']
 assert len(group['trajectory_ids'])==len(set(group['trajectory_ids']))==10
 old_ids.update(old['trajectory_ids']);new_ids.update(new['trajectory_ids'])
 a=sum(old['labels']);z=sum(new['labels']);oldK[a]+=1;newK[z]+=1;K[a+z]+=1
 def label(k,g):return 'wrong' if k==0 else 'correct' if k==g else 'mixed'
 transitions[label(a,5)+'->'+label(a+z,10)]+=1
assert len(old_ids)==len(new_ids)==640 and not old_ids&new_ids
summary=dict(status='PILOT_G10_INDEPENDENT_CPU_SHA_IDENTITY_K_AUDIT_PASS',observed_epoch=time.time(),source_commit=c['source_commit'],base=str(b),complete=fp(p),groups=complete['groups'],extra_feature_sources=complete['extra_G5_feature_sources'],original_feature_manifest=complete['original_feature_manifest'],original_G5_bank=complete['original_G5_bank'],extra_native_feature_manifests=sources,original_selected_native_manifest_count=len(original_native),original_selected_native_manifest_proof_sha256=canonical_sha(sorted(original_native.values(),key=lambda x:x['path'])),fresh_extra_feature_payload_files=payload_count,fresh_extra_feature_payload_bytes=payload_bytes,questions=128,trajectories=1280,old5_trajectories=640,new5_trajectories=640,old_new_ID_intersection=0,response_indices=list(range(5,10)),K10_histogram={str(k):K[k] for k in range(11)},original_K5_histogram={str(k):oldK[k] for k in range(6)},extra_K5_histogram={str(k):newK[k] for k in range(6)},original_mixed_questions=sum(n for k,n in oldK.items() if 0<k<5),extra_mixed_questions=sum(n for k,n in newK.items() if 0<k<5),G10_mixed_questions=sum(n for k,n in K.items() if 0<k<10),original5_to_G10_group_transitions=dict(transitions),original5_preserved_exactly=True,extra_real_G5_complete=True,actor_optimizer_updates=0,model_or_tensor_loaded=False,full_original_feature_payload_bank_rehashed=False)
atomic(b/'bank/independent-g10-audit.json',summary)
print(json.dumps(summary))
