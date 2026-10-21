"""Independent stdlib source/index/retained-row audit after an owned capture.

Reads metadata and identity sidecars only. Inherits the producer's in-allocation
full payload SHA and reconstructed feature-row verification; never loads tensors.
"""
import argparse,fcntl,gzip,hashlib,itertools,json,os,resource,subprocess,time
from pathlib import Path

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def proof(path):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def read(path):return json.loads(Path(path).read_bytes())
def check(ref):
 actual=proof(ref['path'])
 if any(actual[k]!=ref[k]for k in('path','bytes','sha256')):raise ValueError('Metadata SHA mismatch '+ref['path'])
 return Path(ref['path'])
def rows(path):
 with gzip.open(path,'rt')as f:
  for line in f:yield json.loads(line)
def audit(base,job,questions):
 start=time.time();b=Path(base);complete_ref=proof(b/'bank/complete.json');done=read(complete_ref['path']);n=questions*10
 assert done['status']=='FIXED_HPC3_TAIL%d_OWN_G10_LAYER26_SOURCE_RESOLVED_FEATURE_BANK_ACCEPTED'%questions
 assert(done['questions'],done['trajectories'],done['capture_layer'],done['prefix_tokens'],done['question_max_tokens'])==(questions,n,26,20,273)
 assert done['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
 for flag in('physical_execution_prefix_and_retained_row_SHA_binding_complete','producer_full_payload_SHA_preserved','metadata_join_rehashes_tensor_payload','all_retained_rows_reconstructed_SHA_and_finite_verified'):assert done[flag]is True
 assert done['donor_feature_union_performed']is False and done['actor_updates']==done['probe_updates']==done['new_responses_generated']==0
 submission=read(b/'submission-result.json');assert str(submission['job_id'])==str(job)
 terminal=subprocess.check_output(['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State,ExitCode'],universal_newlines=True).strip().splitlines();main=[x for x in terminal if x.split('|')[0]==str(job)];assert len(main)==1 and main[0].split('|')[1:3]==['COMPLETED','0:0']
 handles=[]
 for p in(b/'single-gpu-capture.lock',b/'bank/merge.lock'):
  f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
 prep=read(check(done['preparation']));raw=read(check(done['raw_complete']));check(done['source_resolution_contract']);raw_groups=read(check(raw['outputs']['groups']))['train'];groups=read(check(done['outputs']['groups.json']))['train'];assert groups==raw_groups and len(groups)==questions
 expected={t:g['qid']for g in groups for t in g['trajectory_ids']};assert len(expected)==n and len({g['qid']for g in groups})==questions and all(len(g['trajectory_ids'])==len(g['labels'])==10 and set(g['labels'])<=set((0,1))for g in groups)
 manifests=read(check(done['outputs']['features-layer26.json']));side_refs=read(check(done['outputs']['source-bound-retained-rows.json']));assert len(side_refs)==len(prep['selections'])==len(done['shard_completes'])==4
 feature_rows={};by_manifest={};packed=0
 for ref in manifests:
  m=read(check(ref));assert(m['capture_layer'],m['prefix_tokens'],m['question_max_tokens'],m['dimension'])==(26,20,273,3584)and m['layout']=='lossless_question_base_xor_policy_v1'and m['model_revision']==done['model_revision'];count=0
  for item in m['files'].values():assert Path(item['path']).stat().st_size==item['bytes'];packed+=item['bytes']
  index_ref=m['files']['index.jsonl']
  with check(index_ref).open()as f:
   for line in f:
    row=json.loads(line);t=row['trajectory_id'];assert t not in feature_rows and expected[t]==row['qid']and len(row['retained_feature_sha256'])==64 and len(row['source_prefix_sha256'])==64
    feature_rows[t]=(row['qid'],row['source_prefix_sha256'],row['retained_feature_sha256'],ref,index_ref);count+=1
  assert count==m['count'];by_manifest[ref['path']]=ref
 assert len(feature_rows)==n and len(manifests)==done['manifests']and packed==done['packed_bytes']
 seen=set();executions={};physical_prefix={};physical_seal={};shards=[]
 for i,(selection_ref,side_ref,cap_ref)in enumerate(zip(prep['selections'],side_refs,done['shard_completes'])):
  selection=read(check(selection_ref));cap=read(check(cap_ref));assert cap['source_selection']==selection_ref and cap['feature_manifests_by_capture_layer']['26']==[r for r in manifests if r in cap['feature_manifests_by_capture_layer']['26']];wanted={x['trajectory_id']:x['qid']for x in selection['trajectories']};count=0
  original={x['trajectory_id']:x for x in rows(check(selection['chosen_physical_prefix_bindings']))};bound=rows(check(side_ref));shard_seen=set();assert set(original)==set(wanted)
  for z in bound:
   t=z['trajectory_id'];a=original[t];assert all(z[k]==v for k,v in a.items());assert t not in seen and t not in shard_seen and wanted[t]==z['qid'];qid,psha,rsha,mref,iref=feature_rows[t];assert z['qid']==qid and z['prefix_body_sha256']==psha and z['feature_retained_row_sha256']==rsha and z['feature_manifest']==mref and z['feature_index']==iref and len(z['canonical_prefix_row_sha256'])==64
   kind=z['chosen_execution'];executions[kind]=executions.get(kind,0)+1
   for key,dest in(('physical_prefix',physical_prefix),('physical_seal',physical_seal)):
    item=z[key]
    if item['path']in dest:assert dest[item['path']]==item
    else:dest[item['path']]=item
   shard_seen.add(t);count+=1
  assert shard_seen==set(wanted)and count==cap['trajectories'];seen.update(shard_seen);shards.append(count)
 assert seen==set(expected)and executions.get('original_canonical_old5')==questions*5 and sum(executions.values())==n
 # Physical bytes have producer/pre-submit full-SHA proofs; do not repeat large files.
 for item in list(physical_prefix.values())+list(physical_seal.values()):assert Path(item['path']).stat().st_size==item['bytes']
 assert proof(complete_ref['path'])==complete_ref
 qdigest=hashlib.sha256(canonical(sorted(g['qid']for g in groups))).hexdigest();tdigest=hashlib.sha256(canonical(sorted(expected))).hexdigest()
 out=dict(status='INDEPENDENT_SOURCE_BOUND_LAYER26_METADATA_AND_ALL_RETAINED_ID_LINKS_ACCEPTED',complete=complete_ref,groups=done['outputs']['groups.json'],features_layer26=done['outputs']['features-layer26.json'],source_bound_retained_rows=done['outputs']['source-bound-retained-rows.json'],raw_complete=done['raw_complete'],source_resolution_contract=done['source_resolution_contract'],preparation=done['preparation'],questions=questions,trajectories=n,manifests=len(manifests),packed_bytes=packed,shard_rows=shards,execution_counts=executions,physical_prefix_files=len(physical_prefix),physical_seal_files=len(physical_seal),sorted_qid_json_sha256=qdigest,sorted_own_trajectory_ids_json_sha256=tdigest,all_own_IDs_and_EM_same_as_qualified_raw=True,all_physical_prefix_canonical_row_and_retained_feature_row_SHA_links_verified=True,producer_in_allocation_full_payload_SHA_and_row_reconstruction_inherited=True,metadata_indices_and_sidecars_fresh_full_SHA=True,large_payload_rehashed_by_this_auditor=False,model_or_tensor_loaded=False,auditor_source_sha256=os.environ.get('AUDITOR_SOURCE_SHA256'),job_id=str(job),main_sacct=main[0],capture_and_merge_locks_free=True,CPU_affinity=sorted(os.sched_getaffinity(0)),RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-start,epoch=time.time())
 p=b/'bank/independent-source-bound-reader-audit.json';data=canonical(out)+bytes([10])
 if p.exists():
  old=read(p);assert old['complete']==complete_ref;return dict(receipt=proof(p),audit=old)
 with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 return dict(receipt=proof(p),audit=out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--job-id',required=True);p.add_argument('--questions',type=int,required=True);a=p.parse_args();allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));print(json.dumps(audit(a.base,a.job_id,a.questions),sort_keys=True))
