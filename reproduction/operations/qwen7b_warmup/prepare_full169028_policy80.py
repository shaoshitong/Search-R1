"""Persistent CPU-only authentic own-G10 Prefix80 preparation and readback.

No generation, GPU, hidden feature, model or checkpoint access. Original assets
are read-only. Finished chunks survive interruption and are SHA-bound to RAW.
"""
import argparse,fcntl,gc,gzip,hashlib,io,json,os,time,traceback
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
FULL=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
PILOT=A/'online-packed-matched-L15P80-8192-20261023-v1/fixture'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
STATUS='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY'

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for raw in iter(lambda:f.read(8388608),b''):h.update(raw)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=Path(ref['path']);assert proof(p)=={k:ref[k]for k in('path','bytes','sha256')},'Source SHA/size mismatch: '+str(p);return p
def atomic(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid());tmp.write_bytes(canonical(x)+b'\n');tmp.replace(p)
def once(p,x):
 p=Path(p);raw=canonical(x)+b'\n'
 if p.exists():assert p.read_bytes()==raw,'Immutable own-task binding differs: '+str(p)
 else:atomic(p,x)
def refs(value,keys=()):
 if isinstance(value,dict):
  if {'path','bytes','sha256'}<=value.keys():
   if not any(x in value['path']for x in('final4096','confirmation-materialization','confirmation2048','stability2048')):yield keys,{k:value[k]for k in('path','bytes','sha256')}
   return
  for k,v in value.items():yield from refs(v,keys+(str(k),))
 elif isinstance(value,list):
  for v in value:yield from refs(v,keys+('[]',))
def policy80(row):
 """Exactly retained policy tokens, excluding observations and discarded suffix."""
 result=[];retained=0
 assert isinstance(row['behavior_turns'],list)and row['behavior_turns']
 for turn in row['behavior_turns']:
  tokens=turn['raw_generated_token_ids'];kept=turn['kept_token_count'];assert type(kept)is int and 0<=kept<=len(tokens)
  retained+=kept;result.extend(tokens[:min(kept,80-len(result))])
 assert 0<len(result)==min(retained,80)<=80
 return result,retained
def compact_raw(row):
 tokens,total=policy80(row);prompt=row['initial_prompt_token_ids'];assert prompt and all(type(t)is int and t>=0 for t in prompt+tokens)
 return dict(qid=row['qid'],trajectory_id=row['trajectory_id'],role='train',prompt=prompt,policy_token_ids=tokens),total
def writer(path):
 path=Path(path);raw=path.open('wb');packed=gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0)
 return io.TextIOWrapper(packed,encoding='utf8'),raw
def row_fingerprint(row):return hashlib.sha256(canonical(row)).digest()

def run(base,chunk_sources=64):
 base=Path(base);assert base.parent.is_dir()and(FULL/'full-training-descriptor.json').is_file()and(PILOT/'accepted.json').is_file(),'Real existing shared mount required'
 base.mkdir(exist_ok=True);guard=(base/'prepare.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 fixture=base/'fixture';inputs=fixture/'inputs';pieces=fixture/'token-shards';receipts=base/'chunk-receipts'
 for p in(fixture,inputs,pieces,receipts):p.mkdir(exist_ok=True)
 publication=read(base/'source-publication.json');code=Path(__file__).resolve();assert proof(code)==dict(publication['files'][publication['producer_relative']],path=str(code))
 if(fixture/'accepted.json').is_file():
  accepted=read(fixture/'accepted.json');assert accepted['status']=='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_ACCEPTED'and accepted['fixture']==proof(fixture/'fixture-complete.json')and accepted['producer']==proof(code)
  if not(base/'complete.json').exists():atomic(base/'complete.json',dict(status='FULL169028_POLICY80_CPU_PREPARATION_AND_ACCEPTANCE_COMPLETE',fixture=accepted['fixture'],acceptance=proof(fixture/'accepted.json'),time=time.time(),recovered_existing_acceptance=True,new_rollouts_generated=0,final4096_read=False))
  print(json.dumps(dict(status='EXISTING_COMPLETE_POLICY80_ACCEPTANCE_REUSED',acceptance=proof(fixture/'accepted.json'))),flush=True);return
 controller=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source_publication=proof(base/'source-publication.json'),code=proof(code),started=time.time(),final4096_read=False,GPU_used=False,new_rollouts_generated=0);atomic(base/'controller.json',controller);started=time.time()
 assert proof(FULL/'full-training-descriptor.json')['sha256']=='fa55a95f6fe2601519f7678519324f89aa5037bfc5ed4cb4c1fd44c92e84b436'
 descriptor=read(FULL/'full-training-descriptor.json');assert descriptor['questions']==169028 and descriptor['trajectories']==1690280
 raw=read(checked(descriptor['groups']));old=raw['train']if isinstance(raw,dict)and'train'in raw else raw;old=list(old.values())if isinstance(old,dict)else old
 original_qids=[g['qid']for g in old];assert digest(original_qids)==descriptor['canonical_ordered_qid_sha256']
 groups={g['qid']:{k:g[k]for k in('qid','trajectory_ids','labels','question_hash','data_source')if k in g}for g in old};del old,raw;gc.collect();qids=sorted(groups);assert len(qids)==169028
 lookup={};labels=bytearray();prompt_hashes={}
 for qi,q in enumerate(qids):
  g=groups[q];assert len(g['trajectory_ids'])==len(g['labels'])==10 and len(set(g['trajectory_ids']))==10
  for tid,y in zip(g['trajectory_ids'],g['labels']):
   assert tid not in lookup and type(y)is int and y in(0,1);lookup[tid]=len(labels);labels.append(y)
 assert len(lookup)==1690280
 seen=bytearray(len(lookup));fingerprints=bytearray(32*len(lookup));unique_rows=0;duplicate_rows=0;pilot_compared=set();lengths={};retained_bins={'shorter_than80':0,'at_least80':0}
 pilot=read(PILOT/'fixture-complete.json');accepted=read(PILOT/'accepted.json');assert accepted['fixture']==proof(PILOT/'fixture-complete.json')and accepted['source_policy80_exact_match']
 validation=read(checked(pilot['validation_groups']));fixed=read(checked(pilot['fixed_train_G5']));pilot_train=read(checked(pilot['train_groups']));assert len(validation)==len(fixed)==2048 and len(pilot_train)==8192
 assert not(set(groups)&set(validation))
 for q,g in pilot_train.items():assert groups[q]['trajectory_ids']==g['trajectory_ids']and groups[q]['labels']==g['labels']
 for g in fixed:
  by=dict(zip(groups[g['qid']]['trajectory_ids'],groups[g['qid']]['labels']));assert len(g['trajectory_ids'])==5 and g['labels']==[by[t]for t in g['trajectory_ids']]
 qrank={q:i%2 for role,gs in(('train',groups),('validation',validation))for i,q in enumerate(sorted(gs))};expected_dev={t:(q,y)for q,g in validation.items()for t,y in zip(g['trajectory_ids'],g['labels'])};assert len(expected_dev)==10240
 for name,ref in(('validation-groups.json',pilot['validation_groups']),('fixed-train-G5.json',pilot['fixed_train_G5'])):
  source=checked(ref);dest=inputs/name
  if dest.exists():assert dest.read_bytes()==source.read_bytes()
  else:dest.write_bytes(source.read_bytes())
 once(inputs/'train-groups.json',groups)
 pilot_expected={};validation_refs=[];dev_seen=set();dev_streams=[];dev_raw=[];dev_temps=[];dev_hashers=[hashlib.sha256(),hashlib.sha256()]
 for rank in(0,1):
  dest=pieces/('validation-rank-%d.jsonl.gz'%rank);tmp=dest.with_name(dest.name+'.tmp');stream,handle=writer(tmp);dev_streams.append(stream);dev_raw.append(handle);dev_temps.append((tmp,dest))
 for ref in pilot['token_shards']:
  with gzip.open(checked(ref),'rt')as stream:
   for line in stream:
    r=json.loads(line);tid=r['trajectory_id'];pilot_expected[tid]=row_fingerprint(r)
    if r['role']=='validation':
     assert tid in expected_dev and tid not in dev_seen and expected_dev[tid][0]==r['qid'];dev_seen.add(tid);rank=qrank[r['qid']];encoded=canonical(r)+b'\n';dev_streams[rank].write(encoded.decode());dev_hashers[rank].update(encoded)
 assert len(pilot_expected)==92160 and len(dev_seen)==10240
 for rank in(0,1):
  dev_streams[rank].close();dev_raw[rank].close();tmp,dest=dev_temps[rank]
  if dest.exists():assert proof(dest)['sha256']==proof(tmp)['sha256'];tmp.unlink()
  else:tmp.replace(dest)
  validation_refs.append(proof(dest))
 full_refs={};source_metadata=[]
 for ref in descriptor['source_metadata']:
  x=read(checked(ref));source_metadata.append(ref)
  for keys,item in refs(x):
   if keys[-1:]in(('full',),('raw',))and item['path'].endswith(('.gz','.jsonl')):
    previous=full_refs.setdefault(item['path'],item);assert previous==item
 sources=sorted(full_refs.values(),key=lambda r:r['path']);assert len(sources)==2678 and sum(r['bytes']for r in sources)==12017369444
 once(base/'input-contract.json',dict(descriptor=proof(FULL/'full-training-descriptor.json'),canonical_groups=descriptor['groups'],pilot_fixture=proof(PILOT/'fixture-complete.json'),pilot_acceptance=proof(PILOT/'accepted.json'),source_metadata=source_metadata,RAW_files=sources,RAW_compressed_bytes=sum(r['bytes']for r in sources),training_questions=169028,training_trajectories=1690280,validation_questions=2048,validation_trajectories=10240,prefix=80,chunk_sources=chunk_sources,rank_routing='sorted train qids parity and sorted validation qids parity independently',new_generation=False,final4096_read=False))
 total_bytes=sum(r['bytes']for r in sources);processed_bytes=0;token_refs=[];chunk_refs=[]
 def progress(stage,source_index,**extra):atomic(base/'progress.json',dict(stage=stage,source_files_completed=source_index,source_files_total=len(sources),RAW_compressed_bytes_completed=processed_bytes,RAW_compressed_bytes_total=total_bytes,unique_train_trajectories=unique_rows,expected_train_trajectories=1690280,duplicate_source_rows_verified=duplicate_rows,seconds=time.time()-started,time=time.time(),final4096_read=False,GPU_used=False,new_rollouts_generated=0,**extra))
 def accept_row(r):
  nonlocal unique_rows,duplicate_rows
  tid=r['trajectory_id'];index=lookup[tid];q=qids[index//10];assert r['qid']==q and r['role']=='train'and 0<len(r['policy_token_ids'])<=80
  h=row_fingerprint(r);ph=digest(r['prompt']);old=prompt_hashes.setdefault(q,ph);assert old==ph,'Own trajectories disagree on initial question prompt'
  if tid in pilot_expected:assert pilot_expected[tid]==h;pilot_compared.add(tid)
  if seen[index]:assert bytes(fingerprints[32*index:32*(index+1)])==h;duplicate_rows+=1;return False
  seen[index]=1;fingerprints[32*index:32*(index+1)]=h;unique_rows+=1;return True
 progress('ACCEPTED_CANONICAL_IDS_AND_PILOT_POLICY80_LOADED',0)
 for chunk_index,start in enumerate(range(0,len(sources),chunk_sources)):
  chunk=sources[start:start+chunk_sources];receipt_path=receipts/('chunk-%05d.json'%chunk_index);paths=[pieces/('train-chunk-%05d-rank-%d.jsonl.gz'%(chunk_index,rank))for rank in(0,1)]
  if receipt_path.exists():
   receipt=read(receipt_path);assert receipt['sources']==chunk and receipt['producer']==controller['code']and receipt['canonical_groups']==descriptor['groups'];actual_hashers=[hashlib.sha256(),hashlib.sha256()]
   for ref in chunk:checked(ref);processed_bytes+=ref['bytes']
   for rank,ref in enumerate(receipt['token_shards']):
    assert ref['path']==str(paths[rank])
    with gzip.open(checked(ref),'rt')as stream:
     for line in stream:
      r=json.loads(line);assert qrank[r['qid']]==rank and accept_row(r);actual_hashers[rank].update(canonical(r)+b'\n')
   assert [h.hexdigest()for h in actual_hashers]==receipt['RAW_derived_compact_row_SHA_by_rank'];token_refs.extend(receipt['token_shards']);chunk_refs.append(proof(receipt_path));progress('RESUMED_FINISHED_CHUNK_FRESH_RAW_AND_OUTPUT_SHA',start+len(chunk),chunks_completed=chunk_index+1);continue
  streams=[];handles=[];temps=[];hashers=[hashlib.sha256(),hashlib.sha256()];counts=[0,0];chunk_lengths={};chunk_bins={'shorter_than80':0,'at_least80':0}
  for path in paths:
   tmp=path.with_name(path.name+'.tmp');stream,handle=writer(tmp);streams.append(stream);handles.append(handle);temps.append(tmp)
  try:
   for source_index,ref in enumerate(chunk,start):
    source=checked(ref)
    with gzip.open(source,'rt')as stream:
     for line in stream:
      row=json.loads(line);tid=row['trajectory_id']
      if tid not in lookup:continue
      index=lookup[tid];assert row['qid']==qids[index//10]and row['actor_version']=='official7b-'+REVISION and row['strict_em']==labels[index],'RAW question/actor/own strictEM differs from canonical'
      compact,total=compact_raw(row)
      if not accept_row(compact):continue
      rank=qrank[compact['qid']];encoded=canonical(compact)+b'\n';streams[rank].write(encoded.decode());hashers[rank].update(encoded);counts[rank]+=1;key=str(len(compact['policy_token_ids']));chunk_lengths[key]=chunk_lengths.get(key,0)+1;chunk_bins['shorter_than80'if total<80 else'at_least80']+=1
    processed_bytes+=ref['bytes']
    if(source_index+1)%8==0:progress('STREAMING_TRUE_RAW_POLICY80',source_index+1,chunks_completed=chunk_index,current_chunk=chunk_index)
  finally:
   for stream,handle in zip(streams,handles):stream.close();handle.close()
  for tmp,dest in zip(temps,paths):tmp.replace(dest)
  output_refs=[proof(path)for path in paths];receipt=dict(status='RAW_MATCHED_POLICY80_CHUNK_SHA_VERIFIED',sources=chunk,producer=controller['code'],canonical_groups=descriptor['groups'],token_shards=output_refs,train_rows_by_rank=counts,RAW_derived_compact_row_SHA_by_rank=[h.hexdigest()for h in hashers],real_prefix_length_histogram=chunk_lengths,retained_length_bins=chunk_bins,strictEM_matches_canonical=True,initial_prompt_and_true_retained_policy_only=True,padding_deferred_to_original_runtime_pack_group=True);once(receipt_path,receipt);token_refs.extend(output_refs);chunk_refs.append(proof(receipt_path));progress('CHUNK_SAVED_AND_SHA_VERIFIED',start+len(chunk),chunks_completed=chunk_index+1)
 assert unique_rows==1690280 and all(seen)and len(pilot_compared)==81920
 # Independent readback: compare every written row with its RAW-derived digest,
 # own canonical ID, rank route and question prompt; do not recompute EM labels.
 read_seen=bytearray(len(lookup));read_train=0;read_dev=set();rank_counts=[0,0];full_lengths={}
 all_tokens=token_refs+validation_refs
 for shard_index,ref in enumerate(all_tokens):
  rank=shard_index%2
  with gzip.open(checked(ref),'rt')as stream:
   for line in stream:
    r=json.loads(line);tid=r['trajectory_id'];assert qrank[r['qid']]==rank;rank_counts[rank]+=1
    if r['role']=='train':
     index=lookup[tid];assert not read_seen[index]and r['qid']==qids[index//10]and bytes(fingerprints[32*index:32*(index+1)])==row_fingerprint(r);read_seen[index]=1;read_train+=1;key=str(len(r['policy_token_ids']));full_lengths[key]=full_lengths.get(key,0)+1
    else:assert tid in expected_dev and tid not in read_dev and pilot_expected[tid]==row_fingerprint(r);read_dev.add(tid)
 assert read_train==1690280 and all(read_seen)and len(read_dev)==10240 and rank_counts==[850260,850260]
 for ref in chunk_refs:
  rc=read(checked(ref))
  for k,v in rc['retained_length_bins'].items():retained_bins[k]+=v
 fixture_data=dict(status=STATUS,training_questions=169028,training_trajectories=1690280,validation_questions=2048,validation_trajectories=10240,max_real_policy_prefix=80,real_policy_only_no_20_to_80_fabrication=True,token_shards=all_tokens,train_groups=proof(inputs/'train-groups.json'),validation_groups=proof(inputs/'validation-groups.json'),fixed_train_G5=proof(inputs/'fixed-train-G5.json'),fixed_G5_sha256=pilot['fixed_G5_sha256'],canonical_groups=descriptor['groups'],canonical_ordered_qid_sha256=descriptor['canonical_ordered_qid_sha256'],source_contract=proof(base/'input-contract.json'),RAW_chunk_receipts=chunk_refs,initial_prompt_alignment_all_own_G10_verified=True,canonical_own_strictEM_preserved=True,rank_routing='sorted train parity separately from sorted validation parity; shardindex%2=rank',rows_by_rank=rank_counts,real_prefix_length_histogram=full_lengths,retained_length_bins=retained_bins,pilot_train_own_policy80_rows_exact_compared=81920,hidden_features_saved=False,new_rollouts_generated=0,final4096_read=False)
 once(fixture/'fixture-complete.json',fixture_data)
 acceptance=dict(status='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_ACCEPTED',fixture=proof(fixture/'fixture-complete.json'),producer=controller['code'],source_contract=proof(base/'input-contract.json'),all_RAW_sources_fresh_SHA_verified=True,all_output_tokens_fresh_SHA_verified=True,all_train_own_G10_coverage=1690280,all_canonical_question_groups=169028,all_train_labels_exact_canonical=True,all_RAW_derived_first80_and_prompt_readback_equal=True,real_shorter_answers_not_fabricated=True,original_fixed_dev_G5_questions_and_trajectories=2048,original_fixed_train_B_G5_questions_and_trajectories=2048,validation_G5_metadata_SHA_unchanged=proof(inputs/'validation-groups.json')['sha256']==pilot['validation_groups']['sha256'],fixed_train_B_metadata_SHA_unchanged=proof(inputs/'fixed-train-G5.json')['sha256']==pilot['fixed_train_G5']['sha256'],pilot_train_policy80_exact_comparison_rows=81920,rows_by_rank=rank_counts,no_model_or_GPU_or_hidden_features=True,new_rollouts_generated=0,final4096_read=False,seconds=time.time()-started,time=time.time())
 once(fixture/'accepted.json',acceptance);atomic(base/'complete.json',dict(status='FULL169028_POLICY80_CPU_PREPARATION_AND_ACCEPTANCE_COMPLETE',fixture=acceptance['fixture'],acceptance=proof(fixture/'accepted.json'),source_contract=proof(base/'input-contract.json'),time=time.time(),seconds=time.time()-started,new_rollouts_generated=0,final4096_read=False));progress('COMPLETE_ALL_OWN_G10_POLICY80_SHA_ACCEPTED',len(sources),chunks_completed=len(chunk_refs));print(json.dumps(acceptance),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--chunk-sources',type=int,default=64);a=p.parse_args();assert a.chunk_sources>0
 try:run(a.base,a.chunk_sources)
 except BaseException:
  if a.base.is_dir()and(FULL/'full-training-descriptor.json').is_file():atomic(a.base/'failure.json',dict(time=time.time(),traceback=traceback.format_exc(),new_rollouts_generated=0,final4096_read=False))
  raise
if __name__=='__main__':main()
