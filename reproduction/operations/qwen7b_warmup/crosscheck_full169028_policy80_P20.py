"""Bounded CPU full P20 compact-bank versus authentic RAW-derived P80 audit.

Keeps only tid->ordinal, one-byte labels and32-byte row digests, never a complete
token dictionary. Waits for the existing producer and never restarts it.
"""
import argparse,fcntl,gc,gzip,hashlib,json,os,time,traceback
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');P20=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=Path(ref['path']);assert proof(p)=={k:ref[k]for k in('path','bytes','sha256')};return p
def atomic(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid());tmp.write_bytes(canonical(x)+b'\n');tmp.replace(p)
def rss():
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'):return int(line.split()[1])
def normalized_sha(row,policy,y):return hashlib.sha256(canonical([row['qid'],row['trajectory_id'],row['prompt'],policy,y])).digest()
def finalize_existing(base):
 base=Path(base);report=read(base/'P20-crosscheck-complete.json');fixture=base/'fixture';assert report['fixture']==proof(fixture/'fixture-complete.json');original=fixture/'accepted-producer-only.json';assert report['producer_acceptance']==proof(original);producer=read(original);enhanced=dict(producer,fresh_full_P20_prefix_prompt_and_own_EM_exact=True,P20_alignment=proof(base/'P20-crosscheck-complete.json'),producer_acceptance=proof(original));atomic(fixture/'accepted.json',enhanced);complete=read(base/'complete-producer-only.json');complete.update(acceptance=proof(fixture/'accepted.json'),P20_alignment=proof(base/'P20-crosscheck-complete.json'),fresh_full_P20_prefix_prompt_and_own_EM_exact=True);atomic(base/'complete.json',complete);return report

def run(base):
 base=Path(base);assert base.is_dir()and(P20/'fixture-complete.json').is_file();lock=(base/'P20-crosscheck.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);publication=read(base/'P20-crosscheck-source-publication.json');code=Path(__file__).resolve();assert proof(code)==dict(publication['files'][publication['producer_relative']],path=str(code));started=time.time()
 if(base/'P20-crosscheck-complete.json').is_file():
  old=finalize_existing(base);print(json.dumps(dict(status='EXISTING_FULL_P20_ALIGNMENT_REUSED')),flush=True);return
 atomic(base/'P20-crosscheck-controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],code=proof(code),source_publication=proof(base/'P20-crosscheck-source-publication.json'),started=started,no_GPU=True,final4096_read=False))
 def progress(stage,old_rows=0,new_rows=0,**extra):atomic(base/'P20-crosscheck-progress.json',dict(stage=stage,P20_train_rows=old_rows,P80_train_rows_compared=new_rows,expected_train_rows=1690280,RSS_KiB=rss(),seconds=time.time()-started,time=time.time(),full_token_dictionary_kept=False,no_GPU=True,final4096_read=False,**extra))
 progress('READING_ACCEPTED_P20_CANONICAL_IDS')
 old_fixture=read(P20/'fixture-complete.json');assert old_fixture['training_questions']==169028;old=read(checked(old_fixture['train_groups']));new_groups_path=base/'fixture/inputs/train-groups.json'
 while not new_groups_path.is_file():progress('WAIT_FOR_EXISTING_PRODUCER_CANONICAL_GROUPS');time.sleep(5)
 new=read(new_groups_path);assert len(old)==len(new)==169028 and set(old)==set(new)
 for q,g in old.items():assert g['trajectory_ids']==new[q]['trajectory_ids']and g['labels']==new[q]['labels']and g['question_hash']==new[q]['question_hash']
 qids=sorted(old);lookup={};labels=bytearray()
 for q in qids:
  g=old[q];assert len(g['trajectory_ids'])==len(g['labels'])==10
  for tid,y in zip(g['trajectory_ids'],g['labels']):assert tid not in lookup and type(y)is int and y in(0,1);lookup[tid]=len(labels);labels.append(y)
 assert len(lookup)==1690280;del old,new;gc.collect();digests=bytearray(32*len(lookup));seen=bytearray(len(lookup));old_count=0;old_refs=[]
 for shard_index,ref in enumerate(old_fixture['token_shards']):
  with gzip.open(checked(ref),'rt')as stream:
   for line in stream:
    row=json.loads(line)
    if row['role']!='train':continue
    tid=row['trajectory_id'];index=lookup[tid];assert not seen[index]and row['qid']==qids[index//10]and 0<len(row['policy20'])<=20;digests[32*index:32*(index+1)]=normalized_sha(row,row['policy20'],labels[index]);seen[index]=1;old_count+=1
    if old_count%65536==0:progress('FRESH_P20_COMPACT_PREFIX_SHA_INDEX',old_count)
  old_refs.append(ref)
 assert old_count==1690280 and all(seen);del seen;gc.collect();progress('P20_INDEX_READY_WAITING_FOR_EXISTING_P80_PRODUCER',old_count,index_digest_bytes=len(digests))
 while not(base/'complete.json').is_file():
  if(base/'failure.json').exists():raise RuntimeError('Existing P80 producer reported a failure; audit will not restart it')
  time.sleep(10)
 # Serialize final receipt augmentation with the producer, without touching any
 # active writer. The small producer may still be flushing its final progress.
 main_guard=(base/'prepare.lock').open('a+')
 while True:
  try:fcntl.flock(main_guard,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(2)
 fixture=base/'fixture';current=read(fixture/'fixture-complete.json');producer=read(fixture/'accepted.json');assert current['status']=='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY'and producer['fixture']==proof(fixture/'fixture-complete.json');assert current['train_groups']==proof(new_groups_path);assert producer['all_train_labels_exact_canonical']and producer['all_RAW_derived_first80_and_prompt_readback_equal']
 seen=bytearray(len(lookup));new_count=0;new_refs=[];identity=hashlib.sha256()
 for shard_index,ref in enumerate(current['token_shards']):
  with gzip.open(checked(ref),'rt')as stream:
   for line in stream:
    row=json.loads(line)
    if row['role']!='train':continue
    tid=row['trajectory_id'];index=lookup[tid];assert not seen[index]and row['qid']==qids[index//10]and 0<len(row['policy_token_ids'])<=80;assert bytes(digests[32*index:32*(index+1)])==normalized_sha(row,row['policy_token_ids'][:20],labels[index]),'Fresh full P20 initial_prompt/policy[:20]/ownEM mismatch';seen[index]=1;new_count+=1;identity.update(canonical([row['qid'],tid,labels[index]])+b'\n')
    if new_count%65536==0:progress('FRESH_FULL_P20_VS_P80_PREFIX_PROMPT_OWN_EM_ALIGNMENT',old_count,new_count)
  new_refs.append(ref)
 assert new_count==1690280 and all(seen)
 # Preserve producer-only receipts before adding this separate full comparison.
 producer_path=fixture/'accepted-producer-only.json'
 if producer_path.exists():assert producer_path.read_bytes()==(fixture/'accepted.json').read_bytes()
 else:producer_path.write_bytes((fixture/'accepted.json').read_bytes())
 original_complete=base/'complete-producer-only.json'
 if not original_complete.exists():original_complete.write_bytes((base/'complete.json').read_bytes())
 report=dict(status='FULL169028_AUTHENTIC_POLICY80_ALL_P20_PREFIX_PROMPT_OWN_EM_CROSSCHECK_ACCEPTED',time=time.time(),seconds=time.time()-started,fixture=proof(fixture/'fixture-complete.json'),producer_acceptance=proof(producer_path),source_publication=proof(base/'P20-crosscheck-source-publication.json'),accepted_P20_fixture=proof(P20/'fixture-complete.json'),accepted_P20_train_groups=old_fixture['train_groups'],new_train_groups=current['train_groups'],P20_token_shards=old_refs,P80_token_shards=new_refs,training_questions=169028,all_train_own_trajectory_ids=1690280,own_G10_order_and_EM_metadata_exact=True,all_initial_prompt_token_ids_exact=True,all_true_retained_policy_prefix20_exact=True,all_label_associations_exact=True,P80_first20_is_actual_RAW_derivation=True,streamed_P80_identity_EM_sha256=identity.hexdigest(),index_digest_bytes=len(digests),full_token_dictionary_kept=False,RSS_KiB=rss(),no_GPU=True,new_rollouts_generated=0,final4096_read=False)
 atomic(base/'P20-crosscheck-complete.json',report);finalize_existing(base);progress('COMPLETE_FULL_P20_PREFIX_PROMPT_OWN_EM_ALIGNMENT',old_count,new_count);print(json.dumps(dict(status=report['status'],P20_alignment=proof(base/'P20-crosscheck-complete.json'),enhanced_acceptance=proof(fixture/'accepted.json'))),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
 try:run(a.base)
 except BaseException:
  if a.base.is_dir():atomic(a.base/'P20-crosscheck-failure.json',dict(time=time.time(),traceback=traceback.format_exc(),no_GPU=True,final4096_read=False))
  raise
if __name__=='__main__':main()
