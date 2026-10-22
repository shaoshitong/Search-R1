"""CPU audit of all existing NQ pilot own-G10 actual full/seal/EM/Prefix80.

Only own audit reports are written. No changes to source assets, model, probe,
score files or Slurm; no final holdout or new inference. Bound by physical SHA.
"""
import argparse,collections,fcntl,gzip,hashlib,json,os,time,traceback
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');REV='official7b-a09a35458c702b33eeacc393d103063234e8bc28'
def safe(p):
 p=Path(p);assert not any(s in str(p)for s in('final4096','confirmation','stability2048'));return p
def read(p):return json.loads(safe(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def proof(p):
 p=safe(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=safe(r['path']);assert proof(p)=={k:r[k]for k in('path','bytes','sha256')},'Source SHA differs '+str(p);return p
def refs(x):
 if isinstance(x,dict):
  if {'path','bytes','sha256'}<=x.keys():yield {k:x[k]for k in('path','bytes','sha256')}
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
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_bytes(canonical(x)+b'\n');q.replace(p)
def policy80(r):
 out=[]
 for turn in r['behavior_turns']:
  n=turn['kept_token_count'];assert 0<=n<=len(turn['raw_generated_token_ids']);out+=turn['raw_generated_token_ids'][:min(n,80-len(out))]
  if len(out)==80:break
 assert 0<len(out)<=80;return out

def run(base):
 base.mkdir(exist_ok=True);guard=(base/'audit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time();publication=read(base/'source-publication.json');assert proof(__file__)==dict(publication['runtime'],path=str(Path(__file__)))
 atomic(base/'controller.json',dict(time=started,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],code=proof(__file__),CPU_only=True,final4096_read=False));pf=A/'online-packed-matched-L15P80-8192-20261023-v1/fixture/fixture-complete.json';pilot=read(pf);assert proof(pf)['sha256']=='a269f9d4bf22bef0e30afc7cbc0e9a708fb83cb73230fefa95106b56ec54d936';groups=read(checked(pilot['train_groups']));assert len(groups)==8192 and all(json.loads(q)[0]=='nq'for q in groups)
 expected={t:(q,y)for q,g in groups.items()for t,y in zip(g['trajectory_ids'],g['labels'])};assert len(expected)==81920 and all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in groups.values());records={};token_refs=[]
 for r in pilot['token_shards']:
  p=checked(r);token_refs.append(r)
  with gzip.open(p,'rt')as f:
   for line in f:
    row=json.loads(line)
    if row['trajectory_id']not in expected:continue
    tid=row['trajectory_id'];assert tid not in records and row['role']=='train'and row['qid']==expected[tid][0];records[tid]=row
 assert len(records)==81920
 causal=read(checked(pilot['source_fixture']));selected={(r['path'],r['bytes'],r['sha256'])for r in causal['source_full']};ti_path=A/'peer-newseed8192-g10-train-v1/training-input.json';ti=read(ti_path);raw=read(checked(ti['raw_complete']));events_ref=raw['outputs']['source-events'];events=read(checked(events_ref));batches=[dict(b,event=e['event'])for e in events['events']for b in e['batches']if(b['full']['path'],b['full']['bytes'],b['full']['sha256'])in selected];assert len(batches)==492
 own={};seen={};mismatches=[];binding_rows=[];full_bytes=0;full_rows=0;own_rows=0;domains=collections.Counter();execids=set()
 def progress(stage,index):atomic(base/'progress.json',dict(time=time.time(),seconds=time.time()-started,stage=stage,source_files_completed=index,source_files_total=len(batches),full_compressed_bytes_verified=full_bytes,full_rows_read=full_rows,own_EM_rows_read=own_rows,selected_own_EM_rows=len(own),selected_P80_rows_verified=len(seen),expected_selected_rows=81920,mismatches=len(mismatches),CPU_only=True,final4096_read=False))
 progress('PILOT8192_ACTUAL_COMPACT_P80_AND_G10_METADATA_LOADED',0)
 for i,b in enumerate(batches):
  seal=read(checked(b['seal']));sealed=list(refs(seal))
  for r in(b['full'],b['own_em']):assert any(z['bytes']==r['bytes']and z['sha256']==r['sha256']for z in sealed),'Source payload not bound to actual complete seal'
  ids=runids(seal);execids.update(ids)
  with checked(b['own_em']).open()as f:
   for line in f:
    r=json.loads(line);own_rows+=1;tid=r['trajectory_id']
    if tid not in expected:continue
    assert tid not in own,'Ambiguous selected trajectory in two physical execution sources';q,y=expected[tid];assert r['qid']==q and r['actor_version']==REV
    own[tid]=(int(r['strict_em']),b['full']['sha256'],b['seal']['sha256'])
    if int(r['strict_em'])!=y:mismatches.append(dict(kind='own_EM_label',qid=q,trajectory_id=tid,canonical=y,own_EM=int(r['strict_em']),source=b['full']))
  binding_rows.append(dict(event=b['event'],full=b['full'],own_em=b['own_em'],seal=b['seal'],execution_run_ids=ids))
  if(i+1)%16==0:progress('JOINING_ALL_SELECTED_ACTUAL_OWN_EM_AND_SEALS',i+1)
 assert set(own)==set(expected)
 for i,b in enumerate(batches):
  p=checked(b['full']);full_bytes+=b['full']['bytes']
  with gzip.open(p,'rt')as f:
   for line in f:
    r=json.loads(line);full_rows+=1;tid=r['trajectory_id']
    if tid not in expected:continue
    q,y=expected[tid];assert tid not in seen,'Two selected physical FULL payloads for the same pilot tid';assert own[tid][1:]==(b['full']['sha256'],b['seal']['sha256']);assert r['qid']==q and r['actor_version']==REV;domains[json.loads(q)[0]]+=1;compact=records[tid];p80=policy80(r);bad=[]
    if int(r['strict_em'])!=y or int(r['strict_em'])!=own[tid][0]:bad.append('full_own_EM_canonical_label')
    if r['initial_prompt_token_ids']!=compact['prompt']:bad.append('actual_initial_prompt')
    if p80!=compact['policy_token_ids']:bad.append('actual_retained_policy80')
    if r.get('question_hash')!=groups[q].get('question_hash'):bad.append('question_hash')
    if bad:mismatches.append(dict(kind=bad,qid=q,trajectory_id=tid,canonical=y,full_EM=int(r['strict_em']),actual_policy80_sha=digest(p80),pilot_policy80_sha=digest(compact['policy_token_ids']),source=b['full'],seal=b['seal']))
    seen[tid]=(b['full']['sha256'],b['seal']['sha256'],digest(p80),digest(r['initial_prompt_token_ids']))
  progress('FRESH_FULL_SHA_AND_RETAINED_POLICY80_PROMPT_LABEL_COMPARISON',i+1)
 assert set(seen)==set(expected)
 result=dict(time=time.time(),seconds=time.time()-started,status='ALL81920_NQ_PILOT_OWN_G10_PHYSICAL_SEAL_EM_PREFIX80_VERIFIED'if not mismatches else'PILOT_PHYSICAL_SOURCE_OR_LABEL_MISMATCH_FOUND',questions=8192,own_G10_trajectories=81920,domains=dict(domains),selected_sources=len(batches),fresh_full_compressed_bytes_SHA_verified=full_bytes,full_rows_read=full_rows,fresh_own_EM_rows_read=own_rows,selected_own_EM_rows=len(own),selected_actual_P80_rows=len(seen),mismatch_count=len(mismatches),mismatch_examples=mismatches[:12],execution_run_ids=sorted(execids),pilot_fixture=proof(pf),pilot_train_groups=pilot['train_groups'],pilot_compact_P80_shards=token_refs,source_causal_fixture=pilot['source_fixture'],source_training_input=proof(ti_path),source_events=events_ref,source_binding_SHA256=digest(binding_rows),selected_trajectory_physical_P80_and_prompt_identity_SHA256=digest(sorted(seen.items())),source_binding_examples=binding_rows[:2]+binding_rows[-2:],all_source_payloads_fresh_SHA_verified=True,all_own_EM_and_full_seals_fresh_SHA_verified=True,all_compact_P80_shards_fresh_SHA_verified=True,canonical_labels_mutated=False,full169028_scope_claim=False,model_or_tensor_loaded=False,new_inference=False,GPU_used=False,Slurm_mutated=False,final4096_read=False);atomic(base/'complete.json',result);progress('COMPLETE_ALL81920_PHYSICAL_SOURCE_CHECKED',len(batches));print(json.dumps(result),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
 try:run(a.base)
 except BaseException:atomic(a.base/'failure.json',dict(time=time.time(),traceback=traceback.format_exc(),CPU_only=True,final4096_read=False));raise
if __name__=='__main__':main()
