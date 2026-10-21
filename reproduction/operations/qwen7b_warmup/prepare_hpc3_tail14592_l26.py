"""Prepared-only delta01 adapter; actual complete SHA required. CPU-only fixed14592 qualified G10 -> four source-resolved L26 input shards.

Imports the existing qualified replay helpers without changing their source.
Never starts GPU capture. Whole newG5 execution is selected before any feature
manifest union; projection binds each tid to its selected physical prefix.
"""
import argparse,fcntl,gzip,hashlib,json,os,shutil,sys,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';TARGET=A/'hpc3-tail14592-independent-L26-prepared-v1';RAW=A/'hpc3-tail-completion-delta01-v1/complete.json'
QUALIFIED=A/'peer-joint124194-audited243-L26-v1/code-db5b5ac04173'
PUB={'path':str(QUALIFIED.parent/'source-publication.json'),'bytes':7106,'sha256':'c8547a3dae899547fd4620afde9350c8a034c0dfee5bebd1dd9b5323a4dbdff3'}
GOLDEN={'path':str(A/'replay-batched-golden-v3/output/complete.json'),'bytes':12350,'sha256':'3eeeb79b750d2e0bfee81c9a283c7428d0629b0d56815f91909c93e2bd9f5715'}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def fp(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 actual=fp(ref['path']);assert all(actual[k]==ref[k]for k in('path','bytes','sha256')),'Bound accepted metadata/file differs';return Path(actual['path'])
def read(p):return json.loads(Path(p).read_bytes())
def once(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=canonical(v)+b'\n'
 if p.exists():assert p.read_bytes()==raw,'Immutable L26 preparation differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return fp(p)
def physical_binding(row,source,event_name,selected,seal):
 if row['qid']!=selected['qid']or source['seal']!=selected['seal']or event_name!=selected['event']:
  raise ValueError('Unselected execution physical prefix encountered')
 if event_name=='HPC3_newseed20261021'and(seal['execution_run_id']!=selected['execution']or seal['execution_host_cluster']!='HPC3'):
  raise ValueError('Chosen HPC3 execution differs')
 if any(k in row for k in('strict_em','labels','reward_model','ground_truth','response_token_ids')):
  raise ValueError('Prefix-only feature input carries response/label')
 return dict(trajectory_id=row['trajectory_id'],qid=row['qid'],chosen_execution=selected['execution'],physical_seal=source['seal'],physical_prefix=source['prefix'],prefix_body_sha256=row['prefix_sha256'],canonical_prefix_row_sha256=hashlib.sha256(canonical(row)).hexdigest(),future_retained_row_SHA_must_bind_this_source=True)

def prepare(raw_sha):
 assert sys.platform!='darwin'and not os.environ.get('CUDA_VISIBLE_DEVICES'),'Remote CPU metadata/projection only';TARGET.mkdir(exist_ok=True);start=time.time()
 with(TARGET/'prepare.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rawref=fp(RAW);assert rawref['sha256']==raw_sha
  if(TARGET/'preparation.json').exists():
   old=read(TARGET/'preparation.json');assert old['source_raw_complete']==rawref;return old
  pub=read(checked(PUB))
  for name,ref in pub['files'].items():
   path=QUALIFIED/name;actual=fp(path);assert actual['bytes']==ref['bytes']and actual['sha256']==ref['sha256'],'Qualified helper changed'
  sys.path.insert(0,str(QUALIFIED))
  from reproduction.operations.qwen7b_warmup import replay_probe_layers as r
  from reproduction.operations.qwen7b_warmup.peer_audited122_l26_replay import geometry
  from reproduction.operations.qwen7b_warmup.replay_bank_prepare import q1_model_binding
  from reproduction.operations.qwen7b_warmup.validation_fixedg5_replay import batching_binding
  done=read(RAW);assert done['status']=='HPC3_OWN_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_FULL_STRICT_EM_CAUSAL_SEEDS_QUALIFIED'and done['raw_G10_full_qualified']and done['questions']==14592 and done['trajectories']==145920 and done['strict_EM_mismatches']==0 and done['actual_same_question_old_new_seed_collisions']==0
  contract=read(checked(done['outputs']['source-resolution-contract']));assert contract['qualified_execution_source_selection_before_L26']and contract['donor_and_HPC3_feature_manifest_union_before_source_selection_forbidden']
  groups=read(checked(done['outputs']['groups']))['train'];events=read(checked(done['outputs']['source-events']));assert len(groups)==14592
  identity=[{k:g[k]for k in('qid','question_hash','trajectory_ids')}for g in groups];qidset={g['qid']for g in groups};tids={t for g in groups for t in g['trajectory_ids']};assert len(qidset)==14592 and len(tids)==145920 and all(len(g['trajectory_ids'])==10 for g in groups)
  free=shutil.disk_usage(TARGET).free;assert free>=80*1024**3,'Shared space reserve must cover future L26 bank'
  idref=once(TARGET/'identity-plan.json',dict(role='train',groups=identity,label_blind=True,source_raw_complete=rawref,current124194_fit_expanded=False));parts=[identity[i::4]for i in range(4)];owner={t:i for i,part in enumerate(parts)for g in part for t in g['trajectory_ids']};wanted={}
  for g in groups:
   for index,t in enumerate(g['trajectory_ids']):wanted[t]=dict(qid=g['qid'],seal=g['old5_seal']if index<5 else g['chosen_new5_seal'],event='original'if index<5 else'HPC3_newseed20261021',execution='original_canonical_old5'if index<5 else g['chosen_new5_execution_run_id'])
  streams=[];trace=[];paths=[];seen=set();counts=[0]*4;overlength=[0]*4;stats={};q1,modelbinding=q1_model_binding();batching=batching_binding(GOLDEN,32)
  try:
   for i in range(4):
    d=TARGET/('shard-%02d'%i)/'inputs';d.mkdir(parents=True,exist_ok=True);p=d/'prefix.jsonl.gz';t=d/'chosen-physical-prefix-bindings.jsonl.gz';assert not any(x.exists()for x in(p,t,p.with_suffix('.gz.partial'),t.with_suffix('.gz.partial'))),'Partial old preparation preserved; do not overwrite';paths.append((p,t));streams.append(gzip.open(p.with_suffix('.gz.partial'),'xb'));trace.append(gzip.open(t.with_suffix('.gz.partial'),'xb'))
   for event in events['events']:
    for source in event['batches']:
     seal=read(checked(source['seal']));path=checked(source['prefix'])
     with gzip.open(path,'rt')as f:
      for line in f:
       row=json.loads(line);t=row['trajectory_id']
       if t not in wanted:continue
       selected=wanted[t];assert t not in seen,'Duplicate selected physical own trajectory';binding=physical_binding(row,source,event['event'],selected,seal)
       i=owner[t];measure=geometry(row);r.normalize_prefix(row,t,row['qid'],True);assert not measure['overlength'],'No causal context truncation allowed';streams[i].write(canonical(row)+b'\n');trace[i].write(canonical(binding)+b'\n');seen.add(t);counts[i]+=1
       if len(seen)%1024==0:(TARGET/'progress.json').write_text(json.dumps(dict(stage='SOURCE_RESOLVED_CAUSAL_PREFIX_PROJECTION_ADVANCING',trajectories=len(seen),expected=145920,counts=counts,epoch=time.time()))+'\n')
  finally:
   for f in streams+trace:f.close()
  assert seen==tids and counts==[36480]*4
  selections=[]
  for i,(p,t)in enumerate(paths):
   os.replace(str(p.with_suffix('.gz.partial')),str(p));os.replace(str(t.with_suffix('.gz.partial')),str(t));sel=dict(role='peer_newseed_train_G10',model_revision=r.REVISION,questions=3648,trajectories=[dict(qid=g['qid'],trajectory_id=t)for g in parts[i]for t in g['trajectory_ids']],real_trajectories_per_question=10,label_blind_selection=True,prefix_tokens=20,capture_layers=list(r.LAYERS),persist_layers=[26],prefix_files=[fp(p)],model_files=q1['model_files'],source_model_binding=modelbinding,source_ingest=rawref,question_plan=idref,chosen_physical_prefix_bindings=fp(t),source_resolution_contract=done['outputs']['source-resolution-contract'],source_execution_selection_completed=True,donor_feature_manifest_union_forbidden=True,domain_scope=dict(kind='qualified_HPC3_tail_new5_plus_original_old5',overall_question_counts={'hotpotqa':14592},original_warmup_unseen_claim=False),expected_code_root=str(QUALIFIED),shard=i,shards=4,batching=batching,geometry_checked_no_truncation=True,actor_updates=0,new_responses_generated=0);selections.append(once(p.parent/'selection.json',sel))
  result=dict(status='FIXED14592_FULL_QUALIFIED_G10_SOURCE_RESOLVED_L26_INPUTS_PREPARED_NO_GPU',source=fp(Path(__file__)),source_raw_complete=rawref,identity_plan=idref,source_resolution_contract=done['outputs']['source-resolution-contract'],qualified_runtime_root=str(QUALIFIED),qualified_runtime_publication=PUB,qualified_runtime_source_commit=pub.get('commit'),selections=selections,questions=14592,trajectories=145920,prefix20_causal_ownIDs_and_physical_source_selection_verified=True,chosen_execution_prefix_SHA_bound=True,feature_bank_created=False,GPU_submitted=False,new_rollouts_generated=0,actor_updates=0,current124194_fit_expanded=False,batching=batching,free_bytes=free,elapsed_seconds=time.time()-start,epoch=time.time());once(TARGET/'preparation.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true',required=True);p.add_argument('--raw-sha256',required=True);args=p.parse_args();print(json.dumps(prepare(args.raw_sha256),sort_keys=True))
