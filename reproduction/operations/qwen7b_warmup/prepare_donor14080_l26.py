"""Qualified donor138 minus chosen HPC3 qids: source-resolved L26 inputs only."""
import argparse,fcntl,gzip,hashlib,itertools,json,os,resource,shutil,signal,sys,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021'
TARGET=A/'donor14080-independent-L26-prepared-v1'
DONOR=A/'peer-newseed-ingest-audited138-after971-20261021-51057f37'
DONOR_COMPLETE=dict(path=str(DONOR/'complete.json'),bytes=9066,sha256='e7056d4eb952151b95241dc87ac732fb641daa5aea633f09de4373f536d69ed3')
TAIL_COMPLETE=dict(path=str(A/'hpc3-tail9344-old5-qualified-g10-v1/complete.json'),bytes=24395,sha256='0898b54516068d5b2c0fd1e2d95befb47d58ed2e96ca1065fc9b556387abbbfd')
TAIL_CONTRACT=dict(path=str(A/'hpc3-tail9344-old5-qualified-g10-v1/raw-g10/source-resolution-contract.json'),bytes=485,sha256='229710cbfccc53c92be96081535eeeee308cc52b969b07713258212c2b0d86c2')
QUALIFIED=A/'peer-joint124194-audited243-L26-v1/code-db5b5ac04173'
PUB=dict(path=str(QUALIFIED.parent/'source-publication.json'),bytes=7106,sha256='c8547a3dae899547fd4620afde9350c8a034c0dfee5bebd1dd9b5323a4dbdff3')
GOLDEN=dict(path=str(A/'replay-batched-golden-v3/output/complete.json'),bytes=12350,sha256='3eeeb79b750d2e0bfee81c9a283c7428d0629b0d56815f91909c93e2bd9f5715')
DONOR_RAW=B/'peer-newseed-h100-20261021/audited138-after971-20261021-51057f37/raw'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
RUN='70188b8e50fd4b45bc420ce58014f46e'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def fp(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as stream:
  for raw in iter(lambda:stream.read(8*1024**2),b''):h.update(raw)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=Path(ref['path']);assert p.is_file()and not p.is_symlink();actual=fp(p);assert all(actual[k]==ref[k]for k in('path','bytes','sha256')),'Bound source file changed';return p
def read(p):return json.loads(Path(p).read_bytes())
def once(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=canonical(v)+b'\n'
 if p.exists():assert p.read_bytes()==raw,'Existing immutable preparation differs'
 else:
  with p.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return fp(p)
def progress(v):
 p=TARGET/'progress.json';q=p.with_suffix('.tmp');q.write_bytes(canonical(dict(v,epoch=time.time()))+b'\n');os.replace(str(q),str(p))
def select_groups(donor,tail,expected=(17664,9344,3584,14080)):
 def index(gs,n):
  out={};tids=set()
  for g in gs:
   q=g['qid'];ids=g['trajectory_ids']
   if q in out or len(ids)!=10 or len(set(ids))!=10 or tids.intersection(ids):raise ValueError('Duplicate/incomplete canonical qid/G10 identity')
   out[q]=g;tids.update(ids)
  if len(out)!=n:raise ValueError('Frozen accepted source question count differs')
  return out
 d=index(donor,expected[0]);t=index(tail,expected[1]);overlap=set(d)&set(t)
 if len(overlap)!=expected[2]:raise ValueError('Actual canonical-qid exclusion differs')
 for q in overlap:
  if d[q]['question_hash']!=t[q]['question_hash']or d[q]['trajectory_ids'][:5]!=t[q]['trajectory_ids'][:5]:raise ValueError('Overlapping canonical qid/old5 identity differs')
 chosen=[g for g in donor if g['qid']not in t]
 if len(chosen)!=expected[3]:raise ValueError('Retained whole-G10 question count differs')
 for g in chosen:
  events=g['real_G5_events']
  if len(events)!=2 or [x['event']for x in events]!=['original','newseed20261021']or any(events[i]['trajectory_ids']!=g['trajectory_ids'][5*i:5*i+5]for i in range(2)):raise ValueError('G5 event must remain intact and ordered')
 return chosen,dict(excluded_canonical_qids=len(overlap),excluded_ordered_qid_sha256=digest([g['qid']for g in donor if g['qid']in overlap]),retained_ordered_qid_sha256=digest([g['qid']for g in chosen]),retained_ordered_own10_ids_sha256=digest([v for g in chosen for v in g['trajectory_ids']]),selected_labels_or_scores_consulted=False)
def physical_binding(row,source,event,expected,seal):
 if row['trajectory_id']!=expected['trajectory_id']or row['qid']!=expected['qid']or event!=expected['event']or source.get('event',event)!=event:raise ValueError('Wrong selected physical event/qid/own trajectory')
 prefix=Path(source['prefix']['path']);sp=Path(source['seal']['path'])
 if event=='newseed20261021':
  if not str(prefix).startswith(str(DONOR_RAW)+'/')or not str(sp).startswith(str(DONOR_RAW)+'/')or seal.get('run_id')!=RUN or seal.get('model_revision')!=REVISION or seal.get('actor_optimizer_updates')!=0 or seal.get('execution_host_cluster')=='HPC3':raise ValueError('Donor event must bind its accepted physical execution')
 else:
  if event!='original'or not str(prefix).startswith(str(B/'sampling')+'/')or not str(sp).startswith(str(B/'sampling')+'/'):raise ValueError('Original old5 physical source differs')
 if any(k in row for k in('strict_em','labels','reward_model','ground_truth','response_token_ids')):raise ValueError('Feature prefix contains future response or labels')
 return dict(trajectory_id=row['trajectory_id'],qid=row['qid'],chosen_execution='original_canonical_old5'if event=='original'else'donor_qualified_newseed20261021_accepted138',chosen_event=event,physical_seal=source['seal'],physical_prefix=source['prefix'],prefix_body_sha256=row['prefix_sha256'],canonical_prefix_row_sha256=digest(row),future_retained_row_SHA_must_bind_this_source=True)
def qualified():
 pub=read(checked(PUB))
 for name,item in pub['files'].items():
  actual=fp(QUALIFIED/name);assert all(actual[k]==item[k]for k in('bytes','sha256')),'Qualified runtime drift'
 sys.path.insert(0,str(QUALIFIED))
 from reproduction.operations.qwen7b_warmup import replay_probe_layers as r
 from reproduction.operations.qwen7b_warmup import peer_audited122_l26_replay as core
 return pub,r,core
def prepare():
 assert sys.platform!='darwin'and not os.environ.get('CUDA_VISIBLE_DEVICES'),'Remote CPU-only preparation'
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_CPU,(300,300));signal.alarm(420);TARGET.mkdir(exist_ok=True);started=time.time()
 with(TARGET/'prepare.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(TARGET/'preparation.json').exists():
   old=read(TARGET/'preparation.json');assert old['source_raw_complete']==DONOR_COMPLETE and old['excluded_HPC3_raw_complete']==TAIL_COMPLETE
   for ref in old['selections']:checked(ref)
   return old
  donor=read(checked(DONOR_COMPLETE));tail=read(checked(TAIL_COMPLETE));contract=read(checked(TAIL_CONTRACT))
  assert donor['status']=='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'and donor['questions']==17664 and donor['trajectories']==176640 and donor['strict_EM_mismatches']==0 and donor['actual_same_question_old_new_seed_collisions']==0
  assert tail['raw_G10_full_qualified']and tail['questions']==9344 and tail['trajectories']==93440 and tail['strict_EM_mismatches']==0
  assert tail['outputs']['source-resolution-contract']==TAIL_CONTRACT and contract['whole_G5_per_qid_no_mixing']and contract['qualified_execution_source_selection_before_L26']and contract['donor_and_HPC3_feature_manifest_union_before_source_selection_forbidden']and contract['source_priority']['HPC3_fully_qualified']<contract['source_priority']['donor_fully_qualified']
  donor_groups=read(checked(donor['outputs']['groups']))['train'];tail_groups=read(checked(tail['outputs']['groups']))['train'];truth,exclusion=select_groups(donor_groups,tail_groups)
  pub,r,core=qualified()
  from reproduction.operations.qwen7b_warmup.replay_bank_prepare import q1_model_binding
  from reproduction.operations.qwen7b_warmup.validation_fixedg5_replay import batching_binding
  free=shutil.disk_usage(TARGET).free;assert free>=120*1024**3,'Shared space guard for future L26 output'
  identity=[{k:g[k]for k in('qid','question_hash','trajectory_ids')}for g in truth];domains={}
  for g in identity:dom=json.loads(g['qid'])[0];domains[dom]=domains.get(dom,0)+1
  assert domains=={'hotpotqa':14080}
  exclusion.update(source_raw_complete=DONOR_COMPLETE,source_donor_groups=donor['outputs']['groups'],excluded_HPC3_raw_complete=TAIL_COMPLETE,excluded_HPC3_groups=tail['outputs']['groups'],parent_source_resolution_contract=TAIL_CONTRACT,whole_own_G5_preserved=True,training_text_hash_repeats_retained=True,current124194_fit_expanded=False)
  exref=once(TARGET/'canonical-qid-exclusion.json',exclusion)
  resolution=dict(role='train',whole_G5_per_qid_no_mixing=True,same_protocol_tid_not_payload_identity=True,donor_and_HPC3_feature_manifest_union_before_source_selection_forbidden=True,qualified_execution_source_selection_before_L26=True,every_selected_tid_must_bind_chosen_execution_physical_prefix_and_retained_row_SHA=True,parent_source_resolution_contract=TAIL_CONTRACT,canonical_qid_exclusion=exref,chosen_old5='original_canonical_old5',chosen_new5='donor_qualified_newseed20261021_accepted138',source_priority=contract['source_priority'],current124194_fit_expanded=False)
  resolution_ref=once(TARGET/'source-resolution-contract.json',resolution)
  idref=once(TARGET/'identity-plan.json',dict(role='train',groups=identity,label_blind=True,source_raw_complete=DONOR_COMPLETE,source_raw_groups=donor['outputs']['groups'],source_resolution_contract=resolution_ref,canonical_qid_exclusion=exref,current124194_fit_expanded=False))
  parts=[identity[i*3520:(i+1)*3520]for i in range(4)];owner={t:i for i,part in enumerate(parts)for g in part for t in g['trajectory_ids']};wanted={t:dict(trajectory_id=t,qid=g['qid'],event='original'if j<5 else'newseed20261021')for g in identity for j,t in enumerate(g['trajectory_ids'])}
  events_ref=donor['outputs']['source-events'];events=read(checked(events_ref));q1,modelbinding=q1_model_binding();batching=batching_binding(GOLDEN,32);streams=[];traces=[];paths=[];seen=set();counts=[0]*4;sources={};seals={};geometry={};source_rows=0
  try:
   for i in range(4):
    folder=TARGET/('shard-%02d'%i)/'inputs';folder.mkdir(parents=True,exist_ok=True);p=folder/'prefix.jsonl.gz';t=folder/'chosen-physical-prefix-bindings.jsonl.gz';assert not any(x.exists()for x in(p,t,p.with_suffix('.gz.partial'),t.with_suffix('.gz.partial'))),'Prior partial projection preserved; use a new explicit attempt';paths.append((p,t));streams.append(gzip.open(p.with_suffix('.gz.partial'),'xb'));traces.append(gzip.open(t.with_suffix('.gz.partial'),'xb'))
   for event in events['events']:
    assert event['event']in('original','newseed20261021')
    for source in event['batches']:
     for ref,store in((source['prefix'],sources),(source['seal'],seals)):
      if ref['path']in store:assert store[ref['path']]==ref
      else:store[ref['path']]=ref
     seal=read(checked(source['seal']));path=checked(source['prefix'])
     with gzip.open(path,'rt')as inp:
      for line in inp:
       row=json.loads(line);t=row['trajectory_id'];source_rows+=1
       if t not in wanted:continue
       assert t not in seen,'Repeated selected physical trajectory source';binding=physical_binding(row,source,event['event'],wanted[t],seal);r.normalize_prefix(row,t,row['qid'],True);measure=core.geometry(row);assert not measure['overlength'],'No question/context truncation permitted';key=event['event'];st=geometry.setdefault(key,dict(trajectories=0,max_prompt=0,max_context=0,short_policy=0));st['trajectories']+=1;st['max_prompt']=max(st['max_prompt'],measure['prompt']);st['max_context']=max(st['max_context'],measure['context']);st['short_policy']+=int(measure['policy']<20)
       i=owner[t];streams[i].write(canonical(row)+b'\n');traces[i].write(canonical(binding)+b'\n');seen.add(t);counts[i]+=1
       if len(seen)%1024==0:progress(dict(stage='DONOR14080_SOURCE_RESOLVED_CAUSAL_PREFIX_PROJECTION_ADVANCING',trajectories=len(seen),expected=140800,counts=counts))
  finally:
   for stream in streams+traces:stream.close()
  assert seen==set(wanted)and counts==[35200]*4
  selections=[]
  for i,(p,t)in enumerate(paths):
   os.replace(str(p.with_suffix('.gz.partial')),str(p));os.replace(str(t.with_suffix('.gz.partial')),str(t));sel=dict(role='peer_newseed_train_G10',model_revision=REVISION,questions=3520,trajectories=[dict(qid=g['qid'],trajectory_id=t)for g in parts[i]for t in g['trajectory_ids']],real_trajectories_per_question=10,label_blind_selection=True,prefix_tokens=20,capture_layers=list(r.LAYERS),persist_layers=[26],prefix_files=[fp(p)],model_files=q1['model_files'],source_model_binding=modelbinding,source_ingest=DONOR_COMPLETE,question_plan=idref,chosen_physical_prefix_bindings=fp(t),source_resolution_contract=resolution_ref,source_execution_selection_completed=True,donor_feature_manifest_union_forbidden=True,domain_scope=dict(kind='qualified_donor_new5_plus_original_old5_after_excluding_selected_HPC3_qids',overall_question_counts=domains,original_warmup_unseen_claim=False),expected_code_root=str(QUALIFIED),shard=i,shards=4,batching=batching,geometry_checked_no_truncation=True,actor_updates=0,new_responses_generated=0);selections.append(once(p.parent/'selection.json',sel))
  # Final projected-row linkage and physical-source readback; no tensor decoding.
  readback_seen=set();samples={}
  for ref in selections:
   sel=read(checked(ref));want={x['trajectory_id']:x['qid']for x in sel['trajectories']};n=0
   with gzip.open(checked(sel['prefix_files'][0]),'rt')as inp,gzip.open(checked(sel['chosen_physical_prefix_bindings']),'rt')as trace:
    for pl,tl in itertools.zip_longest(inp,trace):
     assert pl is not None and tl is not None;row=json.loads(pl);binding=json.loads(tl);tid=row['trajectory_id'];assert tid not in readback_seen and want.get(tid)==row['qid']==binding['qid']and tid==binding['trajectory_id']and digest(row)==binding['canonical_prefix_row_sha256']and row['prefix_sha256']==binding['prefix_body_sha256'];readback_seen.add(tid);n+=1
     if binding['chosen_event']not in samples:normalized=r.normalize_prefix(row,tid,row['qid'],True);samples[binding['chosen_event']]=dict(canonical_prefix_row_sha256=binding['canonical_prefix_row_sha256'],causal_prefix_sha256=normalized['source_prefix_sha256'],question_tokens=normalized['question_length'],policy_tokens=normalized['policy_length'])
   assert n==35200
  assert readback_seen==seen and set(samples)=={'original','newseed20261021'}
  for ref in list(sources.values())+list(seals.values()):checked(ref)
  for ref in(DONOR_COMPLETE,TAIL_COMPLETE,TAIL_CONTRACT,events_ref,donor['outputs']['groups'],tail['outputs']['groups']):checked(ref)
  readback=once(TARGET/'physical-source-final-readback.json',dict(status='DONOR14080_ALL_PROJECTED_ROWS_AND_FINAL_PHYSICAL_PREFIX_SEAL_SHA_BINDINGS_PASSED',questions=14080,trajectories=140800,shard_counts=counts,source_prefix_files=len(sources),source_seal_files=len(seals),source_prefix_seal_manifest=once(TARGET/'physical-source-files.json',dict(prefixes=list(sources.values()),seals=list(seals.values()))),both_actual_causal_input_examples=samples,qualified_runtime_publication=PUB,canonical_qid_exclusion=exref,source_resolution_contract=resolution_ref,model_or_tensor_loaded=False,GPU_submitted=False,epoch=time.time()))
  result=dict(status='DONOR14080_FULL_QUALIFIED_G10_SOURCE_RESOLVED_L26_INPUTS_PREPARED_NO_GPU',source=fp(Path(__file__)),source_raw_complete=DONOR_COMPLETE,source_raw_groups=donor['outputs']['groups'],source_events=events_ref,excluded_HPC3_raw_complete=TAIL_COMPLETE,excluded_HPC3_groups=tail['outputs']['groups'],parent_source_resolution_contract=TAIL_CONTRACT,source_resolution_contract=resolution_ref,canonical_qid_exclusion=exref,identity_plan=idref,physical_source_final_readback=readback,qualified_runtime_root=str(QUALIFIED),qualified_runtime_publication=PUB,qualified_runtime_source_commit=pub.get('commit'),questions=14080,trajectories=140800,selections=selections,shard_order='four contiguous retained raw-G10 qid-order segments; 3520 questions each',domains=domains,geometry=geometry,batching=batching,compute_layers=list(r.LAYERS),persist_layers=[26],writer_layout='lossless_question_base_xor_policy_v1',GPU_submitted=False,feature_bank_created=False,actor_updates=0,probe_updates=0,new_rollouts_generated=0,current124194_fit_expanded=False,free_bytes=free,CPU_affinity=sorted(os.sched_getaffinity(0)),RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started,epoch=time.time());once(TARGET/'preparation.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true',required=True);p.parse_args()
 try:print(json.dumps(prepare(),sort_keys=True))
 except BaseException as exc:
  if TARGET.exists():progress(dict(stage='PREPARE_FAILED_PARTIAL_OUTPUTS_PRESERVED',error=type(exc).__name__,message=str(exc)))
  raise
