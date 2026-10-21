"""Freeze already-sealed own HPC3 G5 metadata, never mutate the running writer.

This is a candidate intake, not raw/tokenizer acceptance or a feature bank.
Large response/hidden/model files are never read. Full strict-EM/causal ingest
must precede any future replay. Cross-execution whole-G5 dedup is label blind.
"""
import argparse,collections,fcntl,hashlib,json,os,resource,stat,sys,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';TAIL=A/'hpc3-tail-newseedG5-20261021-v1';TARGET=A/'hpc3-tail-sealed-intake-v1'
REV='a09a35458c702b33eeacc393d103063234e8bc28';CONFIG_SHA='368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835'
MAIN_COMPLETE={'path':str(A/'joint124194-native-training-bank-v1/complete.json'),'bytes':6562,'sha256':'a8fe2c2ea823087d50ac86b283eb361b51ae7ba5ba0571860193214338200801'}
FIELDS=('sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype')
STATUS='HPC3_SEALED_G5_METADATA_AND_CANONICAL_OLD5_LINKS_VERIFIED_PENDING_FULL_INGEST'

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def stamp(p):
 s=Path(p).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def regular(p):
 p=Path(p);require(not p.is_symlink()and stat.S_ISREG(p.stat().st_mode),'Only regular metadata files');return p
def load(p):
 p=regular(p);before=stamp(p);raw=p.read_bytes();value=json.loads(raw);require(before==stamp(p),'Metadata changed during read')
 return value,dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 value,actual=load(ref['path']);require(all(actual[k]==ref[k]for k in('path','bytes','sha256')),'Bound metadata SHA differs');return value,actual
def write_once(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=canonical(value)+b'\n'
 if p.exists():require(p.read_bytes()==raw,'Conflicting immutable intake metadata')
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def tid(run,ordinal,j):return'%s-outer000001-c%04d-g%d'%(run,ordinal,j)
def seed(run,trajectory,base,attempt=None):
 values=[run,1,trajectory,0,0,base]
 if attempt is not None:values.append(attempt)
 return int.from_bytes(hashlib.sha256(canonical(values)).digest()[:8],'big')%(2**31-1)
def own5(candidate):
 ids=candidate['trajectory_ids'];labels=candidate['labels']
 require(isinstance(candidate['qid'],str)and len(candidate['question_hash'])==64 and len(ids)==len(set(ids))==len(labels)==5,'Whole own G5 required')
 require(all(isinstance(t,str)and t for t in ids)and all(type(x)is int and x in(0,1)for x in labels),'Own IDs/binary labels invalid')
 require(candidate['metadata_verified']is True and candidate['execution_run_id']and candidate['execution_host_cluster'],'Explicit execution metadata required')
 return candidate

def choose_whole_g5(candidates):
 """Preserve one entire source G5 per qid; priority is explicit, never labels.

 Distinct canonical qids with the same text hash are retained. Across execution
 namespaces the protocol IDs may coincide; this never proves payload equality.
 """
 by={}
 for c in candidates:own5(c);by.setdefault(c['qid'],[]).append(c)
 selected=[];dropped=[]
 for qid in sorted(by):
  values=by[qid];require(len({x['question_hash']for x in values})==1,'Same qid has conflicting canonical text hash')
  winner=min(values,key=lambda x:(x['source_priority'],x['execution_run_id'],x['seal']['path']))
  selected.append(winner)
  dropped.extend(dict(qid=qid,execution_run_id=x['execution_run_id'],seal=x['seal'],reason='one_whole_G5_chosen_by_explicit_priority_no_label_use')for x in values if x is not winner)
 allids=[t for c in selected for t in c['trajectory_ids']];require(len(allids)==len(set(allids)),'Selected distinct canonical qids share own trajectory IDs')
 return selected,dropped

def phase(receipt,mode,conf,n,config_sha):
 expected=dict(run_id=conf['run_id'],outer=1,rank=0,phase=mode,actor_version='official7b-'+REV,config_sha256=config_sha,questions=n,trajectories=5*n,sampling_attempt_id=conf.get('sampling_attempt_id'))
 require(all(receipt.get(k)==v for k,v in expected.items()),'Phase identity/config/count differs')
 require(receipt['engine_seed']==seed(conf['run_id'],'engine',conf['seed'],conf.get('sampling_attempt_id')),'Native engine seed differs')
 return receipt

def payload_refs(seal,dest):
 result={k:seal[ph][key]for k,ph,key in [('states','prefix','states'),('prefix','prefix','prefix'),('full','full','full'),('own_em','full','own_em')]}
 for item in result.values():
  p=regular(item['path']);require(str(p.resolve()).startswith(str(dest.resolve())+'/')and p.stat().st_size==item['bytes']and len(item['sha256'])==64,'Source payload reference/size differs')
 return result

def batch_metadata(inp,ref,conf,new=False):
 batch,actual=checked(ref);require(Path(actual['path'])==inp,'Frozen input path differs');dest=inp.parents[2]/'sampling'/inp.parent.name/'batches'/inp.stem
 seal,sref=load(dest/'complete.json');cfg,cref=load(dest/'config.json');n=len(batch['rows'])
 require(seal['input_sha256']==actual['sha256']and seal['run_id']==conf['run_id']and seal['questions']==n and seal['trajectories']==5*n and seal['actor_optimizer_updates']==0 and seal['model_revision']==REV,'Sealed input/model/count differs')
 require(cfg['run_id']==conf['run_id']and cfg['seed']==conf['seed']and cfg.get('sampling_attempt_id')==conf.get('sampling_attempt_id')and cfg['actor_optimizer_updates']==0,'Per-batch source sampling config differs')
 pref=phase(seal['prefix'],'prefix',cfg,n,cref['sha256']);full=phase(seal['full'],'continue',cfg,n,cref['sha256'])
 require(all(pref[k]==full[k]for k in FIELDS),'Prefix/full scientific protocol differs')
 require(pref['actual_compute_dtype']=='torch.bfloat16','Production BF16 protocol differs')
 outer=dest/'outers/outer-000001';scan,scref=load(outer/'prefix/scan.json');selection,selref=load(outer/'probe-score/selection.json')
 require(scan['ranks']==[pref]and full['prefix_scan_sha256']==scref['sha256']and full['selection_sha256']==selref['sha256'],'Current sealed prefix/full metadata chain differs')
 require(selection['kind']=='ALL_G5_NO_SELECTION','Label-based source selection forbidden')
 if new:
  require(seal['execution_run_id']==conf['execution_run_id']and seal['execution_host_cluster']=='HPC3'and seal['execution_source_commit']==conf['execution_source_commit']and seal['execution_job']in('745518','745519')and seal['potentialOverlap']and seal['donor_original_receipt_claim']is False and not seal['hidden_cached'],'Own HPC3 execution provenance differs')
  require(cfg['prefix_policy_tokens']==20 and cfg['capture_prefix_tokens']==20 and cfg['seed']==20261021,'Actual Prefix20/new seed config differs')
 refs=payload_refs(seal,dest)
 event=dict(event='HPC3_newseed20261021'if new else'original',input=actual,seal=sref,config=cref,prefix=refs['prefix'],full=refs['full'],own_em=refs['own_em'],states=refs['states'],scan=scref,selection=selref,questions=n,trajectories=5*n,protocol={k:pref[k]for k in FIELDS},producer_payload_SHA_inherited=True,payload_bytes_stat_verified=True,full_payload_SHA_or_strictEM_recomputed_here=False)
 return batch,seal,event

def prepare():
 require(sys.platform!='darwin','Run metadata intake on remote Linux only');resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120));start=time.time()
 TARGET.mkdir(exist_ok=True)
 with(TARGET/'prepare.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  conf,config=load(TAIL/'config.json');require(config['sha256']==CONFIG_SHA,'Fixed producer config changed')
  require(conf['model_revision']==REV and conf['actor_optimizer_updates']==0 and conf['potentialOverlap']and conf['frozen_main_cohort_questions']==124194 and not conf['frozen_main_cohort_expanded'],'Frozen actor/main cohort differs')
  manifest,mref=load(TAIL/'inputs/manifest.json');by={x['path']:x for x in manifest['batches']};frozen=[]
  for path in sorted((TAIL/'sampling').glob('shard-*/batches/batch-*/complete.json')):
   seal,sref=load(path);inp=TAIL/'inputs'/path.parent.parent.parent.name/(path.parent.name+'.json');require(str(inp)in by,'Unfrozen source batch encountered')
   frozen.append(dict(input=by[str(inp)],seal=sref))
  require(frozen,'No actual own sealed G5 batches available')
  snapshot=TARGET/('snapshot-%d-%s'%(int(start),digest(frozen)[:12]));snapshot.mkdir()
  main,mcomplete=checked(MAIN_COMPLETE);require(main['questions']==124194 and main['status']=='FROZEN_FOUR_BANK_REAL_G10_124194_NATIVE_JOINT_INPUTS_ACCEPTED','Accepted main bank identity differs')
  excluded,eref=checked(main['evaluation_exclusions']);require(len(excluded['groups'])==6656 and excluded['role']=='joint_training_fixed_evaluation_identity_exclusions'and excluded['final_labels_or_rollouts_read']is False and all(set(g)=={'qid','question_hash'}for g in excluded['groups']),'True6656 identity-only exclusions required')
  eq={g['qid']for g in excluded['groups']};eh={g['question_hash']for g in excluded['groups']}
  main_groups,mgref=checked(main['groups']);mq={g['qid']for g in main_groups['train']};mh={g['question_hash']for g in main_groups['train']};require(len(mq)==124194,'Main qid count differs');del main_groups
  assets,_=checked(conf['assets_metadata']);oldconf,ocref=checked(assets['original_config']);oldbank,obref=load(B/'bank/complete.json');oldgroups,ogref=checked(oldbank['groups']);old={g['qid']:g for g in oldgroups['train']};require(len(old)==169028 and oldconf['seed']==1729 and oldconf['run_id']!=conf['run_id'],'Canonical original old5 source differs');del oldgroups
  oldmanifest,omref=checked(assets['input_manifest']);oldrefs={x['path']:x for x in oldmanifest['batches']if x['split']=='train'}
  newevents=[];candidates=[];seenq=set();seent=set();oldinputs={};oldwanted={};domains=collections.Counter();meta_rows=[]
  for item in frozen:
   inp=Path(item['input']['path']);batch,seal,event=batch_metadata(inp,item['input'],conf,True);require(event['seal']==item['seal'],'Frozen seal changed')
   require(not any(g['qid']in eq or g['question_hash']in eh for g in batch['rows']),'True evaluation identity in sealed input; no EM opened')
   expected={tid(conf['run_id'],g['candidate_ordinal'],j):(g,j)for g in batch['rows']for j in range(5)};labels={};empath=regular(event['own_em']['path']);before=stamp(empath);raw=empath.read_bytes();require(hashlib.sha256(raw).hexdigest()==event['own_em']['sha256'],'Own EM metadata fresh SHA differs')
   for line in raw.splitlines():
    row=json.loads(line);t=row['trajectory_id'];require(t in expected and t not in labels,'Missing/repeated/foreign own EM ID');g,j=expected[t]
    require(row['qid']==g['qid']and type(row['strict_em'])is int and row['strict_em']in(0,1)and row['prefix_scan_sha256']==event['scan']['sha256']and row['selection_sha256']==event['selection']['sha256']and len(row['prefix_sha256'])==64,'Own EM identity/causal receipt link differs');labels[t]=row['strict_em']
   require(set(labels)==set(expected)and stamp(empath)==before,'Incomplete/changed own EM metadata')
   for g in batch['rows']:
    q,h,o=g['qid'],g['question_hash'],g['source_ordinal'];require(q not in seenq and g['candidate_ordinal']==o and g['split']=='train','Repeated/noncanonical own question')
    source=old[q];require(source['question_hash']==h and source['source_ordinal']==o,'Canonical old5 membership differs');oldids=source['trajectory_ids'];newids=[tid(conf['run_id'],o,j)for j in range(5)]
    require(len(oldids)==len(set(oldids))==5 and oldids==[tid(oldconf['run_id'],o,j)for j in range(5)]and not set(oldids).intersection(newids)and not seent.intersection(newids),'Old/new whole G5 ID association differs')
    seenq.add(q);seent.update(newids);domains[json.loads(q)[0]]+=1
    oldpath=str(B/'inputs'/('shard-%02d'%(o%8))/('batch-%06d.json'%((o//8)//128)));require(oldpath in oldrefs,'Original canonical input batch missing');oldinputs[oldpath]=oldrefs[oldpath];oldwanted[q]=g
    candidates.append(dict(qid=q,question_hash=h,source_ordinal=o,trajectory_ids=newids,labels=[labels[t]for t in newids],metadata_verified=True,source_priority=10,execution_run_id=seal['execution_run_id'],execution_namespace=str(TAIL),execution_source_commit=seal['execution_source_commit'],execution_host_cluster='HPC3',execution_job=seal['execution_job'],sampling_protocol_source_run_id=conf['sampling_protocol_source_run_id'],sampling_seed=conf['seed'],sampling_attempt_id=conf['sampling_attempt_id'],seal=event['seal'],source_event_index=len(newevents),old5_trajectory_ids=oldids,old5_labels=source['labels'],old5_source_ordinal=o,old5_source_input=oldrefs[oldpath],old5_source_run_id=oldconf['run_id'],old5_seed=oldconf['seed'],old5_source_namespace=str(B),already_in_current124194=q in mq,text_hash_overlap_current124194=h in mh,strict_EM_recompute_pending=True))
   newevents.append(event)
   meta_rows.append(dict(batch=inp.name,input=event['input'],seal=event['seal'],questions=len(batch['rows'])))
  oldevents=[];old_seen=set();candidate_by_qid={g['qid']:g for g in candidates}
  for path,ref in sorted(oldinputs.items()):
   batch,seal,event=batch_metadata(Path(path),ref,oldconf)
   require(event['protocol']==newevents[0]['protocol'],'Original/new scientific rollout protocol differs')
   for g in batch['rows']:
    if g['qid']not in oldwanted:continue
    wanted=oldwanted[g['qid']];require(all(g[k]==wanted[k]for k in('qid','question_hash','source_ordinal','candidate_ordinal','initial_prompt_token_ids','reward_model')),'Old/new exact question/prompt/gold source differs');old_seen.add(g['qid']);candidate_by_qid[g['qid']].update(old5_source_event_index=len(oldevents),old5_seal=event['seal'])
   oldevents.append(event)
  require(old_seen==seenq,'Original old5 token-input coverage missing')
  selected,dropped=choose_whole_g5(candidates);eligible=[g for g in selected if not g['already_in_current124194']];already=[g for g in selected if g['already_in_current124194']]
  overlaps=dict(current124194_qid_overlap=len(already),current124194_hash_overlap_questions=sum(g['text_hash_overlap_current124194']for g in selected),canonical_train_text_duplicates=len(selected)-len({g['question_hash']for g in selected}),true6656_qid_overlap=0,true6656_hash_overlap=0,donor_current_net_new_coverage_unknown=True,shared_donor_atomic_claim=False,cross_donor_potentialOverlap=True)
  outputs={};outputs['frozen-seals']=write_once(snapshot/'frozen-seals.json',dict(config=config,input_manifest=mref,batches=meta_rows,frozen_at=start,do_not_follow_newly_sealed_batches=True))
  outputs['identities']=write_once(snapshot/'whole-G5-identities.json',dict(role='train',identity_only=True,labels_read_from_this_file=False,groups=[{k:g[k]for k in('qid','question_hash','trajectory_ids','execution_run_id','execution_host_cluster','seal')}for g in selected],questions=len(selected),trajectories=5*len(selected),potentialOverlap=True,ordered_qid_own5_execution_sha256=digest([[g['qid'],g['trajectory_ids'],g['execution_run_id']]for g in selected])))
  outputs['candidates']=write_once(snapshot/'canonical-g10-candidates.json',dict(role='train_candidate_metadata_not_raw_acceptance',groups=eligible,old5_original_config=ocref,old5_original_bank=ogref,newseed_config=config,complete_raw_acceptance=False,whole_G5_choice_policy=dict(existing_accepted=0,HPC3_own_sealed=10,donor_sealed=20,tie_break='execution_run_id_then_seal_path',labels_or_scores_used=False,identical_protocol_trajectory_IDs_do_not_prove_identical_payload=True)))
  outputs['source-events']=write_once(snapshot/'source-events.json',dict(events=[dict(event='original',batches=oldevents),dict(event='HPC3_newseed20261021',batches=newevents)],repacked=False,hidden_present=False))
  outputs['future-feature-plan']=write_once(snapshot/'future-feature-intake.json',dict(role='independent_L26_P20_feature_intake_after_full_ingest_only',candidate_groups=outputs['candidates'],source_events=outputs['source-events'],prefix_tokens=20,feature_layer=26,actor_revision=REV,actor_updates=0,feature_bank_created=False,full_strict_EM_causal_seed_ingest_required=True,current124194_fit_must_remain_unchanged=True,label_blind_whole_G5_choice=True,original_warmup_unseen_claim=False))
  outputs['overlap']=write_once(snapshot/'overlap.json',dict(overlaps,whole_source_duplicate_G5_dropped=dropped))
  result=dict(status=STATUS,snapshot=str(snapshot),outputs=outputs,source_commit=os.environ.get('INTAKE_SOURCE_COMMIT'),auditor_source=load_source(),producer_config=config,original_config=ocref,original_bank_complete=obref,canonical_training_groups=ogref,original_inputs=omref,current_main_complete=mcomplete,current_main_groups=mgref,true_evaluation_identity_exclusions=eref,frozen_sealed_batches=len(frozen),sealed_questions=len(selected),sealed_trajectories=5*len(selected),future_eligible_questions=len(eligible),future_eligible_G10_trajectories=10*len(eligible),domains=dict(domains),overlap=overlaps,own_EM_records_fresh_SHA_verified=5*len(selected),own_EM_binary_ID_and_metadata_links_verified=True,full_strict_EM_recomputed=False,full_causal_native_seed_journals_revalidated=False,payload_tensor_or_model_loaded=False,full_response_payload_read=False,new_rollouts_or_GPU_jobs=0,feature_bank_created=False,current124194_fit_expanded=False,original_warmup_saw_all_169028=True,elapsed_seconds=time.time()-start,epoch=time.time())
  result['complete']=write_once(snapshot/'complete.json',result);return result

def load_source():
 p=Path(__file__);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-sealed-metadata',action='store_true',required=True);p.parse_args();print(json.dumps(prepare(),sort_keys=True))
