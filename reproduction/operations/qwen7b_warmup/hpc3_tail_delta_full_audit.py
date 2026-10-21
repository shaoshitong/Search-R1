"""Explicit frozen next-tail delta: reuse qualified new5 audit, verify old5 too.

No writer mutations, model loading, GPU calls or source mixing. Two independent
CPU workers each audit complete real new G5 then its matching original old G5.
"""
import argparse,collections,fcntl,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';OUT=A/'hpc3-tail-completion-delta01-v1'
FROZEN={'path':str(OUT/'frozen-seals.json'),'bytes':79129,'sha256':'aa7646573bb6c57059ceaf7699ea594aab1e61c5e7acb81bdb6320636825a10b'}
NEW=A/'hpc3-tail73-full-new5-audit-v1/code-a23ea4265633/audit.py';NEW_SHA='a21a0e522e44290ad08ebccd21e841f429d5402f89436b83fdf2f71f86001ad2'
META=A/'hpc3-tail-sealed-intake-v1/code-48ad7d6ac534/hpc3_tail_sealed_intake.py';META_SHA='87298598d5e6a523f6814bf6e31e3918b04eb11a7ad8c1e08003dff4df34bdc4'
STATUS='HPC3_OWN_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_FULL_STRICT_EM_CAUSAL_SEEDS_QUALIFIED'
def module(path,sha,name):
 if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Published dependency changed '+name)
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');raw=json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
 with tmp.open('wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(str(tmp),str(p))
def prepare():
 m=module(META,META_SHA,'frozen_intake');n=module(NEW,NEW_SHA,'qualified_new');a=n.inherited()
 with(OUT/'prepare.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(OUT/'prepared.json').exists():return dict(prepared=a.proof(OUT/'prepared.json'),summary=a.read(OUT/'prepared.json'))
  frozen,_=m.checked(FROZEN);m.require(len(frozen['batches'])==114 and frozen['questions']==14592,'Published fixed114 delta differs');conf,cref=m.checked(frozen['producer_config']);assets,_=m.checked(conf['assets_metadata']);oldconf,ocref=m.checked(assets['original_config']);bank,obref=m.load(B/'bank/complete.json');groups,ogref=m.checked(m.local_bank_ref(bank['groups'],B/'bank/groups.json'));old={g['qid']:g for g in groups['train']};del groups
  manifest,omref=m.checked(assets['input_manifest']);oldrefs={x['path']:x for x in manifest['batches']if x['split']=='train'};prior,_=m.checked(frozen['prior_qualified_G10']);pq,_=m.checked(prior['outputs']['groups']);priorq={g['qid']for g in pq['train']};del pq
  main,_=m.checked(m.MAIN_COMPLETE);ex,exref=m.checked(main['evaluation_exclusions']);eq={g['qid']for g in ex['groups']};eh={g['question_hash']for g in ex['groups']};m.require(len(eq)==6656,'True evaluation identity count differs')
  events=[];candidates=[];oldinputs={};wanted={};seen=set()
  for item in frozen['batches']:
   batch,seal,event=m.batch_metadata(Path(item['input']['path']),item['input'],conf,True);m.require(event['seal']==item['seal'],'Frozen physical seal changed');labels={};raw=Path(event['own_em']['path']).read_bytes();m.require(hashlib.sha256(raw).hexdigest()==event['own_em']['sha256'],'new ownEM SHA changed')
   for line in raw.splitlines():
    v=json.loads(line);m.require(v['trajectory_id']not in labels and type(v['strict_em'])is int and v['strict_em']in(0,1),'New own EM repeated/nonbinary');labels[v['trajectory_id']]=v['strict_em']
   expected={m.tid(conf['run_id'],g['candidate_ordinal'],j)for g in batch['rows']for j in range(5)};m.require(set(labels)==expected,'New ownEM complete G5 ID set differs')
   for g in batch['rows']:
    q,h,o=g['qid'],g['question_hash'],g['source_ordinal'];source=old[q];m.require(q not in seen and q not in priorq and q not in eq and h not in eh and source['question_hash']==h and source['source_ordinal']==o==g['candidate_ordinal'],'Original canonical/disjoint/evaluation identity differs');seen.add(q);oldpath=str(B/'inputs'/('shard-%02d'%(o%8))/('batch-%06d.json'%((o//8)//128)));m.require(oldpath in oldrefs,'Canonical old input missing');oldinputs[oldpath]=oldrefs[oldpath];wanted[q]=g;ids=[m.tid(conf['run_id'],o,j)for j in range(5)]
    candidates.append(dict(qid=q,question_hash=h,source_ordinal=o,trajectory_ids=ids,labels=[labels[t]for t in ids],old5_trajectory_ids=source['trajectory_ids'],old5_labels=source['labels'],seal=event['seal'],execution_run_id=seal['execution_run_id'],worker=item['worker'],old5_input=oldrefs[oldpath]))
   events.append(event)
  oldevents=[];by={g['qid']:g for g in candidates};covered=set()
  for path,ref in sorted(oldinputs.items()):
   batch,seal,event=m.batch_metadata(Path(path),ref,oldconf);m.require(event['protocol']==events[0]['protocol'],'Old/new rollout protocol changed')
   for g in batch['rows']:
    if g['qid']not in wanted:continue
    m.require(all(g[k]==wanted[g['qid']][k]for k in('qid','question_hash','source_ordinal','candidate_ordinal','initial_prompt_token_ids','reward_model')),'Old/new exact prompt/gold changed');covered.add(g['qid']);by[g['qid']]['old5_seal']=event['seal']
   oldevents.append(event)
  m.require(covered==seen and len(seen)==14592,'Frozen full delta question coverage changed');gp=OUT/'metadata/candidates.json';sp=OUT/'metadata/source-events.json';atomic(gp,dict(groups=candidates));atomic(sp,dict(events=[dict(event='original',batches=oldevents),dict(event='HPC3_newseed20261021',batches=events)]));v=dict(status='FIXED_NEXT114_METADATA_PREPARED_PENDING_FULL_AUDIT',frozen=FROZEN,questions=len(seen),new5_batches=114,old5_source_batches=len(oldevents),candidates=a.proof(gp),source_events=a.proof(sp),producer_config=cref,original_config=ocref,canonical_old_groups=ogref,evaluation_exclusions=exref,prior_full_qualified=frozen['prior_qualified_G10'],source=a.proof(Path(__file__)),current124194_fit_expanded=False,full_qualified=False,epoch=time.time());atomic(OUT/'prepared.json',v);return dict(prepared=a.proof(OUT/'prepared.json'),summary=v)
def run(worker):
 if sys.platform=='darwin'or os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Own remote CPU0GPU only')
 n=module(NEW,NEW_SHA,'qualified_new');a=n.inherited();d=OUT/('worker-%d'%worker);d.mkdir(exist_ok=True)
 with(d/'audit.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(d/'complete.json').exists():return dict(complete=a.proof(d/'complete.json'),summary=a.read(d/'complete.json'))
  start=time.time();prep=a.read(OUT/'prepared.json');a.require(prep['frozen']==FROZEN,'Wrong immutable delta prepared');frozen=a.read(a.checked(FROZEN));main=a.read(a.checked(prep['producer_config']));oldconf=a.read(a.checked(prep['original_config']));candidates=[g for g in a.read(a.checked(prep['candidates']))['groups']if g['worker']==worker];by={g['qid']:g for g in candidates};events=a.read(a.checked(prep['source_events']));items=[x for x in frozen['batches']if x['worker']==worker];job=('745518','745519')[worker];a.require(os.environ.get('SLURM_JOB_ID')==job,'Wrong own allocation');atomic(d/'controller.json',dict(pid=os.getpid(),birth_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],job_id=job,step_id=os.environ.get('SLURM_STEP_ID'),questions=len(by),new_batches=len(items),source=a.proof(Path(__file__)),CPU_only=True,epoch=time.time()));score,tok,strict=n.scorer(a,main);ledger={};newproof=[]
  for i,item in enumerate(items):
   proof=n.audit_batch(a,item,worker,main,score,tok,strict,ledger);p=d/'new5'/('batch-%04d.json'%i);atomic(p,proof);newproof.append(a.proof(p));v=dict(stage='NEXT_DELTA_NEW5_FULL_SHA_STRICT_EM_CAUSAL_SEEDS_ADVANCING',worker=worker,completed_batches=i+1,expected_batches=len(items),questions=sum(a.read(r['path'])['questions']for r in newproof),trajectories=sum(a.read(r['path'])['trajectories']for r in newproof),elapsed_seconds=time.time()-start,epoch=time.time());atomic(d/'progress.json',v);print(json.dumps(v),flush=True)
  oldevents=[]
  for e in events['events']:
   if e['event']=='original':oldevents=e['batches']
  oldevents=[e for e in oldevents if any(g['old5_input']['path']==e['input']['path']for g in candidates)];oldseen=set();oldproof=[]
  for i,event in enumerate(oldevents):
   inp=a.read(a.checked(event['input']));conf=a.read(a.checked(event['config']));seal=a.read(a.checked(event['seal']));a.require(conf['seed']==1729 and conf['run_id']==oldconf['run_id']and conf['actor_optimizer_updates']==0 and conf['model_revision']==a.REVISION,'Old source actor/seed differs');prefix,full=seal['prefix'],seal['full'];scan=a.read(a.checked(event['scan']));selection=a.read(a.checked(event['selection']));a.require(scan['ranks']==[prefix]and selection['kind']=='ALL_G5_NO_SELECTION_NO_OVERSAMPLING'and full['prefix_scan_sha256']==event['scan']['sha256']and full['selection_sha256']==event['selection']['sha256'],'Original source receipt chain differs')
   for key in('sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype'):a.require(prefix[key]==full[key]==event['protocol'][key],'Old scientific protocol differs')
   artifacts={k:event[k]for k in('states','prefix','full','own_em')}
   for r in artifacts.values():a.checked(r)
   before={r['path']:(Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns)for r in artifacts.values()};sources={g['qid']:g for g in inp['rows']};wanted={t:(g['qid'],y)for g in candidates if g['qid']in sources for t,y in zip(g['old5_trajectory_ids'],g['old5_labels'])};prefrows=[v for v in a.records(artifacts['prefix']['path'])if v['trajectory_id']in wanted];emrows=[v for v in a.records(artifacts['own_em']['path'])if v['trajectory_id']in wanted];pref={v['trajectory_id']:v for v in prefrows};em={v['trajectory_id']:v for v in emrows};a.require(len(prefrows)==len(pref)==len(emrows)==len(em)==len(wanted)and set(pref)==set(em)==set(wanted),'Original selected ownG5 missing/repeated');seen=set();counts=collections.Counter()
   for row in a.records(artifacts['full']['path']):
    t=row['trajectory_id']
    if t not in wanted:continue
    a.require(t not in seen and t not in oldseen,'Repeated original own ID');seen.add(t);oldseen.add(t);q,y=wanted[t];g=sources[q];a.require(g['question_hash']==by[q]['question_hash']and g['source_ordinal']==by[q]['source_ordinal'],'Original canonical metadata mismatch');projected={k:row[k]for k in('trajectory_id','qid','strict_em','actor_version','prefix_sha256','prefix_scan_sha256','selection_sha256')};a.require(em[t]==projected and em[t]['strict_em']==y,'Original ownEM linkage mismatch');counts.update(a.audit_trajectory(pref[t],row,g,conf,y,score,ledger,'original_old5_for_next_tail_delta'))
   a.require(seen==set(wanted)and before=={r['path']:(Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns)for r in artifacts.values()},'Original selected wholeG5 incomplete/changed');v=dict(status='NEXT_DELTA_ORIGINAL_OLD5_SUBSET_FULL_SHA_STRICT_EM_CAUSAL_SEEDS_PASSED',source_event=event,questions=len(wanted)//5,trajectories=len(wanted),strict_EM_recomputed=len(wanted),strict_EM_mismatches=0,**dict(counts),epoch=time.time());p=d/'old5'/('batch-%04d.json'%i);atomic(p,v);oldproof.append(a.proof(p));v=dict(stage='NEXT_DELTA_OLD5_FULL_SHA_STRICT_EM_CAUSAL_SEEDS_ADVANCING',worker=worker,completed_batches=i+1,expected_batches=len(oldevents),questions=len(oldseen)//5,trajectories=len(oldseen),elapsed_seconds=time.time()-start,epoch=time.time());atomic(d/'progress.json',v);print(json.dumps(v),flush=True)
  a.require(oldseen=={t for g in candidates for t in g['old5_trajectory_ids']},'OldG5 own IDs incomplete');groups=[dict(qid=g['qid'],question_hash=g['question_hash'],source_ordinal=g['source_ordinal'],trajectory_ids=g['old5_trajectory_ids']+g['trajectory_ids'],labels=g['old5_labels']+g['labels'],chosen_new5_execution_run_id=g['execution_run_id'],chosen_new5_seal=g['seal'],old5_seal=g['old5_seal'])for g in candidates];a.require(len({t for g in groups for t in g['trajectory_ids']})==len(groups)*10,'Own realG10 ID duplication');p=d/'raw-g10-groups.json';atomic(p,dict(train=groups))
  import torch
  a.require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA');result=dict(status=STATUS,worker=worker,job_id=job,questions=len(groups),trajectories=10*len(groups),new5_batches=len(items),old5_source_batches=len(oldevents),new5_audits=newproof,old5_audits=oldproof,new5_strict_EM_recomputed=5*len(groups),old5_strict_EM_recomputed=5*len(groups),strict_EM_mismatches=0,actual_same_question_old_new_seed_collisions=0,raw_G10_full_qualified=True,groups=a.proof(p),prepared=a.proof(OUT/'prepared.json'),frozen=FROZEN,source=a.proof(Path(__file__)),inherited_new5_auditor=a.proof(NEW),inherited_trajectory_auditor=a.proof(n.BASE_AUDIT),tokenizer_files=tok,tokenizer_strict_scorer=strict,actor_updates=0,probe_updates=0,feature_bank_created=False,current124194_fit_expanded=False,CUDA_initialized=False,model_tensors_loaded=False,elapsed_seconds=time.time()-start,epoch=time.time());atomic(d/'complete.json',result);return dict(complete=a.proof(d/'complete.json'),summary=result)
def finalize():
 n=module(NEW,NEW_SHA,'qualified_new');a=n.inherited();prep=a.read(OUT/'prepared.json');values=[];refs=[];groups=[]
 for w in(0,1):
  p=OUT/('worker-%d'%w)/'complete.json';v=a.read(p);a.require(v['status']==STATUS and v['worker']==w and v['frozen']==FROZEN and v['strict_EM_mismatches']==0 and v['raw_G10_full_qualified'],'Actual delta worker incomplete');refs.append(a.proof(p));values.append(v);groups+=a.read(a.checked(v['groups']))['train']
 a.require(len(groups)==len({g['qid']for g in groups})==prep['questions']==14592 and len({t for g in groups for t in g['trajectory_ids']})==len(groups)*10,'Final rawG10 identity/count mismatch');p=OUT/'raw-g10/groups.json';atomic(p,dict(train=groups));events=a.read(a.checked(prep['source_events']));sp=OUT/'raw-g10/source-events.json';atomic(sp,events);cp=OUT/'raw-g10/source-resolution-contract.json';contract=dict(qualified_execution_source_selection_before_L26=True,donor_and_HPC3_feature_manifest_union_before_source_selection_forbidden=True,same_protocol_tid_not_payload_identity=True,every_selected_tid_must_bind_chosen_execution_physical_prefix_and_retained_row_SHA=True,whole_G5_per_qid_no_mixing=True,qualification_tier_before_source_priority=True,current124194_fit_expanded=False);atomic(cp,contract);v=dict(status=STATUS,questions=len(groups),trajectories=10*len(groups),new5_batches=114,worker_completes=refs,new5_strict_EM_recomputed=5*len(groups),old5_strict_EM_recomputed=5*len(groups),strict_EM_mismatches=0,raw_G10_full_qualified=True,actual_same_question_old_new_seed_collisions=0,outputs={'groups':a.proof(p),'source-events':a.proof(sp),'source-resolution-contract':a.proof(cp)},frozen=FROZEN,prepared=a.proof(OUT/'prepared.json'),source=a.proof(Path(__file__)),feature_bank_created=False,current124194_fit_expanded=False,epoch=time.time());atomic(OUT/'complete.json',v);return dict(complete=a.proof(OUT/'complete.json'),summary=v)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','worker','finalize'),required=True);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
 if args.action=='worker'and args.worker is None:p.error('--worker required')
 print(json.dumps(prepare()if args.action=='prepare'else run(args.worker)if args.action=='worker'else finalize(),sort_keys=True))
