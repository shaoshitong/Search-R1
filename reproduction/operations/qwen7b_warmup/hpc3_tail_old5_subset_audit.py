"""Complete only fixed9344 old5 + qualified-new5 raw G10; no model/capture."""
import argparse,collections,fcntl,gzip,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';OUT=A/'hpc3-tail9344-old5-qualified-g10-v1'
NEW_MODULE=A/'hpc3-tail73-full-new5-audit-v1/code-a23ea4265633/audit.py';NEW_SHA='a21a0e522e44290ad08ebccd21e841f429d5402f89436b83fdf2f71f86001ad2'
NEW_DONE={'path':str(A/'hpc3-tail73-full-new5-audit-v1/complete.json'),'bytes':1060,'sha256':'c9d97adf60b52ea79a5b17f0d52b06497529764cb0db98694532d4b29ad26523'}
SNAP={'path':str(A/'hpc3-tail-sealed-intake-v1/snapshot-1792589071-f81516f41b5f/complete.json'),'bytes':5298,'sha256':'94c28dfce72db5b21654a5676494555e614bed9b1c23be5d03a8041d6558087d'}
STATUS='HPC3_OWN_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_FULL_STRICT_EM_CAUSAL_SEEDS_QUALIFIED'
def atomic(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n';tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(raw);os.replace(str(tmp),str(p))
def run():
 if sys.platform=='darwin'or os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Remote CPU0GPU only')
 if hashlib.sha256(NEW_MODULE.read_bytes()).hexdigest()!=NEW_SHA:raise ValueError('Qualified new5 auditor source changed')
 s=importlib.util.spec_from_file_location('accepted_new5',NEW_MODULE);n=importlib.util.module_from_spec(s);s.loader.exec_module(n);a=n.inherited();OUT.mkdir(exist_ok=True)
 with(OUT/'audit.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(OUT/'complete.json').exists():return dict(complete=a.proof(OUT/'complete.json'),summary=a.read(OUT/'complete.json'))
  start=time.time();newdone=a.read(a.checked(NEW_DONE));a.require(newdone['new5_full_source_qualified']and newdone['strict_EM_recomputed']==46720 and newdone['strict_EM_mismatches']==0,'Actual whole73 new5 qualification required')
  snap=a.read(a.checked(SNAP));candidates=a.read(a.checked(snap['outputs']['candidates']))['groups'];events=a.read(a.checked(snap['outputs']['source-events']));oldconf=a.read(a.checked(snap['original_config']));newconf=a.read(a.checked(snap['producer_config']));by={g['qid']:g for g in candidates};a.require(len(by)==9344 and oldconf['seed']==1729 and oldconf['run_id']!=newconf['run_id'],'Fixed9344 canonical old/new identity differs')
  controller=dict(pid=os.getpid(),birth_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),phase='QUALIFIED_NEW5_NATIVE_SEED_LEDGER',questions=9344,CPU_only=True,source=a.proof(Path(__file__)),epoch=time.time());atomic(OUT/'controller.json',controller)
  score,tok,strict=n.scorer(a,newconf);ledger={};new_seen=set();new_events=next(x['batches']for x in events['events']if x['event']=='HPC3_newseed20261021');old_events=next(x['batches']for x in events['events']if x['event']=='original')
  expected_new={t:g['qid']for g in candidates for t in g['trajectory_ids']}
  for index,event in enumerate(new_events):
   p=a.checked(event['full']);before=(p.stat().st_size,p.stat().st_mtime_ns)
   for row in a.records(p):
    t=row['trajectory_id'];a.require(t in expected_new and t not in new_seen and row['qid']==expected_new[t],'Qualified new5 own IDs differ');new_seen.add(t);q=row['qid'];own=ledger.setdefault(q,{})
    for chunk in row['sampling_journal']:
     value=a.seed_for(newconf['run_id'],t,chunk['logical_turn_index'],chunk['native_chunk_index'],newconf['seed'],newconf.get('sampling_attempt_id'));a.require(value==chunk['seed']and value not in own,'Qualified new5 native seed ledger differs');own[value]=dict(event='qualified_HPC3_new5',trajectory_id=t)
   a.require(before==(p.stat().st_size,p.stat().st_mtime_ns),'Qualified source changed during native-seed projection')
   atomic(OUT/'progress.json',dict(stage='QUALIFIED_NEW5_NATIVE_SEED_LEDGER_ADVANCING',batches=index+1,expected_batches=len(new_events),trajectories=len(new_seen),epoch=time.time()))
  a.require(new_seen==set(expected_new),'New5 native seed ledger incomplete');accepted=[];old_seen=set();counts=collections.Counter()
  for index,event in enumerate(old_events):
   inp=a.read(a.checked(event['input']));conf=a.read(a.checked(event['config']));seal=a.read(a.checked(event['seal']));a.require(conf['seed']==1729 and conf['run_id']==oldconf['run_id']and conf['actor_optimizer_updates']==0 and conf['model_revision']==a.REVISION,'Original frozen actor/seed source differs')
   prefix,full=seal['prefix'],seal['full'];scan=a.read(a.checked(event['scan']));selection=a.read(a.checked(event['selection']));a.require(scan['ranks']==[prefix]and selection['kind']=='ALL_G5_NO_SELECTION_NO_OVERSAMPLING'and full['prefix_scan_sha256']==event['scan']['sha256']and full['selection_sha256']==event['selection']['sha256'],'Original whole-G5 source chain differs')
   for key in('sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype'):a.require(prefix[key]==full[key]==event['protocol'][key],'Original scientific protocol differs')
   artifacts={k:event[k]for k in('states','prefix','full','own_em')}
   for ref in artifacts.values():a.checked(ref)
   before={r['path']:(Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns)for r in artifacts.values()};source={g['qid']:g for g in inp['rows']};wanted={t:(g['qid'],y)for g in candidates if g['qid']in source for t,y in zip(g['old5_trajectory_ids'],g['old5_labels'])};a.require(wanted,'Original subset batch has no selected own question')
   prefix_rows=[row for row in a.records(artifacts['prefix']['path'])if row['trajectory_id']in wanted];em_rows=[row for row in a.records(artifacts['own_em']['path'])if row['trajectory_id']in wanted];pref={row['trajectory_id']:row for row in prefix_rows};ems={row['trajectory_id']:row for row in em_rows};a.require(len(prefix_rows)==len(pref)==len(em_rows)==len(ems)==len(wanted)and set(pref)==set(ems)==set(wanted),'Original selected prefix/ownEM G5 missing/repeated');seen=set();local=collections.Counter()
   for row in a.records(artifacts['full']['path']):
    t=row['trajectory_id']
    if t not in wanted:continue
    a.require(t not in seen and t not in old_seen,'Repeated original selected own ID');seen.add(t);old_seen.add(t);q,label=wanted[t];g=source[q];candidate=by[q];a.require(g['question_hash']==candidate['question_hash']and g['source_ordinal']==candidate['source_ordinal'],'Canonical original input differs')
    projected={key:row[key]for key in('trajectory_id','qid','strict_em','actor_version','prefix_sha256','prefix_scan_sha256','selection_sha256')};a.require(ems[t]==projected and ems[t]['strict_em']==label,'Original ownEM actual row linkage differs');result=a.audit_trajectory(pref[t],row,g,conf,label,score,ledger,'original_old5_for_fixed9344');local.update(result)
   a.require(seen==set(wanted)and before=={r['path']:(Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns)for r in artifacts.values()},'Original selected wholeG5 incomplete/changed');counts.update(local);proof=dict(status='FIXED9344_ORIGINAL_OLD5_BATCH_SUBSET_FULL_SHA_STRICT_EM_CAUSAL_SEEDS_PASSED',source_event=event,questions=len(wanted)//5,trajectories=len(wanted),strict_EM_recomputed=len(wanted),strict_EM_mismatches=0,**dict(local),epoch=time.time());path=OUT/'batches'/('batch-%03d.json'%index);atomic(path,proof);accepted.append(a.proof(path));progress=dict(stage='ORIGINAL_OLD5_FIXED9344_FULL_AUDIT_ADVANCING',completed_batches=index+1,expected_batches=len(old_events),questions=len(old_seen)//5,trajectories=len(old_seen),elapsed_seconds=time.time()-start,epoch=time.time());atomic(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
  expected_old={t for g in candidates for t in g['old5_trajectory_ids']};a.require(old_seen==expected_old and len(old_seen)==46720,'Fixed old5 coverage incomplete')
  groups=[dict(qid=g['qid'],question_hash=g['question_hash'],source_ordinal=g['source_ordinal'],trajectory_ids=g['old5_trajectory_ids']+g['trajectory_ids'],labels=g['old5_labels']+g['labels'],chosen_new5_execution_run_id=g['execution_run_id'],chosen_new5_seal=g['seal'],old5_seal=g['old5_seal'])for g in candidates]
  a.require(len({t for g in groups for t in g['trajectory_ids']})==93440,'Whole canonical G10 own IDs not unique');gp=OUT/'raw-g10/groups.json';atomic(gp,dict(train=groups));sp=OUT/'raw-g10/source-events.json';atomic(sp,events)
  contract=dict(qualified_execution_source_selection_before_L26=True,donor_and_HPC3_feature_manifest_union_before_source_selection_forbidden=True,same_protocol_tid_not_payload_identity=True,every_selected_tid_must_bind_chosen_execution_physical_prefix_and_retained_row_SHA=True,whole_G5_per_qid_no_mixing=True,qualification_tier_before_source_priority=True,source_priority={'accepted_existing':0,'HPC3_fully_qualified':10,'donor_fully_qualified':20},current124194_fit_expanded=False);cp=OUT/'raw-g10/source-resolution-contract.json';atomic(cp,contract)
  import torch
  a.require(not torch.cuda.is_initialized(),'CPU audit initialized CUDA');result=dict(status=STATUS,role='train',source_kind='HPC3_own_execution_no_HF_receiver_claim',source=a.proof(Path(__file__)),inherited_auditor=a.proof(n.BASE_AUDIT),fixed_intake=SNAP,new5_full_qualification=NEW_DONE,old5_batch_audits=accepted,questions=9344,trajectories=93440,new5_strict_EM_recomputed=46720,old5_strict_EM_recomputed=46720,own_strict_EM_recomputed=93440,strict_EM_mismatches=0,actual_same_question_old_new_seed_collisions=0,raw_G10_full_qualified=True,outputs={'groups':a.proof(gp),'source-events':a.proof(sp),'source-resolution-contract':a.proof(cp)},tokenizer_files=tok,tokenizer_strict_scorer=strict,actor_updates=0,probe_updates=0,feature_bank_created=False,current124194_fit_expanded=False,CUDA_initialized=False,model_tensors_loaded=False,all_question_texts_seen_in_original_warmup=True,elapsed_seconds=time.time()-start,epoch=time.time());atomic(OUT/'complete.json',result);return dict(complete=a.proof(OUT/'complete.json'),summary=result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit-fixed-old5',action='store_true',required=True);p.parse_args();print(json.dumps(run(),sort_keys=True))
