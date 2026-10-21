"""Full CPU qualification of the fixed73 own new-G5 batches; old5 unchanged."""
import argparse,collections,fcntl,hashlib,importlib.util,inspect,json,os,sys,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';TAIL=A/'hpc3-tail-newseedG5-20261021-v1';OUT=A/'hpc3-tail73-full-new5-audit-v1'
SNAP={'path':str(A/'hpc3-tail-sealed-intake-v1/snapshot-1792589071-f81516f41b5f/complete.json'),'bytes':5298,'sha256':'94c28dfce72db5b21654a5676494555e614bed9b1c23be5d03a8041d6558087d'}
BASE_AUDIT=A/'hpc3-tail-firstbatch-audit-v1/code-8599d1fdc7ff/audit.py';BASE_SHA='3bc69a9e5258a3cb46b719f82654bc9ddeefd7a2c9a0d188daf0db49735fae2e'
require=None

def inherited():
 global require
 raw=BASE_AUDIT.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=BASE_SHA:raise ValueError('Published complete first-batch auditor changed')
 s=importlib.util.spec_from_file_location('qualified_first_batch',BASE_AUDIT);a=importlib.util.module_from_spec(s);s.loader.exec_module(a);require=a.require;return a

def atomic(p,v):
 raw=json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n';tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(raw);os.replace(str(tmp),str(p))
def source_meta(a):return dict(path=str(Path(__file__)),bytes=Path(__file__).stat().st_size,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inherited_auditor=a.proof(BASE_AUDIT))
def inputs(a):
 snap=a.read(a.checked(SNAP));a.require(snap['frozen_sealed_batches']==73 and snap['sealed_questions']==9344 and snap['sealed_trajectories']==46720,'Fixed actual73 snapshot differs')
 frozen=a.read(a.checked(snap['outputs']['frozen-seals']));main=a.read(a.checked(snap['producer_config']));a.require(snap['producer_config']['sha256']==a.CONFIG_SHA and main['actor_optimizer_updates']==0,'Fixed own producer config differs');return snap,frozen,main

def scorer(a,main):
 model=Path(main['model_dir']);snapshot=a.read(a.checked(main['model_snapshot']));a.require(snapshot['revision']==a.REVISION,'Official tokenizer revision differs');refs={}
 for name,item in snapshot['files'].items():
  if name.endswith(('.json','.txt','.model')):refs[name]=a.proof(a.checked(dict(path=str(model/name),bytes=item['bytes'],sha256=item['sha256'])))
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
 from reproduction.prefix20_online.rollout import strict_label
 return lambda row:strict_label(tokenizer,row['response_token_ids'],row['response_info_mask'],row['reward_model']['ground_truth']),refs,a.proof(inspect.getfile(strict_label))

def audit_batch(a,item,worker,main,score,tokenizer_refs,strict_scorer,ledger):
 path=a.checked(item['seal']);seal=a.read(path);source_input=a.checked(item['input']);batch=a.read(source_input);confpath=path.parent/'config.json';conf=a.read(confpath);outer=path.parent/'outers/outer-000001';job=('745518','745519')[worker]
 a.require(seal['execution_job']==job and seal['input_sha256']==a.proof(source_input)['sha256']and seal['questions']==len(batch['rows'])and seal['trajectories']==5*len(batch['rows']),'Actual wholeG5 input/job binding differs')
 a.require(seal['run_id']==seal['sampling_protocol_source_run_id']==main['sampling_protocol_source_run_id']and seal['execution_run_id']==main['execution_run_id']and seal['execution_host_cluster']=='HPC3'and seal['execution_source_commit']==main['execution_source_commit']and seal['potentialOverlap']and seal['donor_original_receipt_claim']is False and seal['actor_optimizer_updates']==0,'Own execution/source provenance differs')
 a.require(conf['seed']==20261021 and conf['sampling_attempt_id']=='independent-seed20261021-g5-20261020'and conf['actor_optimizer_updates']==0 and conf['actor_hidden_native']is False and seal['model_revision']==a.REVISION,'Seed/revision/frozen actor contract differs')
 prefix,full=seal['prefix'],seal['full'];scan=outer/'prefix/scan.json';selection=outer/'probe-score/selection.json'
 a.require(prefix['config_sha256']==full['config_sha256']==a.proof(confpath)['sha256'],'Rank config binding differs')
 a.require(full['prefix_scan_sha256']==a.proof(scan)['sha256']and full['selection_sha256']==a.proof(selection)['sha256'],'Current prefix/full receipt chain differs')
 a.require(a.read(selection)['kind']=='ALL_G5_NO_SELECTION'and a.read(scan)['ranks']==[prefix],'No-selection/scan own rank contract differs')
 for key in ['sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype']:a.require(prefix[key]==full[key],'Prefix/full protocol differs')
 a.require(prefix['actual_compute_dtype']=='torch.bfloat16'and prefix['sampling_attempt_id']==full['sampling_attempt_id']==conf['sampling_attempt_id'],'Actual BF16/attempt differs')
 artifacts={k:dict(v)for k,v in [('states',prefix['states']),('prefix',prefix['prefix']),('full',full['full']),('own_em',full['own_em'])]}
 for value in artifacts.values():a.checked(value)
 before={x['path']:(Path(x['path']).stat().st_size,Path(x['path']).stat().st_mtime_ns)for x in artifacts.values()}
 byqid={g['qid']:g for g in batch['rows']};expected={('%s-outer000001-c%04d-g%d'%(conf['run_id'],g['candidate_ordinal'],j)):(g,j)for g in batch['rows']for j in range(5)}
 prefix_rows=list(a.records(artifacts['prefix']['path']));em_rows=list(a.records(artifacts['own_em']['path']));pref={g['trajectory_id']:g for g in prefix_rows};labels={g['trajectory_id']:g['strict_em']for g in em_rows};a.require(len(prefix_rows)==len(em_rows)==len(expected)==len(pref)==len(labels)and all(type(v)is int and v in(0,1)for v in labels.values())and set(pref)==set(labels)==set(expected),'Actual ownG5 prefix/EM count or ID set differs')
 seen=set();counts=collections.Counter();groups={}
 for row in a.records(artifacts['full']['path']):
  t=row['trajectory_id'];a.require(t in expected and t not in seen,'Repeated/foreign actual full ID');seen.add(t);source,index=expected[t]
  counts.update(a.audit_trajectory(pref[t],row,source,conf,labels[t],score,ledger,'HPC3_OWN_FIXED73_NEW5'));groups.setdefault(source['qid'],[]).append(labels[t])
 a.require(seen==set(expected)and len(groups)==len(byqid)and all(len(x)==5 for x in groups.values()),'Whole authentic ownG5 incomplete')
 a.require(before=={x['path']:(Path(x['path']).stat().st_size,Path(x['path']).stat().st_mtime_ns)for x in artifacts.values()},'Sealed payload changed during independent audit')
 return dict(status='FIXED_OWN_NEW5_BATCH_FULL_SHA_STRICT_EM_CAUSAL_NATIVE_SEEDS_PASSED',worker=worker,job_id=job,seal=a.proof(path),input=a.proof(source_input),config=a.proof(confpath),artifacts=artifacts,questions=len(groups),trajectories=len(seen),K5_histogram={str(k):sum(sum(x)==k for x in groups.values())for k in range(6)},strict_EM_recomputed=len(seen),strict_EM_mismatches=0,**dict(counts),execution_run_id=main['execution_run_id'],sampling_protocol_source_run_id=main['sampling_protocol_source_run_id'],actor_updates=0,potentialOverlap=True,tokenizer_files=tokenizer_refs,tokenizer_strict_scorer=strict_scorer,auditor_source=source_meta(a),inherited_function_binding=a.INHERITED_SOURCE_BINDING,epoch=time.time())

def run(worker):
 a=inherited();a.require(sys.platform!='darwin'and not os.environ.get('CUDA_VISIBLE_DEVICES'),'Remote CPU-only allocation required');OUT.mkdir(exist_ok=True);d=OUT/('worker-%d'%worker);d.mkdir(exist_ok=True)
 with(d/'audit.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);snap,frozen,main=inputs(a);job=('745518','745519')[worker];items=[x for x in frozen['batches']if a.read(a.checked(x['seal']))['execution_job']==job];a.require(items,'Own frozen batches absent');started=time.time();atomic(d/'controller.json',dict(pid=os.getpid(),birth_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),worker=worker,expected_batches=len(items),auditor_source=source_meta(a),CPU_only=True,epoch=time.time()));score,tok,strict=scorer(a,main);accepted=[];ledger={}
  for item in items:
   batch=Path(item['seal']['path']).parent.name;dest=d/(batch+'.json')
   if dest.exists():
    value=a.read(dest);a.require(value['seal']==item['seal']and value['status']=='FIXED_OWN_NEW5_BATCH_FULL_SHA_STRICT_EM_CAUSAL_NATIVE_SEEDS_PASSED','Existing audit belongs to other sealed source')
   else:value=audit_batch(a,item,worker,main,score,tok,strict,ledger);atomic(dest,value)
   accepted.append(a.proof(dest));progress=dict(stage='ACTUAL_FROZEN_NEW5_FULL_AUDIT_ADVANCING',worker=worker,job_id=job,completed_batches=len(accepted),expected_batches=len(items),questions=sum(a.read(x['path'])['questions']for x in accepted),trajectories=sum(a.read(x['path'])['trajectories']for x in accepted),latest_batch=batch,elapsed_seconds=time.time()-started,epoch=time.time());atomic(d/'progress.json',progress);print(json.dumps(progress),flush=True)
  import torch
  a.require(not torch.cuda.is_initialized(),'Audit must not initialize CUDA');values=[a.read(x['path'])for x in accepted];result=dict(status='OWN_FROZEN_NEW5_WORKER_FULL_PAYLOAD_SHA_STRICT_EM_CAUSAL_SEEDS_QUALIFIED',worker=worker,job_id=job,batches=len(accepted),questions=sum(x['questions']for x in values),trajectories=sum(x['trajectories']for x in values),strict_EM_recomputed=sum(x['strict_EM_recomputed']for x in values),strict_EM_mismatches=0,batch_audits=accepted,frozen_intake=SNAP,tokenizer_files=tok,tokenizer_strict_scorer=strict,auditor_source=source_meta(a),actor_updates=0,old5_qualification='metadata_only_not_revalidated_here',G10_raw_accepted=False,current124194_fit_expanded=False,CUDA_initialized=False,model_tensors_loaded=False,elapsed_seconds=time.time()-started,epoch=time.time());atomic(d/'complete.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=(0,1),required=True);args=p.parse_args();print(json.dumps(run(args.worker),sort_keys=True))
