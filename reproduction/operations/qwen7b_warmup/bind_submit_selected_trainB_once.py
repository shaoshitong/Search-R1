"""Root-invoked fixed-trainB binder/submit-once; import/help/stage have no effects.

Default action binds small metadata only. Only explicit --action submit calls
Slurm after actual Zseed2 terminal CPU coordinator acceptance. No CP tensors,
final data, training changes or blind retry of an uncertain submission.
"""
import argparse,datetime,fcntl,hashlib,importlib,json,os,re,subprocess,sys,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASE=A/'selected-state-fixed-trainB2048-unified-20261023-v2'
SCORER_CODE=BASE/'code-ab8060368cc4'
SCORER_SOURCE='ab8060368cc49c1313493463642350dd5912dc0e'
EXPECTED_PUBLICATION=dict(path=str(BASE/'source-publication.json'),bytes=2642,sha256='61eaa510f24f6830304ddcd81aed8090e75de7b9e7683404d0ee276f44dc067c')
EXPECTED_ENTRY=dict(path=str(BASE/'scoring.sbatch'),bytes=1375,sha256='07be86fa0369e79187d0193b42df12c616bf1febb309468611a6a29443b1b498')
COORDINATOR=A/'fullZ-seed2-749935-terminal-CPU-coordinator-20261023-v1'
ACCEPTED={
 'Z_seed1':A/'packed-full169028-variance-CPU-acceptance-20261022-v1/z_mse_anchor/accepted.json',
 'Z_seed2':A/'full169028-Z-seed2-terminal-CPU-acceptance-20261023-v1/epoch10/accepted.json'}


def read(p):return json.loads(Path(p).read_bytes())

def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def checked(ref):
 assert proof(ref['path'])==ref,'Small metadata/source changed: '+ref['path'];return Path(ref['path'])

def durable_once(path,value):
 path=Path(path);raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
 if path.exists():assert path.read_bytes()==raw,'Previously bound descriptor/intent differs';return
 with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 descriptor=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(descriptor)
 finally:os.close(descriptor)

def command(argv):
 r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 if r.returncode:raise RuntimeError('Command failed: '+repr(argv)+': '+r.stdout[-700:]+r.stderr[-1000:])
 return r.stdout.strip()

def terminal_rows(text,job):return any(row.split('|')[:3]==[job,'COMPLETED','0:0']for row in text.splitlines())

def matching_queue_ids(text,name):
 rows=[]
 for line in text.splitlines():
  fields=line.strip().split('|')
  if len(fields)==3 and fields[1]==name and fields[2]=='mou176':
   if not fields[0].isdigit():raise ValueError('Unexpected array/step/heterogeneous job handle')
   rows.append(fields[0])
 if len(rows)>1:raise RuntimeError('Multiple existing same-name owned jobs; never submit another')
 return rows

def verify_job(details,job,entry,name,comment):
 if not all(value in details for value in('JobId='+job+' ','JobName='+name+' ','UserId=mou176(','Comment='+comment))or not re.search(r'(?:^| )Command='+re.escape(str(entry))+r'(?: |$)',details):
  raise RuntimeError('Same-name job is not the exact owned bound entry/comment')
 return job

def unresolved(intent,accounting):
 return dict(status='PRIOR_SUBMISSION_OUTCOME_UNKNOWN_NO_RESUBMISSION',intent=intent,accounting=accounting,Slurm_submitted=False,requires_root_reconciliation=True)


def actual_binding(scorer):
 """Small source/proof/score metadata only, after the actual terminal CPU gate."""
 complete=COORDINATOR/'complete.json'
 if not complete.exists():
  if(COORDINATOR/'failure.json').exists():raise RuntimeError('Zseed2 terminal CPU coordinator failed: '+str(COORDINATOR/'failure.json'))
  return dict(status='WAITING_ACTUAL_Z_SEED2_TERMINAL_CPU_ACCEPTANCE_NO_SUBMISSION',time=time.time(),required=str(complete),Slurm_submitted=False,final4096_read=False)
 done=read(complete);assert done['status']=='FULL_Z_SEED2_E10_TERMINAL_CPU_ACCEPTANCE_COMPLETE'and str(done['job_id'])=='749935'
 assert done['accepted']==proof(ACCEPTED['Z_seed2']);accepted2=read(checked(done['accepted']));assert done['checkpoint']==accepted2['checkpoint']
 accounting=command(['sacct','-X','-n','-P','-j','749935','--format=JobIDRaw,State,ExitCode'])
 if not terminal_rows(accounting,'749935'):raise RuntimeError('CPU coordinator exists but actual Zseed2 is not COMPLETED0:0')
 cfg=scorer.template(EXPECTED_PUBLICATION);cfg.update(status=scorer.STATUS,root_authorized_actual_execution=True)
 for key,path in dict(full_fixture=scorer.FULL/'fixture-complete.json',causal_fixture=scorer.CAUSAL,E1_dev_evaluation=scorer.E1_EVAL,E1_dev_predictions=scorer.E1_DEV).items():cfg[key]=proof(path)
 full=read(checked(cfg['full_fixture']));assert full['fixed_train_G5']['sha256']==scorer.TRAIN_SHA;views=[scorer.explicit(v)for v in read(checked(full['fixed_train_G5']))];assert len(views)==2048
 assert scorer.native.E1['bytes']==Path(scorer.native.E1['path']).stat().st_size
 dev=read(checked(full['validation_groups']));replay=[scorer.explicit(dev[q])for q in sorted(dev)[:32]]
 with scorer.gzip.open(checked(cfg['E1_dev_predictions']),'rt')as f:saved={r['qid']:r for r in(json.loads(line)for line in f)}
 assert sorted(saved)[:32]==[r['qid']for r in replay]and scorer.identity(replay)==scorer.identity([saved[v['qid']]for v in replay])
 reuse={}
 for model in cfg['models'][1:]:
  name=model['name'];accept_ref=proof(ACCEPTED[name]);accept=read(checked(accept_ref));assert accept['status']=='TERMINAL_FULL169028_PACKED_VARIANCE_CPU_SHA_MODEL_ADAM_RNG_PLAN_ACCEPTED'and accept['cursor']==13210
  parent=scorer.Z[name];checkpoint=accept['checkpoint'];assert checkpoint['path']==str(parent/'checkpoint-epoch-10.pt')and Path(checkpoint['path']).stat().st_size==checkpoint['bytes'];assert checkpoint==read(parent/'latest.json')
  model.update(checkpoint=checkpoint,CPU_acceptance=accept_ref,training_publication=accept['source_training_publication'],training_complete=proof(parent/'complete.json'),evaluation=proof(parent/'evaluation-epoch-10.json'),saved_predictions=proof(parent/'epoch-10-fixed_train2048_G5-predictions.jsonl.gz'))
  rows,source=scorer.reused(model,views);assert len(rows)==2048;reuse[name]=source
 scorer.descriptor(cfg)
 previous=read(BASE/'root-bound-trainB.json')if(BASE/'root-bound-trainB.json').exists()else None
 cfg.update(bound_at=previous['bound_at']if previous else time.time(),binding_evidence=dict(Zseed2_terminal_CPU_coordinator=proof(complete),Zseed2_terminal_accounting=accounting,E1_native_dispatch_source=proof(scorer.native.__file__),E1_adapter_replay_qid_trajectory_EM_SHA256=scorer.packing.digest(scorer.identity(replay)),fixed_train_G5=full['fixed_train_G5'],Z_existing_trainB_identity_checked=True,model_tensor_or_checkpoint_payload_opened=False,checkpoint_SHA_reused_from_independent_CPU_acceptance=True,E1_fresh_entire_CP_SHA_deferred_to_owned_GPU_scorer=True,final4096_read=False))
 scorer.descriptor(cfg);return cfg


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('bind','submit'),default='bind');args=parser.parse_args()
 assert os.uname().sysname=='Linux';assert proof(BASE/'source-publication.json')==EXPECTED_PUBLICATION and proof(BASE/'scoring.sbatch')==EXPECTED_ENTRY
 pub=read(checked(EXPECTED_PUBLICATION));assert pub['commit']==SCORER_SOURCE and pub['code_root']==str(SCORER_CODE)
 for name,ref in pub['files'].items():checked(dict(ref,path=str(SCORER_CODE/name)))
 lock=(BASE/'root-bind-submit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);receipt=BASE/'root-submission.json';config_path=BASE/'root-bound-trainB.json';intent=BASE/'root-submit-intent.json'
 if receipt.exists():
  r=read(receipt);assert r['entry']==EXPECTED_ENTRY and r['source_publication']==EXPECTED_PUBLICATION and r['config']==proof(config_path);print(json.dumps(r));return
 sys.path.insert(0,str(SCORER_CODE));scorer=importlib.import_module('reproduction.operations.qwen7b_warmup.score_selected_fixed_trainB');assert Path(scorer.__file__).resolve()==SCORER_CODE/'reproduction/operations/qwen7b_warmup/score_selected_fixed_trainB.py'
 cfg=actual_binding(scorer)
 if cfg['status']!=scorer.STATUS:print(json.dumps(cfg));return
 durable_once(config_path,cfg);config=proof(config_path)
 if args.action=='bind':print(json.dumps(dict(status='ACTUAL_FIXED_TRAIN_B_DESCRIPTOR_BOUND_NOT_SUBMITTED',config=config,entry=EXPECTED_ENTRY,time=time.time(),Slurm_submitted=False,final4096_read=False)));return
 entry=Path(EXPECTED_ENTRY['path']);name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1);comment='q7b-trainB-'+config['sha256'][:20]
 candidates=matching_queue_ids(command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%j|%u']),name)
 if candidates:
  job=verify_job(command(['scontrol','show','job',candidates[0],'-o']),candidates[0],entry,name,comment);submitted_now=False
 else:
  if intent.exists():
   prior=read(intent);assert prior['config']==config and prior['entry']==EXPECTED_ENTRY and prior['name']==name
   since=datetime.datetime.fromtimestamp(prior['time']-3600).strftime('%Y-%m-%dT%H:%M:%S');accounting=command(['sacct','-X','-n','-P','-u','mou176','-S',since,'--format=JobIDRaw,JobName%80,State,ExitCode']);matched='\n'.join(row for row in accounting.splitlines()if len(row.split('|'))>1 and row.split('|')[1]==name);print(json.dumps(unresolved(prior,matched)));return
  assert not(BASE/'controller.json').exists()and not(BASE/'complete.json').exists(),'Do not duplicate prior unreceipted run'
  durable_once(intent,dict(time=time.time(),name=name,entry=EXPECTED_ENTRY,source_publication=EXPECTED_PUBLICATION,config=config,comment=comment,exclude='ACD1-13'))
  job=command(['sbatch','--parsable','--exclude=ACD1-13','--comment='+comment,str(entry),str(config_path),config['sha256']]).split(';')[0];assert job.isdigit();submitted_now=True
 result=dict(status='FIXED_TRAIN_B_ROOT_OWNED_JOB_SUBMISSION_RECORDED',time=time.time(),job_id=job,name=name,comment=comment,entry=EXPECTED_ENTRY,source_publication=EXPECTED_PUBLICATION,source_commit=SCORER_SOURCE,config=config,submitted_now=submitted_now,no_dependencies=True,exclude='ACD1-13',partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time_limit='01:00:00',questions=2048,role='fixed_trainB_only_plus_E1_fixed32_adapter_replay',models=['E1_EMA','Z_seed1_E10_raw','Z_seed2_E10_raw'],Z_saved_trainB_predictions_reused=True,new_rollouts=0,training_changed=False,final4096_read=False)
 durable_once(receipt,result);print(json.dumps(result))
if __name__=='__main__':main()
