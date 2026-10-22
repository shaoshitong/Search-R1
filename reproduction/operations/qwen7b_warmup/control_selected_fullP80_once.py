"""Root-only small-metadata binding/submission; staging/help never bind or submit.

No checkpoint hashing/loading and no final journal/label access. An uncertain
submission remains unresolved, never an invitation to submit again.
"""
import argparse,datetime,fcntl,hashlib,json,os,re,subprocess,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASE=A/'fullP80-selected-root-control-20261023-v1'
SELECTOR=A/'full169028-P80-MSE-actual-family-selection-20261023-v1/actual-family-selection-and-paired.json'
FIXTURE=A/'full169028-policy80-physical-ownG10-20261023-v2/fixture/fixture-complete.json'
ACCEPT=A/'full169028-L15P80-MSE-terminal-CPU-acceptance-20261023-v1'
REPLAY=A/'selected-state-fixed-trainB2048-unified-20261023-v2'
RULE_SHA='8f4fead8f40a7e14f7ddaffb2950bea3f3e405d435c544388519c507e7ce3a27'
TRAIN_COMMIT='7a8ff6ae8a33324846d85f77ffc28e9ab699c028'
SEEDS={'seed1':(9137,2026102227,'750189'),'seed2':(19137,2026102302,'750190')}
PARENTS={s:A/('full169028-L15P80-MSE-ownG5-physical-seed%d-plan%d-20261023-v3'%v[:2])for s,v in SEEDS.items()}
TARGETS={
 'audit':dict(base=A/'full169028-L15P80-MSE-viewed-permutation-audit-20261023-v1',code='code-97f57e4fbed1',module='packed_full169028_trajectory_permutation_audit',validator='descriptor',entry='audit.sbatch',template='future-unbound-audit-template.json',publication_SHA='c773c3cd574002a606c6617efa232e70faea95574091515edea24bb620eb4936',entry_SHA='0f7bbb120f4500cf8f6b6c23cfa99e1dc0c86e61bf02174e7448e80bd8b57142'),
 'final':dict(base=A/'frozen-final4096-full169028-L15P80-MSE-20261023-v2',code='code-a4d4ec0e0e65',module='score_frozen_full169028_MSE_final4096',validator='freeze_contract',entry='scoring.sbatch',template='candidate-freeze-TEMPLATE.json',publication_SHA='3651b977567457f5f1d60f3bb126f5d01b701b3900836771a8506dfb3561900e',entry_SHA='49b62c2ed86adc7eb5fb591c1f97902c614c342dc562664023dd267875dbeeeb')}
ACCEPT_STATUS='TERMINAL_FULL169028_P80_MSE_CPU_SHA_MODEL_ADAM_RNG_PLAN_PHYSICAL_GATE_ACCEPTED'
class Waiting(Exception):pass

def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p)
 if p.suffix in('.pt','.pth','.bin','.safetensors')or p.stat().st_size>32*1024**2:raise ValueError('Only bounded metadata/source may be hashed by root controller')
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 if proof(ref['path'])!=ref:raise ValueError('Metadata/source proof changed: '+ref['path'])
 return Path(ref['path'])
def durable_once(p,value):
 p=Path(p);raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
 if p.exists():
  if p.read_bytes()!=raw:raise ValueError('Previously bound immutable metadata differs')
  return
 with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(str(p.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def command(argv):
 r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 if r.returncode:raise RuntimeError('Command failed: '+repr(argv)+': '+r.stdout[-700:]+r.stderr[-1000:])
 return r.stdout.strip()
def terminal_rows(text,job):return any(row.split('|')[:3]==[job,'COMPLETED','0:0']for row in text.splitlines())
def selection(value):
 if value.get('status')!='ACTUAL_FULL169028_P80_FAMILY_SELECTED_AND_VIEWED_PAIRED_ANALYSIS_COMPLETE'or value.get('selection_rule')!='maxseed1count_then_earliest_epoch_then_EMA'or value.get('seed2_used_for_selection')is not False or value.get('same_epoch_state_on_seed2')is not True or value.get('final4096_read')is not False:raise ValueError('Actual seed1-only complete family selector required')
 choices=value['all20_seed1_choices']
 if len(choices)!=20 or {(r['epoch'],r['state'])for r in choices}!={(e,s)for e in range(1,11)for s in('raw','EMA')}:raise ValueError('All20 actual seed1 choices required')
 if any(type(r['Top512_Mixed'])is not int or not 0<=r['Top512_Mixed']<=512 for r in choices):raise ValueError('Actual Top512 counts required')
 best=sorted(choices,key=lambda r:(-r['Top512_Mixed'],r['epoch'],r['state']!='EMA'))[0]
 if value['actual_selected']!=best or(value['selected_epoch'],value['selected_state'])!=(best['epoch'],best['state']):raise ValueError('Selection violates seed1 Top512/earliest/EMA rule')
 return best['epoch'],best['state']
def complete_metadata(v,seed):
 init,plan,job=SEEDS[seed]
 expected=dict(status='FULL169028_P80_OWNG5_MSE_TRAINING_COMPLETE',epochs=10,cursor=13210,batch_attempts=13210,actual_Adam_updates=13210,EMA_updates=13210,training_questions=169028,margin_satisfied_batches=0,final4096_scored=False)
 b=v.get('binding',{})
 if any(v.get(k)!=x for k,x in expected.items())or any(b.get(k)!=x for k,x in dict(init_seed=init,train_plan_seed=plan,diagnostic_seed=2026102228,layer=15,prefix=80,objective='mse',fixed_m=5,training_questions=169028).items()):raise ValueError('Actual full10E13210 seed/geometry completion differs')
def accepted_metadata(v,seed,epoch,complete):
 init,plan,job=SEEDS[seed];clock=1321*epoch
 expected=dict(status=ACCEPT_STATUS,job_id=job,epoch=epoch,batch_attempts=clock,actual_Adam_updates=clock,EMA_updates=clock,epoch_complete=True,next_epoch=epoch+1,within_next_epoch_question_position=0,execution_phase='epoch_complete_pre_evaluation',CPU_only=True,final4096_read=False)
 if any(v.get(k)!=x for k,x in expected.items())or v.get('binding')!=complete['binding']:raise ValueError('Selected epoch independent CPU acceptance differs')
 cp=v['checkpoint'];parent=PARENTS[seed]
 if cp['path']!=str(parent/('checkpoint-epoch-%02d.pt'%epoch))or not re.fullmatch('[0-9a-f]{64}',cp['sha256']):raise ValueError('Selected CP path/SHA differs')
 if not re.fullmatch('[0-9a-f]{64}',v['epoch_plan_SHAs'][str(epoch)]):raise ValueError('Selected epoch plan SHA absent')
 if set(v.get('models',{}))!={'raw','EMA'}or len(v.get('RNG_by_rank',[]))!=2 or v.get('Adam_CPU_restore')is not True:raise ValueError('Actual raw/EMA/Adam/two RNG proof required')
 return cp

def prerequisite_gate():
 if not SELECTOR.is_file():raise Waiting('Actual completed family selection is not present')
 selected=read(SELECTOR);epoch,state=selection(selected);sources=selected['source_proofs'];accepts={};accounting={}
 for seed,parent in PARENTS.items():
  job=SEEDS[seed][2];accounting[seed]=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'])
  if not terminal_rows(accounting[seed],job):raise Waiting('Actual '+job+' not COMPLETED0:0')
  done=read(parent/'complete.json');complete_metadata(done,seed)
  if str(read(parent/'root-submission.json')['job_id'])!=job:raise ValueError('Actual training job changed')
  if sources[seed]['complete']!=proof(parent/'complete.json')or sources[seed]['source_publication']!=proof(parent/'source-publication.json'):raise ValueError('Selector complete/source proof changed')
  ap=ACCEPT/parent.name/('epoch-%02d'%epoch)/'accepted.json'
  if not ap.is_file():raise Waiting('Selected epoch CPU acceptance missing: '+str(ap))
  accept=read(ap);cp=accepted_metadata(accept,seed,epoch,done)
  if Path(cp['path']).stat().st_size!=cp['bytes']:raise ValueError('Accepted checkpoint size changed; no controller CP hash/load allowed')
  if accept['source_complete']!=proof(parent/'complete.json')or accept['source_publication']!=proof(parent/'source-publication.json')or accept['source_evaluation']!=proof(parent/('evaluation-epoch-%02d.json'%epoch)):raise ValueError('Selected acceptance source/complete/evaluation differs')
  pub=read(parent/'source-publication.json')
  if pub['commit']!=TRAIN_COMMIT:raise ValueError('Wrong actual full training publication')
  ev=read(parent/('evaluation-epoch-%02d.json'%epoch))
  if ev['epoch']!=epoch or ev['cursor']!=1321*epoch or ev['binding']!=accept['binding']:raise ValueError('Selected evaluation epoch/source differs')
  if seed=='seed1':
   for c in selected['all20_seed1_choices']:
    if c['evaluation']!=proof(parent/('evaluation-epoch-%02d.json'%c['epoch']))or c['Top512_Mixed']!=read(c['evaluation']['path'])['metrics']['fixed_validation2048_G5'][c['state']]['Top512']['Mixed']:raise ValueError('Seed1 choice is not the actual saved epoch/state result')
  accepts[seed]=(proof(ap),accept)
 return dict(selection=proof(SELECTOR),value=selected,epoch=epoch,state=state,accepts=accepts,accounting=accounting)
def target(kind):
 t=TARGETS[kind];b=t['base'];pubref=proof(b/'source-publication.json');entryref=proof(b/t['entry']);pub=read(pubref['path'])
 if pubref['sha256']!=t['publication_SHA']or entryref['sha256']!=t['entry_SHA']or pub['code_root']!=str(b/t['code']):raise ValueError('Immutable existing GPU source/entry changed')
 for name,ref in pub['files'].items():checked(dict(ref,path=str(Path(pub['code_root'])/name)))
 return t,pubref,entryref

def validate_descriptor(kind,cfg):
 t=TARGETS[kind];code=t['base']/t['code'];script='import json,sys;from reproduction.operations.qwen7b_warmup import '+t['module']+' as m;m.'+t['validator']+'(json.load(sys.stdin));print("DESCRIPTOR_ACCEPTED")'
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],input=json.dumps(cfg),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env,timeout=30)
 if r.returncode or'DESCRIPTOR_ACCEPTED'not in r.stdout:raise ValueError('Existing immutable descriptor validator rejected binding: '+r.stderr[-1500:])
def audit_binding(gate):
 t,pub,entry=target('audit');cfg=read(t['base']/t['template']);path=BASE/'audit-bound.json';previous=read(path)if path.exists()else None
 cfg.update(status=cfg.pop('required_status'),root_authorized_actual_execution=True,selected_at=previous['selected_at']if previous else time.time(),epoch=gate['epoch'],state=gate['state'],fixture=proof(FIXTURE),audit_source_publication=pub)
 for r in cfg['replications']:
  seed=r['seed_id'];parent=PARENTS[seed];ap,accept=gate['accepts'][seed];epoch=gate['epoch'];r.update(checkpoint=accept['checkpoint'],CPU_acceptance=ap,training_complete=proof(parent/'complete.json'),training_publication=proof(parent/'source-publication.json'),source_evaluation=proof(parent/('evaluation-epoch-%02d.json'%epoch)),source_saved_predictions=proof(parent/('epoch-%02d-fixed_validation2048_G5-predictions.jsonl.gz'%epoch)),ready=proof(parent/'ready.json'))
 cfg.update(binding_evidence=dict(family_selection=gate['selection'],terminal_accounting=gate['accounting'],checkpoint_SHA_inherited_from_independent_CPU_acceptance=True,checkpoint_tensor_opened=False,final4096_read=False));validate_descriptor('audit',cfg);return cfg

def audit_and_replay_gate(gate):
 audit=TARGETS['audit']['base'];p=audit/'complete.json'
 if not p.is_file():raise Waiting('Selected P80 order audit not complete')
 done=read(p);bound=BASE/'audit-bound.json'
 if not bound.is_file():raise Waiting('Owned selected audit binding absent')
 if done.get('status')!='FULL169028_L15P80_MSE_VIEWED_G5_PERMUTATION_AUDIT_COMPLETE'or done.get('epoch')!=gate['epoch']or done.get('state')!=gate['state']or done.get('config')!=proof(bound)or done.get('order_selected_by_audit')is not False or done.get('final4096_read')is not False:raise ValueError('Actual selected audit completion differs')
 for seed in SEEDS:
  r=done['replications'][seed];a=r['original_vs_saved_evaluation']
  if r['checkpoint']!=gate['accepts'][seed][1]['checkpoint']or a.get('passed')is not True or a.get('questions')!=2048 or a.get('question_trajectory_EM_order_exact')is not True or a.get('absolute_score_tolerance')!=.001 or not 0<=a.get('absolute_score_delta_max',float('inf'))<=.001:raise ValueError('Actual selected original-order replay must pass<=.001')
 replay=REPLAY/'complete.json'
 if not replay.is_file():raise Waiting('Actual E1 fixed32 adapter replay absent')
 rr=read(replay);rp=rr['E1_adapter_fixed32_viewed_replay'];job=str(read(REPLAY/'root-submission.json')['job_id'])
 if job!='750334'or not terminal_rows(command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode']),job):raise Waiting('Actual adapter replay750334 not COMPLETED0:0')
 if rr.get('status')!='FIXED_TRAIN_B2048_UNIFIED_SELECTED_STATE_DIAGNOSTICS_COMPLETE'or rp.get('passed')is not True or rp.get('questions')!=32 or rp.get('same_E1_EMA_CP_and_own_G5')is not True or rp.get('absolute_probability_tolerance')!=.001 or not 0<=rp.get('absolute_probability_delta_max',float('inf'))<=.001 or rr.get('final4096_read')is not False:raise ValueError('Actual same-E1 fixed32 numeric replay required')
 return dict(selected_order_audit=proof(p),E1_fixed32_numeric_replay=proof(replay))
def final_binding(gate):
 pre=audit_and_replay_gate(gate);t,pub,entry=target('final');cfg=read(t['base']/t['template']);path=BASE/'candidate-freeze.json';previous=read(path)if path.exists()else None
 rule=proof(BASE/'PREDECLARED-FINAL-CONFIRMATION.json')
 if rule['sha256']!=RULE_SHA:raise ValueError('Prospective confirmation rule changed')
 cfg.update(status=cfg.pop('required_status'),freeze_epoch=previous['freeze_epoch']if previous else time.time(),scorer_publication=pub)
 cfg['configuration'].update(epoch=gate['epoch'],primary_state=gate['state'],secondary_states=[],primary_metric='Top512_true_Mixed_fraction',threshold=.08,threshold_fitted=False)
 fixture=read(FIXTURE);view=fixture['validation_groups'];checked(view)
 if view['sha256']!='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292':raise ValueError('Fixed viewed ownG5 identity differs')
 for r in cfg['replications']:
  seed=r['seed_id'];ap,accept=gate['accepts'][seed];parent=PARENTS[seed];r.update(init_seed=SEEDS[seed][0],train_plan_seed=SEEDS[seed][1],train_plan_sha256=accept['epoch_plan_SHAs'][str(gate['epoch'])],checkpoint_binding=accept['binding'],training_runtime_relative='reproduction/operations/qwen7b_warmup/online_packed_full169028_MSE_P80.py',checkpoint=accept['checkpoint'],source_publication=accept['source_publication'],training_complete=accept['source_complete'],CPU_acceptance=ap,viewed_validation_evaluation=accept['source_evaluation'],viewed_G5_identity_proof=view)
 cfg.update(binding_evidence=dict(family_selection=gate['selection'],terminal_accounting=gate['accounting'],prospective_confirmation_rule=rule,checkpoint_SHA_inherited_from_independent_CPU_acceptance=True,checkpoint_tensor_opened=False,final4096_read=False,**pre));validate_descriptor('final',cfg);return cfg

def matching_queue_ids(text,name):
 out=[]
 for line in text.splitlines():
  f=line.strip().split('|')
  if len(f)==3 and f[1]==name and f[2]=='mou176':
   if not f[0].isdigit():raise ValueError('Unexpected array/step job handle')
   out.append(f[0])
 if len(out)>1:raise RuntimeError('Multiple same-name owned jobs; never submit another')
 return out
def verify_job(details,job,entry,name,comment):
 if not all(v in details for v in('JobId='+job+' ','JobName='+name+' ','UserId=mou176(','Comment='+comment))or not re.search(r'(?:^| )Command='+re.escape(str(entry))+r'(?: |$)',details):raise RuntimeError('Same-name owned job is not exact bound entry/comment')
 return job
def unresolved(intent,accounting):return dict(status='PRIOR_SUBMISSION_OUTCOME_UNKNOWN_NO_RESUBMISSION',intent=intent,accounting=accounting,Slurm_submitted=False,requires_root_reconciliation=True)
def submit_once(kind,config_path):
 t,pub,entryref=target(kind);entry=Path(entryref['path']);config=proof(config_path);receipt=BASE/(kind+'-submission.json');intent=BASE/(kind+'-submit-intent.json');name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1);comment='q7b-fullP80-'+kind+'-'+config['sha256'][:20]
 if receipt.exists():
  v=read(receipt)
  if(v['entry'],v['source_publication'],v['config'])!=(entryref,pub,config):raise ValueError('Existing submission differs')
  return v
 candidates=matching_queue_ids(command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%j|%u']),name)
 if candidates:job=verify_job(command(['scontrol','show','job',candidates[0],'-o']),candidates[0],entry,name,comment);submitted=False
 else:
  if intent.exists():
   prior=read(intent)
   if(prior['config'],prior['entry'],prior['name'])!=(config,entryref,name):raise ValueError('Prior submission intent differs')
   since=datetime.datetime.fromtimestamp(prior['time']-3600).strftime('%Y-%m-%dT%H:%M:%S');accounting=command(['sacct','-X','-n','-P','-u','mou176','-S',since,'--format=JobIDRaw,JobName%80,State,ExitCode']);matched='\n'.join(r for r in accounting.splitlines()if len(r.split('|'))>1 and r.split('|')[1]==name);return unresolved(prior,matched)
  if(t['base']/'controller.json').exists()or(t['base']/'complete.json').exists():raise RuntimeError('Existing unreceipted actual run; do not duplicate')
  durable_once(intent,dict(time=time.time(),config=config,entry=entryref,source_publication=pub,name=name,comment=comment))
  job=command(['sbatch','--parsable','--comment='+comment,str(entry),str(config_path),config['sha256']]).split(';')[0]
  if not job.isdigit():raise RuntimeError('Unknown sbatch response; persistent intent prevents resubmission')
  submitted=True
 result=dict(status='FULLP80_SELECTED_ROOT_OWNED_SUBMISSION_RECORDED',kind=kind,job_id=job,name=name,comment=comment,entry=entryref,source_publication=pub,config=config,submitted_now=submitted,no_dependencies=True,partition='acd_u',qos='normal',GPUs=2,training_changed=False,final4096_read=False,time=time.time());durable_once(receipt,result);return result

def execute(action):
 kind='audit'if action.endswith('audit')else'final';path=BASE/('audit-bound.json'if kind=='audit'else'candidate-freeze.json')
 try:
  gate=prerequisite_gate();cfg=audit_binding(gate)if kind=='audit'else final_binding(gate)
 except(FileNotFoundError,Waiting)as e:return dict(status='WAITING_ACTUAL_PREREQUISITES_NO_BIND_FREEZE_OR_SUBMISSION',reason=str(e),action=action,Slurm_submitted=False,candidate_frozen=False,checkpoint_or_final_opened=False,time=time.time())
 durable_once(path,cfg)
 if action.startswith('submit-'):return submit_once(kind,path)
 return dict(status='SELECTED_AUDIT_BOUND_NOT_SUBMITTED'if kind=='audit'else'ACTUAL_FULLP80_CANDIDATE_FROZEN_NOT_SUBMITTED',config=proof(path),Slurm_submitted=False,candidate_frozen=kind=='final',checkpoint_or_final_opened=False,time=time.time())
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('bind-audit','submit-audit','freeze-final','submit-final'),default='bind-audit');a=p.parse_args()
 if os.uname().sysname!='Linux':raise RuntimeError('Existing remote root control only')
 pub=read(BASE/'source-publication.json');here=Path(__file__).resolve();relative='reproduction/operations/qwen7b_warmup/'+here.name
 if dict(path=str(here),**pub['files'][relative])!=proof(here)or pub['code_root']!=str(here.parents[3]):raise ValueError('Controller is not its published immutable source')
 with(BASE/'control.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);print(json.dumps(execute(a.action),sort_keys=True))
if __name__=='__main__':main()
