"""Full124194 E4 native block27/readout joint continuation, main-only.

The separately completed seven-update qualification is not this training source.
All G10 inputs must be sealed; four epochs use a stateful class-balanced sampler.
No Count postblock cache, former5120 unseen claim, actor update, or new loss/LR.
"""
import argparse,copy,fcntl,json,math,os,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
import torch
from . import probe_ablation_train as u,hard_enrichment_train as inherited
from .balanced_hard_g10_sampler import BalancedHardG10Sampler
from .joint_balanced_native_update import balanced_native_update
SOURCE_SHA=inherited.SOURCE_SHA;SOURCE_STEPS=14418;LOSS=dict(trajectory_weight=1.,binary_mixed_weight=1.);STOP=False
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';MAIN=A/'joint-balanced-native-E4-124194-v1'


def own_groups(c):
 from .g10_grouping import _validate
 value=u.read(u.checked(c['groups']));groups=value['train'];_validate(groups,True)
 if len(groups)!=c['fit_questions']:raise ValueError('Exact realG10 question cohort required')
 forbidden=u.read(u.checked(c['exclusions']));blocked={r['qid']for r in forbidden['groups']};hashes={r['question_hash']for r in forbidden['groups']}
 if forbidden['role']!='joint_training_fixed_evaluation_identity_exclusions'or c['expected_exclusion_questions']!=len(blocked)or any(g['qid']in blocked or g['question_hash']in hashes for g in groups):raise ValueError('True diagnostic/selection/final identity exclusion required')
 return groups,blocked,hashes


def source_parts(model):
 return dict(block=u.tensor_hash({k:v for k,v in model.state_dict().items()if k.startswith('block.')}),readout=u.tensor_hash({k:v for k,v in model.state_dict().items()if k.startswith('output.')}))


def snapshot(model,ema,opt,device,sampler,cursor,binding,ranks,seen):
 state=u.checkpoint_blob(model,ema,opt,device,cursor,binding,ranks);state.update(role='BALANCED_HARD_REAL_G10_NATIVE_JOINT_E4_CONTINUATION',source_steps=SOURCE_STEPS,sampler_state=sampler.state_dict(),sampler_state_sha256=u.digest(sampler.state_dict()),training_seen_qids=sorted(seen),training_seen_questions=len(seen),phase='accepted_update_complete',original_E4_world_size=4,historical_four_rank_archive_only=True,immediate_resume_world_size=1,current_world_size=1,data_cursor=dict(cursor=cursor,sampler_binding=sampler.binding,sampler_state_sha256=u.digest(sampler.state_dict())))
 if state['steps']!=SOURCE_STEPS+cursor or state['sampler_state']['accepted_updates']!=cursor:raise ValueError('Native Adam and sampler accepted clocks differ')
 return state


def restore(state,model,ema,opt,device,sampler,binding):
 if state['role']!='BALANCED_HARD_REAL_G10_NATIVE_JOINT_E4_CONTINUATION'or state['config_sha256']!=binding['config_sha256']or state['matched_binding_sha256']!=binding['matched_binding_sha256']or state['steps']!=SOURCE_STEPS+state['cursor']or state['sampler_state']['accepted_updates']!=state['cursor']or u.digest(state['sampler_state'])!=state['sampler_state_sha256']or state['phase']!='accepted_update_complete':raise ValueError('Exact native optimizer/sampler/source binding differs')
 if state.get('data_cursor')!=dict(cursor=state['cursor'],sampler_binding=sampler.binding,sampler_state_sha256=state['sampler_state_sha256']):raise ValueError('Explicit sampler data cursor binding differs')
 u.restore_blob(state,model,ema,opt,device);sampler.load_state_dict(state['sampler_state'])
 if u.state_hashes(model,ema,opt,device)!=state['accepted_hashes']or sampler.state_dict()!=state['sampler_state']:raise ValueError('Actual full native and sampler restore differs')
 return state['cursor'],set(state['training_seen_qids'])


def do_update(model,ema,opt,reader,lookup,sampler,device):
 batch=sampler.next_batch();groups=[lookup[q]for q in batch['source_qids']]
 def supplied(actual,role,**kwargs):
  if role!='train'or[g['qid']for g in actual]!=batch['source_qids']:raise ValueError('Trusted pending sampler batch identity differs')
  return batch['grouping']
 value=balanced_native_update(model,ema,opt,reader,groups,device,grouping_kwargs={},loss_kwargs=LOSS,micro_trajectories=5,ema_decay=.99,max_virtual_views=128,max_inter_view_pairs=16384,grouping_builder=supplied)
 sampler.observe(batch,value.pop('per_view_mixed_BCE'));value.update(batch_token=batch['token'],source_qids_sha256=u.digest(batch['source_qids']),sampler_epoch=sampler.epoch,sampler_accepted_updates=sampler.accepted_updates,sampler_state_sha256=u.digest(sampler.state_dict()))
 return value,batch


def independent_cpu(base):
 if not os.environ.get('SLURM_JOB_ID')or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or torch.cuda.is_initialized():raise ValueError('Independent fullcheckpoint audit only inside owned allocation CPU step')
 c=u.read(base/'config.json');groups,blocked,hashes=own_groups(c);sampler=BalancedHardG10Sampler(groups,seed=1729,batch_questions=32,steps_per_epoch=c['steps_per_epoch'],forbidden_qids=blocked,forbidden_hashes=hashes)
 p=base/'checkpoints/cursor-000001.pt';ref=dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p));state=torch.load(p,map_location='cpu',mmap=True,weights_only=False);actual=dict(raw=u.model_mapping_hash(state['raw']),ema=u.model_mapping_hash(state['ema']),adam=u.optimizer_blob_hash(state['optimizer']),raw_buffers=u.tensor_hash(state['raw_buffers']),ema_buffers=u.tensor_hash(state['ema_buffers']),rng=u.rng_hash(state['rng']))
 if actual!=state['accepted_hashes']or state['steps']!=14419 or state['cursor']!=1 or state['sampler_state']['accepted_updates']!=1 or u.digest(state['sampler_state'])!=state['sampler_state_sha256']:raise ValueError('Independent full native six-state/sampler reload failed')
 sampler.load_state_dict(state['sampler_state']);before=copy.deepcopy(sampler.state_dict());expected=sampler.next_batch();sampler.load_state_dict(before)
 if sampler.next_batch()!=expected:raise ValueError('Independent sampler next-batch restore differs')
 replay=u.read(base/'first-update-replay.json')
 if replay['expected']!=actual or replay['actual']!=actual or not replay['block_changed']or not replay['readout_changed']:raise ValueError('Actual replay and native parameter-change proofs differ')
 if p.stat().st_size!=ref['bytes']or u.sha(p)!=ref['sha256']:raise ValueError('Saved checkpoint changed during independent read')
 result=dict(status='NATIVE_JOINT_CP1_INDEPENDENT_CPU_FULL_SIX_STATE_AND_SAMPLER_RELOAD_PASSED',checkpoint=ref,components=actual,sampler_state_sha256=state['sampler_state_sha256'],next_sampler_batch_identity_exact=True,old_E4_unchanged=True,block_and_readout_changed=True,CUDA_initialized=False,GPU=0,host=platform.node(),allocation_job_id=os.environ['SLURM_JOB_ID']);u.atomic(base/'first-cp-independent-audit.json',result);return result


def selected_reader(c,groups,sampler):
 """Qualification selects only immutable source chunks covering its seven batches.

    Separate epoch0 metadata lookahead observes zeros only to advance the sampler
    cursor; these are not used as actual training hardness or persisted learner
    observations. Epoch0 question/views are loss-independent and rechecked below.
    """
 from .replay_feature_reader import ReplayFeatureReader
 refs=u.read(u.checked(c['features']));needed={t for g in groups for t in g['trajectory_ids']};lookahead=None
 if c['mode']=='qualification':
  state=sampler.state_dict();temporary=copy.deepcopy(sampler);temporary.load_state_dict(state);wanted=set();lookahead=[]
  for _ in range(c['update_budget']):
   batch=temporary.next_batch();lookahead.append(dict(qids=batch['source_qids'],views_sha256=u.digest(batch['grouping']['groups'])));wanted.update(batch['source_qids']);temporary.observe(batch,[0.]*128)
  needed={t for g in groups if g['qid']in wanted for t in g['trajectory_ids']}
 selected=[]
 for ref in refs:
  manifest=u.read(u.checked(ref));index=Path(manifest['files']['index.jsonl']['path'])
  with index.open()as stream:hit=any(json.loads(row)['trajectory_id']in needed for row in stream)
  if hit:selected.append(ref)
 u.atomic(Path(c['base'])/'reader-selection.json',dict(scope=c['mode'],selected_manifest_count=len(selected),actual_required_trajectory_count=len(needed),reader_full_payload_SHA_required=True,lookahead_actual_training_hardness=False))
 reader=ReplayFeatureReader(selected,26)
 if not needed<=set(reader.records):raise ValueError('Accepted full source identities missing actual retained HF layer26 rows')
 return reader,lookahead


def run(base):
 global STOP
 if(base/'complete.json').exists():return u.read(base/'complete.json')
 c=u.read(base/'config.json')
 if c['mode']!='main'or base!=MAIN or c['source_checkpoint']['sha256']!=SOURCE_SHA or c['loss_kwargs']!=LOSS or c['lr']!=5e-6 or c['ema_decay']!=.99:raise ValueError('Typed exact E4 native joint recipe required')
 if c['fit_questions']!=124194 or c['steps_per_epoch']!=3882 or c['update_budget']!=15528:raise ValueError('Actual qualification scope or124194 main budget differs')
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1 or'H100'not in torch.cuda.get_device_properties(0).name:raise ValueError('Own oneH100 allocation required')
 for name,sha in c['code_hashes'].items():
  if u.sha(Path(__file__).resolve().parents[3]/name)!=sha:raise ValueError('Immutable source drift '+name)
 device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 def stop(*_):
  global STOP
  STOP=True
 for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 groups,blocked,hashes=own_groups(c);lookup={g['qid']:g for g in groups};sampler=BalancedHardG10Sampler(groups,seed=1729,batch_questions=32,steps_per_epoch=c['steps_per_epoch'],forbidden_qids=blocked,forbidden_hashes=hashes)
 binding=dict(config_sha256=u.sha(base/'config.json'),matched_binding_sha256=u.digest(c),source_checkpoint=c['source_checkpoint'],training_groups=c['groups'],sampler_source=c['sampler_source'],mode=c['mode'],loss_kwargs=LOSS)
 u.atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],started=time.time(),stage='INPUT_SHA_AND_NATIVE_RESTORE',**binding))
 source=torch.load(u.checked(c['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False);ranks=inherited.verify_source(c,u.read(u.checked(c['source_receipt'])),u.read(u.checked(c['source_config'])),source)
 from .model import Probe
 model=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);opt=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0);u.restore_blob(source,model,ema,opt,device);inherited.verify_restored_source(source,model,ema,opt,device)
 source_before=source_parts(model);reader=None;seen=set();cursor=0
 try:
  reader,lookahead=selected_reader(c,groups,sampler)
  # All reader/model initialization is outside the inherited RNG clock.
  u.restore_blob(source,model,ema,opt,device);inherited.verify_restored_source(source,model,ema,opt,device)
  if(base/'latest.json').exists():
   latest=u.read(base/'latest.json');saved=torch.load(u.checked(latest),map_location='cpu',mmap=True,weights_only=False);cursor,seen=restore(saved,model,ema,opt,device,sampler,binding)
   for name in ('progress.json','update-intent.json'):
    if(base/name).exists()and u.read(base/name).get('cursor',0)>cursor:raise ValueError('Unsaved accepted or unresolved later native transaction; no rollback')
  elif any((base/n).exists()for n in ('progress.json','update-intent.json')):raise ValueError('No complete resume checkpoint for existing transaction')
  def save():
   from .native_joint_rolling_checkpoint import save_native
   state=snapshot(model,ema,opt,device,sampler,cursor,binding,ranks,seen)
   pin=cursor in (0,1)or cursor%c['steps_per_epoch']==0
   return save_native(base,state,pin=pin)
  def evaluate_due():
   if c['mode']=='main'and cursor%c['steps_per_epoch']==0:
    if STOP:
     u.atomic(base/'pending-evaluation.json',dict(cursor=cursor,epoch=cursor//c['steps_per_epoch'],checkpoint=u.read(base/'latest.json')));return None
    from .joint_balanced_native_evaluation import evaluate
    return evaluate(base,c,model,ema,opt,device,save())
  if c['mode']=='main':
   if 'evaluation'not in c:raise ValueError('Actual fixed512/2048 evaluation descriptors required for main')
   for ep in range(cursor//c['steps_per_epoch']):
    if not(base/'evaluations'/('epoch-%02d'%ep)/'complete.json').exists():raise ValueError('Missing past evaluation cannot use future weights')
   evaluate_due()
  def advance():
   nonlocal cursor
   u.atomic(base/'update-intent.json',dict(cursor=cursor+1,phase='started',time=time.time()));metrics,batch=do_update(model,ema,opt,reader,lookup,sampler,device)
   if lookahead is not None and(batch['source_qids']!=lookahead[cursor]['qids']or u.digest(batch['grouping']['groups'])!=lookahead[cursor]['views_sha256']):raise ValueError('Actual uniform epoch0 sampler differs from bounded metadata reader scope')
   seen.update(batch['source_qids']);cursor+=1;u.atomic(base/'update-intent.json',dict(cursor=cursor,phase='accepted_update_complete'));row=dict(cursor=cursor,steps=u.steps(opt),budget=c['update_budget'],seen_questions=len(seen),time=time.time(),**metrics);u.atomic(base/'progress.json',row)
   with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(row),flush=True);return row
  if STOP:
   checkpoint=save();u.atomic(base/'interrupted.json',dict(cursor=cursor,steps=u.steps(opt),checkpoint=checkpoint,sampler_state_sha256=u.digest(sampler.state_dict())));return
  if cursor==0:
   initial_sampler=sampler.state_dict();metrics,batch=do_update(model,ema,opt,reader,lookup,sampler,device);expected=u.state_hashes(model,ema,opt,device);expected_sampler=sampler.state_dict();u.restore_blob(source,model,ema,opt,device);sampler.load_state_dict(initial_sampler);actual_metrics,actual_batch=do_update(model,ema,opt,reader,lookup,sampler,device)
   if u.state_hashes(model,ema,opt,device)!=expected or sampler.state_dict()!=expected_sampler or batch!=actual_batch:raise ValueError('Actual first native fullstate+sampler replay mismatch')
   after=source_parts(model)
   if any(after[key]==source_before[key]for key in after):raise ValueError('Native block and readout must both change')
   cursor=1;seen.update(batch['source_qids']);u.atomic(base/'first-update-replay.json',dict(status='PASSED',expected=expected,actual=u.state_hashes(model,ema,opt,device),sampler_state_sha256=u.digest(expected_sampler),block_changed=True,readout_changed=True,before=source_before,after=after,block_gradient_L2=metrics['block_gradient_L2'],readout_gradient_L2=metrics['readout_gradient_L2'],source_steps=14418,accepted_steps=14419));u.atomic(base/'update-intent.json',dict(cursor=1,phase='accepted_update_complete'));save();u.atomic(base/'progress.json',dict(cursor=1,steps=14419,budget=c['update_budget'],seen_questions=len(seen),**actual_metrics,time=time.time()))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   with(base/'first-cp-independent-audit.log').open('ab')as log:child=subprocess.run([sys.executable,'-u','-m',__package__+'.joint_balanced_native_main','--base',str(base),'--action','audit-first'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=150)
   if child.returncode or not(base/'first-cp-independent-audit.json').exists():raise ValueError('Independent allocation full native CP1+sampler reload failed; stop before update2')
  if cursor>=1 and not(base/'first-cp-independent-audit.json').exists():raise ValueError('Prior full CP1 independent CPU audit missing; do not advance resume')
  if cursor==1:
   saved=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False);before=sampler.state_dict();metrics,batch=do_update(model,ema,opt,reader,lookup,sampler,device);expected=u.state_hashes(model,ema,opt,device);sampler_expected=sampler.state_dict();restore(saved,model,ema,opt,device,sampler,binding);metrics2,batch2=do_update(model,ema,opt,reader,lookup,sampler,device)
   if batch!=batch2 or expected!=u.state_hashes(model,ema,opt,device)or sampler.state_dict()!=sampler_expected:raise ValueError('Actual next update after durable CP1 restore differs')
   seen.update(batch2['source_qids']);cursor=2;save();u.atomic(base/'durable-next-update-replay.json',dict(status='PASSED',cursor=2,steps=14420,expected=expected,actual=u.state_hashes(model,ema,opt,device),sampler_state_sha256=u.digest(sampler_expected)));u.atomic(base/'update-intent.json',dict(cursor=2,phase='accepted_update_complete'));u.atomic(base/'progress.json',dict(cursor=2,steps=14420,budget=c['update_budget'],seen_questions=len(seen),**metrics2,time=time.time()))
  if 2<cursor<c['update_budget'] and not STOP and(base/'latest.json').exists():
   saved=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False);prior_cursor=cursor;metrics,batch=do_update(model,ema,opt,reader,lookup,sampler,device);expected=u.state_hashes(model,ema,opt,device);sampler_expected=sampler.state_dict();restore(saved,model,ema,opt,device,sampler,binding);metrics,batch2=do_update(model,ema,opt,reader,lookup,sampler,device)
   if batch!=batch2 or expected!=u.state_hashes(model,ema,opt,device)or sampler.state_dict()!=sampler_expected:raise ValueError('Actual resumed native next update differs')
   cursor+=1;seen.update(batch2['source_qids']);save();u.atomic(base/('resume-next-update-%06d.json'%prior_cursor),dict(status='PASSED',source_cursor=prior_cursor,accepted_cursor=cursor,actual=expected,sampler_state_sha256=u.digest(sampler_expected)));u.atomic(base/'update-intent.json',dict(cursor=cursor,phase='accepted_update_complete'));u.atomic(base/'progress.json',dict(cursor=cursor,steps=u.steps(opt),budget=c['update_budget'],seen_questions=len(seen),**metrics,time=time.time()));evaluate_due()
  del source
  if 'saved'in locals():del saved
  while cursor<c['update_budget']and not STOP:
   advance()
   if cursor<=7 or cursor%c['save_every']==0 or cursor%c['steps_per_epoch']==0 or cursor==c['update_budget']:save()
   if cursor%c['steps_per_epoch']==0:evaluate_due()
  if u.read(base/'latest.json')['cursor']!=cursor:save()
  if STOP:u.atomic(base/'interrupted.json',dict(cursor=cursor,steps=u.steps(opt),checkpoint=u.read(base/'latest.json'),sampler_state_sha256=u.digest(sampler.state_dict())));return
  result=dict(status='NATIVE_BALANCED_HARD_REAL_G10_FOUR_EPOCH_COMPLETE',mode=c['mode'],cursor=cursor,steps=u.steps(opt),checkpoint=u.read(base/'latest.json'),sampler_state_sha256=u.digest(sampler.state_dict()),source_E4_unchanged=True,block27_and_readout_both_updated=True,training_seen_questions=len(seen),fit_questions=len(groups),full124194_training_claim=c['mode']=='main'and len(seen)==c['fit_questions'],declared_complete_input_cohort=c['fit_questions'],unseen_training_question_count=c['fit_questions']-len(seen),actor_updates=0,fixed512_and_selection2048_evaluation_performed=c['mode']=='main',untouched_final_evaluation_performed=False,postblock_cache_used=False,source_config_sha256=c['source_config']['sha256']);u.atomic(base/'complete.json',result);return result
 finally:
  if reader is not None:reader.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--action',choices=('train','audit-first'),default='train');a=p.parse_args();base=a.base.resolve()
 if base!=MAIN:p.error('Only declared124194 full-cohort main namespace')
 base.mkdir(parents=True,exist_ok=True)
 with(base/('independent-audit.lock'if a.action=='audit-first'else'training.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:result=independent_cpu(base)if a.action=='audit-first'else run(base);print(json.dumps(result),flush=True)
  except BaseException as e:
   if a.action=='train':u.atomic(base/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()))
   raise
if __name__=='__main__':main()
