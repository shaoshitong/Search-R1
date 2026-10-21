"""Immutable asynchronous continuation of the accepted official Mixed-only run.

The original complete checkpoint is restored under its ORIGINAL config binding.
Only then is the same nine-component state saved under this runtime's explicit
binding migration. B32/four authentic G5 views/two Adam/EMA/sampler/global budget
are unchanged. CPU prefetch is disposable, never optimizer or sampler state.
"""
import argparse,fcntl,gc,json,os,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
import torch
from . import probe_ablation_train as u,mixed_group_checkpoint as ck
from . import joint_mixed_direct_pretrain as source
from .balanced_hard_g10_sampler import BalancedHardG10Sampler
from .direct_update_fast import direct_update_fast
from .native_mixed_group_classifier import OFFICIAL_REVISION
BASE=source.B/'algorithm-improvement-20261021/joint-mixed-direct-official-169028-fast-v1'
SOURCE_CONFIG_SHA='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393'
READER_MODE='inherited_full_seal_with_fresh_metadata_and_each_row_SHA'
NINE={'raw','ema','native_adam','head_adam','raw_buffers','ema_buffers','rng','sampler','official_source'}
STOP=False


def proof(path):
 p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def scientific_config(c):
 """Only the declared runtime/transport leaves may change at handoff."""
 value={k:v for k,v in c.items()if k not in('base','mode','code_hashes','continuation','reader_integrity_mode','prefetch_batches','parallel_workers')}
 value['architecture']=dict(value['architecture']);value['architecture'].pop('micro_questions',None)
 return value


def validate_receipts(c,base=BASE):
 spec=c['continuation'];old=spec['source_checkpoint_original'];pinned=spec['source_checkpoint'];cfg=spec['source_config'];main=spec['source_main_config'];cursor=old.get('cursor')
 if (cfg.get('path')!=str(base/'source-config.json')or cfg.get('bytes')!=9615 or cfg.get('sha256')!=SOURCE_CONFIG_SHA or main!=dict(cfg,path=str(source.MAIN/'config.json'))or type(cursor)is not int or not 128<=cursor<21132 or old.get('native_steps')!=cursor or old.get('head_steps')!=cursor or old.get('config_sha256')!=SOURCE_CONFIG_SHA or old.get('full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed')is not True or set(old.get('accepted_hashes',{}))!=NINE):raise ValueError('Highest accepted official source checkpoint/config/full-nine-state receipt required')
 if not str(Path(old['path'])).startswith(str(source.MAIN/'checkpoints')+'/')or pinned.get('path')!=str(base/'inputs/source-checkpoint.pt')or{key:value for key,value in pinned.items()if key!='path'}!={key:value for key,value in old.items()if key!='path'}:raise ValueError('Source pin may change only pathname; no older checkpoint substitution')
 return cursor


def validate_handoff(spec):
 handoff=u.read(u.checked(spec['source_handoff_receipt']));old=spec['source_checkpoint_original'];pinned=spec['source_checkpoint']
 if (str(handoff.get('job_id'))!='746083'or handoff.get('cursor')!=old['cursor']or any(handoff.get(k)is not True for k in('terminal_exit_0_0','highest_accepted_complete_cursor_equal','no_unresolved_tail','training_lock_free'))or handoff.get('checkpoint')!=old or handoff.get('pinned_checkpoint')!=pinned):raise ValueError('Terminal original job/highest cursor/no-unresolved-tail/lock-free handoff required')
 return handoff


def typed_config(base,c):
 if (base!=BASE or c.get('base')!=str(base)or c.get('mode')!='fast_continuation'or c.get('reader_integrity_mode')!=READER_MODE or c.get('prefetch_batches')!=8 or c.get('parallel_workers')!=8 or c.get('architecture',{}).get('micro_questions')!=32):raise ValueError('Exact independent asynchronous micro32 continuation namespace required')
 validate_receipts(c,base);spec=c['continuation'];copied=u.checked(spec['source_config']);original=u.checked(spec['source_main_config'])
 if copied.read_bytes()!=original.read_bytes():raise ValueError('Original source config bytes/base must remain unchanged')
 old=u.read(copied);source.typed_config(source.MAIN,old)
 if scientific_config(c)!=scientific_config(old):raise ValueError('Only runtime/base/microbatch/explicit transport binding may change; science and global budget stay exact')
 for name,sha in c['code_hashes'].items():
  if u.sha(Path(__file__).resolve().parents[3]/name)!=sha:raise ValueError('Immutable continuation source drift '+name)
 validate_handoff(spec);return old


def source_binding(c):
 return dict(config_sha256=SOURCE_CONFIG_SHA,matched_binding_sha256=u.digest(c),training_groups=c['groups'],features=c['features'],sampler_source=source.SAMPLER,initialization_source='official_initial',official_revision=OFFICIAL_REVISION,official_inventory=c['official_inventory'])


def continuation_binding(c,config_sha):
 return dict(config_sha256=config_sha,matched_binding_sha256=u.digest(c),training_groups=c['groups'],features=c['features'],sampler_source=source.SAMPLER,initialization_source='official_initial',official_revision=OFFICIAL_REVISION,official_inventory=c['official_inventory'],source_binding_archive=dict(source_config=c['continuation']['source_config'],source_main_config=c['continuation']['source_main_config'],source_checkpoint_original=c['continuation']['source_checkpoint_original']),binding_migration=dict(source_cursor=c['continuation']['source_checkpoint']['cursor'],source_checkpoint=c['continuation']['source_checkpoint'],source_handoff_receipt=c['continuation']['source_handoff_receipt'],preserved_component_names=sorted(NINE),sampler_and_two_Adam_reset=False,changed_only_runtime_microbatch_and_explicit_input_integrity_policy=True))


def migrate_snapshot(saved,model,ema,native_opt,head_opt,device,sampler,old_binding,new_binding):
 """Restore original binding before creating any new-binding checkpoint."""
 cursor,seen=ck.restore(saved,model,ema,native_opt,head_opt,device,sampler,old_binding)
 value=ck.snapshot(model,ema,native_opt,head_opt,device,sampler,cursor,new_binding,saved['official_initialization'],seen,saved['stage'])
 if value['accepted_hashes']!=saved['accepted_hashes']:raise ValueError('Binding migration changed actual raw/EMA/twoAdam/RNG/buffers/sampler/source')
 return value


def make_sampler(c,groups,blocked,hashes):
 return BalancedHardG10Sampler(groups,seed=c['seed'],batch_questions=32,steps_per_epoch=c['steps_per_epoch'],forbidden_qids=blocked,forbidden_hashes=hashes)


def assert_no_unresolved_tail(base,cursor):
 for name in('progress.json','update-intent.json'):
  if(base/name).exists()and u.read(base/name).get('cursor',0)>cursor:raise ValueError('Unresolved later accepted/started update; never silently roll back')
 transaction=base/'mixed-checkpoint-transaction.json'
 if transaction.exists()and u.read(transaction).get('phase')!='published':raise ValueError('Unresolved full-state checkpoint publication; preserve it')


def inherit_evaluations(base,c,start):
 """Past epoch scores keep their original source/config, never future weights."""
 inherited={}
 last=start//c['steps_per_epoch']
 for epoch in range(last+1):
  path=source.MAIN/'evaluations'/('epoch-%02d'%epoch)/'complete.json'
  if path.exists():
   value=u.read(path)
   if value.get('cursor')!=epoch*c['steps_per_epoch']or value.get('config_sha256')!=SOURCE_CONFIG_SHA or value.get('checkpoint_saved_before_evaluation')is not True:raise ValueError('Inherited completed epoch receipt has wrong original cursor/config')
   inherited[str(epoch)]=proof(path)
  elif epoch*c['steps_per_epoch']<start:raise ValueError('Past source evaluation missing; do not score it with future weights')
 result=dict(source_config=c['continuation']['source_config'],source_cursor=start,evaluations=inherited,score_payloads_not_copied=True)
 path=base/'inherited-evaluations.json'
 if path.exists()and u.read(path)!=result:raise ValueError('Inherited epoch provenance changed')
 u.atomic(path,result);return inherited


def evaluation_history(base,c,cursor,inherited):
 for epoch in range((cursor+c['steps_per_epoch']-1)//c['steps_per_epoch']):
  if str(epoch)in inherited:continue
  path=base/'evaluations'/('epoch-%02d'%epoch)/'complete.json'
  if not path.exists():raise ValueError('Past fast evaluation missing; preserve its pending exact state')
  value=u.read(path)
  if value.get('cursor')!=epoch*c['steps_per_epoch']or value.get('config_sha256')!=u.sha(base/'config.json'):raise ValueError('Past fast evaluation cursor/config differs')


def independent_cpu(base):
 if not os.environ.get('SLURM_JOB_ID')or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or torch.cuda.is_initialized():raise ValueError('Independent full checkpoint audit requires owned allocation CPU-only child')
 c=u.read(base/'config.json');typed_config(base,c);start=validate_receipts(c,base);groups,blocked,hashes=source.own_groups(c);sampler=make_sampler(c,groups,blocked,hashes)
 path=base/'checkpoints'/('cursor-%06d.pt'%(start+1));ref=proof(path);blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False);ck.validate(blob,continuation_binding(c,u.sha(base/'config.json')))
 sampler.load_state_dict(blob['sampler_state']);state=sampler.state_dict();batch=sampler.next_batch();sampler.load_state_dict(state)
 if sampler.next_batch()!=batch or blob['cursor']!=start+1 or blob['native_steps']!=start+1 or blob['head_steps']!=start+1:raise ValueError('Independent first fast full checkpoint clocks or next sampler identity differ')
 replay=u.read(base/'first-fast-update-replay.json')
 if replay['expected']!=blob['accepted_hashes']or replay['actual']!=blob['accepted_hashes']or not replay['native_changed']or not replay['group_head_changed']:raise ValueError('Independent first fast actual replay/gradient proofs differ')
 if proof(path)!=ref:raise ValueError('Pinned first fast checkpoint changed during CPU reload')
 result=dict(status='FAST_CONTINUATION_FIRST_CP_INDEPENDENT_CPU_FULL_NINE_STATE_PASSED',checkpoint=ref,components=blob['accepted_hashes'],source_cursor=start,cursor=start+1,native_steps=start+1,head_steps=start+1,next_sampler_batch_identity_exact=True,global_clocks_not_reset=True,GPU=0,host=platform.node(),job_id=os.environ['SLURM_JOB_ID']);u.atomic(base/'first-fast-cp-independent-audit.json',result);return result


def run(base):
 global STOP
 c=u.read(base/'config.json');old=typed_config(base,c);start=validate_receipts(c,base);spec=c['continuation']
 if(base/'complete.json').exists():return u.read(base/'complete.json')
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1 or'H100'not in torch.cuda.get_device_properties(0).name:raise ValueError('Owned oneH100 continuation allocation required')
 device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 for sig in(signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:globals().__setitem__('STOP',True))
 groups,blocked,hashes=source.own_groups(c);lookup={g['qid']:g for g in groups};sampler=make_sampler(c,groups,blocked,hashes);binding=continuation_binding(c,u.sha(base/'config.json'))
 u.atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],time=time.time(),stage='OFFICIAL_FULL_SHA_AND_HIGHEST_FULL_STATE_RESTORE',config_sha256=binding['config_sha256'],source_cursor=start))
 model,ema,native_opt,head_opt,_=source.initialize(old,device);source.warm_gradient_buckets(model)
 resumed=(base/'latest.json').exists()
 if resumed:
  receipt=u.read(base/'latest.json');saved=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False);cursor,seen=ck.restore(saved,model,ema,native_opt,head_opt,device,sampler,binding);provenance=saved['official_initialization'];del saved;gc.collect();assert_no_unresolved_tail(base,cursor)
  migration=u.read(base/'startup-binding-migration.json')
  if migration['original_checkpoint']!=spec['source_checkpoint']or migration['source_accepted_hashes']!=spec['source_checkpoint']['accepted_hashes']:raise ValueError('Original full-state migration provenance differs on continuation resume')
 else:
  if any((base/name).exists()for name in('progress.json','update-intent.json')):raise ValueError('No accepted complete state for existing continuation transaction')
  receipt=spec['source_checkpoint'];saved=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
  if saved['cursor']!=start or saved['accepted_hashes']!=receipt['accepted_hashes']:raise ValueError('Pinned source complete state is not exact highest handoff')
  migrated=migrate_snapshot(saved,model,ema,native_opt,head_opt,device,sampler,source_binding(old),binding);cursor=start;seen=set(saved['training_seen_qids']);provenance=saved['official_initialization']
  restoration=dict(full_checkpoint_fresh_SHA_and_nine_state_restore_passed=True,accepted_hashes=saved['accepted_hashes'])
  migration=dict(status='HIGHEST_SOURCE_FULL_STATE_RESTORED_BINDING_ONLY_MIGRATED',source_cursor=start,original_checkpoint=receipt,source_accepted_hashes=saved['accepted_hashes'],new_accepted_hashes=migrated['accepted_hashes'],source_config=spec['source_config'],new_config=proof(base/'config.json'),restoration_receipt=restoration,global_Adam_sampler_RNG_preserved=True)
  u.atomic(base/'startup-binding-migration.json',migration);ck.save(base,migrated,pin=True);del saved,migrated;gc.collect()
 # The original immutable full-SHA reader completed before this accepted source
 # cursor. Fresh metadata/index/size + EACH used retained row SHA remain strict.
 # This explicitly inherits full payload seals; no new whole276GB scan is claimed.
 from .replay_feature_reader_fast import PrefetchedReplayFeatureReader
 seal=dict(source_config=spec['source_config'],source_checkpoint=spec['source_checkpoint'],restoration_receipt=migration['restoration_receipt'])
 reader=PrefetchedReplayFeatureReader.from_inherited_seal(u.read(u.checked(c['features'])),26,seal)
 if reader.parallel_workers!=c['parallel_workers']or reader.prefetch_batches!=c['prefetch_batches']:raise ValueError('Declared producer queue/worker count differs')
 needed={tid for g in groups for tid in g['trajectory_ids']}
 if not needed<=set(reader.records):raise ValueError('All169028 authentic ownG10 rows required')
 u.atomic(base/'reader-integrity.json',dict(mode=READER_MODE,proof=reader.proof,whole_payload_fresh_SHA_claimed=False,per_used_row_SHA_required=True,disposable_CPU_queue_not_checkpoint_state=True))
 inherited=inherit_evaluations(base,c,start)
 evaluation_history(base,c,cursor,inherited)
 def stage():return 'evaluation_due'if source.eval_node(cursor,c['steps_per_epoch'])is not None else'training'
 def state():return ck.snapshot(model,ema,native_opt,head_opt,device,sampler,cursor,binding,provenance,seen,stage())
 def save():return ck.save(base,state(),pin=cursor<=start+7 or source.eval_node(cursor,c['steps_per_epoch'])is not None)
 def due():
  epoch=source.eval_node(cursor,c['steps_per_epoch'])
  if epoch is not None and str(epoch)not in inherited and not STOP:
   reader.reset_prefetch()
   from .mixed_direct_evaluation import evaluate
   return evaluate(base,c,model,ema,native_opt,head_opt,device,save(),cursor)
 def raw_update():
  reader.prepare(sampler)
  return direct_update_fast(model,ema,native_opt,head_opt,reader,lookup,sampler,device,c['ema_decay'],micro_questions=c['architecture']['micro_questions'])
 def update():
  nonlocal cursor
  u.atomic(base/'update-intent.json',dict(cursor=cursor+1,phase='started',time=time.time()));metrics,batch=raw_update();cursor+=1;seen.update(batch['source_qids']);u.atomic(base/'update-intent.json',dict(cursor=cursor,phase='accepted_update_complete'))
  row=dict(cursor=cursor,budget=c['update_budget'],source_cursor=start,new_updates=cursor-start,seen_questions=len(seen),time=time.time(),prefetch=dict(getattr(reader,'prefetch_statistics',{})),reader_integrity_mode=READER_MODE,**metrics);u.atomic(base/'progress.json',row)
  with(base/'curve.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
  print(json.dumps(row),flush=True);return metrics,batch
 def replay_update(first=False):
  origin=cursor;checkpoint=u.read(base/'latest.json');saved=torch.load(u.checked(checkpoint),map_location='cpu',mmap=True,weights_only=False)
  before_native=u.tensor_hash({k:v for k,v in saved['raw'].items()if k.startswith('block.')});before_head=u.tensor_hash({k:v for k,v in saved['raw'].items()if k.startswith('group_head.')})
  reader.reset_prefetch();u.atomic(base/'update-intent.json',dict(cursor=cursor+1,phase='fast_update_replay_trial',time=time.time()));_,batch1=raw_update();expected=ck.snapshot(model,ema,native_opt,head_opt,device,sampler,origin+1,binding,provenance,seen|set(batch1['source_qids']),'training')['accepted_hashes']
  ck.restore(saved,model,ema,native_opt,head_opt,device,sampler,binding);reader.reset_prefetch();_,batch2=update();actual=state()['accepted_hashes']
  if batch1!=batch2 or expected!=actual:raise ValueError('Complete state next-update replay differs under fixed fast micro32')
  changed=dict(native_changed=before_native!=u.tensor_hash({k:v for k,v in model.state_dict().items()if k.startswith('block.')}),group_head_changed=before_head!=u.tensor_hash({k:v for k,v in model.state_dict().items()if k.startswith('group_head.')}))
  if not all(changed.values()):raise ValueError('Native block and original direct group head must both change')
  name='first-fast-update-replay.json'if first else'fast-durable-next-update-replay.json'if origin==start+1 else'resume-next-update-%06d.json'%origin
  u.atomic(base/name,dict(status='PASSED',source_cursor=origin,cursor=cursor,expected=expected,actual=actual,micro_questions=32,numerical_bitexact_vs_old_micro1_claim=False,**changed));save();del saved;gc.collect()
 try:
  due()
  if cursor==start and not STOP:replay_update(first=True)
  if cursor>=start+1 and not(base/'first-fast-cp-independent-audit.json').exists():
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   with(base/'first-fast-cp-independent-audit.log').open('ab')as log:r=subprocess.run([sys.executable,'-u','-m',__package__+'.joint_mixed_direct_fast_continuation','--base',str(base),'--action','audit-first'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   if r.returncode or not(base/'first-fast-cp-independent-audit.json').exists():raise ValueError('Independent full first-fast CP audit failed')
  if cursor<c['update_budget']and not STOP and(cursor==start+1 or resumed):replay_update();due()
  while cursor<c['update_budget']and not STOP:
   update()
   if cursor<=start+7 or cursor%c['save_every']==0 or source.eval_node(cursor,c['steps_per_epoch'])is not None:save()
   if cursor>=start+6 and not(base/'startup-qualification.json').exists():
    required=('first-fast-update-replay.json','first-fast-cp-independent-audit.json','fast-durable-next-update-replay.json')
    if any(not(base/name).exists()for name in required):raise ValueError('First full-state fast continuation qualifications missing')
    u.atomic(base/'startup-qualification.json',dict(status='SIX_REAL_FAST_CONTINUATION_UPDATES_AND_FULL_STATE_REPLAY_PASSED',source_cursor=start,cursor=cursor,new_complete_updates=cursor-start,checkpoint=save(),proofs={name:proof(base/name)for name in required},micro_questions=32,asynchronous_CPU_producer=True,global_clocks_not_reset=True))
   due()
  if u.read(base/'latest.json')['cursor']!=cursor:save()
  if STOP:
   result=dict(status='INTERRUPTED_AFTER_COMPLETE_ACCEPTED_UPDATE',source_cursor=start,cursor=cursor,native_steps=cursor,head_steps=cursor,checkpoint=u.read(base/'latest.json'),pending_evaluation=source.eval_node(cursor,c['steps_per_epoch'])is not None);u.atomic(base/'interrupted.json',result);return result
  result=dict(status='OFFICIAL_DIRECT_MIXED_ASYNC_CONTINUATION_FOUR_GLOBAL_EPOCH_COMPLETE',source_cursor=start,cursor=cursor,native_steps=cursor,head_steps=cursor,checkpoint=u.read(base/'latest.json'),fit_questions=len(groups),training_seen_questions=len(seen),original_global_budget_preserved=21132,reader_integrity_mode=READER_MODE,actor_updates=0,trajectory_accuracy_computed=False,final4096_scored=False);u.atomic(base/'complete.json',result);return result
 finally:reader.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path);p.add_argument('--action',choices=('train','audit-first'),default='train');a=p.parse_args();base=a.base.resolve()
 if base!=BASE:p.error('Only independent official169028 asynchronous continuation namespace')
 base.mkdir(parents=True,exist_ok=True)
 with(base/('independent-audit.lock'if a.action=='audit-first'else'training.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:print(json.dumps(independent_cpu(base)if a.action=='audit-first'else run(base)),flush=True)
  except BaseException as e:
   if a.action=='train':u.atomic(base/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()))
   raise
if __name__=='__main__':main()
