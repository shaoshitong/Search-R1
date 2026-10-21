"""Continue actual native/direct-Mixed training at user-requested global B128.

Preserve the highest accepted model/EMA/twoAdam/RNG/data order. Epochs retain
original question-visit budget; only optimizer batch size changes. CPU prefetch
runs concurrently and the chosen successful GPU microbatch persists thereafter.
"""
import argparse,copy,fcntl,gc,json,math,os,signal,time,traceback
from pathlib import Path
import torch
from . import probe_ablation_train as u,mixed_group_checkpoint as ck
from . import joint_mixed_direct_fast_continuation as oldfast
from . import joint_mixed_direct_pretrain as source
from .balanced_hard_g10_sampler import BalancedHardG10Sampler,ROLES
from .direct_update_b128 import direct_update_b128
from .replay_feature_reader_fast import PrefetchedReplayFeatureReader
BASE=source.B/'algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1'
STOP=False

def proof(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))

class B128Sampler(BalancedHardG10Sampler):
 def __init__(self,groups,c,blocked,hashes):
  self.start=c['b128_source']['source_checkpoint']['cursor'];self.boundaries=tuple(c['epoch_end_cursors'])
  super().__init__(groups,seed=c['seed'],batch_questions=128,steps_per_epoch=1321,forbidden_qids=blocked,forbidden_hashes=hashes)
  self.binding=u.digest(dict(base_sampler=self.binding,source_cursor=self.start,epoch_end_cursors=self.boundaries,batch_questions=128))
 @property
 def steps_per_epoch(self):
  # Existing prefetch uses steps-cursor%steps: with the NEXT global boundary,
  # lookahead ends exactly before a hard-pool/epoch change.
  cursor=getattr(self,'accepted_updates',self.start)
  return next((x for x in self.boundaries if x>cursor),self.boundaries[-1]+1)
 @steps_per_epoch.setter
 def steps_per_epoch(self,value):self.nominal_steps=value
 def epoch_at(self,cursor):return sum(cursor>=x for x in self.boundaries)
 def load_state_dict(self,state):
  epoch=self.epoch_at(state['accepted_updates'])
  if state['epoch']!=epoch:raise ValueError('Question-visit epoch differs from saved position')
  self.accepted_updates=state['accepted_updates']
  adapted=copy.deepcopy(state);adapted['epoch']=0
  super().load_state_dict(adapted);self.epoch=epoch
 def observe(self,batch,losses):
  if self.pending!=batch or len(losses)!=len(batch['grouping']['groups']):raise ValueError('Actual128 pending batch/loss count differs')
  by={q:[] for q in batch['source_qids']}
  for view,loss in zip(batch['grouping']['groups'],losses):by[view['qid']].append(float(loss))
  for qid,role in zip(batch['source_qids'],batch['roles']):self.difficulty[role][qid]=sum(by[qid])/4
  self.accepted_updates+=1;epoch=self.epoch_at(self.accepted_updates)
  if epoch!=self.epoch:
   self.epoch=epoch
   for role in ROLES:
    ranked=sorted(self.difficulty[role],key=lambda q:(-self.difficulty[role][q],q));self.hard_pools[role]=ranked[:max(1,math.ceil(len(ranked)/4))]
  self.pending=None

def binding(c):
 return dict(config_sha256=u.sha(BASE/'config.json'),matched_binding_sha256=u.digest(c),training_groups=c['groups'],features=c['features'],sampler_source='user_global128_question_visit_preserving_v1',initialization_source='official_initial',official_revision=c['official_revision'],official_inventory=c['official_inventory'],b128_source=c['b128_source'],epoch_end_cursors=c['epoch_end_cursors'])

def evaluate(base,c,model,ema,native,head,device,sampler,provenance,receipt,epoch):
 folder=base/'evaluations'/('epoch-%02d'%epoch);done=folder/'complete.json'
 if done.exists():return u.read(done)
 from .mixed_direct_evaluation import live_state_hashes,score_reader
 from .joint_balanced_native_evaluation import accepted_HF_validation
 from .replay_feature_reader import ReplayFeatureReader
 spec=c['evaluation'];diagnostic=u.read(u.checked(spec['diagnostic_groups']))['diagnostic'];validation,vr,vproof=accepted_HF_validation(spec);dr=ReplayFeatureReader(u.read(u.checked(spec['diagnostic_features'])),26)
 folder.mkdir(parents=True,exist_ok=True);rng=u.rng_state(device);buffers=(u.buffers(model),u.buffers(ema));before=live_state_hashes(model,ema,native,head,device);metrics={}
 try:
  for label,probe in(('raw',model),('EMA',ema)):
   metrics[label]={}
   for name,groups,reader in(('diagnostic512',diagnostic,dr),('fixed_selection2048',validation,vr)):
    metrics[label][name]=score_reader(probe,reader,groups,device,folder/(label+'-'+name+'.jsonl.gz'),base/'evaluation-progress.json',label+'-'+name,micro_groups=4)
 finally:
  u.restore_rng(rng,device);u.restore_buffers(model,buffers[0]);u.restore_buffers(ema,buffers[1]);vr.close();dr.close()
 after=live_state_hashes(model,ema,native,head,device)
 if before!=after:raise ValueError('Evaluation changed accepted training state')
 result=dict(status='DIRECT_MIXED_B128_FIXED512_2048_EVALUATION_COMPLETE',epoch=epoch,cursor=receipt['cursor'],checkpoint=receipt,config_sha256=u.sha(base/'config.json'),checkpoint_saved_before_evaluation=True,models_two_Adam_RNG_native_buffers_unchanged=True,metrics=metrics,validation_proof=vproof,final4096_scored=False,threshold_fitted=False,actor_updates=0,completed_epoch_time=time.time());u.atomic(done,result);return result

def run(base):
 c=u.read(base/'config.json');spec=c['b128_source'];old=u.read(u.checked(spec['source_config']));start=spec['source_checkpoint']['cursor']
 if c['mode']!='batch128_continuation' or c['batch_questions']!=128 or c['architecture']['micro_questions']!=128:raise ValueError('User global B128 configuration required')
 for name,sha in c['code_hashes'].items():
  if u.sha(Path(__file__).resolve().parents[3]/name)!=sha:raise ValueError('Published source differs '+name)
 if not torch.cuda.is_available()or torch.cuda.device_count()!=1:raise ValueError('Owned oneGPU training required')
 device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 for sig in(signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:globals().__setitem__('STOP',True))
 u.atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],stage='RESTORING_HIGHEST_STATE_FOR_BATCH128',time=time.time(),source_cursor=start))
 groups,blocked,hashes=source.own_groups(c);lookup={g['qid']:g for g in groups};sampler=B128Sampler(groups,c,blocked,hashes);bind=binding(c)
 model,ema,native,head,_=source.initialize(old,device);source.warm_gradient_buckets(model)
 if(base/'latest.json').exists():
  saved=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False);cursor,seen=ck.restore(saved,model,ema,native,head,device,sampler,bind);provenance=saved['official_initialization'];migration=u.read(base/'startup-binding-migration.json')
 else:
  saved=torch.load(u.checked(spec['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False);old_sampler=oldfast.make_sampler(old,groups,blocked,hashes);cursor,seen=ck.restore(saved,model,ema,native,head,device,old_sampler,oldfast.continuation_binding(old,u.sha(u.checked(spec['source_config']))));provenance=saved['official_initialization']
  s=old_sampler.state_dict();s['binding']=sampler.binding;sampler.load_state_dict(s)
  migration=dict(status='BATCH128_HIGHEST_COMPLETE_STATE_RESTORED',source_cursor=cursor,source_checkpoint=spec['source_checkpoint'],source_accepted_hashes=saved['accepted_hashes'],raw_EMA_twoAdam_RNG_buffers_and_sampling_position_preserved=True,sampler_binding_changed_for_user_batch128=True,source_config=spec['source_config'],restoration_receipt=dict(full_checkpoint_fresh_SHA_and_nine_state_restore_passed=True,accepted_hashes=saved['accepted_hashes']))
  u.atomic(base/'startup-binding-migration.json',migration)
 del saved;gc.collect()
 seal=dict(source_config=spec['source_config'],source_checkpoint=spec['source_checkpoint'],restoration_receipt=migration['restoration_receipt'])
 reader=PrefetchedReplayFeatureReader.from_inherited_seal(u.read(u.checked(c['features'])),26,seal);reader.proof['max_prefetched_feature_bytes']=8*128*10*293*3584*2;u.atomic(base/'reader-integrity.json',reader.proof)
 micro=128
 if(base/'progress.json').exists():micro=u.read(base/'progress.json').get('micro_questions',128)
 def epoch_due():return sampler.epoch_at(cursor)if cursor in c['epoch_end_cursors']else None
 def save():return ck.save(base,ck.snapshot(model,ema,native,head,device,sampler,cursor,bind,provenance,seen,'evaluation_due'if epoch_due()else'training'),pin=cursor==start or epoch_due()is not None)
 save()
 try:
  while cursor<c['update_budget']and not STOP:
   epoch=epoch_due()
   if epoch and not(base/'evaluations'/('epoch-%02d'%epoch)/'complete.json').exists():reader.reset_prefetch();evaluate(base,c,model,ema,native,head,device,sampler,provenance,save(),epoch)
   reader.prepare(sampler);u.atomic(base/'update-intent.json',dict(cursor=cursor+1,phase='started',time=time.time()))
   metrics,batch=direct_update_b128(model,ema,native,head,reader,lookup,sampler,device,c['ema_decay'],micro_questions=micro);micro=metrics['micro_questions'];cursor+=1;seen.update(batch['source_qids'])
   visits=start*32+(cursor-start)*128;row=dict(cursor=cursor,budget=c['update_budget'],source_cursor=start,new_updates=cursor-start,question_visits=visits,question_visit_budget=c['total_question_visit_budget'],seen_questions=len(seen),time=time.time(),prefetch=dict(getattr(reader,'prefetch_statistics',{})),**metrics);u.atomic(base/'update-intent.json',dict(cursor=cursor,phase='accepted_update_complete'));u.atomic(base/'progress.json',row)
   with(base/'curve.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
   print(json.dumps(row),flush=True)
   if cursor==start+1 or cursor%128==0 or epoch_due():save()
  receipt=save();epoch=epoch_due()
  if epoch and not STOP:reader.reset_prefetch();evaluate(base,c,model,ema,native,head,device,sampler,provenance,receipt,epoch)
  result=dict(status='INTERRUPTED_AFTER_COMPLETE_ACCEPTED_UPDATE'if STOP else'OFFICIAL_DIRECT_MIXED_B128_FOUR_EPOCH_QUESTION_BUDGET_COMPLETE',cursor=cursor,native_steps=cursor,head_steps=cursor,checkpoint=receipt,source_cursor=start,question_visits=start*32+(cursor-start)*128,global_batch_questions=128,final4096_scored=False,actor_updates=0);u.atomic(base/('interrupted.json'if STOP else'complete.json'),result);return result
 finally:reader.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base.resolve();assert base==BASE
 with(base/'training.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:print(json.dumps(run(base)),flush=True)
  except BaseException as e:u.atomic(base/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
