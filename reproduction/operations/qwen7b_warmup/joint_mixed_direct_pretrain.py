"""Official block27 + direct real-G5 Mixed-only joint pretraining.

This is a new phase, not a continuation of any trained probe checkpoint. Native
block and new group head have separate fresh Adam0. One G10 native forward per
question supplies its four authentic class-balanced G5 views; the microquestion
loss is immediately backpropagated, retaining no other question's native graph.
All tensor/checkpoint/prediction IO requires the owned remote allocation.
"""
import argparse,copy,fcntl,gc,gzip,json,math,os,platform,random,signal,subprocess,sys,time,traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
import numpy as np
import torch
import torch.nn.functional as F
from . import probe_ablation_train as u,mixed_group_checkpoint as ck
from .balanced_hard_g10_sampler import BalancedHardG10Sampler
from .native_mixed_group_classifier import fresh_official_training_state,block_state_hash,OFFICIAL_REVISION
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
MAIN=B/'algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1'
SAMPLER='c98ddcd7861b7023f3efa0d82d3c1fbea0664a13';STOP=False


def typed_config(base,c):
 if (base!=MAIN or c.get('base')!=str(base)or c.get('mode')!='main'or c.get('initialization_source')!='official_initial'or c.get('official_revision')!=OFFICIAL_REVISION or c.get('fit_questions')!=169028 or c.get('steps_per_epoch')!=5283 or c.get('epochs')!=4 or c.get('update_budget')!=21132 or c.get('lr')!=5e-6 or c.get('ema_decay')!=.99 or c.get('seed')!=1729 or c.get('head_seed')!=9137 or c.get('save_every')!=128 or c.get('sampler_source')!=SAMPLER or c.get('expected_exclusion_questions')!=6656):raise ValueError('Exact official169028 direct Mixed-only phase required')
 if any(k in c for k in('source_checkpoint','source_receipt','source_config','resume_from_E4')):raise ValueError('No old trained probe checkpoint may initialize official-start phase')
 if c.get('loss')!={'binary_mixed_weight':1.}or c.get('architecture')!={'capture_layer':26,'block_layer':27,'question_max_tokens':273,'prefix_tokens':20,'head_dimension':256,'head_layers':2,'head_heads':8,'micro_questions':1,'views_per_question':4,'batch_questions':32}:raise ValueError('Mixed-only causal native architecture/loss differs')
 if not isinstance(c.get('evaluation'),dict)or not c.get('code_hashes'):raise ValueError('Published immutable code and fixed512/2048 receipts required')
 for name,sha in c['code_hashes'].items():
  if u.sha(Path(__file__).resolve().parents[3]/name)!=sha:raise ValueError('Immutable source drift '+name)
 return True


def own_groups(c):
 from .g10_grouping import _validate
 groups=u.read(u.checked(c['groups']))['train'];_validate(groups,True)
 if len(groups)!=c['fit_questions']or any(not g.get('question_hash')for g in groups):raise ValueError('Exact169028 authentic ownG10 qid/hash/labels required')
 exclusion=u.read(u.checked(c['exclusions']));blocked={g['qid']for g in exclusion['groups']};hashes={g['question_hash']for g in exclusion['groups']}
 if exclusion.get('role')!='joint_training_fixed_evaluation_identity_exclusions'or len(blocked)!=c['expected_exclusion_questions']or any(g['qid']in blocked or g['question_hash']in hashes for g in groups):raise ValueError('True diag512/selection2048/final4096 identity exclusion required')
 return groups,blocked,hashes


def initialize(c,device):
 """Fresh official native block only; never opens any probe checkpoint."""
 from .replay_probe_layers import verify_model
 from .model import Probe
 proofs,_=verify_model(c['model_dir'],dict(model_files=c['model_files']))
 inventory=dict(path=str(Path(c['model_dir'])/'official-inventory.json'),**c['model_files']['official-inventory.json'])
 if inventory!=c['official_inventory']:raise ValueError('Official inventory differs from declared actual model directory')
 legacy=Probe(c['model_dir'],273,20)
 state=fresh_official_training_state(legacy,official_inventory=inventory,expected_block_hash=block_state_hash(legacy.block),head_seed=c['head_seed'],learning_rate=c['lr'],official_revision=OFFICIAL_REVISION)
 model=state['model'].to(device);ema=state['ema'].to(device).eval().requires_grad_(False)
 # Construct device optimizers AFTER .to(device); no CPU Parameter alias risk.
 if state['block_adam'].state or state['head_adam'].state:raise ValueError('Factory Adam must be fresh0')
 native_opt=torch.optim.Adam(model.block.parameters(),lr=c['lr']);head_opt=torch.optim.Adam(model.group_head.parameters(),lr=c['lr'])
 provenance=dict(state['provenance'],initialization_source='official_initial_Qwen2.5-7B-Instruct_block27',official_model_files=proofs,seed=c['seed'],head_seed=c['head_seed'],initial_raw_hash=u.model_hash(model),initial_EMA_hash=u.model_hash(ema),initial_native_hash=block_state_hash(model.block))
 if provenance['initial_raw_hash']!=provenance['initial_EMA_hash']or native_opt.state or head_opt.state:raise ValueError('Official raw=EMA and two Adam0 required')
 # The unused legacy scalar head is never registered/run/saved in this phase.
 del legacy;gc.collect();return model,ema,native_opt,head_opt,provenance


def warm_gradient_buckets(model):
 """Persistent zero gradients make first/restored accumulation use the same path."""
 for p in model.parameters():
  if p.requires_grad and p.grad is None:p.grad=torch.zeros_like(p)


def selected_views(batch,lookup,expected_questions):
 qids=batch['source_qids'];views=batch['grouping']['groups'];by=defaultdict(list)
 if len(qids)!=expected_questions or len(set(qids))!=len(qids)or len(views)!=4*expected_questions:raise ValueError('One distinct B32 question batch with four own views required')
 for offset,v in enumerate(views):
  q=v['qid'];g=lookup[q];indices=v['source_indices']
  if(q not in qids or len(indices)!=5 or len(set(indices))!=5 or any(type(i)is not int or not 0<=i<10 for i in indices)or v['trajectory_ids']!=[g['trajectory_ids'][i]for i in indices]or v['labels']!=[g['labels'][i]for i in indices]or v['K']!=sum(v['labels'])or v['true_mixed']!=(0<sum(v['labels'])<5)or v['weight_numerator']!=1 or v['weight_denominator']!=4 or v['weight']!=.25):raise ValueError('Sampler G5 must retain own positional IDs/strictEM and quarter weight')
  by[q].append((offset,v))
 if set(by)!=set(qids)or any(len(v)!=4 for v in by.values())or sum(v['true_mixed']for v in views)!=2*expected_questions:raise ValueError('Actual real-G5 positive/negative50:50 and question weight1 required')
 return by


def direct_update(model,ema,native_opt,head_opt,reader,lookup,sampler,device,ema_decay=.99):
 """No scalar p_j or trajectory objective; autograd graph spans only one G10."""
 batch=sampler.next_batch();qids=batch['source_qids'];n=sampler.batch_questions;by=selected_views(batch,lookup,n)
 model.train();native_opt.zero_grad(set_to_none=False);head_opt.zero_grad(set_to_none=False)
 losses=[None]*(4*n);confusion=dict(tp=0,tn=0,fp=0,fn=0);total=0.;started=time.monotonic()
 for qid in qids:
  g=lookup[qid];data=reader.get(g['trajectory_ids'],device)
  if data.get('capture_layer',26)!=26 or data.get('trajectory_ids',g['trajectory_ids'])!=g['trajectory_ids']or data.get('qids',[qid]*10)!=[qid]*10:raise ValueError('Actual layer26 feature/ownG10 identities differ')
  items=by[qid];indices=torch.tensor([v['source_indices']for _,v in items],device=device,dtype=torch.long)
  context=torch.autocast('cuda',dtype=torch.bfloat16)if device.type=='cuda'else nullcontext()
  with context:
   encoded=model.encode_trajectories(data)
   logits=model.group_logits({k:v[indices]for k,v in encoded.items()}).float()
  target=torch.tensor([float(v['true_mixed'])for _,v in items],device=device)
  per=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
  if per.shape!=(4,)or not bool(torch.isfinite(per).all()):raise ValueError('Finite four-view direct binary Mixed BCE required')
  (per.mean()/n).backward()
  numbers=per.detach().cpu().tolist();pred=logits.detach().ge(0).cpu().tolist()
  for(offset,v),loss,p in zip(items,numbers,pred):
   losses[offset]=loss;truth=v['true_mixed'];confusion['tp'if p and truth else'fp'if p else'fn'if truth else'tn']+=1;total+=loss/(4*n)
  del data,encoded,logits,per,target,indices
 if any(p.grad is None or not bool(torch.isfinite(p.grad).all())for p in model.parameters()if p.requires_grad):raise ValueError('Missing or nonfinite direct native/head gradient')
 norms={name:math.sqrt(sum(float(p.grad.detach().float().square().sum())for p in module.parameters()))for name,module in(('native',model.block),('head',model.group_head))}
 if any(not math.isfinite(v)or v<=0 for v in norms.values()):raise ValueError('Native block and new group head must both receive nonzero gradients')
 clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));native_opt.step();head_opt.step()
 with torch.no_grad():
  for raw,avg in zip(model.parameters(),ema.parameters()):avg.mul_(ema_decay).add_(raw,alpha=1-ema_decay)
 sampler.observe(batch,losses)
 native_steps=ck.optimizer_steps(native_opt.state_dict(),True);head_steps=ck.optimizer_steps(head_opt.state_dict(),True)
 if native_steps!=sampler.accepted_updates or head_steps!=native_steps:raise ValueError('Two real Adam updates and sampler clock must agree')
 return dict(binary_Mixed_BCE=total,loss=total,native_gradient_L2=norms['native'],group_head_gradient_L2=norms['head'],gradient_norm_before_clip=clip,gradient_clip=1.,native_steps=native_steps,head_steps=head_steps,questions=n,real_native_trajectories=10*n,views=4*n,mixed_views=2*n,nonmixed_views=2*n,per_question_view_total_weight=1.,micro_questions=1,native_forward_calls=n,trajectory_loss_computed=False,trajectory_accuracy_computed=False,confusion=confusion,seconds=time.monotonic()-started,batch_token=batch['token'],source_qids_sha256=u.digest(qids),sampler_epoch=sampler.epoch,sampler_state_sha256=u.digest(sampler.state_dict())),batch


def eval_node(cursor,steps_per_epoch):
 return cursor//steps_per_epoch if cursor%steps_per_epoch==0 else None


def proof(path):
 p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def independent_cpu(base):
 if not os.environ.get('SLURM_JOB_ID')or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or torch.cuda.is_initialized():raise ValueError('Actual fullCP independent reload requires owned allocation CPU-only child')
 c=u.read(base/'config.json');typed_config(base,c);groups,blocked,hashes=own_groups(c);sampler=BalancedHardG10Sampler(groups,seed=c['seed'],batch_questions=32,steps_per_epoch=c['steps_per_epoch'],forbidden_qids=blocked,forbidden_hashes=hashes)
 p=base/'checkpoints/cursor-000001.pt';ref=proof(p);blob=torch.load(p,map_location='cpu',mmap=True,weights_only=False);ck.validate(blob,dict(config_sha256=u.sha(base/'config.json'),matched_binding_sha256=u.digest(c)));sampler.load_state_dict(blob['sampler_state']);initial=sampler.state_dict();batch=sampler.next_batch();sampler.load_state_dict(initial)
 if sampler.next_batch()!=batch or blob['cursor']!=1 or blob['native_steps']!=1 or blob['head_steps']!=1:raise ValueError('Independent fullCP1 clock and next sampler identity differ')
 replay=u.read(base/'first-update-replay.json')
 if replay['expected']!=blob['accepted_hashes']or replay['actual']!=blob['accepted_hashes']or not replay['native_changed']or not replay['group_head_changed']:raise ValueError('Independent first actual replay/gradient proofs differ')
 if proof(p)!=ref:raise ValueError('Actual fullCP1 changed during independent reload')
 result=dict(status='DIRECT_MIXED_CP1_INDEPENDENT_CPU_FULL_RAW_EMA_TWO_ADAM_RNG_SAMPLER_SOURCE_PASSED',checkpoint=ref,components=blob['accepted_hashes'],sampler_state_sha256=blob['sampler_state_sha256'],next_sampler_batch_identity_exact=True,initialization_source='official_initial',native_steps=1,head_steps=1,GPU=0,host=platform.node(),job_id=os.environ['SLURM_JOB_ID']);u.atomic(base/'first-cp-independent-audit.json',result);return result


def run(base):
 global STOP
 c=u.read(base/'config.json');typed_config(base,c)
 if(base/'complete.json').exists():return u.read(base/'complete.json')
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1 or'H100'not in torch.cuda.get_device_properties(0).name:raise ValueError('Owned oneH100 allocation required')
 device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 for sig in(signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:globals().__setitem__('STOP',True))
 groups,blocked,hashes=own_groups(c);lookup={g['qid']:g for g in groups};sampler=BalancedHardG10Sampler(groups,seed=c['seed'],batch_questions=32,steps_per_epoch=c['steps_per_epoch'],forbidden_qids=blocked,forbidden_hashes=hashes)
 binding=dict(config_sha256=u.sha(base/'config.json'),matched_binding_sha256=u.digest(c),training_groups=c['groups'],features=c['features'],sampler_source=SAMPLER,initialization_source='official_initial',official_revision=OFFICIAL_REVISION,official_inventory=c['official_inventory'])
 u.atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],time=time.time(),stage='OFFICIAL_SOURCE_SHA_AND_LAYER26_READER',**binding))
 model,ema,native_opt,head_opt,provenance=initialize(c,device);warm_gradient_buckets(model)
 from .replay_feature_reader import ReplayFeatureReader
 reader=ReplayFeatureReader(u.read(u.checked(c['features'])),26);needed={t for g in groups for t in g['trajectory_ids']}
 if not needed<=set(reader.records):raise ValueError('All169028 authentic ownG10 layer26 features required')
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
 cursor=0;seen=set();resumed=False
 def stage():return 'evaluation_due'if eval_node(cursor,c['steps_per_epoch'])is not None else'training'
 def state():return ck.snapshot(model,ema,native_opt,head_opt,device,sampler,cursor,binding,provenance,seen,stage())
 def save():return ck.save(base,state(),pin=cursor in(0,1,2)or eval_node(cursor,c['steps_per_epoch'])is not None)
 def due():
  if eval_node(cursor,c['steps_per_epoch'])is not None and not STOP:
   from .mixed_direct_evaluation import evaluate
   return evaluate(base,c,model,ema,native_opt,head_opt,device,save(),cursor)
 def publish(metrics,batch):
  u.atomic(base/'update-intent.json',dict(cursor=cursor,phase='accepted_update_complete'));row=dict(cursor=cursor,budget=c['update_budget'],seen_questions=len(seen),time=time.time(),**metrics);u.atomic(base/'progress.json',row)
  with(base/'curve.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
  print(json.dumps(row),flush=True)
 def update():
  nonlocal cursor
  u.atomic(base/'update-intent.json',dict(cursor=cursor+1,phase='started',time=time.time()));metrics,batch=direct_update(model,ema,native_opt,head_opt,reader,lookup,sampler,device,c['ema_decay']);cursor+=1;seen.update(batch['source_qids']);publish(metrics,batch);return metrics,batch
 try:
  if(base/'latest.json').exists():
   checkpoint=u.read(base/'latest.json');saved=torch.load(u.checked(checkpoint),map_location='cpu',mmap=True,weights_only=False);cursor,seen=ck.restore(saved,model,ema,native_opt,head_opt,device,sampler,binding);provenance=saved['official_initialization'];del saved;gc.collect();resumed=True
   for name in('progress.json','update-intent.json'):
    if(base/name).exists()and u.read(base/name).get('cursor',0)>cursor:raise ValueError('Unresolved later accepted/started update; never silently roll back')
  elif any((base/name).exists()for name in('progress.json','update-intent.json')):raise ValueError('No verified complete restore state for existing transaction')
  else:save()
  for epoch in range(cursor//c['steps_per_epoch']+(0 if cursor%c['steps_per_epoch']==0 else 1)):
   if not(base/'evaluations'/('epoch-%02d'%epoch)/'complete.json').exists():raise ValueError('Past evaluation missing; never score old epoch using future weights')
  due()
  if cursor==0 and not STOP:
   cp0=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False);before=cp0['raw'];before_native=u.tensor_hash({k:v for k,v in before.items()if k.startswith('block.')});before_head=u.tensor_hash({k:v for k,v in before.items()if k.startswith('group_head.')})
   u.atomic(base/'update-intent.json',dict(cursor=1,phase='first_update_replay_trial'))
   m1,b1=direct_update(model,ema,native_opt,head_opt,reader,lookup,sampler,device);expected=ck.snapshot(model,ema,native_opt,head_opt,device,sampler,1,binding,provenance,set(b1['source_qids']),'training')['accepted_hashes'];ck.restore(cp0,model,ema,native_opt,head_opt,device,sampler,binding);m2,b2=update();actual=state()['accepted_hashes']
   if b1!=b2 or expected!=actual:raise ValueError('First actual official native/head/twoAdam/RNG/sampler replay differs')
   changed=dict(native_changed=before_native!=u.tensor_hash({k:v for k,v in model.state_dict().items()if k.startswith('block.')}),group_head_changed=before_head!=u.tensor_hash({k:v for k,v in model.state_dict().items()if k.startswith('group_head.')}))
   if not all(changed.values()):raise ValueError('Actual native block and group head must both change')
   u.atomic(base/'first-update-replay.json',dict(status='PASSED',expected=expected,actual=actual,native_steps=1,head_steps=1,**changed));save();del cp0;gc.collect()
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   with(base/'first-cp-independent-audit.log').open('ab')as log:r=subprocess.run([sys.executable,'-u','-m',__package__+'.joint_mixed_direct_pretrain','--base',str(base),'--action','audit-first'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   if r.returncode or not(base/'first-cp-independent-audit.json').exists():raise ValueError('Independent allocation fullCP1 audit failed before update2')
  if cursor>=1 and not(base/'first-cp-independent-audit.json').exists():raise ValueError('Missing prior actual independent CP1 full-state acceptance')
  if cursor<c['update_budget']and not STOP and(cursor==1 or resumed):
   origin=cursor;saved=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False);m1,b1=direct_update(model,ema,native_opt,head_opt,reader,lookup,sampler,device);expected=ck.snapshot(model,ema,native_opt,head_opt,device,sampler,cursor+1,binding,provenance,seen|set(b1['source_qids']),'training')['accepted_hashes'];ck.restore(saved,model,ema,native_opt,head_opt,device,sampler,binding);m2,b2=update();actual=state()['accepted_hashes']
   if b1!=b2 or expected!=actual:raise ValueError('Actual next update after complete durable restore differs')
   save();u.atomic(base/('durable-next-update-replay.json'if origin==1 else'resume-next-update-%06d.json'%origin),dict(status='PASSED',source_cursor=origin,cursor=cursor,expected=expected,actual=actual));del saved;gc.collect();due()
  while cursor<c['update_budget']and not STOP:
   update()
   if cursor<=7 or cursor%c['save_every']==0 or eval_node(cursor,c['steps_per_epoch'])is not None:save()
   due()
  if u.read(base/'latest.json')['cursor']!=cursor:save()
  if STOP:
   result=dict(status='INTERRUPTED_AFTER_COMPLETE_ACCEPTED_UPDATE',cursor=cursor,native_steps=cursor,head_steps=cursor,checkpoint=u.read(base/'latest.json'),pending_evaluation=eval_node(cursor,c['steps_per_epoch'])is not None);u.atomic(base/'interrupted.json',result);return result
  result=dict(status='OFFICIAL_NATIVE_DIRECT_MIXED_ONLY_FOUR_EPOCH_COMPLETE',cursor=cursor,native_steps=cursor,head_steps=cursor,checkpoint=u.read(base/'latest.json'),fit_questions=len(groups),training_seen_questions=len(seen),balanced_sampling_every_question_once_claim=False,actor_updates=0,old_probe_checkpoint_loaded=False,trajectory_accuracy_computed=False,final4096_scored=False);u.atomic(base/'complete.json',result);return result
 finally:reader.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path);p.add_argument('--action',choices=('train','audit-first'),default='train');a=p.parse_args();base=a.base.resolve()
 if base!=MAIN:p.error('Only independent official169028 direct Mixed namespace')
 base.mkdir(parents=True,exist_ok=True)
 with(base/('independent-audit.lock'if a.action=='audit-first'else'training.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:print(json.dumps(independent_cpu(base)if a.action=='audit-first'else run(base)),flush=True)
  except BaseException as e:
   if a.action=='train':u.atomic(base/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()))
   raise
if __name__=='__main__':main()
