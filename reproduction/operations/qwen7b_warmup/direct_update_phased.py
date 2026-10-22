"""Phased B128 update: joint low LR, then frozen-native/head-only high LR.

Every accepted step averages512 authentic own-G5 losses (four per question).
Micro128->64->32 changes activation residency, never effective batch size. An
OOM can be retried ONLY during forward/backward/checks, before either Adam step.
Optimizer-stage errors propagate without retry or a second optimizer step.
"""
import gc,math,time
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from . import mixed_group_checkpoint as ck,probe_ablation_train as u
from .joint_mixed_direct_pretrain import selected_views
GLOBAL_QUESTIONS=128
MICRO_QUESTIONS=(128,64,32)


def _gradient_attempt(model,native_opt,head_opt,reader,lookup,batch,device,micro,frozen):
 n=GLOBAL_QUESTIONS;qids=batch['source_qids'];by=selected_views(batch,lookup,n);count=4*n
 model.train();native_opt.zero_grad(set_to_none=False);head_opt.zero_grad(set_to_none=False)
 losses_gpu=torch.empty(count,device=device,dtype=torch.float32);logits_gpu=torch.empty_like(losses_gpu);targets_gpu=torch.tensor([float(v['true_mixed'])for v in batch['grouping']['groups']],device=device)
 for offset in range(0,n,micro):
  part=qids[offset:offset+micro];ids=[tid for q in part for tid in lookup[q]['trajectory_ids']];data=reader.get(ids,device);expected_qids=[q for q in part for _ in range(10)]
  if data.get('capture_layer',26)!=26 or data.get('trajectory_ids',ids)!=ids or data.get('qids',expected_qids)!=expected_qids:raise ValueError('Global128 actual layer26/ownG10 feature identity differs')
  own=[(original,view,qi)for qi,q in enumerate(part)for original,view in by[q]];positions=torch.tensor([original for original,_,_ in own],device=device,dtype=torch.long);indices=torch.tensor([[qi*10+i for i in view['source_indices']]for _,view,qi in own],device=device,dtype=torch.long)
  with(torch.autocast('cuda',dtype=torch.bfloat16)if device.type=='cuda'else nullcontext()):
   with(torch.no_grad()if frozen else nullcontext()):encoded=model.encode_trajectories(data)
   logits=model.group_logits({key:value[indices]for key,value in encoded.items()}).float()
  if logits.shape!=(4*len(part),):raise ValueError('Exactly four direct ownG5 logits per source question required')
  per=F.binary_cross_entropy_with_logits(logits,targets_gpu[positions],reduction='none')
  (per.sum()/(4*n)).backward();losses_gpu.index_copy_(0,positions,per.detach());logits_gpu.index_copy_(0,positions,logits.detach())
  del data,encoded,logits,per,indices,positions
 trainable=[p for p in model.parameters()if p.requires_grad]
 if any(p.grad is None for p in trainable):raise ValueError('Missing global128 native/head gradient')
 native=[p for p in model.block.parameters()if p.requires_grad];head=[p for p in model.group_head.parameters()if p.requires_grad]
 if (bool(native)==frozen)or not head or{id(p)for p in native+head}!={id(p)for p in trainable}:raise ValueError('Only the original native block and direct group head may train')
 native_norm=torch.zeros((),device=device)if frozen else torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float(),2)for p in native]),2);head_norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float(),2)for p in head]),2)
 finite=torch.stack([torch.isfinite(p.grad).all()for p in trainable]).all()&torch.isfinite(losses_gpu).all()&torch.isfinite(logits_gpu).all();clip=torch.nn.utils.clip_grad_norm_(trainable,1.)
 predicted=logits_gpu.ge(0);truth=targets_gpu.bool();confusion=torch.stack(((predicted&truth).sum(),((~predicted)&(~truth)).sum(),(predicted&(~truth)).sum(),((~predicted)&truth).sum())).float()
 packed=torch.cat((losses_gpu,torch.stack((native_norm,head_norm,clip,finite.float())),confusion)).detach().cpu().tolist();losses=packed[:count];native_l2,head_l2,clip_value,isfinite=packed[count:count+4]
 if isfinite!=1. or not(math.isfinite(native_l2)and(native_l2==0 if frozen else native_l2>0)and math.isfinite(head_l2)and head_l2>0)or not math.isfinite(clip_value):raise ValueError('Nonfinite global128 direct Mixed loss/gradient or zero native/head gradient')
 return losses,native_l2,head_l2,clip_value,tuple(int(x)for x in packed[count+4:count+8])


def direct_update_phased(model,ema,native_opt,head_opt,reader,lookup,sampler,device,ema_decay=.99,*,micro_questions=64,phase='phase2_joint_low_lr'):
 """One real B128 accepted update; native Adam steps only in phase2.

 Caller prepares the asynchronous reader at the accepted sampler boundary.
 All OOM retries restore that same boundary, its global RNG/native buffers and
 exact authentic batch token. No retry includes optimizer.step or EMA mutation.
 """
 if sampler.batch_questions!=GLOBAL_QUESTIONS or micro_questions not in MICRO_QUESTIONS:raise ValueError('Actual sampler B128 and micro128/64/32 required')
 if phase not in('phase2_joint_low_lr','phase3_head_only'):raise ValueError('Unknown update phase')
 frozen=phase=='phase3_head_only'
 if any(p.requires_grad==frozen for p in model.block.parameters()):raise ValueError('Actual native trainability differs from requested phase')
 if any(g['lr']!=5e-7 for g in native_opt.param_groups)or any(g['lr']!=(5e-5 if frozen else 5e-7)for g in head_opt.param_groups):raise ValueError('Actual phase Adam learning rate differs')
 started=time.monotonic();sampler_before=sampler.state_dict();rng_before=u.rng_state(device);buffers_before=u.buffers(model);native_before=ck.optimizer_steps(native_opt.state_dict(),True);head_before=ck.optimizer_steps(head_opt.state_dict(),True)
 if native_before!=(3898 if frozen else sampler.accepted_updates)or head_before!=sampler.accepted_updates:raise ValueError('Before global128 update, twoAdam/sampler clocks must agree')
 batch=sampler.next_batch();initial_batch=batch;selected_views(batch,lookup,GLOBAL_QUESTIONS)
 if hasattr(reader,'accept_batch'):reader.accept_batch(batch)
 attempted=[];oom=[];candidates=MICRO_QUESTIONS[MICRO_QUESTIONS.index(micro_questions):]
 for attempt,micro in enumerate(candidates):
  attempted.append(micro)
  try:
   losses,native_l2,head_l2,clip_value,confusion=_gradient_attempt(model,native_opt,head_opt,reader,lookup,batch,device,micro,frozen)
  except torch.OutOfMemoryError as error:
   message=str(error)[:1000];error.__traceback__=None
   native_opt.zero_grad(set_to_none=False);head_opt.zero_grad(set_to_none=False);u.restore_buffers(model,buffers_before);u.restore_rng(rng_before,device);sampler.load_state_dict(sampler_before)
   if hasattr(reader,'reset_prefetch'):reader.reset_prefetch()
   gc.collect()
   if device.type=='cuda':torch.cuda.empty_cache()
   if ck.optimizer_steps(native_opt.state_dict(),True)!=native_before or ck.optimizer_steps(head_opt.state_dict(),True)!=head_before:raise ValueError('Unexpected optimizer mutation during pre-Adam OOM retry')
   oom.append(dict(micro_questions=micro,error=message,phase='before_either_optimizer_step',same_global128_retried=True))
   if attempt+1==len(candidates):raise
   if hasattr(reader,'prepare'):reader.prepare(sampler)
   batch=sampler.next_batch()
   if batch!=initial_batch:raise ValueError('OOM retry changed the actual128 questions/ownG5views or sampler token')
   if hasattr(reader,'accept_batch'):reader.accept_batch(batch)
   continue
  break
 # Deliberately OUTSIDE the OOM retry handler. Never retry after either Adam
 # could have stepped; an optimizer error remains a failed unaccepted update.
 if not frozen:native_opt.step()
 head_opt.step()
 with torch.no_grad():
  for raw,average in zip(model.parameters(),ema.parameters()):average.mul_(ema_decay).add_(raw,alpha=1-ema_decay)
 sampler.observe(batch,losses);native_steps=ck.optimizer_steps(native_opt.state_dict(),True);head_steps=ck.optimizer_steps(head_opt.state_dict(),True)
 if native_steps!=native_before+(not frozen)or head_steps!=head_before+1 or sampler.accepted_updates!=head_steps:raise ValueError('One actual global128 twoAdam/EMA/sampler update required')
 tp,tn,fp,fn=confusion;n=GLOBAL_QUESTIONS;total=sum(losses)/(4*n)
 return dict(binary_Mixed_BCE=total,loss=total,native_gradient_L2=native_l2,group_head_gradient_L2=head_l2,gradient_norm_before_clip=clip_value,gradient_clip=1.,native_steps=native_steps,head_steps=head_steps,questions=n,effective_batch_questions=n,real_native_trajectories=10*n,views=4*n,mixed_views=2*n,nonmixed_views=2*n,per_question_view_total_weight=1.,micro_questions=micro,native_forward_calls=n//micro,gradient_accumulation_microbatches=n//micro,attempted_micro_questions=attempted,pre_Adam_OOM_retries=oom,native_Adam_updates_per_global_update=int(not frozen),head_Adam_updates_per_global_update=1,native_frozen=frozen,native_lr=5e-7,head_lr=5e-5 if frozen else 5e-7,phase=phase,EMA_updates_per_global_update=1,trajectory_loss_computed=False,trajectory_accuracy_computed=False,confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn),mixed_recall=tp/(2*n),nonmixed_recall=tn/(2*n),mixed_balanced_accuracy=(tp+tn)/(4*n),seconds=time.monotonic()-started,batch_token=batch['token'],source_qids_sha256=u.digest(batch['source_qids']),sampler_epoch=sampler.epoch,sampler_state_sha256=u.digest(sampler.state_dict()),GPU_to_CPU_metric_vectors=1,updater_variant='global128_lowLR_then_frozen_block_headonly_v1',numerical_bitexact_vs_different_microbatch_claim=False),batch
