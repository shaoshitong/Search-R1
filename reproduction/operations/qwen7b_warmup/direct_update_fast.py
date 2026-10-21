"""Batched real-G10 direct-Mixed update, no per-question metric device sync.

The sampler, authentic four-view objective, native block, original two-layer
head, two Adam optimizers and EMA are unchanged. Numerical batching can differ
from scalar BF16 GEMMs; this is a new immutable implementation, never a hotpatch.
The only host extraction of CUDA metrics is one compact vector after backward.
The model's original geometry/finite guards remain intact at microbatch granularity.
"""
import math,time
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from . import mixed_group_checkpoint as ck,probe_ablation_train as u
from .joint_mixed_direct_pretrain import selected_views
MICRO_QUESTIONS=(1,4,8,16,32)


def direct_update_fast(model,ema,native_opt,head_opt,reader,lookup,sampler,device,ema_decay=.99,*,micro_questions=8):
 if micro_questions not in MICRO_QUESTIONS or sampler.batch_questions!=32:raise ValueError('Preserve B32; micro_questions must be1/4/8/16/32')
 batch=sampler.next_batch()
 # The asynchronous reader checks the real sampler token before ANY model work.
 # Its disposable lookahead is never part of the accepted sampler/RNG state.
 if hasattr(reader,'accept_batch'):reader.accept_batch(batch)
 qids=batch['source_qids'];n=32;by=selected_views(batch,lookup,n);count=4*n;start=time.monotonic()
 model.train();native_opt.zero_grad(set_to_none=False);head_opt.zero_grad(set_to_none=False)
 losses_gpu=torch.empty(count,device=device,dtype=torch.float32);logits_gpu=torch.empty_like(losses_gpu);targets_gpu=torch.tensor([float(v['true_mixed'])for v in batch['grouping']['groups']],device=device)
 for offset in range(0,n,micro_questions):
  part=qids[offset:offset+micro_questions];ids=[tid for q in part for tid in lookup[q]['trajectory_ids']];data=reader.get(ids,device);expected_qids=[q for q in part for _ in range(10)]
  if data.get('capture_layer',26)!=26 or data.get('trajectory_ids',ids)!=ids or data.get('qids',expected_qids)!=expected_qids:raise ValueError('Batched actual layer26/ownG10 feature identities differ')
  own=[(original,view,qi)for qi,q in enumerate(part)for original,view in by[q]]
  positions=torch.tensor([original for original,_,_ in own],device=device,dtype=torch.long);indices=torch.tensor([[qi*10+i for i in view['source_indices']]for _,view,qi in own],device=device,dtype=torch.long)
  with(torch.autocast('cuda',dtype=torch.bfloat16)if device.type=='cuda'else nullcontext()):
   encoded=model.encode_trajectories(data);logits=model.group_logits({key:value[indices]for key,value in encoded.items()}).float()
  if logits.shape!=(4*len(part),):raise ValueError('Exactly four direct ownG5 logits per microquestion required')
  per=F.binary_cross_entropy_with_logits(logits,targets_gpu[positions],reduction='none')
  # All views retain weight1/4 per source question and overall question mean1/32.
  (per.sum()/(4*n)).backward();losses_gpu.index_copy_(0,positions,per.detach());logits_gpu.index_copy_(0,positions,logits.detach())
  del data,encoded,logits,per,indices,positions
 trainable=[p for p in model.parameters()if p.requires_grad]
 if any(p.grad is None for p in trainable):raise ValueError('Missing direct native/head gradient')
 native=[p for p in model.block.parameters()if p.requires_grad];head=[p for p in model.group_head.parameters()if p.requires_grad]
 if not native or not head or {id(p)for p in native+head}!={id(p)for p in trainable}:raise ValueError('Only the native block and original group head are trainable')
 native_norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float(),2)for p in native]),2);head_norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float(),2)for p in head]),2)
 finite=torch.stack([torch.isfinite(p.grad).all()for p in trainable]).all()&torch.isfinite(losses_gpu).all()&torch.isfinite(logits_gpu).all()
 clip=torch.nn.utils.clip_grad_norm_(trainable,1.)
 predicted=logits_gpu.ge(0);truth=targets_gpu.bool();confusion=torch.stack(((predicted&truth).sum(),((~predicted)&(~truth)).sum(),(predicted&(~truth)).sum(),((~predicted)&truth).sum())).float()
 # Exactly one metric D2H vector. No loss.item()/tolist()/bool(cuda) per question
 # or parameter; fixed recipe Adam clocks are CPU scalar state after both steps.
 packed=torch.cat((losses_gpu,torch.stack((native_norm,head_norm,clip,finite.float())),confusion)).detach().cpu().tolist()
 losses=packed[:count];native_l2,head_l2,clip_value,isfinite=packed[count:count+4]
 if isfinite!=1. or not all(math.isfinite(x)and x>0 for x in(native_l2,head_l2))or not math.isfinite(clip_value):raise ValueError('Nonfinite direct Mixed loss/gradient or missing native/head gradient')
 native_opt.step();head_opt.step()
 with torch.no_grad():
  for raw,average in zip(model.parameters(),ema.parameters()):average.mul_(ema_decay).add_(raw,alpha=1-ema_decay)
 sampler.observe(batch,losses);native_steps=ck.optimizer_steps(native_opt.state_dict(),True);head_steps=ck.optimizer_steps(head_opt.state_dict(),True)
 if native_steps!=head_steps or native_steps!=sampler.accepted_updates:raise ValueError('Batched actual twoAdam/sampler accepted clocks differ')
 tp,tn,fp,fn=(int(x)for x in packed[count+4:count+8]);total=sum(losses)/count
 return dict(binary_Mixed_BCE=total,loss=total,native_gradient_L2=native_l2,group_head_gradient_L2=head_l2,gradient_norm_before_clip=clip_value,gradient_clip=1.,native_steps=native_steps,head_steps=head_steps,questions=n,real_native_trajectories=10*n,views=4*n,mixed_views=2*n,nonmixed_views=2*n,per_question_view_total_weight=1.,micro_questions=micro_questions,native_forward_calls=n//micro_questions,trajectory_loss_computed=False,trajectory_accuracy_computed=False,confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn),mixed_recall=tp/(2*n),nonmixed_recall=tn/(2*n),mixed_balanced_accuracy=(tp+tn)/(4*n),seconds=time.monotonic()-start,batch_token=batch['token'],source_qids_sha256=u.digest(qids),sampler_epoch=sampler.epoch,sampler_state_sha256=u.digest(sampler.state_dict()),GPU_to_CPU_metric_vectors=1,updater_variant='batched_direct_mixed_v1',numerical_bitexact_vs_different_microbatch_claim=False),batch
