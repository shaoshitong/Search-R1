"""Bounded first-order joint hidden VJP, no model/actor/job initialization."""
from contextlib import nullcontext
import time
import torch
from . import probe_ablation_train as u
from .g10_grouping import _validate,build_training_views
from .direct_hidden_count import count_loss,native_hidden_forward,optimizer_steps,new_head_hashes


def bounded_update(legacy,legacy_ema,legacy_optimizer,head,head_ema,head_optimizer,
        reader,source_groups,device,*,grouping_kwargs,micro_trajectories=5,
        ema_decay=.99,forward_adapter=native_hidden_forward):
    """Only one transformer micrograph, no additional native model forwards.

    Pass1 detaches old logits +128D trajectory embeddings; the small countMLP
    computes the loss and embedding VJP. Pass2 replays the SAME micro RNG/native
    buffers and logits/embeddings, then backpropagates into block/oldtrajectory
    head and new attention/projector. All gradients are finite before either
    Adam writes. Each optimizer advances once; their old/new clocks differ.
    """
    _validate(source_groups,True)
    if micro_trajectories<1 or not 0<=ema_decay<1:raise ValueError('Positive micro and valid EMA required')
    for optimizer in (legacy_optimizer,head_optimizer):
        if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):
            raise ValueError('Declared old/new Adam LR5e-6/weightdecay0 required')
    grouping=build_training_views(source_groups,role='train',**grouping_kwargs)
    if len(grouping['groups'])!=4*len(source_groups) or not grouping['summary']['per_question_total_weight_exactly_one']:
        raise ValueError('Declared four realG5 views/q and total weight1 required')
    old_before=u.steps(legacy_optimizer);head_before=optimizer_steps(head_optimizer)
    started=time.monotonic();legacy.train();head.train()
    legacy_optimizer.zero_grad(set_to_none=True);head_optimizer.zero_grad(set_to_none=True)
    tids=[t for g in source_groups for t in g['trajectory_ids']];micros=[];logits=[];embeddings=[]
    def forward(data):
        context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with context:return forward_adapter(legacy,data,head)
    for start in range(0,len(tids),micro_trajectories):
        ids=tids[start:start+micro_trajectories];data=reader.get(ids,device)
        micros.append((start,ids,u.rng_state(device),u.buffers(legacy)))
        z,h=forward(data)
        if z.numel()!=len(ids) or h.shape!=(len(ids),head.embedding_dimension):raise ValueError('Actual micro output coverage differs')
        if not bool(torch.isfinite(z).all() and torch.isfinite(h).all()):raise ValueError('Nonfinite joint micro output')
        logits.append(z.detach().clone());embeddings.append(h.detach().clone());del z,h,data
    after_rng=u.rng_state(device);after_buffers=u.buffers(legacy)
    z=torch.cat(logits).reshape(len(source_groups),10).requires_grad_(True)
    h=torch.cat(embeddings).reshape(len(source_groups),10,head.embedding_dimension).requires_grad_(True)
    result=count_loss(z,h,source_groups,grouping['groups'],head)
    if not bool(torch.isfinite(result['loss'])):raise ValueError('Nonfinite count objective')
    # Only classifier parameters are connected to this detached embedding graph.
    result['loss'].backward();dz=z.grad.detach().reshape(-1);dh=h.grad.detach().reshape(-1,head.embedding_dimension)
    if not bool(torch.isfinite(dz).all() and torch.isfinite(dh).all()):raise ValueError('Nonfinite logit/embedding VJP')
    expected_z=z.detach().reshape(-1);expected_h=h.detach().reshape(-1,head.embedding_dimension)
    for start,ids,rng,buffers in micros:
        data=reader.get(ids,device);u.restore_rng(rng,device);u.restore_buffers(legacy,buffers)
        out,embedding=forward(data)
        if not torch.equal(out.detach(),expected_z[start:start+len(ids)]) or not torch.equal(embedding.detach(),expected_h[start:start+len(ids)]):
            raise ValueError('Two-pass joint logits/hidden embeddings differ before any Adam write')
        torch.autograd.backward((out,embedding),(dz[start:start+len(ids)].to(out.dtype),dh[start:start+len(ids)].to(embedding.dtype)))
        del out,embedding,data
    u.restore_rng(after_rng,device);u.restore_buffers(legacy,after_buffers)
    parameters=list(legacy.parameters())+list(head.parameters())
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Missing/nonfinite old or new gradient')
    legacy_optimizer.step();head_optimizer.step()
    if u.steps(legacy_optimizer)!=old_before+1 or optimizer_steps(head_optimizer)!=head_before+1:
        raise ValueError('Old/new Adam clocks must each advance once without relabeling')
    with torch.no_grad():
        for target,value in zip(legacy_ema.parameters(),legacy.parameters()):target.mul_(ema_decay).add_(value,alpha=1-ema_decay)
        for target,value in zip(head_ema.parameters(),head.parameters()):target.mul_(ema_decay).add_(value,alpha=1-ema_decay)
    predicted=result['mixed_probability'].detach()>=.5;truth=result['true_mixed'];w=result['view_weights'].detach()
    tp=float(w[predicted&truth].sum());fn=float(w[~predicted&truth].sum());tn=float(w[~predicted&~truth].sum());fp=float(w[predicted&~truth].sum())
    return dict(loss=float(result['loss'].detach()),trajectory_BCE=float(result['trajectory_loss'].detach()),count_NLL=float(result['count_NLL'].detach()),
        legacy_Adam_steps=u.steps(legacy_optimizer),new_head_Adam_steps=optimizer_steps(head_optimizer),grouping=grouping['summary'],
        original_questions=len(source_groups),real_trajectories=len(tids),virtual_G5_views=len(grouping['groups']),
        original_question_total_weight=1.,actual_forward_trajectory_exposures=2*len(tids),
        maximum_retained_transformer_graphs=1,gradient_strategy='two_pass_same_micro_logit_and_hidden_embedding_VJP',
        joint_micro_replay_exact=True,actor_updates=0,old_optimizer_reset=False,new_head_optimizer_initialized_once=True,
        mixed_recall=tp/(tp+fn) if tp+fn else None,nonmixed_recall=tn/(tn+fp) if tn+fp else None,
        mixed_balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None,
        seconds=time.monotonic()-started)
