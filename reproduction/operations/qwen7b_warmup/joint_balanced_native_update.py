"""Independent balanced/hard real-G10 native block27+trajectory joint update.

Reuses the qualified two-pass native-logit VJP. No postblock cache, no frozen
block, no new classifier, no actor update and no extra scoring forward.
"""
from contextlib import nullcontext
import time
import torch
import torch.nn.functional as F
from . import probe_ablation_train as u
from .g10_grouping import _validate, build_training_views
from .g10_question_loss import g10_question_loss
from .probe_g10_train import _forward

def balanced_native_update(model,ema,optimizer,reader,groups,device,*,
        grouping_kwargs,loss_kwargs,micro_trajectories=5,
        distinguishable_pair_mask=None,ema_decay=.99,max_virtual_views=2048,
        max_inter_view_pairs=1000000,grouping_builder=build_training_views):
    """Same complete-question loss, exact chain rule, bounded activation graphs.

    Pass1 uses the SAME grad-enabled training forward as pass2, but detaches each
    result immediately. Per-micro RNG and nonpersistent native buffers are
    replayed exactly; recomputed logits must be bit-identical before any Adam
    write. Failure never accepts a partial optimizer update. Parameter-gradient
    accumulation roundoff is not claimed bit-identical to joined-graph backward.
    """
    _validate(groups,require_labels=True)
    if type(micro_trajectories) is not int or micro_trajectories<1:
        raise ValueError('Positive explicit micro trajectory budget required')
    if not 0<=float(ema_decay)<1:raise ValueError('EMA decay must be in [0,1)')
    if {'logits','source_groups','views','distinguishable_pair_mask'}&set(loss_kwargs):
        raise ValueError('Recipe cannot inject identities, labels or masks')
    grouping=grouping_builder(groups,role='train',**grouping_kwargs)
    views=grouping['groups'];positive=sum(v['true_mixed'] for v in views)
    if len(views)>max_virtual_views or positive*(len(views)-positive)>max_inter_view_pairs:
        raise ValueError('Explicit virtual-view/loss-pair memory budget exceeded')
    if any(group['lr']!=5e-6 or group['weight_decay']!=0 for group in optimizer.param_groups):
        raise ValueError('Inherited Adam LR/weight decay must remain unchanged before update')
    before=u.steps(optimizer);started=time.monotonic();model.train()
    optimizer.zero_grad(set_to_none=True)
    tids=[tid for g in groups for tid in g['trajectory_ids']]
    micros=[];parts=[];native_replay_bytes=0
    for start in range(0,len(tids),micro_trajectories):
        ids=tids[start:start+micro_trajectories]
        # FeatureReader performs no random sampling. Capture state immediately
        # before actual forward, after any reader work.
        data=reader.get(ids,device)
        native=u.buffers(model)
        native_replay_bytes+=sum(v.numel()*v.element_size() for v in native.values())
        if native_replay_bytes>256*1024**2:
            raise ValueError('Native replay snapshots exceed explicit 256MiB CPU-state budget')
        micros.append((start,ids,u.rng_state(device),native))
        out=_forward(model,data,device)
        if out.numel()!=len(ids) or not bool(torch.isfinite(out).all()):
            raise ValueError('Nonfinite or incomplete actual trajectory logits')
        parts.append(out.detach().clone())
        del out,data  # releases this micro's entire transformer activation graph
    after_forward_rng=u.rng_state(device);after_forward_buffers=u.buffers(model)
    logits=torch.cat(parts).reshape(len(groups),10).requires_grad_(True)
    del parts
    result=g10_question_loss(logits,groups,views,
        distinguishable_pair_mask=distinguishable_pair_mask,**loss_kwargs)
    if not bool(torch.isfinite(result['loss'])):raise ValueError('Nonfinite complete question loss')
    derivative=torch.autograd.grad(result['loss'],logits)[0].detach().reshape(-1)
    if not bool(torch.isfinite(derivative).all()):raise ValueError('Nonfinite dL/dlogits')
    expected=logits.detach().reshape(-1)
    for start,ids,rng,native in micros:
        data=reader.get(ids,device);u.restore_rng(rng,device);u.restore_buffers(model,native)
        out=_forward(model,data,device)
        if not torch.equal(out.detach(),expected[start:start+len(ids)]):
            raise ValueError('Two-pass actual micro logits differ; no Adam update accepted')
        torch.autograd.backward(out,derivative[start:start+len(ids)].to(out.dtype))
        del out,data
    # The second forward is recomputation, not a second RNG/native-buffer clock.
    u.restore_rng(after_forward_rng,device);u.restore_buffers(model,after_forward_buffers)
    parameters=list(model.parameters())
    if not parameters or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('Missing/nonfinite complete-question parameter gradient')
    optimizer.step()
    if u.steps(optimizer)!=before+1:raise ValueError('Adam must advance exactly once')
    with torch.no_grad():
        for target,value in zip(ema.parameters(),parameters):
            target.mul_(float(ema_decay)).add_(value,alpha=1-float(ema_decay))
    pred=result['mixed_probability'].detach()>=.5;truth=result['mixed_label'].detach().bool()
    w=result['view_weights'].detach()
    tp=float(w[pred&truth].sum());fn=float(w[~pred&truth].sum())
    fp=float(w[pred&~truth].sum());tn=float(w[~pred&~truth].sum())
    tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
    per_view=F.binary_cross_entropy_with_logits(result['mixed_event_logit'],result['mixed_label'],reduction='none').detach().cpu().tolist()
    block_grad=sum(float(p.grad.detach().double().square().sum()) for name,p in model.named_parameters() if name.startswith('block.'))**.5
    head_grad=sum(float(p.grad.detach().double().square().sum()) for name,p in model.named_parameters() if name.startswith('output.'))**.5
    if not block_grad>0 or not head_grad>0:raise ValueError('Both native block and trajectory readout need nonzero real gradients')
    return dict(per_view_mixed_BCE=per_view,block_gradient_L2=block_grad,readout_gradient_L2=head_grad,loss=float(result['loss'].detach()),
        components={k:float(v.detach()) for k,v in result['components'].items()},
        grouping=grouping['summary'],original_questions=len(groups),
        unique_real_trajectories=len(tids),virtual_views=len(views),
        view_trajectory_exposures=5*len(views),actual_forward_trajectory_exposures=2*len(tids),
        forward_passes=2,backward_passes=1,micro_trajectories=micro_trajectories,
        maximum_retained_transformer_graph_microbatches=1,
        native_replay_snapshot_bytes=native_replay_bytes,
        gradient_strategy='two_pass_same_micro_RNG_native_buffer_logit_VJP',
        recomputed_logits_bit_identical=True,monolithic_bitexact_claim=False,
        denominators=result['denominators'],
        inter_eligible_weighted_pair_mass_exact=result['inter_eligible_weighted_pair_mass_exact'],
        question_weighted_confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),
        mixed_recall=tpr,nonmixed_recall=tnr,
        mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
        seconds=time.monotonic()-started,actor_updates=0,supports_G20=False)
