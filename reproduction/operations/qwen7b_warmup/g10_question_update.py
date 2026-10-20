"""Runner adapter for authentic G10 question-weighted loss; no job launcher.

Requires a verified reader for all10 distinct trajectories per question. It does
not manufacture the extra5 rollouts/features, reset Adam, or claim G20 support.
No existing or active runner imports this module automatically.
"""
from contextlib import nullcontext
import torch
from .g10_grouping import build_training_views, _validate
from .g10_question_loss import g10_question_loss


def question_weighted_update(model,ema,optimizer,reader,source_groups,device,*,
                             grouping_kwargs,loss_kwargs,micro_trajectories=10,
                             distinguishable_pair_mask=None,ema_decay=.99):
    """One optimizer update over complete original questions, all graphs joined.

    Returned grouping summary binds IDs/own EM and exact weights for audit.
    Source RNG/Adam/checkpoint continuation remains the surrounding runner's job.
    """
    _validate(source_groups,require_labels=True)
    if type(micro_trajectories) is not int or micro_trajectories<1:
        raise ValueError('Positive trajectory microbatch required')
    if not 0<=float(ema_decay)<1:raise ValueError('EMA decay must be in [0,1)')
    forbidden={'logits','source_groups','views','distinguishable_pair_mask'}
    if forbidden&set(loss_kwargs):raise ValueError('Loss configuration may not inject labels/IDs/masks')
    grouping=build_training_views(source_groups,role='train',**grouping_kwargs)
    tids=[tid for g in source_groups for tid in g['trajectory_ids']]
    optimizer.zero_grad(set_to_none=True);model.train();parts=[]
    for start in range(0,len(tids),micro_trajectories):
        data=reader.get(tids[start:start+micro_trajectories],device)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with context:parts.append(model(data).reshape(-1))
    logits=torch.cat(parts).reshape(len(source_groups),10)
    result=g10_question_loss(logits,source_groups,grouping['groups'],
               distinguishable_pair_mask=distinguishable_pair_mask,**loss_kwargs)
    if not bool(torch.isfinite(result['loss'])):raise ValueError('Nonfinite question-weighted loss')
    result['loss'].backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('Nonfinite question-weighted gradient')
    optimizer.step()
    with torch.no_grad():
        for target,value in zip(ema.parameters(),model.parameters()):
            target.mul_(float(ema_decay)).add_(value,alpha=1-float(ema_decay))
    return dict(loss=float(result['loss'].detach()),
                components={k:float(v.detach()) for k,v in result['components'].items()},
                original_questions=result['original_questions'],virtual_views=result['virtual_views'],
                denominators=result['denominators'],
                inter_nominal_weighted_pair_mass=result['inter_nominal_weighted_pair_mass'],
                inter_eligible_weighted_pair_mass=result['inter_eligible_weighted_pair_mass'],
                inter_eligible_weighted_pair_mass_exact=result['inter_eligible_weighted_pair_mass_exact'],
                weighted_mixed_fraction=result['weighted_mixed_fraction'],
                grouping=grouping['summary'],actor_updates=0,supports_G20=False)
