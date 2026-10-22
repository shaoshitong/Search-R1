"""Detached source contributions to the existing masked PPO loss, not gradients."""
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss

@torch.no_grad()
def contributions(old, new, advantages, mask, fallback, cliprange, dual_clip, *, check_finite=True):
    mass=mask.sum()
    assert float(mass)>0
    result={}
    for name,value in [('mixed',False),('all_correct_fallback',True)]:
        part=mask*(fallback.bool()==value).unsqueeze(-1)
        fraction=part.sum()/mass
        if bool(part.any()):
            loss,clip,_=compute_policy_loss(old.detach(),new.detach(),advantages.detach(),part,
                                           cliprange,dual_clip=dual_clip,check_finite=check_finite)
            loss_contribution=float(loss*fraction);clip_contribution=float(clip*fraction)
        else:loss_contribution=clip_contribution=0.0
        result.update({f'actor/{name}/pg_loss_contribution':loss_contribution,
                       f'actor/{name}/clip_fraction_contribution':clip_contribution,
                       f'actor/{name}/policy_token_fraction':float(fraction)})
    return result
