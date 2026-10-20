"""Joint BCE plus correctly normalized distinguishable within-Mixed ranking.

The global batch Mixed fraction is shared with every rank/microbatch. This
keeps ranking a mean over Mixed questions under unequal local class counts.
"""
import math
import torch
import torch.nn.functional as F
from reproduction.prefix20_group_joint100.loss import joint_loss


def balanced_loss(logits,labels,pair_mask,weight=1.,margin=1.,mixed_fraction=1.):
    z=logits if logits.dtype==torch.float64 else logits.float()
    y=labels.to(device=z.device,dtype=z.dtype)
    if y.shape!=z.shape or not bool(((y==0)|(y==1)).all()):raise ValueError('Own binary EM labels required')
    fraction=float(mixed_fraction)
    if not math.isfinite(fraction) or not 0<fraction<=1:raise ValueError('Global Mixed fraction must be in (0,1]')
    if not math.isfinite(float(weight)) or float(weight)<0:raise ValueError('Finite nonnegative ranking weight required')
    expected=(y[:,:,None]==1)&(y[:,None,:]==0)
    valid=pair_mask.to(device=z.device,dtype=torch.bool)
    if valid.shape!=expected.shape or bool((valid&~expected).any()):raise ValueError('Ranking requires same-question own positive-negative pairs')
    diff=z[:,:,None]-z[:,None,:];counts=valid.sum((1,2))
    rank_per_group=(F.softplus(float(margin)-diff)*valid).sum((1,2))/counts.clamp_min(1)
    result=joint_loss(z,y,group_weight=1.)
    result['pairwise_loss']=rank_per_group.mean()/fraction
    result['loss']=result['loss']+float(weight)*result['pairwise_loss'];result['total_loss']=result['loss']
    result['eligible_pairs']=counts.sum();result['pairwise_correct']=((diff>0)&valid).sum();result['eligible_groups']=(counts>0).sum()
    result['mean_p_range']=(z.sigmoid().amax(1)-z.sigmoid().amin(1)).mean()
    return result
