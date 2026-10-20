"""Undeployed candidate: distinguish allwrong/Mixed/allcorrect event targets.

All Mixed examples keep the current objective exactly. Negative event targets
use their known homogeneous outcome instead of merging opposite extremes.
The existing `group_bce` remains a binary diagnostic, while the optimized
component is `group_event_nll`; a future trainer must log this distinction.
No entry point or live config imports this candidate.
"""
import torch
import torch.nn.functional as F
from .balanced_loss import balanced_loss


def event_candidate_loss(logits,labels,pair_mask,weight=1.,margin=1.,mixed_fraction=1.):
    result=balanced_loss(logits,labels,pair_mask,weight,margin,mixed_fraction)
    z=logits if logits.dtype==torch.float64 else logits.float()
    y=labels.to(device=z.device,dtype=z.dtype);k=y.sum(1)
    log_allcorrect=F.logsigmoid(z).sum(1);log_allwrong=F.logsigmoid(-z).sum(1)
    log_extreme=torch.logsumexp(torch.stack((log_allcorrect,log_allwrong),dim=1),dim=1)
    correction=torch.where(k==5,log_extreme-log_allcorrect,
                           torch.where(k==0,log_extreme-log_allwrong,torch.zeros_like(log_extreme))).mean()
    result['event_correction']=correction
    result['group_event_nll']=result['group_bce']+correction
    result['loss']=result['loss']+correction;result['total_loss']=result['loss']
    return result
