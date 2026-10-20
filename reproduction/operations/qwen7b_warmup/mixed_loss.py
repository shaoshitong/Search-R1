"""Label-directed separation for distinguishable positive/negative G5 pairs."""
import torch
import torch.nn.functional as F
from reproduction.prefix20_group_joint100.loss import joint_loss


def mixed_only_loss(logits, labels, pair_mask, weight=.1, margin=1.):
    z=logits if logits.dtype==torch.float64 else logits.float()
    y=labels.to(device=z.device,dtype=z.dtype)
    if not bool(((y.sum(1)>0)&(y.sum(1)<5)).all()):
        raise ValueError('Mixed-only fitting requires genuine K=1..4 own-EM groups')
    expected=(y[:,:,None]==1)&(y[:,None,:]==0)
    valid=pair_mask.to(device=z.device,dtype=torch.bool)
    if valid.shape!=expected.shape or bool((valid&~expected).any()):
        raise ValueError('Ranking mask must contain only own positive-negative pairs')
    diff=z[:,:,None]-z[:,None,:]
    counts=valid.sum((1,2));denom=counts.clamp_min(1)
    rank_per_group=(F.softplus(float(margin)-diff)*valid).sum((1,2))/denom
    result=joint_loss(z,y,group_weight=1.)
    result['pairwise_loss']=rank_per_group.mean()
    result['loss']=result['loss']+float(weight)*result['pairwise_loss']
    result['total_loss']=result['loss']
    result['eligible_pairs']=counts.sum()
    result['pairwise_correct']=((diff>0)&valid).sum()
    result['eligible_groups']=(counts>0).sum()
    result['mean_p_range']=(torch.sigmoid(z).amax(1)-torch.sigmoid(z).amin(1)).mean()
    return result
