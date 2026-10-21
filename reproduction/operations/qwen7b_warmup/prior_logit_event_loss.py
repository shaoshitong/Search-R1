"""Prepared-only prior-adjusted coarse losses over unchanged six count logits.

Training: CE(e(z) + log(pi_train/tau), target), with fixed train-only pi.
Inference: UNADJUSTED event-softmax, preserving raw six-count/p_j outputs.
This is a tau-reference balanced decision score, not an originally distributed
calibrated Mixed probability. No runner, production default or score is changed.
"""
import hashlib,json
from pathlib import Path
import torch
from torch.nn import functional as F

MODES=('binary','three_event')
FULL938_PRIVATE=dict(path='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full938-train-prior-decision-diagnostic-v1/train-only-prior-private.json',bytes=2155,sha256='2f76659c49fa6e280b6ef3a3ef528128d9ab29dd41c4e671951e8df5563eae5e')


def desired_priors(mode,*,device=None,dtype=torch.float64):
 if mode not in MODES:raise ValueError('Explicit binary or three_event recipe required')
 return torch.tensor([.5,.5] if mode=='binary' else [.25,.5,.25],device=device,dtype=dtype)


def event_logits(count_logits,mode):
 if mode not in MODES or count_logits.shape[-1]!=6 or not count_logits.is_floating_point() or not torch.isfinite(count_logits).all():raise ValueError('Finite six real count logits and explicit mode required')
 mixed=torch.logsumexp(count_logits[...,1:5],dim=-1)
 if mode=='binary':return torch.stack((torch.logsumexp(count_logits[...,[0,5]],dim=-1),mixed),dim=-1)
 return torch.stack((count_logits[...,0],mixed,count_logits[...,5]),dim=-1)


def targets_from_own_EM(labels,mode):
 if mode not in MODES or labels.shape[-1]!=5 or labels.requires_grad or not ((labels==0)|(labels==1)).all():raise ValueError('Exactly five own binary strict-EM labels required')
 k=labels.to(torch.long).sum(-1)
 if mode=='binary':return ((k>0)&(k<5)).to(torch.long)
 return torch.where(k==0,0,torch.where(k==5,2,1)).to(torch.long)


def coarse_prior_from_counts(counts,mode,*,device=None,dtype=torch.float64):
 if len(counts)!=6 or any(type(n) is not int or n<0 for n in counts) or not sum(counts):raise ValueError('Six nonnegative true training-view counts required')
 if mode=='binary':v=[counts[0]+counts[5],sum(counts[1:5])]
 elif mode=='three_event':v=[counts[0],sum(counts[1:5]),counts[5]]
 else:raise ValueError('Explicit mode required')
 if min(v)<=0:raise ValueError('Observed zero event prior needs a separately specified policy; no silent smoothing')
 return torch.tensor(v,device=device,dtype=dtype)/sum(v)


def frozen_full938_train_prior(mode,*,device=None,dtype=torch.float64):
 """Optional remote-only verified source. Never reads validation or prints pi."""
 p=Path(FULL938_PRIVATE['path']);raw=p.read_bytes()
 if p.resolve()!=p or len(raw)!=FULL938_PRIVATE['bytes'] or hashlib.sha256(raw).hexdigest()!=FULL938_PRIVATE['sha256']:raise ValueError('Frozen train-only prior receipt changed')
 r=json.loads(raw)
 if (r['role']!='FULL_ORIGINAL30016_TRAIN_ONLY_EXACT_FOUR_VIEW_MIXED_PRIOR' or r['questions']!=30016 or r['views']!=120064 or r['view_weight']!=.25 or r['computed_before_validation_predictions_opened'] is not True or sum(r['K0_to5'])!=120064):raise ValueError('Only exact full938 real four-view training prior accepted')
 return coarse_prior_from_counts(r['K0_to5'],mode,device=device,dtype=dtype)


def prior_adjusted_view_losses(count_logits,own_EM,train_priors,mode):
 raw=event_logits(count_logits,mode);target=targets_from_own_EM(own_EM,mode)
 if target.shape!=raw.shape[:-1] or train_priors.requires_grad or train_priors.ndim!=1 or len(train_priors)!=raw.shape[-1] or not torch.isfinite(train_priors).all() or (train_priors<=0).any() or abs(float(train_priors.sum().item())-1)>1e-6:raise ValueError('Detached fixed positive train prior and aligned real groups required')
 pi=train_priors.to(device=raw.device,dtype=raw.dtype);tau=desired_priors(mode,device=raw.device,dtype=raw.dtype)
 adjusted=raw+torch.log(pi/tau)
 return F.cross_entropy(adjusted.reshape(-1,raw.shape[-1]),target.reshape(-1),reduction='none').reshape(target.shape)


def question_weighted_loss(count_logits,own_EM,train_priors,mode,view_weights=None):
 """Q original questions, exactly4 true G5 views, exactly.25 each; mean over Q."""
 if count_logits.ndim!=3 or count_logits.shape[1:]!=(4,6) or own_EM.shape!=count_logits.shape[:-1]+(5,):raise ValueError('Exact [Q,4,6] logits and [Q,4,5] own labels required')
 if not count_logits.shape[0]:raise ValueError('Nonempty original question batch')
 if view_weights is None:view_weights=count_logits.new_full(count_logits.shape[:2],.25)
 if view_weights.shape!=count_logits.shape[:2] or view_weights.requires_grad or not torch.isfinite(view_weights).all() or not (view_weights==.25).all():raise ValueError('Each original question must retain four .25 views, no class reweighting')
 return (prior_adjusted_view_losses(count_logits,own_EM,train_priors,mode)*view_weights).sum(-1).mean()


def inference_outputs(count_logits,mode,trajectory_probabilities=None):
 """Keep unadjusted six-count outputs/p_j, and clearly label changed semantics."""
 raw=event_logits(count_logits,mode);six=torch.softmax(count_logits,dim=-1);events=torch.softmax(raw,dim=-1)
 if trajectory_probabilities is not None and trajectory_probabilities.shape!=count_logits.shape[:-1]+(5,):raise ValueError('Preserved p_j shape differs')
 return dict(unadjusted_six_count_probabilities=six,unadjusted_event_probabilities=events,Mixed_balanced_decision_score=events[...,1],trajectory_probabilities=trajectory_probabilities,score_semantics='tau-reference balanced decision score; not calibrated original-distribution Mixed probability',target_reference_priors=desired_priors(mode,device=raw.device,dtype=raw.dtype))
