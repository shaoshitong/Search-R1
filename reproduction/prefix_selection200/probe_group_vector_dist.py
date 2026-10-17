"""G5 vector relation + mean-correctness KL + finite-G diversity KL.

No individual-trajectory BCE. All synthetic targets come from one mixed G5
vector. Constant targets have no correlation term; both group KL terms remain.
"""
import math
import random
import torch
import torch.nn.functional as F


def diversity_target(probabilities):
    r=probabilities.float();offdiag=~torch.eye(5,dtype=torch.bool,device=r.device)
    return (4/25)*(r.unsqueeze(2)*(1-r.unsqueeze(1))*offdiag).sum((1,2))


def group_logits(logits):
    z=logits.float()
    lp,ln=F.logsigmoid(z),F.logsigmoid(-z)
    mean_logit=torch.logsumexp(lp,dim=1)-torch.logsumexp(ln,dim=1)
    bits=((torch.arange(32,device=z.device).unsqueeze(1)>>torch.arange(5,device=z.device).unsqueeze(0))&1).bool()
    pattern_logp=torch.where(bits.unsqueeze(0),lp.unsqueeze(1),ln.unsqueeze(1)).sum(-1)
    k=bits.sum(-1).float();d=4*(k/5)*(1-k/5)
    log_d=torch.logsumexp(pattern_logp+d.log().unsqueeze(0),dim=1)
    log_not_d=torch.logsumexp(pattern_logp+torch.log1p(-d).unsqueeze(0),dim=1)
    return mean_logit,log_d-log_not_d


def bernoulli_kl(logit,target):
    target=target.float()
    entropy=-(torch.xlogy(target,target)+torch.xlogy(1-target,1-target))
    return F.binary_cross_entropy_with_logits(logit,target,reduction='none')-entropy


def vector_dist_loss(logits,target,dist_weight=1,mean_weight=1,diversity_weight=5):
    p=torch.sigmoid(logits.float());r=target.float()
    pc,rc=p-p.mean(-1,keepdim=True),r-r.mean(-1,keepdim=True)
    vp,vr=pc.square().sum(-1),rc.square().sum(-1)
    correlation=(pc*rc).sum(-1)/torch.sqrt((vp+1e-6)*(vr+1e-6))
    active=(vr>1e-12).float()
    relation=((1-correlation)*active).mean()
    am,ad=group_logits(logits)
    mean_kl=bernoulli_kl(am,r.mean(-1)).mean()
    diversity_kl=bernoulli_kl(ad,diversity_target(r)).mean()
    total=dist_weight*relation+diversity_weight*diversity_kl
    if mean_weight:total=total+mean_weight*mean_kl
    return dict(loss=total,total_loss=total,dist_loss=relation,mean_correctness_kl=mean_kl,
        diversity_kl=diversity_kl,predicted_mean_correctness=torch.sigmoid(am),predicted_diversity=torch.sigmoid(ad))


def dominant_hidden_mixup(batch,labels,pairs,seed,alpha,with_pair_lambdas=False):
    from .actor_hidden import WIDTH,DIM
    features=batch['features'].reshape(len(labels),5,WIDTH,DIM);mixed=features.clone();target=labels.clone()
    rng=random.Random(seed);coefficients=[]
    for a,b in pairs:
        u=rng.betavariate(alpha,alpha);lam=max(u,1-u)
        if lam==.5:lam=math.nextafter(.5,1.)
        coefficients.append(lam)
        mixed[a]=(lam*features[a].float()+(1-lam)*features[b].float()).to(features.dtype)
        mixed[b]=(lam*features[b].float()+(1-lam)*features[a].float()).to(features.dtype)
        target[a]=lam*labels[a]+(1-lam)*labels[b]
        target[b]=lam*labels[b]+(1-lam)*labels[a]
    augmentation=dict(augmented_questions=2*len(pairs),pairs=len(pairs),lambda_sum=sum(coefficients),
        dominant_lambda_min=min(coefficients)if coefficients else None,dominant_lambda_max=max(coefficients)if coefficients else None)
    if with_pair_lambdas:augmentation['pair_lambdas']=coefficients
    return dict(batch,features=mixed.reshape_as(batch['features'])),target,augmentation


def evaluation_metrics(rows,dist_weight=1,mean_weight=1,diversity_weight=5):
    # CPU only; real unaugmented G5 vectors. No model forward or GPU gate.
    z=torch.tensor([r['trajectory_logits']for r in rows],dtype=torch.float32)
    targets=torch.tensor([r['strict_em']for r in rows],dtype=torch.float32)
    values=vector_dist_loss(z,targets,dist_weight,mean_weight,diversity_weight)
    predicted=values['predicted_diversity'].tolist();true=diversity_target(targets).tolist()
    p=torch.sigmoid(z);pc=p-p.mean(-1,keepdim=True);rc=targets-targets.mean(-1,keepdim=True)
    vr=rc.square().sum(-1);active=vr>1e-12;correlation=(pc*rc).sum(-1)/torch.sqrt((pc.square().sum(-1)+1e-6)*(vr+1e-6))
    tp=fn=tn=fp=0
    for row,d,t in zip(rows,predicted,true):
        label=0<row['correct_count']<5;positive=d>=.5
        tp+=int(positive and label);fn+=int(not positive and label);tn+=int(not positive and not label);fp+=int(positive and not label)
        row.update(diversity_score=d,diversity_target=t,predicted_mean_correctness=sum(row['probabilities'])/5,
            score_semantics='old_mixed_probability_is_event_q;diversity_score_is_expected_4K(5-K)/25')
    return dict(**{k:float(values[k])for k in('dist_loss','mean_correctness_kl','diversity_kl','total_loss')},
        diversity_mae=sum(abs(a-b)for a,b in zip(predicted,true))/len(true),
        diversity_mixed_accuracy=(tp+tn)/len(rows),diversity_mixed_balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,
        diversity_tp=tp,diversity_fn=fn,diversity_tn=tn,diversity_fp=fp,
        vector_mae=float((p-targets).abs().mean()),vector_pearson_mixed_groups=float(correlation[active].mean()),
        group_mean_mae=float((p.mean(-1)-targets.mean(-1)).abs().mean()),
        objective=('G5_Pearson_DIST_plus_mean_KL_plus_finiteG_diversity_KL'if mean_weight else'G5_Pearson_DIST_plus_finiteG_diversity_KL'),
        group_mean_KL_in_training_loss=bool(mean_weight),group_mean_KL_weight=mean_weight,
        trajectory_BCE_in_training_loss=False,binary_mixed_BCE_in_training_loss=False,
        vector_relation_constant_target_term=0,diversity_threshold=.5,
        legacy_trajectory_and_binary_mixed_BCE_are_diagnostics_only=True)
