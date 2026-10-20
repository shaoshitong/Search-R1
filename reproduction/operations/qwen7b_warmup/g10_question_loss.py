"""G10-derived G5 ablations measured per ORIGINAL question, never per view.

All views of question q carry exact rational total weight1. Every ordinary
component is Q^-1 sum_q sum_v w_qv L(v). Inter-question ranking instead uses
ordered DISTINCT original-question pairs: its denominator is Q*(Q-1), and
view-pair weight is w_qv*w_ru. Ineligible pairs contribute zero. This is NOT
the legacy mean-over-eligible-pairs objective. No G20 support is claimed.
"""
from fractions import Fraction
import math
import torch
import torch.nn.functional as F
from .g10_grouping import _validate, fixed_g5
from .probe_ablation_loss import group_event_scores


def _validate_views(source_groups, views, *, source_size, source_validator):
    """Bind each five-subset/own EM to a validated declared source size.

    The caller must have checked the source rollout/EM/feature manifest SHAs.
    Repeated views are permitted only with exact reallocated rational weights;
    this makes a view's representation multiplicity mathematically irrelevant.
    """
    if source_size not in (10,20):raise ValueError('Only explicitly accepted G10 or G20 sources')
    canonical=source_validator(source_groups,True)
    lookup={g['qid']:g for g in canonical}
    order={g['qid']:i for i,g in enumerate(source_groups)}
    if not views:raise ValueError('At least one genuine G5 training view required')
    mass={qid:Fraction(0) for qid in lookup}
    rows=[];indices=[];weights=[];labels=[];question_indices=[]
    for view in views:
        qid=view['qid']
        if qid not in lookup or view.get('source_group_size')!=source_size:
            raise ValueError('View must belong to an authentic declared G%d question'%source_size)
        own=lookup[qid];ix=view['source_indices']
        if (len(ix)!=5 or any(type(i) is not int or not 0<=i<source_size for i in ix)
                or len(set(ix))!=5 or ix!=sorted(ix)):
            raise ValueError('Five distinct canonical own G10 source indices required')
        if len(view['labels'])!=5 or any(type(y) is not int or y not in (0,1) for y in view['labels']):
            raise ValueError('Five strict own integer EM labels required in every G5 view')
        if (view['trajectory_ids']!=[own['trajectory_ids'][i] for i in ix] or
                view['labels']!=[own['labels'][i] for i in ix]):
            raise ValueError('G5 trajectory IDs or own EM do not match authentic source order')
        k=sum(view['labels']);mixed=0<k<5
        if view.get('K',k)!=k or view.get('true_mixed',mixed)!=mixed:
            raise ValueError('G5 count/Mixed label was not recomputed from own EM')
        n,d=view['weight_numerator'],view['weight_denominator']
        if type(n) is not int or type(d) is not int or n<=0 or d<=0:
            raise ValueError('Positive exact rational view weight required')
        weight=Fraction(n,d);mass[qid]+=weight
        rows.append([order[qid]]*5);indices.append(ix);weights.append(weight)
        labels.append(view['labels']);question_indices.append(order[qid])
    if any(weight!=1 for weight in mass.values()):
        raise ValueError('Every original question must be covered with exact total weight1')
    return dict(rows=rows,indices=indices,weights=weights,labels=labels,
                question_indices=question_indices,questions=len(source_groups),
                original_qids=[g['qid'] for g in source_groups])


def validate_views(source_groups,views):
    return _validate_views(source_groups,views,source_size=10,source_validator=_validate)


def _question_loss(
        logits, source_groups, views, *, source_size,source_validator,trajectory_weight=1.,
        binary_mixed_weight=0.,three_event_weight=0.,within_rank_weight=0.,
        inter_rank_weight=0.,stat_mean_weight=0.,stat_variance_weight=0.,
        distinguishable_pair_mask=None,within_margin=1.,inter_margin=1.):
    """Declared-size logits, authentic source groups, and own G5 views.

    Within ranking: mean valid positive/negative pairs within each G5 view,
    zero for an ineligible view, then the SAME per-question weighted mean.
    Inter ranking: all Mixed-view/nonMixed-view pairs from different original
    qids, product view weights, nominal ordered-question-pair denominator.
    """
    plan=_validate_views(source_groups,views,source_size=source_size,source_validator=source_validator)
    if (not isinstance(logits,torch.Tensor) or not logits.is_floating_point()):
        raise TypeError('Floating authentic source-group logits required')
    if logits.shape!=(plan['questions'],source_size) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite logits[Q,%d] in original source-group order required'%source_size)
    z=logits if logits.dtype==torch.float64 else logits.float()
    weights=dict(trajectory=float(trajectory_weight),binary_mixed=float(binary_mixed_weight),
                 three_event=float(three_event_weight),within_rank=float(within_rank_weight),
                 inter_rank=float(inter_rank_weight),stat_mean=float(stat_mean_weight),
                 stat_variance=float(stat_variance_weight))
    if any(not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('Finite nonnegative explicit component weights required')
    if weights['binary_mixed'] and weights['three_event']:
        raise ValueError('Binary and three-event group losses are alternatives')
    if any(not math.isfinite(float(m)) or float(m)<0 for m in (within_margin,inter_margin)):
        raise ValueError('Finite nonnegative margins required')
    rows=torch.tensor(plan['rows'],dtype=torch.long,device=z.device)
    indices=torch.tensor(plan['indices'],dtype=torch.long,device=z.device)
    view_z=z[rows,indices]
    y=torch.tensor(plan['labels'],dtype=z.dtype,device=z.device)
    w=torch.tensor([float(v) for v in plan['weights']],dtype=z.dtype,device=z.device)
    original=torch.tensor(plan['question_indices'],dtype=torch.long,device=z.device)
    qcount=plan['questions'];zero=z.sum()*0.;p=view_z.sigmoid();k=y.sum(1)
    mixed=(k>0)&(k<5);scores=group_event_scores(view_z)
    per_view=dict(
        trajectory=F.binary_cross_entropy_with_logits(view_z,y,reduction='none').mean(1),
        binary_mixed=F.binary_cross_entropy_with_logits(scores['mixed_event_logit'],mixed.to(z.dtype),reduction='none'),
        three_event=-torch.where(k==0,scores['log_allwrong'],torch.where(k==5,scores['log_allcorrect'],scores['log_mixed'])),
        stat_mean=(p.mean(1)-y.mean(1)).square(),
        stat_variance=(p.var(1,unbiased=False)-y.var(1,unbiased=False)).square())
    if distinguishable_pair_mask is None:
        if weights['within_rank']:raise ValueError('Within ranking needs authentic source input-distinguishability mask')
        valid=torch.zeros((len(views),5,5),dtype=torch.bool,device=z.device)
    else:
        mask=distinguishable_pair_mask
        if not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool or mask.shape!=(qcount,source_size,source_size):
            raise ValueError('Boolean authentic pair mask[Q,%d,%d] required'%(source_size,source_size))
        source_y=torch.tensor([g['labels'] for g in source_groups],device=z.device,dtype=z.dtype)
        own=(source_y[:,:,None]==1)&(source_y[:,None,:]==0)
        mask=mask.to(z.device)
        if bool((mask&~own).any()):raise ValueError('Mask may only select own positive-negative pairs')
        valid=mask[original[:,None,None],indices[:,:,None],indices[:,None,:]]
    pair_counts=valid.sum((1,2));difference=view_z[:,:,None]-view_z[:,None,:]
    per_view['within_rank']=(F.softplus(float(within_margin)-difference)*valid).sum((1,2))/pair_counts.clamp_min(1)
    components={name:(values*w).sum()/qcount for name,values in per_view.items()}
    positive=torch.nonzero(mixed,as_tuple=False).flatten()
    negative=torch.nonzero(~mixed,as_tuple=False).flatten()
    nominal_pairs=qcount*(qcount-1)
    if positive.numel() and negative.numel() and nominal_pairs:
        pair_mass=w[positive,None]*w[negative][None,:]
        different=original[positive,None]!=original[negative][None,:]
        pair_mass=pair_mass*different
        difference=scores['mixed_event_logit'][positive,None]-scores['mixed_event_logit'][negative][None,:]
        components['inter_rank']=(F.softplus(float(inter_margin)-difference)*pair_mass).sum()/nominal_pairs
        eligible_mass=pair_mass.sum()
        inter_pairs=int(different.sum())
    else:
        components['inter_rank']=zero;eligible_mass=zero;inter_pairs=0
    # Exact accounting is computed from canonical rational weights, not floats.
    positive_mass=[Fraction(0) for _ in range(qcount)]
    for position,label,weight in zip(plan['question_indices'],plan['labels'],plan['weights']):
        if 0<sum(label)<5:positive_mass[position]+=weight
    exact_eligible=sum((positive_mass[i]*(1-positive_mass[j]) for i in range(qcount)
                        for j in range(qcount) if i!=j),Fraction(0))
    total=sum((weights[name]*value for name,value in components.items()),zero)
    return dict(loss=total,total_loss=total,components=components,weights=weights,
                mixed_probability=scores['mixed_probability'],mixed_event_logit=scores['mixed_event_logit'],
                mixed_label=mixed.to(z.dtype),view_weights=w,view_original_question_indices=original,
                original_questions=qcount,virtual_views=len(views),
                per_original_question_total_weight=[1.]*qcount,
                weighted_mixed_fraction=float(sum(positive_mass,Fraction(0))/qcount),
                within_eligible_view_weight=float((w*(pair_counts>0)).sum().detach()),
                inter_actual_eligible_view_pairs=inter_pairs,
                inter_eligible_weighted_pair_mass=float(eligible_mass.detach()),
                inter_eligible_weighted_pair_mass_exact=str(exact_eligible),
                inter_nominal_weighted_pair_mass=nominal_pairs,
                denominators=dict(question=qcount,inter_ordered_original_question_pairs=nominal_pairs),
                within_reduction='per-view eligible-pair mean (zero if empty), then original-question weighted mean',
                inter_reduction='product canonical view weights; distinct original qids; divide Q*(Q-1), NOT eligible-pair mean',
                supports_G20=source_size==20,source_group_size=source_size)


def g10_question_loss(
        logits, source_groups, views, *, trajectory_weight=1.,
        binary_mixed_weight=0.,three_event_weight=0.,within_rank_weight=0.,
        inter_rank_weight=0.,stat_mean_weight=0.,stat_variance_weight=0.,
        distinguishable_pair_mask=None,within_margin=1.,inter_margin=1.):
    """Unchanged public G10-only contract and per-question loss reductions."""
    return _question_loss(logits,source_groups,views,source_size=10,source_validator=_validate,
        trajectory_weight=trajectory_weight,binary_mixed_weight=binary_mixed_weight,
        three_event_weight=three_event_weight,within_rank_weight=within_rank_weight,
        inter_rank_weight=inter_rank_weight,stat_mean_weight=stat_mean_weight,
        stat_variance_weight=stat_variance_weight,distinguishable_pair_mask=distinguishable_pair_mask,
        within_margin=within_margin,inter_margin=inter_margin)


def freeze_evaluation_g5(groups,**kwargs):
    """Select/freeze G5 IDs label-blind; join strict EM only AFTER this returns."""
    return fixed_g5(groups,**kwargs)
