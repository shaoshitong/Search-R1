"""Exact uniform C(G,5) trajectory-BCE + three-event NLL for real G10/G20.

All views of one authentic question have weight1/C(G,5), total1. Only a
[Q,G] first-order VJP is retained; one bounded combinatorial chunk at a time
uses the shared stable group_event_scores primitive. No native probe forward,
NxN view-pair allocation, per-view Python identity metadata, or sampled labels.
"""
import itertools,math
from fractions import Fraction
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from .probe_ablation_loss import group_event_scores
from .g10_grouping import _validate as validate10
from .g20_grouping import validate as validate20

SUPPORTED_COMPONENTS=('trajectory','three_event')


def choose(n,k):
    return 0 if k<0 or k>n else math.factorial(n)//(math.factorial(k)*math.factorial(n-k))


def validate_source(groups,size=None):
    if not groups:raise ValueError('Authentic nonempty question groups required')
    actual=len(groups[0]['trajectory_ids'])
    if size is not None and actual!=size:raise ValueError('Source real-G size differs')
    if actual==10:validate10(groups,True)
    elif actual==20:validate20(groups,True)
    else:raise ValueError('Exactly10 or20 distinct own real trajectories required')
    return actual


def mixed_combinatorial_mass(labels):
    g=len(labels);k=sum(labels);total=choose(g,5)
    negative=choose(k,5)+choose(g-k,5)
    return dict(K=k,allwrong=choose(g-k,5),allcorrect=choose(k,5),Mixed=total-negative,total=total,
                Mixed_weight=str(Fraction(total-negative,total)))


def weights(trajectory_weight=1.,three_event_weight=1.,**unsupported):
    for key,value in unsupported.items():
        if key not in ('binary_mixed_weight','within_rank_weight','inter_rank_weight','stat_mean_weight','stat_variance_weight') or not math.isfinite(float(value)) or float(value)!=0:
            raise ValueError('Exact uniform supports only trajectory BCE and three-event NLL; unsupported nonzero or unknown '+key)
    a,b=float(trajectory_weight),float(three_event_weight)
    if any(not math.isfinite(v) or v<0 for v in (a,b)) or not a+b:raise ValueError('Finite nonnegative component weights with a nonzero objective required')
    return a,b


class _ExactFirstOrderVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,source,value,gradient):
        ctx.save_for_backward(gradient);return value.clone()
    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        gradient,=ctx.saved_tensors;return upstream*gradient,None,None


def exact_uniform_g5_loss(logits,source_groups,*,trajectory_weight=1.,three_event_weight=1.,
        combination_chunk=64,binary_mixed_weight=0.,within_rank_weight=0.,inter_rank_weight=0.,
        stat_mean_weight=0.,stat_variance_weight=0.,distinguishable_pair_mask=None):
    """Exactly the finite all-five-subset objective, first derivatives only.

    The uniform trajectory term analytically equals BCE over the source G.
    Chunk event graphs are differentiated immediately and discarded. Returned
    component values/confusions are detached statistics; ``loss`` alone is the
    custom first-order VJP used by Adam. No higher-order derivative claim.
    """
    g=validate_source(source_groups);a,b=weights(trajectory_weight,three_event_weight,
        binary_mixed_weight=binary_mixed_weight,within_rank_weight=within_rank_weight,
        inter_rank_weight=inter_rank_weight,stat_mean_weight=stat_mean_weight,
        stat_variance_weight=stat_variance_weight)
    if distinguishable_pair_mask is not None:raise ValueError('Ranking masks are unsupported; no ranking objective is constructed')
    if not isinstance(logits,torch.Tensor) or not logits.is_floating_point() or tuple(logits.shape)!=(len(source_groups),g) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite own logits[Q,G] required')
    if type(combination_chunk) is not int or not 1<=combination_chunk<=256:raise ValueError('Explicit combination_chunk1..256 required')
    q=len(source_groups);total=choose(g,5);dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32
    z=logits.detach().to(dtype).requires_grad_(True);labels=torch.tensor([x['labels'] for x in source_groups],dtype=dtype,device=z.device)
    with torch.no_grad():
        trajectory=F.binary_cross_entropy_with_logits(z,labels);gradient=a*(z.sigmoid()-labels)/(q*g);event=torch.zeros((),dtype=dtype,device=z.device)
    count_tensor=torch.zeros(4,dtype=torch.long,device=z.device);seen=0;maximum=0;combinations=itertools.combinations(range(g),5)
    while True:
        chunk=list(itertools.islice(combinations,combination_chunk))
        if not chunk:break
        n=len(chunk);seen+=n;maximum=max(maximum,q*n);index=torch.tensor(chunk,dtype=torch.long,device=z.device)
        with torch.enable_grad():
            own=z[:,index].reshape(q*n,5);y=labels[:,index].reshape(q*n,5);k=y.sum(1);scores=group_event_scores(own)
            nll=-torch.where(k==0,scores['log_allwrong'],torch.where(k==5,scores['log_allcorrect'],scores['log_mixed'])).sum()/(q*total)
            if b:
                chunk_gradient=torch.autograd.grad(nll,z,create_graph=False,retain_graph=False)[0]
                with torch.no_grad():gradient.add_(chunk_gradient,alpha=b)
            with torch.no_grad():
                event.add_(nll.detach());truth=(k>0)&(k<5);pred=scores['mixed_probability']>=.5
                count_tensor.add_(torch.stack([(pred&truth).sum(),(~pred&truth).sum(),(pred&~truth).sum(),(~pred&~truth).sum()]))
        del own,y,k,scores,nll,index
    if seen!=total:raise AssertionError('Incomplete exact all-combinations coverage')
    with torch.no_grad():
        value=a*trajectory+b*event
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):raise ValueError('Nonfinite all-view exact value or gradient')
    loss=_ExactFirstOrderVJP.apply(logits,value,gradient.to(logits.dtype))
    counts=dict(zip(('tp','fn','fp','tn'),count_tensor.cpu().tolist()))
    capacities=[mixed_combinatorial_mass(x['labels']) for x in source_groups];exact_mixed=sum(x['Mixed'] for x in capacities)
    if counts['tp']+counts['fn']!=exact_mixed:raise AssertionError('Enumerated Mixed labels differ from combinatorial source truth')
    return dict(loss=loss,total_loss=loss,components=dict(trajectory=trajectory.detach(),three_event=event.detach()),
        weights=dict(trajectory=a,three_event=b),original_questions=q,source_group_size=g,virtual_views=q*total,
        combinations_per_question=total,combination_chunk=combination_chunk,maximum_resident_G5_rows=maximum,
        maximum_resident_pattern_rows=32*maximum,first_order_only=True,retained_loss_gradient_elements=q*g,
        all_combinations_exactly_once=True,per_original_question_total_weight_exactly_one=True,view_weight='1/%d'%total,
        weighted_Mixed_fraction=exact_mixed/(q*total),source_combinatorial_class_mass=capacities,
        confusion_integer_all_views=counts,question_weighted_confusion={k:v/total for k,v in counts.items()},
        denominators=dict(question=q,source_trajectory=q*g,event_view=q*total),
        trajectory_reduction='uniform all-G5 trajectory BCE analytically equals mean over own source G',
        event_reduction='1/Q sum_question 1/C(G,5) sum_all_real_five_subsets three-event NLL',
        inter_view_pairs_allocated=0,per_view_python_identity_metadata_allocated=0,
        native_model_forwards_performed_by_loss=0,actor_updates=0,supports_G20=g==20)
