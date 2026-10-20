"""New train-only G10 grouping: exact ORIGINAL-question weighted class balance.

This is independent of the published89102db runner. It does not change that
runner, submit jobs, generate trajectories, or touch validation/final pools.
Each original question has four UNIQUE authentic G5 views and total weight1.
For Q questions, M heterogeneous G10 questions can contribute Mixed mass at
mostM, so exact1:1 needs2M>=Q. Homogeneous questions can NEVER create Mixed.
"""
from fractions import Fraction
from .g10_grouping import (_validate, _partitions, _rng, _hash,
    _capacities,build_training_views as original_training_views)

MODE='global-weighted-balanced'


def equal_view_quotas(own,seed,epoch,on_unbalance):
    """Four real views/Q, all weights1/4, exact positive-count target2Q.

    Quota_i is bounded by the actual unique Mixed/nonMixed combination
    capacities. Starting at all lower bounds, a seeded round-robin fills only
    remaining eligible slots. No EM label is edited or invented.
    """
    q=len(own);capacities=[_capacities(g) for g in own]
    lower=[max(0,4-negative) for positive,negative in capacities]
    upper=[min(4,positive) for positive,negative in capacities]
    assert all(lo<=hi for lo,hi in zip(lower,upper))
    requested=2*q;feasible=sum(lower)<=requested<=sum(upper)
    if not feasible and on_unbalance=='raise':
        raise ValueError('Exact weighted1:1 and view-count1:1 impossible: feasible Mixed-count interval%d..%d, target%d'%
                         (sum(lower),sum(upper),requested))
    # The only infeasibility for authentic G10/V4 is insufficient Mixed
    # capacity. Report its maximum, retaining all four views of EVERY question.
    if not feasible:assert requested>sum(upper)
    target=requested if feasible else sum(upper)
    quota=list(lower);order=list(range(q));_rng(seed,epoch,['equal-view-quota',[g['qid'] for g in own]]).shuffle(order)
    left=target-sum(quota)
    while left:
        before=left
        for index in order:
            if quota[index]<upper[index]:quota[index]+=1;left-=1
            if not left:break
        assert left<before
    return quota,dict(rule='actual-capacity-bounded-seeded-round-robin',requested_Mixed_views=requested,
        actual_Mixed_views=target,minimum_Mixed_views=sum(lower),maximum_Mixed_views=sum(upper),
        equal_class_count_feasible=feasible,quota_by_question=[dict(qid=g['qid'],minimum=lo,maximum=hi,selected=k)
            for g,lo,hi,k in zip(own,lower,upper,quota)])


def build_training_views(groups,mode=MODE,views_per_question=4,seed=1729,
                         epoch=0,on_unbalance='raise',role='train',weighting='equal-view-quota'):
    """Return exact question-weighted G5 views compatible with g10_question_loss.

    Default equal-view-quota gives EVERY view weight1/4. It selects per-question
    Mixed counts in [max(0,4-negative_capacity),min(4,positive_capacity)], with
    totalMixed=2Q. Thus both view counts AND original-question class masses
    are1:1, with four unique real views and totalweight1 for every question.

    Optional heterogeneous-mass-rational retains the earlier definition:
    for H=Q-M homogeneous questions, massMixed=0. When M>Q/2, every
    heterogeneous question contributes t=Q/(2M), using two Mixed views of
    weight t/2 and two nonMixed views of weight(1-t)/2. At M=Q/2 it uses four
    Mixed views of weight1/4, avoiding zero-weight views. All heterogeneousG10
    have>=126 distinct Mixed subsets and>=2 nonMixed subsets, so this4-view
    construction is feasible exactly when2M>=Q. Count balance is not claimed.

    If2M<Q, report mode returns the maximum reachable massM/Q, with four
    Mixed views per heterogeneous question and four negative views per
    homogeneous question; raise mode fails before returning any views.
    """
    if role!='train':raise ValueError('Weighted balancing is training-only; use label-blind fixed validation IDs')
    if mode not in ('random',MODE):raise ValueError('Only random or global-weighted-balanced supported here')
    if views_per_question!=4 or type(views_per_question) is not int:
        raise ValueError('This qualification comparison fixes exactly4 unique views per original question')
    if on_unbalance not in ('raise','report'):raise ValueError('Explicit raise/report required')
    if weighting not in ('equal-view-quota','heterogeneous-mass-rational'):
        raise ValueError('Explicit equal-view-quota or heterogeneous-mass-rational weighting required')
    if mode=='random':
        return original_training_views(groups,mode='random',views_per_question=4,
            seed=seed,epoch=epoch,on_unbalance=on_unbalance,role=role)
    own=_validate(groups,require_labels=True);q=len(own)
    _rng(seed,epoch,'validation')
    m=sum(0<sum(g['labels'])<10 for g in own);possible=2*m>=q
    if not possible and on_unbalance=='raise':
        raise ValueError('Exact weighted1:1 impossible: M%d/Q%d, maximum Mixed fraction%s'%(m,q,Fraction(m,q)))
    t=Fraction(q,2*m) if possible else Fraction(1)
    quotas,quota_summary=equal_view_quotas(own,seed,epoch,on_unbalance) if weighting=='equal-view-quota' else (None,None)
    views=[];per_question=[]
    for ordinal,group in enumerate(own):
        rng=_rng(seed,epoch,[MODE,group['qid']]);positive,negative=_partitions(group)
        if quotas is not None:
            quota=quotas[ordinal]
            chosen=[(indices,Fraction(1,4)) for indices in
                rng.sample(positive,quota)+rng.sample(negative,4-quota)]
        elif not positive:
            chosen=[(indices,Fraction(1,4)) for indices in rng.sample(negative,4)]
        elif t==1:
            chosen=[(indices,Fraction(1,4)) for indices in rng.sample(positive,4)]
        else:
            chosen=([(indices,t/2) for indices in rng.sample(positive,2)]+
                    [(indices,(1-t)/2) for indices in rng.sample(negative,2)])
        rng.shuffle(chosen);mixed_mass=Fraction()
        for indices,weight in chosen:
            labels=[group['labels'][i] for i in indices];k=sum(labels);mixed=0<k<5
            assert weight>0
            if mixed:mixed_mass+=weight
            views.append(dict(qid=group['qid'],source_indices=list(indices),
                trajectory_ids=[group['trajectory_ids'][i] for i in indices],labels=labels,
                K=k,true_mixed=mixed,weight_numerator=weight.numerator,
                weight_denominator=weight.denominator,weight=float(weight),source_group_size=10))
        per_question.append(dict(qid=group['qid'],views=4,
            mixed_views=sum(0<sum(group['labels'][i] for i in ix)<5 for ix,_ in chosen),
            weighted_mixed_mass=str(mixed_mass),weighted_nonmixed_mass=str(1-mixed_mass),
            total_weight_numerator=1,total_weight_denominator=1))
    _rng(seed,epoch,[MODE,'view-order']).shuffle(views)
    mass=sum((Fraction(v['weight_numerator'],v['weight_denominator']) for v in views if v['true_mixed']),Fraction())
    by={g['qid']:[] for g in own}
    for v in views:by[v['qid']].append(v)
    assert all(len(rows)==len({tuple(v['source_indices']) for v in rows})==4 and
        sum((Fraction(v['weight_numerator'],v['weight_denominator']) for v in rows),Fraction())==1
        for rows in by.values())
    assert mass==(Fraction(q,2) if possible else m)
    count=sum(v['true_mixed'] for v in views)
    summary=dict(mode=MODE,seed=seed,epoch=epoch,questions=q,covered_questions=q,
        weighting=weighting,view_weight_always_one_quarter=weighting=='equal-view-quota',
        quota_allocation=quota_summary,
        requested_views_per_question=4,views=4*q,mixed_views=count,nonmixed_views=4*q-count,
        count_balanced=2*count==4*q,count_mixed_fraction=count/(4*q),
        weighted_mixed_mass=str(mass),weighted_nonmixed_mass=str(q-mass),
        weighted_mixed_fraction=float(mass/q),weighted_class_mass_balanced=mass==q-mass,
        exact_balance_feasible=possible,heterogeneous_questions=m,homogeneous_questions=q-m,
        maximum_attainable_weighted_Mixed_fraction=str(Fraction(m,q)),
        heterogeneous_question_Mixed_mass=str(t) if m and weighting=='heterogeneous-mass-rational' else None,
        per_question_total_weight_exactly_one=True,exactly_four_unique_views_per_question=True,
        rational_weights_are_authoritative=True,homogeneous_questions_can_create_Mixed=False,
        all_qids_covered=True,labels_recomputed_from_own_subset=True,synthetic_labels=False,
        source_G10_K_histogram=[sum(sum(g['labels'])==k for g in own) for k in range(11)],
        qid_sha256=_hash([g['qid'] for g in own]),source_groups_sha256=_hash(own),
        selected_views_sha256=_hash(views),training_only=True,supports_G20=False,
        loss_reduction='mean_over_questions(sum_over_views(exact_weight*whole_G5_loss))',
        impossibilities=[] if possible else [dict(reason='heterogeneous_question_weight_capacity_below_Q/2',
            requested_weighted_Mixed_mass=str(Fraction(q,2)),maximum_weighted_Mixed_mass=str(Fraction(m)),
            maximum_weighted_Mixed_fraction=str(Fraction(m,q)),all_questions_preserved=True)])
    return dict(groups=views,per_question=per_question,summary=summary)


def build_weighted_batch_plan(groups,batch_questions=16,seed=1729,epoch=0,
                              on_unbalance='raise',role='train'):
    """One fixed train-only K10-stratified plan, shared by random AND balanced.

    Every question appears exactly once. Homogeneous counts are spread as
    evenly as possible, so everybatch has H<=Q/2 iff the entirecohort permits
    it. This planner DOES consult TRAIN labels/K10; it is never an evaluation
    selector. A reported infeasible plan preservesall qids and exposes each
    batch's maximum attainable weighted Mixed fraction.
    """
    if role!='train':raise ValueError('K10 planning is training-only')
    if type(batch_questions) is not int or not 2<=batch_questions<=50 or batch_questions%2:
        raise ValueError('An explicit even original-question batch2..50 is required')
    if on_unbalance not in ('raise','report'):raise ValueError('Explicit raise/report required')
    own=_validate(groups,require_labels=True);q=len(own)
    if q%batch_questions:raise ValueError('Full equal-size original-question batches required')
    n=q//batch_questions;homogeneous=[g for g in own if sum(g['labels']) in (0,10)]
    heterogeneous=[g for g in own if 0<sum(g['labels'])<10]
    feasible=2*len(heterogeneous)>=q
    if not feasible and on_unbalance=='raise':
        raise ValueError('All-batch exact weighted1:1 impossible; maximum Mixed fraction%s'%Fraction(len(heterogeneous),q))
    rng=_rng(seed,epoch,['weighted-balanced-batch-plan',[g['qid'] for g in own]])
    rng.shuffle(homogeneous);rng.shuffle(heterogeneous)
    order=list(range(n));rng.shuffle(order);batches=[[] for _ in range(n)];hcounts=[0]*n
    for i,g in enumerate(homogeneous):
        index=order[i%n];batches[index].append(g['qid']);hcounts[index]+=1
    offset=0
    for index in order:
        need=batch_questions-len(batches[index]);batches[index].extend(g['qid'] for g in heterogeneous[offset:offset+need]);offset+=need
        rng.shuffle(batches[index])
    assert offset==len(heterogeneous)
    cohort=[qid for batch in batches for qid in batch]
    assert len(cohort)==len(set(cohort))==q and set(cohort)=={g['qid'] for g in own}
    assert all(len(batch)==batch_questions for batch in batches)
    if feasible:assert max(hcounts)<=batch_questions//2
    return dict(cohort_qids=cohort,batches=batches,batch_epochs=[epoch]*n,
        original_questions=q,global_original_questions=batch_questions,updates=n,
        ordered_qid_sha256=_hash(cohort),source_groups_sha256=_hash(own),
        plan_rule='training_K10_stratified_even_homogeneous_allocation_then_same_plan_for_random_and_weighted',
        labels_used_for_plan=True,label_scope='training_only',validation_or_final_labels_read=False,
        supports_G20=False,exact_weighted_balance_feasible_in_every_batch=feasible,
        maximum_global_weighted_Mixed_fraction=str(Fraction(len(heterogeneous),q)),
        homogeneous_questions=len(homogeneous),heterogeneous_questions=len(heterogeneous),
        batch_homogeneous_counts=hcounts,
        per_batch_maximum_weighted_Mixed_fraction=[str(Fraction(batch_questions-h,batch_questions)) for h in hcounts],
        seed=seed,epoch=epoch,smaller_question_budget_acknowledged=batch_questions!=51,
        same_plan_required_for_random_and_balanced=True,all_original_questions_preserved=True)
