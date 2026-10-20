"""Authentic G10 -> G5 training views, with exact per-question total weight.

This module consumes identifiers and already verified own-trajectory binary EM,
never model tensors/features. It validates their positional binding structure;
the caller must first verify the generating rollout/EM manifests and SHA.
No sampling operation changes a label. Label-aware training view selection is
deliberately separate from label-blind fixed validation/final selection.
"""
import bisect
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
import random


MODES = ('random', 'per-question-balanced', 'global-balanced')
COMBINATIONS = tuple(itertools.combinations(range(10), 5))
assert len(COMBINATIONS) == 252


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _rng(seed, epoch, scope):
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError('Integer seed and nonnegative integer epoch required')
    return random.Random(int(_hash(['authentic-g10-v1', seed, epoch, scope]), 16))


def _validate(groups, require_labels):
    if not groups:
        raise ValueError('At least one G10 question required')
    seen_qids, seen_tids, result = set(), set(), []
    for group in groups:
        qid, tids = group['qid'], group['trajectory_ids']
        if not isinstance(qid, str) or not qid or qid in seen_qids:
            raise ValueError('Distinct nonempty string question IDs required')
        if len(tids) != 10 or len(set(tids)) != 10:
            raise ValueError('Exactly10 distinct trajectory IDs required for '+qid)
        if any(not isinstance(tid, str) or not tid for tid in tids):
            raise ValueError('Nonempty string trajectory IDs required')
        if seen_tids.intersection(tids):
            raise ValueError('Trajectory ID reused across questions')
        own = dict(qid=qid, trajectory_ids=list(tids))
        if require_labels:
            labels = group['labels']
            if len(labels) != 10 or any(type(y) is not int or y not in (0, 1) for y in labels):
                raise ValueError('Exactly10 own binary integer EM labels required for '+qid)
            own['labels'] = list(labels)
        seen_qids.add(qid)
        seen_tids.update(tids)
        result.append(own)
    return sorted(result, key=lambda item: item['qid'])


def _partitions(group):
    mixed, nonmixed = [], []
    for indices in COMBINATIONS:
        k = sum(group['labels'][i] for i in indices)
        (mixed if 0 < k < 5 else nonmixed).append(indices)
    assert len(mixed)+len(nonmixed) == 252
    return mixed, nonmixed


def _choose(n, k=5):
    return 0 if n < k else math.factorial(n)//(math.factorial(k)*math.factorial(n-k))


def _capacities(group):
    k = sum(group['labels'])
    negative = _choose(k)+_choose(10-k)
    return 252-negative, negative


def _allocate_remaining(capacities, selected, target, rng):
    """Uniform slots from a virtual flattened category pool; no huge252*N list."""
    available = [capacity-current for capacity, current in zip(capacities, selected)]
    if target < 0 or target > sum(available):
        raise ValueError('Requested category count exceeds real unique G5 capacity')
    cumulative, total = [], 0
    for capacity in available:
        total += capacity
        cumulative.append(total)
    for offset in rng.sample(range(total), target):
        selected[bisect.bisect_right(cumulative, offset)] += 1


def _global_allocations(groups, views_per_question, seed, epoch):
    desired = len(groups)*views_per_question
    if desired % 2:
        raise ValueError('Exact global Mixed/nonMixed count balance requires an even total view budget')
    target = desired//2
    capacities = [_capacities(group) for group in groups]
    positive_capacity = [x[0] for x in capacities]
    negative_capacity = [x[1] for x in capacities]
    homogeneous = [i for i, cap in enumerate(positive_capacity) if cap == 0]
    if target > min(sum(positive_capacity), sum(negative_capacity)):
        raise ValueError('Global balance impossible: actual Mixed/nonMixed capacities are %d/%d, target%d/%d' %
                         (sum(positive_capacity), sum(negative_capacity), target, target))
    if len(homogeneous) > target:
        raise ValueError('Global balance incompatible with all-question coverage: %d homogeneous questions need negative views, target%d' % (len(homogeneous), target))
    positive, negative = [0]*len(groups), [0]*len(groups)
    for index in homogeneous:
        negative[index] = 1
    heterogeneous = [i for i, cap in enumerate(positive_capacity) if cap]
    rng = _rng(seed, epoch, ['global-allocation', [g['qid'] for g in groups]])
    rng.shuffle(heterogeneous)
    # Ensure each heterogeneous question is represented before filling slots.
    positive_used = 0
    for index in heterogeneous:
        if positive_used < target:
            positive[index] = 1
            positive_used += 1
        else:
            negative[index] = 1
    if sum(negative) > target:
        raise ValueError('Global balance incompatible with all-question coverage and requested budget')
    _allocate_remaining(positive_capacity, positive, target-sum(positive), rng)
    _allocate_remaining(negative_capacity, negative, target-sum(negative), rng)
    assert sum(positive) == sum(negative) == target
    assert all(p+n > 0 and p+n <= 252 for p, n in zip(positive, negative))
    return list(zip(positive, negative))


def build_training_views(groups, mode='random', views_per_question=2, seed=1729,
                         epoch=0, on_unbalance='raise', role='train'):
    """Return sampled real G5 groups and an exact rational weight for each.

    random: sample unique five-subsets uniformly per question.
    per-question-balanced: equal Mixed/nonMixed subset counts within each
      eligible question. A homogeneous G10 has no Mixed subset. With
      on_unbalance='report', such a question keeps negative views, and an
      insufficient heterogeneous capacity reduces its budget explicitly.
    global-balanced: exact global view-count balance with every qid covered;
      allocation can vary by question, so count balance != weighted balance.

    Every selected view uses weight_numerator/weight_denominator exactly1/n_q,
    so all views of one question sum to1 in rational arithmetic. 'weight' is
    the floating convenience approximation, not the canonical exact contract.
    """
    if role != 'train':
        raise ValueError('Training grouping cannot select validation/final groups; use fixed_g5')
    if mode not in MODES or on_unbalance not in ('raise', 'report'):
        raise ValueError('Unknown G10 grouping or impossibility handling mode')
    if type(views_per_question) is not int or not 1 <= views_per_question <= 252:
        raise ValueError('views_per_question must be an integer from1 through252')
    if mode == 'per-question-balanced' and views_per_question % 2:
        raise ValueError('Per-question exact count balance requires an even view budget')
    own = _validate(groups, require_labels=True)
    _rng(seed, epoch, 'validation')
    allocations = _global_allocations(own, views_per_question, seed, epoch) if mode == 'global-balanced' else None
    views, exceptions, per_question = [], [], []
    for ordinal, group in enumerate(own):
        rng = _rng(seed, epoch, [mode, group['qid']])
        mixed, nonmixed = _partitions(group)
        if mode == 'random':
            selected = rng.sample(COMBINATIONS, views_per_question)
        elif mode == 'per-question-balanced':
            capacity = 2*min(len(mixed), len(nonmixed))
            if capacity < views_per_question:
                reason = 'homogeneous_G10_has_no_real_Mixed_subset' if not mixed else 'insufficient_unique_subgroups_for_requested_balanced_budget'
                if on_unbalance == 'raise':
                    raise ValueError('%s: %s; Mixed/nonMixed capacities%d/%d, requested%d' %
                                     (group['qid'], reason, len(mixed), len(nonmixed), views_per_question))
                count = min(views_per_question, capacity) if capacity else views_per_question
                exceptions.append(dict(qid=group['qid'], reason=reason,
                                       requested_views=views_per_question, actual_views=count,
                                       mixed_capacity=len(mixed), nonmixed_capacity=len(nonmixed)))
            else:
                count = views_per_question
            selected = (rng.sample(mixed, count//2)+rng.sample(nonmixed, count//2)) if mixed else rng.sample(nonmixed, count)
        else:
            positive, negative = allocations[ordinal]
            selected = rng.sample(mixed, positive)+rng.sample(nonmixed, negative)
        rng.shuffle(selected)
        assert selected and len(selected) == len(set(selected)) <= 252
        own_mixed = 0
        for indices in selected:
            tids = [group['trajectory_ids'][i] for i in indices]
            labels = [group['labels'][i] for i in indices]
            k = sum(labels)
            target = 0 < k < 5
            own_mixed += int(target)
            views.append(dict(qid=group['qid'], source_indices=list(indices),
                              trajectory_ids=tids, labels=labels, K=k, true_mixed=target,
                              weight_numerator=1, weight_denominator=len(selected),
                              weight=1./len(selected), source_group_size=10))
        per_question.append(dict(qid=group['qid'], views=len(selected), mixed_views=own_mixed,
                                 nonmixed_views=len(selected)-own_mixed,
                                 total_weight_numerator=1, total_weight_denominator=1))
    _rng(seed, epoch, [mode, 'view-order']).shuffle(views)
    weight_by_question = Counter()
    positive_mass = Fraction(0)
    negative_mass = Fraction(0)
    for view in views:
        weight = Fraction(view['weight_numerator'], view['weight_denominator'])
        weight_by_question[view['qid']] += weight
        if view['true_mixed']:
            positive_mass += weight
        else:
            negative_mass += weight
    assert set(weight_by_question) == {g['qid'] for g in own}
    assert all(weight == 1 for weight in weight_by_question.values())
    assert positive_mass+negative_mass == len(own)
    positive_count = sum(v['true_mixed'] for v in views)
    qid_sha = _hash([g['qid'] for g in own])
    summary = dict(mode=mode, seed=seed, epoch=epoch, requested_views_per_question=views_per_question,
                   questions=len(own), covered_questions=len(weight_by_question),
                   source_G10_K_histogram=[sum(sum(g['labels']) == k for g in own) for k in range(11)],
                   homogeneous_questions=sum(sum(g['labels']) in (0, 10) for g in own),
                   homogeneous_questions_can_create_Mixed=False,
                   views=len(views), maximum_unique_subgroups_per_question=252,
                   mixed_views=positive_count, nonmixed_views=len(views)-positive_count,
                   count_balanced=positive_count*2 == len(views),
                   count_mixed_fraction=positive_count/len(views),
                   weighted_mixed_mass=str(positive_mass), weighted_nonmixed_mass=str(negative_mass),
                   weighted_class_mass_balanced=positive_mass == negative_mass,
                   weighted_mixed_fraction=float(positive_mass/len(own)),
                   per_question_total_weight_exactly_one=True,
                   rational_weights_are_authoritative=True, all_qids_covered=True,
                   qid_sha256=qid_sha, source_groups_sha256=_hash(own),
                   selected_views_sha256=_hash(views), impossibilities=exceptions,
                   labels_recomputed_from_own_subset=True, synthetic_labels=False,
                   training_only=True, loss_reduction='mean_over_questions(sum_over_views(exact_weight*whole_G5_loss))')
    return dict(groups=views, per_question=per_question, summary=summary)


def fixed_g5(groups, selection='first5', seed=1729, epoch=0, fixed_by_qid=None,
             role='final'):
    """Freeze five IDs without consulting EM; original/designated first5 default.

    Output intentionally contains no labels. Join own EM only after selection is
    frozen. Changing/removing input labels cannot change the selected IDs/hash.
    'random' is a deterministic label-blind five-subset, never a balanced one.
    """
    if role not in ('validation', 'final') or selection not in ('first5', 'random'):
        raise ValueError('Fixed validation/final G5 supports only label-blind first5/random')
    own = _validate(groups, require_labels=False)
    if fixed_by_qid is not None and set(fixed_by_qid) != {g['qid'] for g in own}:
        raise ValueError('Designated fixedG5 must cover exactly the declared question pool')
    _rng(seed, epoch, 'validation')
    selected = []
    for group in own:
        if fixed_by_qid is not None:
            tids = list(fixed_by_qid[group['qid']])
            if len(tids) != 5 or len(set(tids)) != 5 or not set(tids).issubset(group['trajectory_ids']):
                raise ValueError('Designated original fixedG5 is not five distinct own IDs')
        elif selection == 'first5':
            tids = group['trajectory_ids'][:5]
        else:
            indices = sorted(_rng(seed, epoch, ['label-blind-fixed-G5', group['qid']]).sample(range(10), 5))
            tids = [group['trajectory_ids'][i] for i in indices]
        selected.append(dict(qid=group['qid'], trajectory_ids=tids))
    return dict(groups=selected, summary=dict(role=role, selection='designated-original' if fixed_by_qid is not None else selection,
                seed=seed, epoch=epoch, questions=len(own), trajectories=5*len(own),
                qid_sha256=_hash([g['qid'] for g in own]), selected_g5_sha256=_hash(selected),
                labels_consulted=False, original_fixed5_preserved=(fixed_by_qid is not None or selection == 'first5')))
