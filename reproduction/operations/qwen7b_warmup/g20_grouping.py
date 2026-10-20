"""Authentic G20 -> G5 views without enumerating C(20,5) for every question.

Pure identifier/own-EM consumer. The caller must first accept all four real G5
events, their payload SHA, own EM and disjoint native sampling namespaces.
This module never loads features, generates data or selects evaluation labels.
"""
from fractions import Fraction
import hashlib
import json
import math
import random

SIZE = 20
VIEW_SIZE = 5
VIEWS = 4


def choose(n, k):
    if k < 0 or n < k:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n-k))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rng_for(seed, epoch, scope):
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError('Explicit integer seed and nonnegative epoch required')
    return random.Random(int(digest(['authentic-g20-v1', seed, epoch, scope]), 16))


def unrank(items, count, ordinal):
    """Lexicographic unranking of one exact integer combination index."""
    if type(ordinal) is not int or not 0 <= ordinal < choose(len(items), count):
        raise ValueError('Combination index exceeds actual distinct capacity')
    result = []
    start = 0
    for left in range(count, 0, -1):
        for index in range(start, len(items)-left+1):
            capacity = choose(len(items)-index-1, left-1)
            if ordinal < capacity:
                result.append(items[index]); start = index+1
                break
            ordinal -= capacity
        else:
            raise AssertionError('Unreachable combination index')
    return tuple(result)


def validate(groups, labels=True):
    if not groups:
        raise ValueError('Nonempty authentic G20 question cohort required')
    qids, tids, own = set(), set(), []
    for group in groups:
        qid, ids = group['qid'], group['trajectory_ids']
        if not isinstance(qid, str) or not qid or qid in qids:
            raise ValueError('Distinct question IDs required')
        if (len(ids) != SIZE or len(set(ids)) != SIZE or
                any(not isinstance(t, str) or not t for t in ids) or tids.intersection(ids)):
            raise ValueError('Exactly20 own globally distinct trajectory IDs required')
        row = dict(qid=qid, trajectory_ids=list(ids))
        if labels:
            values = group['labels']
            if len(values) != SIZE or any(type(v) is not int or v not in (0, 1) for v in values):
                raise ValueError('Exactly20 own binary integer EM labels required')
            row['labels'] = list(values)
        qids.add(qid); tids.update(ids); own.append(row)
    return sorted(own, key=lambda g:g['qid'])


def class_capacity(group):
    k = sum(group['labels'])
    negative = choose(k, VIEW_SIZE)+choose(SIZE-k, VIEW_SIZE)
    return choose(SIZE, VIEW_SIZE)-negative, negative


def category_subset(group, mixed, ordinal):
    """Uniform category index -> unique G5 using its exact own K multiplicity."""
    positive = [i for i, y in enumerate(group['labels']) if y]
    negative = [i for i, y in enumerate(group['labels']) if not y]
    counts = range(1, VIEW_SIZE) if mixed else (0, VIEW_SIZE)
    for k in counts:
        a, b = choose(len(positive), k), choose(len(negative), VIEW_SIZE-k)
        if ordinal < a*b:
            left, right = divmod(ordinal, b)
            return tuple(sorted(unrank(positive, k, left)+unrank(negative, VIEW_SIZE-k, right)))
        ordinal -= a*b
    raise ValueError('Category index exceeds actual own-label subset capacity')


def quotas_for(own, seed, epoch):
    capacities = [class_capacity(g) for g in own]
    lower = [max(0, VIEWS-n) for p, n in capacities]
    upper = [min(VIEWS, p) for p, n in capacities]
    target = VIEWS*len(own)//2
    if not sum(lower) <= target <= sum(upper):
        raise ValueError('Strict original-question weighted1:1 impossible: Mixed views %d..%d, target%d; no dropping/padding/synthetic labels' %
            (sum(lower), sum(upper), target))
    quotas = list(lower)
    order = list(range(len(own)))
    rng_for(seed, epoch, ['Mixed-quota', [g['qid'] for g in own]]).shuffle(order)
    left = target-sum(quotas)
    while left:
        before = left
        for i in order:
            if quotas[i] < upper[i]:
                quotas[i] += 1; left -= 1
            if not left:
                break
        assert left < before
    return quotas


def build_training_views(groups, mode='random', seed=1729, epoch=0, role='train'):
    if role != 'train':
        raise ValueError('Label-aware G20 grouping is training-only')
    if mode not in ('random', 'global-weighted-balanced'):
        raise ValueError('Explicit random or global-weighted-balanced mode required')
    own = validate(groups)
    rng_for(seed, epoch, 'contract')
    quotas = quotas_for(own, seed, epoch) if mode != 'random' else None
    views, per_question = [], []
    for index, group in enumerate(own):
        rng = rng_for(seed, epoch, [mode, group['qid']])
        if quotas is None:
            selected = [unrank(tuple(range(SIZE)), VIEW_SIZE, i)
                for i in rng.sample(range(choose(SIZE, VIEW_SIZE)), VIEWS)]
        else:
            positive, negative = class_capacity(group)
            quota = quotas[index]
            selected = ([category_subset(group, True, i) for i in rng.sample(range(positive), quota)]+
                [category_subset(group, False, i) for i in rng.sample(range(negative), VIEWS-quota)])
        assert len(selected) == len(set(selected)) == VIEWS
        rng.shuffle(selected)
        count = 0
        for indices in selected:
            labels = [group['labels'][i] for i in indices]
            k = sum(labels); mixed = 0 < k < VIEW_SIZE; count += mixed
            views.append(dict(qid=group['qid'], source_indices=list(indices),
                trajectory_ids=[group['trajectory_ids'][i] for i in indices],
                labels=labels, K=k, true_mixed=mixed,
                weight_numerator=1, weight_denominator=VIEWS, weight=1./VIEWS,
                source_group_size=SIZE))
        per_question.append(dict(qid=group['qid'], views=VIEWS, mixed_views=count,
            weighted_mixed_mass=str(Fraction(count, VIEWS)), total_weight='1'))
    rng_for(seed, epoch, [mode, 'view-order']).shuffle(views)
    positives = sum(v['true_mixed'] for v in views)
    if quotas is not None:
        assert 2*positives == len(views)
    return dict(groups=views, per_question=per_question, summary=dict(
        mode=mode, source_group_size=SIZE, views_per_question=VIEWS,
        questions=len(own), views=len(views), mixed_views=positives,
        nonmixed_views=len(views)-positives,
        weighted_mixed_fraction=positives/float(len(views)),
        count_balanced=2*positives == len(views),
        weighted_class_mass_balanced=2*positives == len(views),
        per_question_total_weight_exactly_one=True,
        all_original_questions_preserved=True, own_EM_recomputed_per_subset=True,
        source_K_histogram=[sum(sum(g['labels']) == k for g in own) for k in range(SIZE+1)],
        source_groups_sha256=digest(own), selected_views_sha256=digest(views),
        seed=seed, epoch=epoch, training_only=True, synthetic_labels=False,
        category_sampling='uniform_without_replacement_by_exact_combinatorial_unranking',
        actual_rollout_SHA_acceptance_is_callers_precondition=True))


def fixed_g5(groups, seed=1729, scope='precommitted-validation'):
    """Deterministic label-blind IDs; call/freeze before joining any own EM."""
    own = validate(groups, labels=False)
    rows = []
    for group in own:
        rng = rng_for(seed, 0, [scope, group['qid']])
        indices = sorted(rng.sample(range(SIZE), VIEW_SIZE))
        rows.append(dict(qid=group['qid'], source_indices=indices,
            trajectory_ids=[group['trajectory_ids'][i] for i in indices]))
    return dict(groups=rows, ordered_selection_sha256=digest(rows),
        labels_read=False, source_group_size=SIZE, selected_group_size=VIEW_SIZE,
        seed=seed, scope=scope)


def build_batch_plan(groups, batch_questions=48, seed=1729, epoch=0, role='train'):
    """Common random/balanced cohort order; spread homogeneous questions evenly.

    This plan deliberately consults TRAIN K20, for both arms identically. Every
    batch must support equal Mixed/nonMixed view mass without dropping a qid.
    """
    if role != 'train':
        raise ValueError('K20-stratified plan is training-only')
    if type(batch_questions) is not int or batch_questions < 2 or batch_questions % 2:
        raise ValueError('Even positive original-question batch required')
    own = validate(groups)
    if len(own) % batch_questions:
        raise ValueError('Equal full batches required; no duplicated/padded questions')
    homogeneous = [g for g in own if sum(g['labels']) in (0, SIZE)]
    heterogeneous = [g for g in own if 0 < sum(g['labels']) < SIZE]
    if len(homogeneous) > len(heterogeneous):
        raise ValueError('Strict balanced G5 views infeasible for complete cohort')
    rng = rng_for(seed, epoch, ['common-K20-plan', [g['qid'] for g in own]])
    rng.shuffle(homogeneous); rng.shuffle(heterogeneous)
    n = len(own)//batch_questions
    order = list(range(n)); rng.shuffle(order)
    batches = [[] for _ in range(n)]
    for i, group in enumerate(homogeneous):
        batches[order[i % n]].append(group['qid'])
    counts = [len(b) for b in batches]
    offset = 0
    for i in order:
        count = batch_questions-len(batches[i])
        batches[i].extend(g['qid'] for g in heterogeneous[offset:offset+count])
        offset += count
        rng.shuffle(batches[i])
    assert offset == len(heterogeneous) and max(counts) <= batch_questions//2
    cohort = [qid for batch in batches for qid in batch]
    assert len(set(cohort)) == len(own)
    return dict(cohort_qids=cohort, batches=batches, batch_epochs=[epoch]*n,
        original_questions=len(own), global_original_questions=batch_questions,
        updates=n, source_group_size=SIZE, seed=seed, epoch=epoch,
        source_groups_sha256=digest(own), ordered_qid_sha256=digest(cohort),
        all_original_questions_preserved=True,
        same_plan_required_for_random_and_balanced=True,
        labels_used_for_plan=True, label_scope='training_only',
        validation_or_final_labels_read=False,
        batch_homogeneous_counts=counts,
        exact_weighted_balance_feasible_in_every_batch=True)
