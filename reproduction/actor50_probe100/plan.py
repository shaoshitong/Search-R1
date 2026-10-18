"""Deterministic oversampling of complete authentic G5 groups only."""
import random


def group_class(labels):
    if len(labels) != 5 or any(type(x) is not int or x not in (0, 1) for x in labels):
        raise ValueError('Five own strict binary EM labels required')
    k = sum(labels)
    return 'all_wrong' if k == 0 else 'all_correct' if k == 5 else 'mixed'


def balanced_epoch(groups, index, seed, cycle, epoch):
    buckets = {k: [] for k in ('all_correct', 'all_wrong', 'mixed')}
    for i, group in enumerate(groups):
        labels = [index[tid]['strict_em'] for tid in group['trajectory_ids']]
        buckets[group_class(labels)].append(i)
    if any(not rows for rows in buckets.values()):
        raise ValueError('A genuine class is absent; no fabricated labels or cross-cycle substitute')
    rng = random.Random(seed + cycle * 1009 + epoch * 100003)
    n = max(map(len, buckets.values()))
    selected = []
    for rows in buckets.values():
        own = list(rows)
        rng.shuffle(own)
        own.extend(rng.choice(rows) for _ in range(n-len(own)))
        selected.extend(own)
    rng.shuffle(selected)
    batches = [selected[i:i+51] for i in range(0, len(selected), 51)]
    if len(batches[-1]) < 4:
        deficit = 4-len(batches[-1])
        batches[-1][:0] = batches[-2][-deficit:]
        del batches[-2][-deficit:]
    if sum(map(len, batches)) != 3*n or min(map(len, batches)) < 4:
        raise RuntimeError('Invalid nonempty four-rank partition')
    return batches, dict(original_class_counts={k:len(v) for k,v in buckets.items()},
                         balanced_class_counts={k:n for k in buckets},
                         balanced_groups=3*n, balanced_trajectories=15*n,
                         sampling='all majority groups once; minority whole-G5 oversampling with replacement')
