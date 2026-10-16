"""The actor-hidden experiment's clocks and complete, unselected G5 epoch."""
import random

PERIOD = 20
UPDATES = 100


def is_probe_update(outer):
    if type(outer) is not int or not 1 <= outer <= 200:
        raise ValueError('Actor outer must be1..200')
    return outer % PERIOD == 0


def probe_boundary(completed_outer):
    if type(completed_outer) is not int or not 0 <= completed_outer <= 200:
        raise ValueError('Completed actor outer must be0..200')
    last = completed_outer // PERIOD * PERIOD
    return dict(last_update_outer=last, actual_adam_per_rank=last // PERIOD * UPDATES)


def complete_pool_plan(seed, outer):
    """5120 own-G5 groups exactly once:80x51 and20x52 groups globally."""
    if not is_probe_update(outer):
        raise ValueError('No probe writes outside20,40,...200')
    shuffled = list(range(5120))
    random.Random(seed + outer).shuffle(shuffled)
    plan, offset, extras_start = [], 0, 0
    for step in range(UPDATES):
        n = 52 if (step + 1) % 5 == 0 else 51
        counts = [6] * 8
        for j in range(n - 48):
            counts[(extras_start + j) % 8] += 1
        plan.append(dict(question_indices=shuffled[offset:offset+n],
                         global_questions=n, rank_questions=counts))
        offset += n
        extras_start = (extras_start + n - 48) % 8
    assert offset == 5120
    assert sorted(i for batch in plan for i in batch['question_indices']) == list(range(5120))
    assert all(sum(batch['rank_questions'][r] for batch in plan) == 640 for r in range(8))
    return plan
