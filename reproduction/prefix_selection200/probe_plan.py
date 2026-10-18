"""Physical-rank whole-G5 partitions with unchanged global batches."""
import random

from .contract import WORLD, FINAL_OUTER, CANDIDATE_QUESTIONS, PROBE_PERIOD, PROBE_UPDATES


def _natural_plan(seed, outer, questions, updates):
    if type(outer) is not int or not 1 <= outer <= FINAL_OUTER or WORLD not in (4, 8):
        raise ValueError('Require the configured physical ranks and outer1..200')
    shuffled = list(range(questions))
    random.Random(seed + outer).shuffle(shuffled)
    plan, offset, extras_start = [], 0, 0
    for step in range(updates):
        n = 52 if (step + 1) % 5 == 0 else 51
        floor = n // WORLD
        counts = [floor] * WORLD
        for j in range(n - floor * WORLD):
            counts[(extras_start + j) % WORLD] += 1
        plan.append(dict(question_indices=shuffled[offset:offset+n],
                         global_questions=n, rank_questions=counts,
                         extras_start=extras_start))
        offset += n
        extras_start = (extras_start + n - floor * WORLD) % WORLD
    if (offset != questions or sorted(i for batch in plan for i in batch['question_indices']) != list(range(questions))
            or any(sum(batch['rank_questions'][rank] for batch in plan) != questions // WORLD
                   for rank in range(WORLD))):
        raise RuntimeError('Natural complete-G5 plan lost, repeated, or unbalanced questions')
    return plan


def online_plan(seed, outer):
    return _natural_plan(seed, outer, 512, 10)


def complete_pool_plan(seed, outer):
    if type(outer) is not int or not 1 <= outer < FINAL_OUTER or outer % PROBE_PERIOD:
        raise ValueError('Probe writes only at10,20,...190; never at200')
    return _natural_plan(seed, outer, CANDIDATE_QUESTIONS, PROBE_UPDATES)


def rank_question_slice(batch, rank):
    counts = batch['rank_questions']
    if (type(rank) is not int or not 0 <= rank < WORLD or len(counts) != WORLD
            or any(type(n) is not int or n not in (batch['global_questions']//WORLD, (batch['global_questions']+WORLD-1)//WORLD) for n in counts)
            or sum(counts) != batch['global_questions']
            or len(batch['question_indices']) != batch['global_questions']):
        raise ValueError('Invalid four-rank complete-G5 partition')
    start = sum(counts[:rank])
    return batch['question_indices'][start:start+counts[rank]]


def ddp_mean_scale(local_questions, global_questions):
    """Four-rank DDP averages gradients; recover the true global question mean."""
    if global_questions not in (51, 52) or local_questions not in (global_questions//WORLD, (global_questions+WORLD-1)//WORLD):
        raise ValueError('Unexpected four-rank whole-G5 batch')
    return WORLD * local_questions / global_questions
