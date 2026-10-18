"""Four-rank whole-G5 partitions with the reference's unchanged global batches."""
import random

from .contract import WORLD, FINAL_OUTER, CANDIDATE_QUESTIONS, PROBE_PERIOD, PROBE_UPDATES, PROBE_EPOCHS


def _natural_plan(seed, outer, questions, updates):
    if type(outer) is not int or not 1 <= outer <= FINAL_OUTER or WORLD != 4:
        raise ValueError('Require an actual four-rank actor outer1..200')
    shuffled = list(range(questions))
    random.Random(seed + outer).shuffle(shuffled)
    plan, offset, extras_start = [], 0, 0
    for step in range(updates):
        n = 52 if (step + 1) % 5 == 0 else 51
        counts = [12] * WORLD
        for j in range(n - 12 * WORLD):
            counts[(extras_start + j) % WORLD] += 1
        plan.append(dict(question_indices=shuffled[offset:offset+n],
                         global_questions=n, rank_questions=counts,
                         extras_start=extras_start))
        offset += n
        extras_start = (extras_start + n - 12 * WORLD) % WORLD
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
    if PROBE_EPOCHS==1:return _natural_plan(seed,outer,CANDIDATE_QUESTIONS,PROBE_UPDATES)
    return [dict(batch,epoch=epoch+1,epoch_step=step+1)
            for epoch in range(PROBE_EPOCHS)
            for step,batch in enumerate(_natural_plan(seed+1000003*epoch,outer,CANDIDATE_QUESTIONS,50))]


def rank_question_slice(batch, rank):
    counts = batch['rank_questions']
    if (type(rank) is not int or not 0 <= rank < WORLD or len(counts) != WORLD
            or any(type(n) is not int or n not in (12, 13) for n in counts)
            or sum(counts) != batch['global_questions']
            or len(batch['question_indices']) != batch['global_questions']):
        raise ValueError('Invalid four-rank complete-G5 partition')
    start = sum(counts[:rank])
    return batch['question_indices'][start:start+counts[rank]]


def ddp_mean_scale(local_questions, global_questions):
    """Four-rank DDP averages gradients; recover the true global question mean."""
    if local_questions not in (12, 13) or global_questions not in (51, 52):
        raise ValueError('Unexpected four-rank whole-G5 batch')
    return WORLD * local_questions / global_questions
