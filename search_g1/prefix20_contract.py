"""The authorized four-rank, trajectory-prefix question acquisition contract."""
import math


CONTRACT = dict(
    schema_version=1, world_size=4, successful_outers=200,
    candidate_questions=5120, selected_questions=512, responses_per_question=5,
    exploration_questions=0, policy_prefix_tokens=20,
    acquisition='one_minus_product_correct_minus_product_incorrect',
    probe_target='own_trajectory_strict_terminal_em',
    probe_learning_rate=1e-6, probe_global_batch=256, probe_local_batch=64,
    probe_epochs_per_outer=1, probe_optimizer_steps_per_outer=10,
    probe_training_trajectories_per_outer=2560, replay=False,
    actor_learning_rate='unchanged_original_search_g1',
    virtual_zero_scope='strict_em_all_correct_only', grpo_std_correction=1,
)


def mixed_probability(probabilities):
    """Probability that independent Bernoulli trajectories contain both labels."""
    values = list(probabilities)
    if len(values) != CONTRACT['responses_per_question']:
        raise ValueError('PREFIX20_REQUIRES_FIVE_OWN_TRAJECTORY_PROBABILITIES')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError('PREFIX20_INVALID_EM_PROBABILITY')
    return max(0.0, min(1.0, 1.0 - math.prod(values) - math.prod(1.0-p for p in values)))


def top_questions(rows, *, count=512):
    """Rank predictions only; gold labels and completed rewards cannot enter."""
    if len(rows) != CONTRACT['candidate_questions'] or count != CONTRACT['selected_questions']:
        raise ValueError('PREFIX20_CANDIDATE_OR_SELECTED_BUDGET_CHANGED')
    ids = [r['qid'] for r in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(q, str) or not q for q in ids):
        raise ValueError('PREFIX20_QUESTION_IDENTITIES_NOT_UNIQUE')
    if any(set(r) != {'qid', 'trajectory_ids', 'em_probabilities'} for r in rows):
        raise ValueError('PREFIX20_PREDICTION_WIRE_FORBIDS_GOLD_OR_REWARD')
    trajectories = [tid for r in rows for tid in r['trajectory_ids']]
    if (any(len(r['trajectory_ids']) != 5 for r in rows) or
            len(set(trajectories)) != 5 * len(rows) or
            any(not isinstance(t, str) or not t for t in trajectories)):
        raise ValueError('PREFIX20_CANDIDATE_TRAJECTORY_IDENTITIES_CHANGED')
    ranked = sorted(enumerate(rows), key=lambda pair: (-mixed_probability(pair[1]['em_probabilities']), pair[0]))
    return [dict(row, mixed_probability=mixed_probability(row['em_probabilities']),
                 candidate_ordinal=ordinal) for ordinal, row in ranked[:count]]


def probe_batch_plan(trajectory_ids, permutation):
    """One complete pass: ten disjoint global256 batches, four local64 shards."""
    if len(trajectory_ids) != 2560 or len(set(trajectory_ids)) != 2560:
        raise ValueError('PREFIX20_REQUIRES_EXACTLY_NEW_SELECTED_2560_TRAJECTORIES')
    if (len(permutation) != 2560 or any(type(i) is not int for i in permutation) or
            sorted(permutation) != list(range(2560))):
        raise ValueError('PREFIX20_PROBE_PASS_IS_NOT_A_PERMUTATION')
    return [[ [trajectory_ids[i] for i in permutation[start+rank*64:start+(rank+1)*64]]
              for rank in range(4)] for start in range(0, 2560, 256)]
