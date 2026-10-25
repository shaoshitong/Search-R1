"""Explicit user-authorized CP50 objective fork; preserve persisted training state."""
REVISION = 'cp50_mathfix_v1'


def bind_transition(run, parent, receipt):
    before, after = parent.get('budget'), run.get('budget')
    if run.get('objective_revision') != REVISION:
        if before != after:
            raise ValueError('A restore cannot change the scientific budget')
        return
    expected = dict(actor_logical_micro_batch_per_rank=4, all_correct_virtual_zero=False,
                    all_correct_reward_one=True, selection_allow_all_wrong=False)
    if any(run.get(k) != v for k,v in expected.items()):
        raise ValueError('Incomplete CP50 objective revision')
    if parent.get('objective_revision') == REVISION:
        if before != after or any(parent.get(k) != v for k,v in expected.items()):
            raise ValueError('Resumed revised objective differs from its saved run')
        return
    if receipt['completed_outer'] != 50 or receipt['status'] != 'DYNRPO7B_ROLLING_READY':
        raise ValueError('Initial objective fork requires the complete CP50 boundary')
    allowed = {'per_rank_actor_logical_micro_batch', 'objective_reduction'}
    if before.keys() != after.keys() or any(before[k] != after[k] for k in before if k not in allowed):
        raise ValueError('CP50 fork changes an unauthorized budget field')
    if before['per_rank_actor_logical_micro_batch'] != 2 or after['per_rank_actor_logical_micro_batch'] != 4:
        raise ValueError('CP50 logical objective must change2 to4')
    run['objective_transition'] = dict(revision=REVISION, source_outer=50,
        source_base=receipt['source_base'], effective_from_outer=51,
        logical_micro_before=2, logical_micro_after=4,
        all_correct_reward=1.0, all_correct_advantage=0.0,
        mixed_soft_reward_unchanged=True, all_wrong_fallback=False,
        model_optimizer_scheduler_rng_probe_cursor_preserved=True,
        bitwise_continuation=False, old_run_suffix_not_imported=True)


def all_correct_rows(group_ids, correctness):
    """Group by identity, not adjacency: DP balancing permutes row order."""
    from collections import defaultdict
    if len(group_ids) != len(correctness):
        raise ValueError('Reward group IDs and labels differ')
    groups = defaultdict(list)
    for row,qid in enumerate(group_ids):
        groups[qid].append(row)
    if any(len(rows) != 5 for rows in groups.values()):
        raise ValueError('All-correct reward requires complete G5 groups')
    return {i for rows in groups.values() if all(correctness[i] for i in rows) for i in rows}
