"""The explicitly authorized outer51..200 sampling and probe clocks."""

START_OUTER = 50
FINAL_OUTER = 200
CANDIDATES = 2560
SELECTED = 512
RERANK_CANDIDATES = 1024
PREFIX40_FROM = 100
RERANK_FROM = 150
MIXED_FLOOR = dict(trigger_numerator=3, trigger_denominator=10,
    target_mixed_questions=154, continuation_batch_questions=128,
    remaining_order='predicted_pMixed_descending_then_original_candidate_ordinal',
    replace_only_nonmixed=True, stop_when='154_mixed_or_candidate_pool_exhausted')

BRANCH_SCHEDULE = dict(actor_start_outer=50, final_outer=200, candidates=2560,
    selected=512, prefix20_from=51, prefix40_from=100, rerank1024_from=150,
    probe_period=10, probe_update_at_final=False,
    rerank_fallback=['mixed','all_correct','all_wrong'], mixed_floor=MIXED_FLOOR)


def validate_branch(config):
    if (config.get('grpo_from_actor50') is not True
            or config.get('actor50_branch_schedule') != BRANCH_SCHEDULE
            or type(config.get('actor50_pool_consumed_candidates')) is not int
            or config['actor50_pool_consumed_candidates'] < 50*2560
            or not isinstance(config.get('candidate_schedule'),dict)):
        raise ValueError('Bind the exact actor50/Prefix20/100/150/200 branch schedule')
    return config


def outer_plan(outer):
    if type(outer) is not int or not START_OUTER < outer <= FINAL_OUTER:
        raise ValueError('This branch continues actor50 at outer51..200')
    periodic = outer < FINAL_OUTER and outer % 10 == 0
    rerank = outer >= RERANK_FROM
    return dict(outer=outer, prefix_tokens=20 if outer < PREFIX40_FROM else 40,
        candidate_questions=CANDIDATES, actor_questions=SELECTED,
        predicted_top_questions=RERANK_CANDIDATES if rerank else SELECTED,
        actual_em_rerank=rerank, probe_update=periodic,
        # The original P10 probe pass retains all2560 G5 labels. On those
        # rounds actor selection after150 is still restricted to Top1024.
        continuation_questions=CANDIDATES if periodic else
            (RERANK_CANDIDATES if rerank else SELECTED),
        probe_training_questions=CANDIDATES if periodic else 0,
        probe_actual_updates=50 if periodic else 0,
        fallback='all_correct_then_all_wrong' if rerank else 'probe_score_fill',
        mixed_floor=MIXED_FLOOR)


def probe_boundary(completed_outer):
    if type(completed_outer) is not int or not START_OUTER <= completed_outer <= FINAL_OUTER:
        raise ValueError('Probe clock starts at the actor50 warmup handoff')
    last = min(completed_outer // 10 * 10, 190)
    return dict(last_update_outer=last, actual_adam_per_rank=(last-START_OUTER)//10*50)


def rank_actual_groups(groups):
    """Keep trueMixed first, then allCorrect, then allWrong; stable score ties."""
    if len(groups) != RERANK_CANDIDATES:
        raise ValueError('Rerank only the complete predicted Top1024 G5 groups')
    ordinals=set();qids=set();trajectories=set();classes={0:[],1:[],2:[]}
    for original in groups:
        row=dict(original);k=row.get('actual_correct_count');ids=row.get('trajectory_ids',[])
        if (type(k) is not int or not 0<=k<=5 or len(ids)!=5 or len(set(ids))!=5
                or type(row['candidate_ordinal']) is not int or not 0<=row['candidate_ordinal']<CANDIDATES
                or row['candidate_ordinal'] in ordinals or row['qid'] in qids
                or trajectories.intersection(ids)):
            raise ValueError('Require distinct, complete, strict-binary ownG5 outcomes')
        ordinals.add(row['candidate_ordinal']);qids.add(row['qid']);trajectories.update(ids)
        category=0 if 0<k<5 else (1 if k==5 else 2)
        classes[category].append(row)
    import math
    for rows in classes.values():
        if any(not isinstance(r['q'],(float,int)) or not math.isfinite(r['q']) or not 0<=r['q']<=1 for r in rows):
            raise ValueError('Scores must be finite probabilities')
        rows.sort(key=lambda r:(-r['q'],r['candidate_ordinal']))
    selected=(classes[0]+classes[1]+classes[2])[:SELECTED]
    assert len(selected)==SELECTED
    return selected,dict(mixed_available=len(classes[0]),all_correct_available=len(classes[1]),
        all_wrong_available=len(classes[2]),selected_mixed=sum(0<r['actual_correct_count']<5 for r in selected),
        selected_all_correct=sum(r['actual_correct_count']==5 for r in selected),
        selected_all_wrong=sum(r['actual_correct_count']==0 for r in selected),
        fallback_order=['mixed','all_correct','all_wrong'])
