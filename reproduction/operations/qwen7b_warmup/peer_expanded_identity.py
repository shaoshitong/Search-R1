"""Identity-only expansion of accepted NQ old5+new5; no feature/model IO.

Preserve every already-qualified8192 question. Exclude controlled/evaluation
identities and duplicate question text hashes BEFORE any correctness use. Fill
complete B32 batches from the same seed1729 order, explicitly record the tail.
"""
import hashlib,json
SEED=1729
BATCH=32

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def identity(g):
    v={k:g[k] for k in ('qid','question_hash','trajectory_ids')}
    if not isinstance(v['qid'],str) or not v['qid']:raise ValueError('Question identity required')
    if len(v['question_hash'])!=64 or any(c not in '0123456789abcdef' for c in v['question_hash']):raise ValueError('Question text hash required')
    tids=v['trajectory_ids']
    if len(tids)!=10 or len(set(tids))!=10 or any(not isinstance(t,str) or not t for t in tids):raise ValueError('Ten genuine distinct IDs required')
    return dict(qid=v['qid'],question_hash=v['question_hash'],trajectory_ids=list(tids))
def expand(groups,existing,controlled,exclusions,*,batch=BATCH):
    if batch!=BATCH:raise ValueError('Declared B32 expansion only')
    rows=[identity(g) for g in groups];prior=[identity(g) for g in existing]
    by={g['qid']:g for g in rows}
    if len(by)!=len(rows) or len({g['qid'] for g in prior})!=len(prior):raise ValueError('Repeated canonical question')
    if not prior or len(prior)%batch:raise ValueError('Existing complete-B32 cohort required')
    excluded_q=set();excluded_h=set()
    for own in [controlled]+list(exclusions):
        for row in own:excluded_q.add(row['qid']);excluded_h.add(row['question_hash'])
    eligible=sorted((g for g in rows if g['qid'] not in excluded_q and g['question_hash'] not in excluded_h),
        key=lambda g:(digest(['peer-newseed-independent8192',SEED,g['qid']]),g['qid']))
    if eligible[:len(prior)]!=prior:raise ValueError('Prior sealed cohort is not exact prefix of unchanged eligible identity ordering')
    unique=[];seen_h=set();seen_t=set();duplicates=[];prior_ids={p['qid'] for p in prior}
    for g in eligible:
        if g['question_hash'] in seen_h:
            if g['qid'] in prior_ids:raise ValueError('Previously accepted cohort contains duplicated question text')
            duplicates.append(g['qid']);continue
        if seen_t.intersection(g['trajectory_ids']):raise ValueError('Trajectory collision across canonical questions')
        seen_h.add(g['question_hash']);seen_t.update(g['trajectory_ids']);unique.append(g)
    budget=len(unique)//batch*batch
    if budget<=len(prior):raise ValueError('No additional complete-B32 question cohort')
    chosen=unique[:budget]
    if chosen[:len(prior)]!=prior:raise ValueError('Expansion changed previously qualified question identities')
    remaining=chosen[len(prior):];tail=unique[budget:]
    return dict(role='train',groups=chosen,remaining_groups=remaining,
        omitted_tail_identities=[dict(qid=g['qid'],question_hash=g['question_hash']) for g in tail],
        duplicate_text_qids_excluded=duplicates,
        summary=dict(status='IDENTITY_ONLY_EXPANDED_REAL_G10_COHORT_FROZEN_NOT_CAPTURED_OR_FIT',
          source_questions=len(rows),eligible_questions_before_hash_dedup=len(eligible),unique_eligible_questions=len(unique),
          questions=budget,existing_questions=len(prior),new_questions=len(remaining),trajectories=10*budget,
          new_trajectories_needing_capture=10*len(remaining),tail_questions_omitted_for_complete_batch=len(tail),
          duplicated_question_text_excluded=len(duplicates),global_questions=batch,onepass_training_updates=budget//batch,
          existing8192_preserved_exactly=True,labels_consulted=False,scores_consulted=False,
          controlled_and_all_evaluation_qid_and_hash_exclusions_applied=True,
          all_original_warmup_question_unseen_claim=False,
          ordered_qid_sha256=digest([g['qid'] for g in chosen]),
          ordered_new_qid_sha256=digest([g['qid'] for g in remaining]),
          ordered_all_trajectory_ids_sha256=digest([t for g in chosen for t in g['trajectory_ids']]),
          selection='same SHA256(seed1729,qid) order; preserve prior prefix; unique text; full B32 tail rule',
          feature_capture_started=False,probe_training_started=False,actor_updates=0))
