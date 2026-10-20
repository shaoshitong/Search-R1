"""Pure append-only plan/state contracts, without loading tensors or starting jobs.

The accepted 8192-question/256-update prefix is immutable history. Only the
additional21824 questions may be trained next:682 new updates,938 cumulative.
No fake E1 metadata, optimizer reset, relabeling, or replay of the old prefix.
"""
import hashlib,json,random

HISTORICAL_QUESTIONS=8192
TOTAL_QUESTIONS=30016
BATCH=32
HISTORICAL_UPDATES=256
SOURCE_ADAM=10720
E1_ADAM=10464
COMPONENTS={'raw','ema','adam','raw_buffers','ema_buffers','rng'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def _extend(old,identity,*,historical_questions,total_questions,batch):
    """Small explicit dimensions are for synthetic tests; production pins them."""
    groups=identity['groups'];prior=historical_questions//batch
    if (historical_questions%batch or total_questions%batch or historical_questions>=total_questions
            or len(groups)!=total_questions or old['questions']!=historical_questions
            or old['updates']!=prior or old['global_questions']!=batch or old['seed']!=1729
            or old['views_per_question']!=4 or old['view_weight']!=.25
            or old['effective_total_weight_per_original_question']!=1.
            or old['labels_used_for_question_order'] is not False or old['all_questions_once'] is not True):
        raise ValueError('Exact completed question-once source plan required')
    # Never inspect correctness labels or a model score to choose/reorder data.
    for g in groups:
        if set(g)!={'qid','question_hash','trajectory_ids'} or len(g['trajectory_ids'])!=10:
            raise ValueError('Identity-only ten-real-trajectory groups required')
    qids=[g['qid'] for g in groups];hashes=[g['question_hash'] for g in groups]
    tids=[t for g in groups for t in g['trajectory_ids']]
    if len(set(qids))!=len(qids) or len(set(hashes))!=len(hashes) or len(set(tids))!=10*len(groups):
        raise ValueError('Question text or actual trajectory identities repeat')
    if old['cohort_qids']!=qids[:historical_questions]:
        raise ValueError('Expanded identity does not preserve the old cohort exactly')
    batches=old['batches'];epochs=old['batch_epochs']
    flat=[q for b in batches for q in b]
    if (len(batches)!=prior or any(len(b)!=batch or len(set(b))!=batch for b in batches)
            or len(flat)!=len(set(flat)) or set(flat)!=set(qids[:historical_questions])
            or epochs!=[0]*prior):
        raise ValueError('Historical updates must cover every old question exactly once')
    tail=qids[historical_questions:]
    new=list(tail)
    random.Random(int(digest(['peer-expanded-append-only-tail',1729,tail]),16)).shuffle(new)
    following=[new[i:i+batch] for i in range(0,len(new),batch)]
    return dict(cohort_qids=qids,batches=[list(b) for b in batches]+following,
        batch_epochs=list(epochs)+[0]*len(following),questions=total_questions,
        updates=total_questions//batch,global_questions=batch,seed=1729,
        views_per_question=4,view_weight=.25,effective_total_weight_per_original_question=1.,
        labels_used_for_question_order=False,all_questions_once=True,
        historical_updates=prior,additional_updates=len(following),
        source_plan_canonical_sha256=digest(old),historical_batches_sha256=digest(batches),
        added_question_order_sha256=digest(new),selection_uses_labels=False,
        historical_prefix_retrained=False,domain_scope='NQ_only_enrichment',
        budget_scope='Larger data and total optimization budget; not a matched256 causal question-count comparison')

def extend_plan(old,identity):
    return _extend(old,identity,historical_questions=HISTORICAL_QUESTIONS,
                   total_questions=TOTAL_QUESTIONS,batch=BATCH)

def validate_source_checkpoint_metadata(saved,receipt,*,source_config_sha,source_plan_sha):
    """Reject a model-only or rewritten source, after its whole file SHA check.

    Tensor/buffer/RNG contents still require actual restore and six-component
    hash comparisons on the allocated runtime. This is not that verification.
    The four original E1 ranks are an archive, not four current training ranks.
    """
    required={'raw','ema','optimizer','raw_buffers','ema_buffers','rng','accepted_hashes','source_all_rank_states'}
    if not required<=set(saved):raise ValueError('All live optimizer/model/RNG components required')
    if (saved.get('cursor')!=HISTORICAL_UPDATES or saved.get('steps')!=SOURCE_ADAM
            or receipt.get('cursor')!=HISTORICAL_UPDATES or receipt.get('steps')!=SOURCE_ADAM
            or saved.get('config_sha256')!=source_config_sha or receipt.get('config_sha256')!=source_config_sha
            or saved.get('plan_sha256')!=source_plan_sha or saved.get('actor_updates')!=0
            or saved.get('optimizer_reset') is not False or saved.get('readout_reset') is not False
            or saved.get('block_reset') is not False):
        raise ValueError('Only an intact accepted CP256/Adam10720 lineage may continue')
    hashes=saved['accepted_hashes']
    if (set(hashes)!=COMPONENTS or hashes!=receipt.get('accepted_hashes')
            or any(not isinstance(v,str) or len(v)!=64 for v in hashes.values())):
        raise ValueError('Accepted full six-component hashes required')
    ranks=saved['source_all_rank_states']
    if len(ranks)!=4 or {r['rank'] for r in ranks}!={0,1,2,3} or any('rng' not in r for r in ranks):
        raise ValueError('Retain all four original source-rank RNG archives')
    return dict(source_cursor=HISTORICAL_UPDATES,source_Adam=SOURCE_ADAM,
        additional_updates=682,cumulative_target_updates=938,cumulative_target_Adam=11402,
        original_E1_Adam=E1_ADAM,metadata_only=True,full_restore_and_next_update_replay_still_required=True,
        optimizer_reset=False,readout_reset=False,block_reset=False)

def validate_cursor(cursor,steps,*,new_config_sha,saved_config_sha,highest_intent=None):
    if (type(cursor) is not int or not HISTORICAL_UPDATES<cursor<=TOTAL_QUESTIONS//BATCH
            or steps!=E1_ADAM+cursor or new_config_sha!=saved_config_sha):
        raise ValueError('Cumulative append-only cursor/Adam/config mismatch')
    if highest_intent is not None and highest_intent>cursor:
        raise ValueError('Unresolved later update; refuse silent rollback')
    return cursor-HISTORICAL_UPDATES
