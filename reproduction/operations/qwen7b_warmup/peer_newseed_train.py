"""Independent NQ8192 old5+peer5 continuation, after real feature acceptance.

No source job or existing native runner is modified. E1 raw/EMA/full Adam/RNG
fork into a new branch, unchanged LR5e-6; 8192 distinct training questions,
B32*256 updates, four unique random G5 views per authentic G10 with total
question weight1. This is NQ enrichment, not a seven-domain controlled claim.
All fitting/evaluation inputs must use the same qualified frozen-HF backend.
"""
import argparse
import fcntl
from pathlib import Path
import random
import time

from . import peer_newseed_ingest as raw
from . import peer_newseed_replay as replay

INPUT_STATUS='PEER8192_AUTHENTIC_RAW_G10_AND_FOUR_HF_FEATURE_SHARDS_ACCEPTED'
TARGET=raw.A/'peer-newseed8192-g10-train-v1'
Q=8192
BATCH=32
UPDATES=256


def select_groups(raw_complete,groups,identity):
    """Identity selection was committed before labels; now join exact own EM."""
    if (raw_complete.get('status')!=raw.STATUS or raw_complete.get('role')!='train'
            or raw_complete.get('actor_updates')!=0 or raw_complete.get('probe_updates')!=0
            or raw_complete.get('strict_EM_mismatches')!=0
            or raw_complete.get('final_labels_or_rollouts_read') is not False
            or raw_complete.get('real_trajectories_per_question')!=10):
        raise ValueError('Actual accepted peer raw G10 with recomputed own strict EM required')
    summary=identity['summary'];chosen=identity['groups']
    if (identity.get('role')!='train' or summary.get('labels_consulted') is not False
            or summary.get('selected_questions')!=Q or summary.get('selected_trajectories')!=Q*10
            or summary.get('controlled6528_and_eval_qids_and_hashes_excluded') is not True
            or summary.get('final_labels_or_rollouts_read') is not False
            or identity.get('domain_scope',{}).get('kind')!='NQ_only_enrichment'):
        raise ValueError('Precommitted identity-only independent NQ8192 selection required')
    lookup={g['qid']:g for g in groups}
    if len(lookup)!=len(groups):raise ValueError('Repeated raw question identity')
    result=[];seen=set();hashes=set();tids=set()
    for own in chosen:
        if set(own)!={'qid','question_hash','trajectory_ids'}:
            raise ValueError('Identity-only selection must not contain labels or scores')
        qid=own['qid']
        if qid in seen or qid not in lookup:raise ValueError('Duplicate or absent selected question')
        group=lookup[qid];raw.own_group(group,10)
        if (own['question_hash']!=group['question_hash'] or own['trajectory_ids']!=group['trajectory_ids']
                or group['question_hash'] in hashes or tids.intersection(group['trajectory_ids'])):
            raise ValueError('Question/hash/ten trajectory binding differs')
        # Selection hashes and an explicit domain scope, not the position in a
        # partially generated source, determine this fitting cohort.
        import json
        if json.loads(qid)[0]!='nq':raise ValueError('This partial peer bank is explicitly NQ-only')
        seen.add(qid);hashes.add(group['question_hash']);tids.update(group['trajectory_ids'])
        result.append(group)
    if (len(result)!=Q or raw.digest([g['qid'] for g in result])!=summary['ordered_qid_sha256']
            or raw.digest([t for g in result for t in g['trajectory_ids']])!=summary['trajectory_ids_sha256']):
        raise ValueError('Selected cohort/trajectory hash differs')
    return result


def build_plan(groups,seed=1729):
    qids=[g['qid'] for g in groups]
    if len(qids)!=Q or len(set(qids))!=Q or seed!=1729:
        raise ValueError('Exactly8192 distinct preselected training questions required')
    order=list(qids);random.Random(int(raw.digest(['peer8192-onepass',seed,qids]),16)).shuffle(order)
    batches=[order[i:i+BATCH] for i in range(0,Q,BATCH)]
    assert len(batches)==UPDATES and all(len(b)==BATCH for b in batches)
    return dict(cohort_qids=qids,batches=batches,batch_epochs=[0]*UPDATES,seed=seed,
        questions=Q,updates=UPDATES,global_questions=BATCH,labels_used_for_question_order=False,
        all_questions_once=True,views_per_question=4,view_weight=0.25,
        effective_total_weight_per_original_question=1.,domain_scope='NQ_only_enrichment')


def collect_features(complete_refs,identity,*,identity_ref,raw_ref):
    if len(complete_refs)!=4:raise ValueError('Four real complete feature shards required')
    desired={t:g['qid'] for g in identity['groups'] for t in g['trajectory_ids']}
    manifests=[];seen=set();shards=set();proofs=[]
    for item in complete_refs:
        complete=raw.read(raw.checked(item))
        if (complete.get('status')!=replay.STATUS or complete.get('actor_updates')!=0
                or complete.get('probe_updates')!=0 or complete.get('new_responses_generated')!=0
                or complete.get('source_runtime')!='huggingface_frozen_causal_prefill'
                or complete.get('model_revision')!=raw.REVISION
                or complete.get('native_bitexact_equivalence_claim') is not False
                or complete.get('domain_scope',{}).get('kind')!='NQ_only_enrichment'):
            raise ValueError('Actual same-backend actor-frozen peer HF replay shard required')
        selection=raw.read(raw.checked(complete['source_selection']))
        if (selection.get('role')!='peer_newseed_train_G10' or selection.get('shards')!=4
                or selection.get('shard') in shards or selection.get('capture_layers')!=[20,23,26]
                or selection.get('label_blind_selection') is not True
                or selection.get('real_trajectories_per_question')!=10
                or selection.get('question_plan')!=identity_ref
                or selection.get('source_ingest')!=raw_ref):
            raise ValueError('Wrong or repeated complete shard selection')
        shards.add(selection['shard']);rows=selection['trajectories']
        if len(rows)!=20480 or complete.get('trajectories')!=len(rows) or complete.get('questions')!=2048:
            raise ValueError('Each real shard must contain2048 questions and20480 trajectories')
        for row in rows:
            tid=row['trajectory_id']
            if tid in seen or desired.get(tid)!=row['qid']:raise ValueError('Feature identity coverage differs')
            seen.add(tid)
        manifests.extend(complete['feature_manifests_by_capture_layer']['26']);proofs.append(item)
    if seen!=set(desired) or shards!={0,1,2,3}:raise ValueError('Incomplete full8192 G10 feature coverage')
    return manifests,proofs


def accept_inputs(output,raw_complete_ref,identity_ref,feature_completes):
    """CPU metadata gate; the reader checks full feature bytes at actual use."""
    output=Path(output)
    if output!=TARGET/'training-input.json':raise ValueError('Independent declared peer training target required')
    complete=raw.read(raw.checked(raw_complete_ref));identity=raw.read(raw.checked(identity_ref))
    if identity['raw_complete']!=raw_complete_ref:raise ValueError('Identity plan belongs to another raw bank')
    groups=raw.read(raw.checked(complete['outputs']['groups']))['train']
    selected=select_groups(complete,groups,identity)
    manifests,proofs=collect_features(feature_completes,identity,identity_ref=identity_ref,raw_ref=raw_complete_ref)
    value=dict(status=INPUT_STATUS,raw_complete=raw_complete_ref,identity_plan=identity_ref,
        feature_shard_completes=proofs,feature_manifests=manifests,
        ordered_qid_sha256=raw.digest([g['qid'] for g in selected]),questions=Q,trajectories=Q*10,
        full_feature_payload_SHA_validation_required_at_training_start=True,
        training_started=False,actor_updates=0,probe_updates=0,domain_scope=identity['domain_scope'],
        backend='huggingface_frozen_causal_prefill',native_bitexact_claim=False,time=time.time())
    raw.immutable(output,value);raw.immutable(TARGET/'plan.json',build_plan(selected));return value


def accepted_peer_g10(reference):
    value=raw.read(raw.checked(reference))
    if (value.get('status')!=INPUT_STATUS or value.get('questions')!=Q or value.get('trajectories')!=10*Q
            or value.get('actor_updates')!=0 or value.get('domain_scope',{}).get('kind')!='NQ_only_enrichment'):
        raise ValueError('Explicit full peer8192 G10 training acceptance required')
    complete=raw.read(raw.checked(value['raw_complete']));identity=raw.read(raw.checked(value['identity_plan']))
    groups=select_groups(complete,raw.read(raw.checked(complete['outputs']['groups']))['train'],identity)
    manifests,_=collect_features(value['feature_shard_completes'],identity,
        identity_ref=value['identity_plan'],raw_ref=value['raw_complete'])
    if manifests!=value['feature_manifests']:raise ValueError('Accepted feature list changed')
    return groups,manifests,value


def replay_reader(manifests):
    from .replay_feature_reader import ReplayFeatureReader
    return ReplayFeatureReader(manifests,26)


def run(base):
    from . import probe_g10_train as train
    c=raw.read(base/'config.json')
    required={'reproduction/operations/qwen7b_warmup/'+name for name in
        ('peer_newseed_train.py','peer_newseed_replay.py','peer_newseed_ingest.py',
         'replay_feature_reader.py','replay_probe_layers.py')}
    if not required<=set(c.get('code_hashes',{})):
        raise ValueError('Complete immutable peer/HF reader source bindings required')
    if (base!=TARGET or c.get('global_questions')!=32 or c.get('update_budget')!=256
            or c.get('fit_questions')!=8192 or c.get('fit_real_trajectories')!=81920
            or c.get('feature_backend')!='huggingface_frozen_causal_prefill'
            or c.get('domain_scope',{}).get('kind')!='NQ_only_enrichment'
            or c.get('grouping')!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise')
            or c.get('recipe',{}).get('loss_kwargs')!=dict(trajectory_weight=1.,three_event_weight=1.)):
        raise ValueError('Declared independent NQ8192 E1 continuation recipe required')
    train.run(base,bank_loader=accepted_peer_g10,reader_factory=replay_reader)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('accept','train'),required=True)
    parser.add_argument('--raw-complete');parser.add_argument('--identity-plan')
    parser.add_argument('--feature-complete',action='append');parser.add_argument('--base',type=Path,default=TARGET)
    args=parser.parse_args();base=args.base.resolve()
    if base!=TARGET:parser.error('Only independent remote peer8192 target accepted')
    base.mkdir(parents=True,exist_ok=True)
    with (base/'peer8192.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='accept':
            if not args.raw_complete or not args.identity_plan or len(args.feature_complete or [])!=4:
                parser.error('Actual raw complete, frozen identity plan, four real feature completes required')
            import json
            print(json.dumps(accept_inputs(base/'training-input.json',raw.ref(args.raw_complete),raw.ref(args.identity_plan),
                [raw.ref(p) for p in args.feature_complete]),sort_keys=True),flush=True)
        else:run(base)


if __name__=='__main__':main()
