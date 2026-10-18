"""Strict own-G5 floor, with label-free ranking of remaining native prefixes.

No new question pool, prefix, probe score, actor update or probe update is
created by a refill. Only disjoint original G5s are continued as necessary.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import time

from .schedule import MIXED_FLOOR, outer_plan, rank_actual_groups


def rank_key(row):return (-row['q'],row['candidate_ordinal'])


def is_mixed(row):return 0<row['actual_correct_count']<5


def refill(initial, remaining_known):
    """Keep all initial mixed; replace lowest-ranked extremes up to154 mixed."""
    if len(initial)!=512 or len({r['qid'] for r in initial})!=512:
        raise ValueError('The actor batch must contain512 distinct completeG5 questions')
    before=sum(is_mixed(r) for r in initial)
    trigger=before*10<=512*3
    chosen=list(initial);used={r['qid'] for r in chosen};replacements=[]
    # Replace all-wrong first, retaining all-correct when the old fallback
    # preferred it. Within either class remove the lowest probe score first.
    removable=sorted((r for r in chosen if not is_mixed(r)),
        key=lambda r:(r['actual_correct_count']==5,r['q'],-r['candidate_ordinal']))
    if trigger:
        for row in remaining_known:
            if before+len(replacements)>=154:break
            if row['qid'] in used:raise ValueError('Refill must exclude the initial selected512')
            if not is_mixed(row):continue
            old=removable.pop(0);chosen[chosen.index(old)]=row;used.add(row['qid'])
            replacements.append(dict(removed=old['candidate_ordinal'],added=row['candidate_ordinal']))
    after=sum(is_mixed(r) for r in chosen)
    assert len(chosen)==512 and len({r['qid'] for r in chosen})==512
    assert {r['qid'] for r in initial if is_mixed(r)}<={r['qid'] for r in chosen}
    return chosen,dict(triggered=trigger,initial_mixed=before,final_mixed=after,
        initial_fraction=before/512,final_fraction=after/512,target_mixed=154,
        target_reached=after>=154,replacements=replacements)


def validate_scores(groups,candidates):
    if len(groups)!=candidates or len({g['qid'] for g in groups})!=candidates:
        raise ValueError('Every original candidate must have one label-free score')
    tids=set();ordinals=set();hashes=set()
    for g in groups:
        if (type(g['candidate_ordinal']) is not int or not 0<=g['candidate_ordinal']<candidates
                or g['candidate_ordinal'] in ordinals or g['question_hash'] in hashes
                or type(g['q']) not in (float,int) or not math.isfinite(g['q']) or not 0<=g['q']<=1
                or len(g['trajectory_ids'])!=5 or len(set(g['trajectory_ids']))!=5
                or tids.intersection(g['trajectory_ids'])):
            raise ValueError('Invalid original question/G5/probability identity')
        tids.update(g['trajectory_ids']);ordinals.add(g['candidate_ordinal']);hashes.add(g['question_hash'])


def complete_pool(config_path,outer):
    from reproduction.prefix_selection200.rollout import binding, root_dir, launch_phase, validate_selection, file_item
    from reproduction.prefix_selection200.periodic_data import read, digest, iter_rows, immutable, project_top512
    from reproduction.prefix20_online.pool import atomic
    c=binding(read(config_path),outer);plan=outer_plan(outer);root=root_dir(c,outer)
    assert c['actor50_branch_schedule']['mixed_floor']==MIXED_FLOOR
    prefix=read(root/'prefix/complete.json');score_path=root/'probe-score/manifest.json';score=read(score_path)
    predicted_path=root/'probe-score/selection.json';predicted=read(predicted_path)
    index_path=root/'probe-score/index.json';scores_path=root/'probe-score/candidate-scores.jsonl'
    if (digest(index_path)!=score['index_sha256'] or digest(scores_path)!=score['candidate_scores_sha256']
            or digest(predicted_path)!=score['selection_sha256'] or score['no_outcome_labels'] is not True
            or score['prefix_scan_sha256']!=prefix['scan_sha256']):
        raise ValueError('Immutable pre-outcome probe scores/prefixes changed')
    groups=list(iter_rows(scores_path));validate_scores(groups,plan['candidate_questions'])
    byid={g['qid']:g for g in groups};index={r['trajectory_id']:r for r in read(index_path)}
    expected_prefixes={tid:r['prefix_sha256'] for tid,r in index.items()}
    assert len(index)==len(groups)*5
    ranked=sorted(groups,key=rank_key);predicted_groups=validate_selection(predicted,prefix,plan['predicted_top_questions'])
    assert [g['qid'] for g in predicted_groups]==[g['qid'] for g in ranked[:plan['predicted_top_questions']]]
    sources=[];known={};seen=set();source_rows=set()

    def progress(stage,**extra):
        atomic(root/'mixed-floor-progress.json',dict(stage=stage,outer=outer,checked_unix=time.time(),
            completed_candidate_questions=len(known),candidate_questions=len(groups),**extra))

    def selection_value(subset,scope):
        return dict(predicted,selected=subset,selected_questions=len(subset),continuation_scope=scope,
            selection_semantics='original_pre_outcome_probe_ranked_disjoint_G5',outcome_labels_used=False)

    def collect(receipt,selection_path,subset):
        selection_sha=digest(selection_path);owners={tid:g for g in subset for tid in g['trajectory_ids']}
        if (receipt['run_id']!=c['run_id'] or receipt['outer']!=outer
                or receipt['actor_version']!=c['actor_version'] or receipt['questions']!=len(subset)
                or receipt['trajectories']!=len(subset)*5 or receipt['selection_sha256']!=selection_sha
                or receipt['prefix_scan_sha256']!=prefix['scan_sha256'] or seen.intersection(owners)):
            raise ValueError('Every refill must continue disjoint original G5s from this actor/prefix')
        item=receipt['own_em'];assert Path(item['path']).stat().st_size==item['bytes'] and digest(item['path'])==item['sha256']
        labels={}
        for row in iter_rows(item['path']):
            tid=row['trajectory_id'];g=owners.get(tid)
            if (g is None or tid in labels or row['qid']!=g['qid'] or type(row['strict_em']) is not int
                    or row['strict_em'] not in (0,1) or row['actor_version']!=c['actor_version']
                    or row['prefix_scan_sha256']!=prefix['scan_sha256'] or row['selection_sha256']!=selection_sha
                    or row['prefix_sha256']!=expected_prefixes[tid]):raise ValueError('StrictEM label provenance differs')
            labels[tid]=row['strict_em']
        assert set(labels)==set(owners)
        for shard in receipt['shards']:
            assert Path(shard['path']).stat().st_size==shard['bytes'] and digest(shard['path'])==shard['sha256']
            for row in iter_rows(shard['path']):
                tid=row['trajectory_id']
                if (tid not in owners or tid in source_rows or row['qid']!=owners[tid]['qid']
                        or row['strict_em']!=labels[tid] or row['actor_version']!=c['actor_version']
                        or row['selection_sha256']!=selection_sha or row['prefix_scan_sha256']!=prefix['scan_sha256']
                        or row['prefix_sha256']!=expected_prefixes[tid]):raise ValueError('Full native rows differ from original EM labels')
                source_rows.add(tid)
        assert set(owners)<=source_rows
        seen.update(owners)
        for g in subset:
            k=sum(labels[t] for t in g['trajectory_ids']);known[g['qid']]=dict(g,actual_correct_count=k,actual_mixed=0<k<5)
        sources.append(dict(receipt=receipt,selection=str(selection_path),selection_sha256=selection_sha))

    def continue_groups(subset,name,selection_path=None):
        assert len(subset)%4==0 and 4<=len(subset)<=2560
        sp=Path(selection_path) if selection_path else root/'probe-score'/(name+'-selection.json')
        if selection_path is None:immutable(sp,selection_value(subset,name))
        cfg=root/(name+'-config.json')
        immutable(cfg,dict(c,mixed_floor_continuation=True,rollout_continuation_questions=len(subset),
            rollout_full_subdirectory='full-pool' if plan['probe_update'] else name))
        progress('CONTINUING_ORIGINAL_G5',batch=name,batch_questions=len(subset))
        result=launch_phase(cfg,outer,'continue',sp);collect(result,sp,subset)
        progress('ORIGINAL_G5_COMPLETE',batch=name,batch_questions=len(subset))

    if plan['probe_update']:
        pool_path=root/'probe-score/pool-selection.json';pool=read(pool_path)
        pool_groups=validate_selection(pool,prefix,len(groups));continue_groups(pool_groups,'floor-initial',pool_path)
        if plan['actual_em_rerank']:initial,_=rank_actual_groups([known[g['qid']] for g in predicted_groups])
        else:
            # Preserve the original periodic all-pool oracle selection and tie
            # order. A depleted full pool can report a truthful floor shortfall.
            initial=(sorted((r for r in known.values() if is_mixed(r)),key=rank_key)+
                sorted((r for r in known.values() if not is_mixed(r)),key=rank_key))[:512]
    else:
        continue_groups(predicted_groups,'floor-initial')
        initial=([known[g['qid']] for g in predicted_groups] if not plan['actual_em_rerank'] else
            rank_actual_groups([known[g['qid']] for g in predicted_groups])[0])
    initial_ids={g['qid'] for g in initial};remaining=[g for g in ranked if g['qid'] not in initial_ids]
    chosen,stats=refill(initial,[known[g['qid']] for g in remaining if g['qid'] in known])
    checked_batches=0
    # Do not skip an unknown higher-ranked candidate in favor of a known lower
    # one. Initial periodic/rerank scopes are contiguous predicted prefixes.
    if stats['triggered'] and not stats['target_reached']:
        unknown=[g for g in remaining if g['qid'] not in known]
        for offset in range(0,len(unknown),MIXED_FLOOR['continuation_batch_questions']):
            batch=unknown[offset:offset+MIXED_FLOOR['continuation_batch_questions']]
            continue_groups(batch,'floor-extra-%04d'%(checked_batches+1));checked_batches+=1
            chosen,stats=refill(initial,[known[g['qid']] for g in remaining if g['qid'] in known])
            if stats['target_reached']:break
    stats.update(extra_native_batches=checked_batches,completed_candidate_questions=len(known),
        candidate_questions=len(groups),remaining_uncontinued_questions=len(groups)-len(known),
        exhausted=len(known)==len(groups),shortfall=max(0,154-stats['final_mixed']) if stats['triggered'] else 0,
        initial_scope_questions=plan['continuation_questions'],native_prefix_or_score_regenerated=False,
        policy=MIXED_FLOOR)
    if stats['triggered'] and not stats['target_reached']:assert stats['exhausted']

    if plan['probe_update']:
        pool_path=root/'probe-score/pool-selection.json';source_path=root/'full-pool/complete.json';source=read(source_path)
    else:
        # Merge existing completed events for projection only. Generated token
        # IDs, masks, sampled log probabilities and strictEM remain unchanged.
        out=root/'full-pool';out.mkdir(exist_ok=False);all_done=sorted((byid[q] for q in known),key=lambda r:r['candidate_ordinal'])
        pool_path=root/'probe-score/floor-completed-selection.json';immutable(pool_path,selection_value(all_done,'completed_floor_subset'))
        merged_sha=digest(pool_path);own=out/'own-em.jsonl';shards=[];ranks=[]
        paths=[out/('rank-%d-full.jsonl.gz'%r) for r in range(4)]
        destinations={tid:r for r in range(4) for g in all_done[r::4] for tid in g['trajectory_ids']}
        streams=[gzip.open(path,'xt',encoding='utf8',compresslevel=1) for path in paths];counts=[0]*4
        try:
            with own.open('x') as labels_out:
                for source_batch in sources:
                    for shard in source_batch['receipt']['shards']:
                        assert digest(shard['path'])==shard['sha256']
                        for original in iter_rows(shard['path']):
                            rank=destinations[original['trajectory_id']]
                            row=dict(original,original_batch_selection_sha256=original['selection_sha256'],selection_sha256=merged_sha)
                            streams[rank].write(json.dumps(row,sort_keys=True)+'\n');counts[rank]+=1
                            label={k:row[k] for k in ('trajectory_id','qid','strict_em','actor_version','prefix_sha256','prefix_scan_sha256','selection_sha256')}
                            labels_out.write(json.dumps(label,sort_keys=True)+'\n')
        finally:
            for stream in streams:stream.close()
        assert sum(counts)==len(all_done)*5
        for rank,path in enumerate(paths):
            item=file_item(path,counts[rank]);shards.append(item);ranks.append(dict(rank=rank,questions=counts[rank]//5,trajectories=counts[rank],full=item))
        source=dict(sources[0]['receipt'],questions=len(all_done),trajectories=len(all_done)*5,ranks=ranks,shards=shards,
            selection_sha256=merged_sha,own_em=file_item(own,len(all_done)*5),
            native_generation_requests=sum(s['receipt']['native_generation_requests'] for s in sources),
            actual_sampled_tokens=sum(s['receipt']['actual_sampled_tokens'] for s in sources),
            completed_native_batch_sources=[dict(selection=s['selection'],selection_sha256=s['selection_sha256'],
                own_em=s['receipt']['own_em'],shards=s['receipt']['shards']) for s in sources],
            merged_existing_events_only=True,all_candidate_mixed_accuracy_available=len(all_done)==len(groups))
        source_path=out/'complete.json';immutable(source_path,source)
    selected_mixed=sum(is_mixed(g) for g in chosen)
    value=dict(predicted,selected=chosen,selected_questions=512,outcome_labels_used=True,
        selection_scope='actor_only_mixed_floor30_actual_EM',predicted_selection_path=str(predicted_path),
        predicted_selection_sha256=digest(predicted_path),original_pool_selection_path=str(pool_path),
        original_pool_selection_sha256=digest(pool_path),full_pool_complete_path=str(source_path),
        full_pool_complete_sha256=digest(source_path),full_pool_own_em_path=source['own_em']['path'],
        full_pool_own_em_sha256=source['own_em']['sha256'],actual_em_used_only_after_full_pool_complete=True,
        actual_mixed_candidate_count=sum(is_mixed(g) for g in known.values()),chosen_mixed_count=selected_mixed,
        fallback_count=512-selected_mixed,fallback_mode=plan['fallback'],selection_mode='predicted_selection_with_conditional_mixed30_floor',
        mixed_floor=stats,original_continued_pool_questions=source['questions'])
    target=root/'probe-score/actual-em-selection.json';immutable(target,value)
    immutable(root/'mixed-floor-complete.json',dict(stats,actor_selection_sha256=digest(target),
        predicted_selection_sha256=digest(predicted_path),full_pool_complete_sha256=digest(source_path)))
    result=project_top512(config_path,outer,target);progress('MIXED_FLOOR_COMPLETE',mixed_floor=stats)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--outer',type=int,required=True)
    a=p.parse_args();complete_pool(a.config,a.outer)


if __name__=='__main__':main()
