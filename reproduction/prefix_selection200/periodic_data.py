"""On update rounds select actual mixed G5s from the completed candidate pool.

The original predicted selection and label-free score manifest are immutable.
Only10..190 update rounds may use own EM for actor selection; ordinary rounds
continue their existing predicted Top512 without reading this module's labels.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

from .pool import fixed_test_identities, identity, immutable, read, digest
from .rollout import binding, root_dir, launch_phase, iter_rows, file_item, validate_selection
from .contract import PERIODIC_CANDIDATES, G, SELECTED_QUESTIONS, WORLD
from .candidate_schedule import candidate_count, schedule_metadata


FALLBACK_MODES = ('probe_score_fill', 'require_512_mixed')


def select_actual_mixed_groups(groups, records, *, fallback_mode, provenance, expected_prefixes, expected_questions=PERIODIC_CANDIDATES, ordinal_limit=None):
    """Actual mixed first, with explicit policy for a pool containing <512 mixed.

Every label is bound to the same authentic candidate G5/prefix/actor and the
original full-pool continuation selection. Ranking within the mixed pool and
the optional nonmixed fill remains predicted pMixed descending/ordinal ties.
"""
    if fallback_mode not in FALLBACK_MODES:
        raise ValueError('An explicit actual-EM mixed shortfall fallback mode is required')
    ordinal_limit=expected_questions if ordinal_limit is None else ordinal_limit
    if expected_questions not in (1024,PERIODIC_CANDIDATES) or len(groups) != expected_questions:
        raise ValueError('Actual-EM selection requires the complete2560 candidate questions')
    owners, qids, ordinals, hashes = {}, set(), set(), set()
    for group in groups:
        qid, ordinal, probability = group['qid'], group['candidate_ordinal'], group['q']
        if (qid in qids or ordinal in ordinals or group['question_hash'] in hashes
                or type(ordinal) is not int or not 0 <= ordinal < ordinal_limit
                or type(probability) not in (float, int) or not math.isfinite(probability)
                or not 0 <= probability <= 1 or len(group['trajectory_ids']) != G
                or len(set(group['trajectory_ids'])) != G):
            raise ValueError('Candidate group identity, predicted score or authentic G5 differs')
        qids.add(qid); ordinals.add(ordinal); hashes.add(group['question_hash'])
        for tid in group['trajectory_ids']:
            if tid in owners:
                raise ValueError('A candidate trajectory occurs in more than one G5')
            owners[tid] = qid
    if set(expected_prefixes) != set(owners):
        raise ValueError('Every candidate requires its exact scored prefix identity')
    labels = {}
    for row in records:
        tid = row['trajectory_id']
        if (tid not in owners or tid in labels or row['qid'] != owners[tid]
                or type(row.get('strict_em')) is not int or row['strict_em'] not in (0, 1)
                or row.get('actor_version') != provenance['actor_version']
                or row.get('prefix_scan_sha256') != provenance['prefix_scan_sha256']
                or row.get('selection_sha256') != provenance['pool_selection_sha256']
                or row.get('prefix_sha256') != expected_prefixes[tid]):
            raise ValueError('Actual EM label changed its actor, original pool selection or scored prefix')
        labels[tid] = row['strict_em']
    if set(labels) != set(owners):
        raise ValueError('Actual-EM selection requires all12800 original own-trajectory labels')
    histogram = {k: 0 for k in range(G + 1)}
    mixed, extreme = [], []
    for group in groups:
        correct = sum(labels[tid] for tid in group['trajectory_ids'])
        histogram[correct] += 1
        actual = dict(group, actual_correct_count=correct, actual_mixed=0 < correct < G)
        (mixed if actual['actual_mixed'] else extreme).append(actual)
    rank = lambda row: (-row['q'], row['candidate_ordinal'])
    mixed.sort(key=rank); extreme.sort(key=rank)
    if len(mixed) < SELECTED_QUESTIONS and fallback_mode == 'require_512_mixed':
        raise ValueError('The completed candidate pool has fewer than512 actual mixed G5s; no implicit fill')
    selected = mixed[:SELECTED_QUESTIONS]
    fallback_count = SELECTED_QUESTIONS - len(selected)
    if fallback_count:
        selected.extend(extreme[:fallback_count])
    if len(selected) != SELECTED_QUESTIONS:
        raise ValueError('Actual-EM selector failed to retain exactly512 complete G5s')
    return selected, dict(actual_mixed_candidate_count=len(mixed),
        actual_nonmixed_candidate_count=len(extreme), chosen_mixed_count=min(len(mixed), SELECTED_QUESTIONS),
        fallback_count=fallback_count, fallback_mode=fallback_mode,
        selection_mode='actual_mixed_then_probe_score_fill' if fallback_count else 'actual_mixed_only_probe_ranked',
        ranking_within_actual_class='predicted_pMixed_descending_then_original_candidate_ordinal',
        actual_candidate_correct_count_histogram=histogram,
        all_selected_are_actual_mixed=fallback_count == 0)


def actual_em_selection(config_path, outer, *, allow_two_stage=False):
    """Write a distinct final actor selection after all candidate G5s completed."""
    from .periodic import is_probe_update
    if not is_probe_update(outer) and not allow_two_stage:
        raise ValueError('Only periodic update rounds10..190 may use final EM without an explicit two-stage entry')
    c = binding(read(config_path), outer); root = root_dir(c, outer)
    from .two_stage_selection import ordinary
    two_stage=ordinary(c,outer)
    if not is_probe_update(outer) and not two_stage:
        raise ValueError('Actual-EM selection needs an authorized periodic or two-stage scope')
    prefix_candidates=candidate_count(c,outer)
    candidates = 1024 if two_stage else prefix_candidates
    if not two_stage and candidates != PERIODIC_CANDIDATES:
        raise ValueError('Actual-EM selection is limited to the scheduled periodic2560 pool')
    fallback_mode = c.get('actual_em_selection_fallback')
    if fallback_mode not in FALLBACK_MODES:
        raise ValueError('Bind an explicit actual_em_selection_fallback before reading outcomes')
    prefix = read(root / 'prefix/complete.json')
    scored_path = root / 'probe-score/pool-selection.json'
    predicted_path = root / 'probe-score/selection.json'
    scored = read(scored_path)
    source_path = root / 'full-pool/complete.json'; source = read(source_path)
    groups = validate_selection(scored, prefix, candidates)
    expected = dict(run_id=c['run_id'], outer=outer, actor_version=c['actor_version'],
        questions=candidates, trajectories=candidates * G,
        prefix_scan_sha256=prefix['scan_sha256'], selection_sha256=digest(scored_path))
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError('Full-pool outcomes belong to another actual actor/prefix/selection')
    manifest = read(root / 'probe-score/manifest.json')
    index_path = root / 'probe-score/index.json'
    if (manifest['index_sha256'] != digest(index_path)
            or manifest['selection_sha256'] != digest(predicted_path)
            or manifest['prefix_scan_sha256'] != prefix['scan_sha256']):
        raise ValueError('Original label-free score index or provisional predicted selection changed')
    indexed = read(index_path)
    chosen_ids={tid for group in groups for tid in group['trajectory_ids']}
    expected_prefixes = {row['trajectory_id']: row['prefix_sha256'] for row in indexed if row['trajectory_id'] in chosen_ids}
    if len(expected_prefixes) != candidates * G:
        raise ValueError('Scored full-pool prefix identities are missing or duplicated')
    own = Path(source['own_em']['path'])
    if own.stat().st_size != source['own_em']['bytes'] or digest(own) != source['own_em']['sha256']:
        raise ValueError('Complete full-pool own-EM file differs from its producer SHA')
    selected, stats = select_actual_mixed_groups(groups, iter_rows(own), fallback_mode=fallback_mode,
        provenance=dict(actor_version=c['actor_version'], prefix_scan_sha256=prefix['scan_sha256'],
                        pool_selection_sha256=digest(scored_path)), expected_prefixes=expected_prefixes,
        expected_questions=candidates,ordinal_limit=prefix_candidates)
    result = dict(schema_version=1, run_id=c['run_id'], outer=outer, actor_version=c['actor_version'],
        prefix_scan_sha256=prefix['scan_sha256'], selected=selected, selected_questions=SELECTED_QUESTIONS,
        candidate_questions=candidates, trajectories_per_question=G, exploration=0,
        outcome_labels_used=True, selection_scope=('actor_only_probe_top1024_actual_EM' if two_stage else 'actor_only_periodic_full_pool_actual_EM'),
        prefix_candidate_questions=prefix_candidates,full_rollout_candidate_questions=candidates,
        predicted_selection_path=str(predicted_path), predicted_selection_sha256=digest(predicted_path),
        original_pool_selection_path=str(scored_path), original_pool_selection_sha256=digest(scored_path),
        full_pool_complete_path=str(source_path), full_pool_complete_sha256=digest(source_path),
        full_pool_own_em_path=str(own), full_pool_own_em_sha256=source['own_em']['sha256'],
        actual_em_used_only_after_full_pool_complete=True, **schedule_metadata(c, outer), **stats)
    target = root / 'probe-score/actual-em-selection.json'
    immutable(target, result)
    return target


def project_top512(config_path, outer, selection_path=None):
    """No generation: select exact completed full-pool rows for actor PPO."""
    c=binding(read(config_path),outer);root=root_dir(c,outer)
    from .two_stage_selection import ordinary
    two_stage=ordinary(c,outer)
    candidates = 1024 if two_stage else candidate_count(c, outer)
    if not two_stage and candidates != PERIODIC_CANDIDATES:
        raise ValueError('Only authorized full-pool outcomes can be projected for the actor')
    source=read(root/'full-pool/complete.json')
    prefix=read(root/'prefix/complete.json')
    top_path=Path(selection_path or root/'probe-score/actual-em-selection.json');selection=read(top_path)
    if selection.get('outcome_labels_used') is not True or selection.get('selection_scope') != ('actor_only_probe_top1024_actual_EM' if two_stage else 'actor_only_periodic_full_pool_actual_EM'):
        raise ValueError('Periodic actor projection requires its separate actual-EM selection')
    groups=validate_selection(selection,prefix)
    wanted={tid for group in groups for tid in group['trajectory_ids']}
    own_labels = {row['trajectory_id']: row for row in iter_rows(source['own_em']['path'])}
    rows={}
    for shard in source['shards']:
        if Path(shard['path']).stat().st_size != shard['bytes'] or digest(shard['path']) != shard['sha256']:
            raise ValueError('Full-pool native continuation shard changed before actor projection')
        for row in iter_rows(shard['path']):
            if row['trajectory_id'] in wanted:
                if row['trajectory_id'] in rows:
                    raise ValueError('Selected original trajectory repeats in full-pool native rows')
                label = own_labels.get(row['trajectory_id'])
                if label is None or any(row[key] != label[key] for key in (
                        'qid', 'strict_em', 'actor_version', 'prefix_sha256', 'prefix_scan_sha256', 'selection_sha256')):
                    raise ValueError('Projected native trajectory differs from the actual own-EM event used for selection')
                rows[row['trajectory_id']]=row
    if (set(rows)!=wanted or source['trajectories']!=candidates * G
            or source['actor_version']!=c['actor_version']
            or selection['full_pool_complete_sha256'] != digest(root/'full-pool/complete.json')
            or selection['original_pool_selection_sha256'] != source['selection_sha256']
            or selection['full_pool_own_em_sha256'] != source['own_em']['sha256']):
        raise ValueError('Projection requires every original Top512 trajectory from this same pool rollout')
    out=root/'full';out.mkdir(exist_ok=False)
    ranks=[];labels=[]
    for rank in range(WORLD):
        selected=[rows[tid] for group in groups[rank::WORLD] for tid in group['trajectory_ids']]
        full=out/('rank-%d-full.jsonl.gz'%rank);own=out/('rank-%d-own-em.jsonl'%rank)
        with gzip.open(full,'xt',encoding='utf8',compresslevel=1) as f,own.open('x') as g:
            for original in selected:
                # Preserve origin, all sampled IDs/LP and strictEM. Only the
                # subset's bookkeeping selection binding changes explicitly.
                row=dict(original,full_pool_origin_selection_sha256=original['selection_sha256'],
                         full_pool_complete_sha256=digest(root/'full-pool/complete.json'),
                         full_pool_own_em_sha256=source['own_em']['sha256'],
                         predicted_selection_sha256=selection['predicted_selection_sha256'],
                         actor_selection_outcome_labels_used=True,
                         selection_sha256=digest(top_path))
                f.write(json.dumps(row,sort_keys=True)+'\n')
                label={k:row[k] for k in ('trajectory_id','qid','strict_em','actor_version',
                    'prefix_sha256','prefix_scan_sha256','selection_sha256','full_pool_origin_selection_sha256',
                    'full_pool_complete_sha256','full_pool_own_em_sha256','predicted_selection_sha256')}
                labels.append(label);g.write(json.dumps(label,sort_keys=True)+'\n')
        ranks.append(dict(rank=rank,trajectories=len(selected),questions=len(selected)//5,
            actor_version=c['actor_version'],full=file_item(full,len(selected)),own_em=file_item(own,len(selected)),
            native_generation_performed=False,full_pool_origin_complete_sha256=digest(root/'full-pool/complete.json')))
    own=out/'own-em.jsonl'
    with own.open('x') as f:
        for label in labels:f.write(json.dumps(label,sort_keys=True)+'\n')
    label_by_id={x['trajectory_id']:x['strict_em'] for x in labels}
    hist={k:sum(sum(label_by_id[tid] for tid in group['trajectory_ids'])==k for group in groups) for k in range(6)}
    value=dict(source,schema_version=2,phase='continue',config_sha256=digest(config_path),
        ranks=ranks,trajectories=2560,questions=512,shards=[r['full'] for r in ranks],
        own_em=file_item(own,2560),selection_sha256=digest(top_path),
        actor_selection_path=str(top_path.resolve()),actor_selection_sha256=digest(top_path),
        actor_selection_outcome_labels_used=True,
        predicted_selection_sha256=selection['predicted_selection_sha256'],
        original_pool_selection_sha256=selection['original_pool_selection_sha256'],
        full_pool_origin_selection_sha256=selection['original_pool_selection_sha256'],
        full_pool_own_em_sha256=source['own_em']['sha256'],
        actual_mixed_candidate_count=selection['actual_mixed_candidate_count'],
        chosen_mixed_count=selection['chosen_mixed_count'],fallback_count=selection['fallback_count'],
        selection_mode=selection['selection_mode'],actual_em_selection_fallback=selection['fallback_mode'],
        projected_from_full_pool=True,full_pool_complete_sha256=digest(root/'full-pool/complete.json'),
        projected_generation_requests=0,actual_selected_mixed_questions=sum(hist[k] for k in (1,2,3,4)),
        actual_selected_mixed_fraction=sum(hist[k] for k in (1,2,3,4))/512,
        selected_strict_correct=sum(label_by_id.values()),selected_correct_count_histogram=hist,
        all_candidate_mixed_accuracy_available=True)
    immutable(out/'complete.json',value)
    return value


def complete_pool(config_path,outer):
    from .periodic import is_probe_update
    if not is_probe_update(outer):raise ValueError('Full-pool outcomes only at10,20,...190; no update at200')
    c=read(config_path);root=root_dir(c,outer)
    candidates = candidate_count(c, outer)
    if candidates != PERIODIC_CANDIDATES:
        raise ValueError('Periodic probe update retains its exact2560G5 pool')
    selection=root/'probe-score/pool-selection.json'
    if not selection.exists():raise ValueError('Pre-update probe must finish its label-free provisional score first')
    scoped=root/'full-pool-config.json'
    immutable(scoped,dict(c,rollout_continuation_questions=candidates,rollout_full_subdirectory='full-pool'))
    launch_phase(scoped,outer,'continue',selection)
    final_selection=actual_em_selection(config_path,outer)
    return project_top512(config_path,outer,final_selection)


def diagnostic(config_path,outer):
    """Fixed heldout question identities, fresh G5 from exactly this actor."""
    import pyarrow.parquet as pq
    c=binding(read(config_path),outer);root=root_dir(c,outer)
    fixed_rows=fixed_test_identities(c)
    if fixed_rows is not None:
        wanted={row['qid']:row['question_hash'] for row in fixed_rows}
        normalized={row['qid']:row['normalized_question_hash'] for row in fixed_rows}
        identity_sha=c['fixed_test_identity_sha256']
    else:
        # LegacyH100 inputs remain compatible; A800 supplies explicit identities.
        from reproduction.prefix20_question_replay.features import FeatureCache
        fixed=FeatureCache(Path(c['fixed_test_features']),c['fixed_test_manifest_sha256'])
        wanted={str(row['qid']):row['question_hash'] for row in fixed.index.values()}
        normalized=None
        identity_sha=c['fixed_test_manifest_sha256']
    if len(wanted)!=512:raise ValueError('Fixed diagnostic question identities changed')
    table=pq.read_table(c['canonical_train'],columns=['prompt','reward_model','data_source','extra_info'])
    rows=[]
    for source_index,original_row in enumerate(table.to_pylist()):
        key=identity(original_row)
        if key['qid'] in wanted:
            if key['question_hash']!=wanted[key['qid']]:raise ValueError('Diagnostic question content changed')
            if normalized is not None and key['normalized_question_hash']!=normalized[key['qid']]:
                raise ValueError('Diagnostic normalized question content changed')
            rows.append(dict(original_row,**key,source_index=source_index,candidate_ordinal=len(rows)))
    if len(rows)!=512:raise ValueError('Fixed diagnostic is not fully represented in canonical questions')
    base=root/'diagnostic';r=root_dir(dict(base=str(base)),outer);r.mkdir(parents=True,exist_ok=False)
    immutable(r/'questions.json',dict(run_id=c['run_id'],outer=outer,questions=512,rows=rows))
    immutable(r/'actor-binding.json',dict(actor_model_dir=c['actor_model_dir'],actor_version=c['actor_version']))
    conf=base/'preparation/launch-config.json'
    immutable(conf,dict(c,base=str(base),diagnostic_only=True,rollout_candidate_questions=512,
        rollout_continuation_questions=512,rollout_full_subdirectory='full'))
    prefix=launch_phase(conf,outer,'prefix')
    scan=read(Path(prefix['scan_path'])/'scan-complete.json')
    from .probe import candidate_groups
    index={x['trajectory_id']:x for x in iter_rows(Path(prefix['scan_path'])/'index.jsonl')}
    groups=candidate_groups(index,512)
    selection=r/'diagnostic-all-selection.json'
    immutable(selection,dict(run_id=c['run_id'],outer=outer,actor_version=c['actor_version'],
        prefix_scan_sha256=prefix['scan_sha256'],selected=groups,selected_questions=512,
        outcome_labels_used=False,diagnostic_only=True))
    full=launch_phase(conf,outer,'continue',selection)
    receipt=dict(run_id=c['run_id'],outer=outer,actor_version=c['actor_version'],questions=512,
        trajectories=2560,scan_path=prefix['scan_path'],scan_sha256=prefix['scan_sha256'],
        full_complete=str(r/'full/complete.json'),full_complete_sha256=digest(r/'full/complete.json'),
        original_fixed_identity_manifest_sha256=identity_sha,
        fixed_test_identity_path=c.get('fixed_test_identity_path'),
        fixed_test_identity_sha256=c.get('fixed_test_identity_sha256'),
        training_questions_used=False,independent_encoder_forward=False)
    immutable(root/'diagnostic-complete.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('complete-pool','diagnostic'))
    p.add_argument('--config',required=True);p.add_argument('--outer',type=int,required=True);a=p.parse_args()
    value=complete_pool(a.config,a.outer) if a.phase=='complete-pool' else diagnostic(a.config,a.outer)
    print(json.dumps(value))


if __name__=='__main__':main()
