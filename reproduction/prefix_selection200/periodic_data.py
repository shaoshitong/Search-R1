"""Complete the unselected pool once, and project the preselected PPO subset."""
import argparse
import gzip
import json
from pathlib import Path

from .pool import identity, immutable, read, digest
from .rollout import binding, root_dir, launch_phase, iter_rows, file_item, validate_selection


def project_top512(config_path, outer):
    """No generation: select exact completed full-pool rows for actor PPO."""
    c=binding(read(config_path),outer);root=root_dir(c,outer)
    source=read(root/'full-pool/complete.json')
    prefix=read(root/'prefix/complete.json')
    top_path=root/'probe-score/selection.json';selection=read(top_path)
    groups=validate_selection(selection,prefix)
    wanted={tid for group in groups for tid in group['trajectory_ids']}
    rows={}
    for shard in source['shards']:
        for row in iter_rows(shard['path']):
            if row['trajectory_id'] in wanted:rows[row['trajectory_id']]=row
    if set(rows)!=wanted or source['trajectories']!=25600 or source['actor_version']!=c['actor_version']:
        raise ValueError('Projection requires every original Top512 trajectory from this same pool rollout')
    out=root/'full';out.mkdir(exist_ok=False)
    ranks=[];labels=[]
    for rank in range(8):
        selected=[rows[tid] for group in groups[rank::8] for tid in group['trajectory_ids']]
        full=out/('rank-%d-full.jsonl.gz'%rank);own=out/('rank-%d-own-em.jsonl'%rank)
        with gzip.open(full,'xt',encoding='utf8',compresslevel=1) as f,own.open('x') as g:
            for original in selected:
                # Preserve origin, all sampled IDs/LP and strictEM. Only the
                # subset's bookkeeping selection binding changes explicitly.
                row=dict(original,full_pool_origin_selection_sha256=original['selection_sha256'],
                         full_pool_complete_sha256=digest(root/'full-pool/complete.json'),
                         selection_sha256=digest(top_path))
                f.write(json.dumps(row,sort_keys=True)+'\n')
                label={k:row[k] for k in ('trajectory_id','qid','strict_em','actor_version',
                                         'prefix_sha256','prefix_scan_sha256','selection_sha256')}
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
        projected_from_full_pool=True,full_pool_complete_sha256=digest(root/'full-pool/complete.json'),
        projected_generation_requests=0,actual_selected_mixed_questions=sum(hist[k] for k in (1,2,3,4)),
        actual_selected_mixed_fraction=sum(hist[k] for k in (1,2,3,4))/512,
        selected_strict_correct=sum(label_by_id.values()),selected_correct_count_histogram=hist,
        all_candidate_mixed_accuracy_available=True)
    immutable(out/'complete.json',value)
    return value


def complete_pool(config_path,outer):
    from .periodic import is_probe_update
    if not is_probe_update(outer):raise ValueError('Full-pool outcomes only at20,40,...200')
    c=read(config_path);root=root_dir(c,outer)
    selection=root/'probe-score/pool-selection.json'
    if not selection.exists():raise ValueError('Pre-update probe must finish Top512 selection first')
    scoped=root/'full-pool-config.json'
    immutable(scoped,dict(c,rollout_continuation_questions=5120,rollout_full_subdirectory='full-pool'))
    launch_phase(scoped,outer,'continue',selection)
    return project_top512(config_path,outer)


def diagnostic(config_path,outer):
    """Fixed heldout question identities, fresh G5 from exactly this actor."""
    import pyarrow.parquet as pq
    c=binding(read(config_path),outer);root=root_dir(c,outer)
    original=Path(c['fixed_test_features'])
    manifest=read(original/'manifest.json')
    # FeatureCache owns the immutable original question/trajectory identity.
    from reproduction.prefix20_question_replay.features import FeatureCache
    fixed=FeatureCache(original,c['fixed_test_manifest_sha256'])
    wanted={str(row['qid']):row['question_hash'] for row in fixed.index.values()}
    if len(wanted)!=512:raise ValueError('Fixed diagnostic question identities changed')
    table=pq.read_table(c['canonical_train'],columns=['prompt','reward_model','data_source','extra_info'])
    rows=[]
    for source_index,original_row in enumerate(table.to_pylist()):
        key=identity(original_row)
        if key['qid'] in wanted:
            if key['question_hash']!=wanted[key['qid']]:raise ValueError('Diagnostic question content changed')
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
        original_fixed_identity_manifest_sha256=c['fixed_test_manifest_sha256'],
        training_questions_used=False,independent_encoder_forward=False)
    immutable(root/'diagnostic-complete.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('complete-pool','diagnostic'))
    p.add_argument('--config',required=True);p.add_argument('--outer',type=int,required=True);a=p.parse_args()
    value=complete_pool(a.config,a.outer) if a.phase=='complete-pool' else diagnostic(a.config,a.outer)
    print(json.dumps(value))


if __name__=='__main__':main()
