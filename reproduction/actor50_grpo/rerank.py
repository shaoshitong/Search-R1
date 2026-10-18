"""After150 restrict actual-EM actor selection to predicted Top1024 G5s."""
import argparse
from pathlib import Path

from .schedule import outer_plan,rank_actual_groups


def actual_selection(config_path,outer):
    from reproduction.prefix_selection200.rollout import binding,root_dir,validate_selection
    from reproduction.prefix_selection200.periodic_data import read,digest,iter_rows,immutable
    c=binding(read(config_path),outer);plan=outer_plan(outer);assert plan['actual_em_rerank']
    root=root_dir(c,outer);prefix=read(root/'prefix/complete.json')
    predicted_path=root/'probe-score/selection.json';predicted=read(predicted_path)
    predicted_groups=validate_selection(predicted,prefix,1024)
    assert predicted['outcome_labels_used'] is False
    score=read(root/'probe-score/manifest.json')
    assert score['selection_sha256']==digest(predicted_path) and score['prefix_scan_sha256']==prefix['scan_sha256']
    index_path=root/'probe-score/index.json';assert digest(index_path)==score['index_sha256']
    index={r['trajectory_id']:r for r in read(index_path)};assert len(index)==2560*5
    source_path=root/'full-pool/complete.json';source=read(source_path)
    pool_path=root/'probe-score/pool-selection.json' if plan['probe_update'] else predicted_path
    pool_sha=digest(pool_path)
    pool_groups=validate_selection(read(pool_path),prefix,plan['continuation_questions'])
    if (source['run_id']!=c['run_id'] or source['outer']!=outer
            or source['actor_version']!=c['actor_version'] or source['prefix_scan_sha256']!=prefix['scan_sha256']
            or source['selection_sha256']!=pool_sha
            or source['questions']!=plan['continuation_questions']
            or source['trajectories']!=plan['continuation_questions']*5):
        raise ValueError('Actual outcomes must bind this exact original native continuation pool')
    owners={tid:group for group in pool_groups for tid in group['trajectory_ids']}
    assert len(owners)==source['trajectories']
    own=Path(source['own_em']['path']);assert own.stat().st_size==source['own_em']['bytes'] and digest(own)==source['own_em']['sha256']
    labels={}
    for row in iter_rows(own):
        tid=row['trajectory_id'];owner=owners.get(tid)
        if (owner is None or tid in labels or row['qid']!=owner['qid']
                or type(row['strict_em']) is not int or row['strict_em'] not in (0,1)
                or row['actor_version']!=c['actor_version']
                or row['prefix_scan_sha256']!=prefix['scan_sha256']
                or row['selection_sha256']!=pool_sha
                or row['prefix_sha256']!=index[tid]['prefix_sha256']):
            raise ValueError('Own-G5 labels are incomplete, duplicated or bound to another actor/prefix')
        labels[tid]=row['strict_em']
    assert set(labels)==set(owners)
    restricted=[]
    for group in predicted_groups:
        correct=sum(labels[tid] for tid in group['trajectory_ids'])
        restricted.append(dict(group,actual_correct_count=correct,actual_mixed=0<correct<5))
    selected,stats=rank_actual_groups(restricted)
    histogram={k:sum(r['actual_correct_count']==k for r in restricted) for k in range(6)}
    value=dict(schema_version=1,run_id=c['run_id'],outer=outer,actor_version=c['actor_version'],
        prefix_scan_sha256=prefix['scan_sha256'],selected=selected,selected_questions=512,
        candidate_questions=2560,trajectories_per_question=5,exploration=0,outcome_labels_used=True,
        selection_scope='actor_only_predicted_top1024_actual_EM',
        predicted_selection_path=str(predicted_path),predicted_selection_sha256=digest(predicted_path),
        original_pool_selection_path=str(pool_path),original_pool_selection_sha256=pool_sha,
        full_pool_complete_path=str(source_path),full_pool_complete_sha256=digest(source_path),
        full_pool_own_em_path=str(own),full_pool_own_em_sha256=source['own_em']['sha256'],
        actual_em_used_only_after_full_pool_complete=True,actor_candidate_scope=1024,
        actual_mixed_candidate_count=stats['mixed_available'],
        actual_nonmixed_candidate_count=1024-stats['mixed_available'],
        chosen_mixed_count=stats['selected_mixed'],fallback_count=512-stats['selected_mixed'],
        fallback_mode='all_correct_then_all_wrong',selection_mode='actual_mixed_then_all_correct_then_all_wrong',
        ranking_within_actual_class='predicted_pMixed_descending_then_original_candidate_ordinal',
        actual_candidate_correct_count_histogram=histogram,all_selected_are_actual_mixed=stats['selected_mixed']==512,
        selection_stats=stats,original_continued_pool_questions=source['questions'])
    target=root/'probe-score/actual-em-selection.json';immutable(target,value);return target


def complete_pool(config_path,outer):
    from reproduction.prefix_selection200.rollout import launch_phase,root_dir
    from reproduction.prefix_selection200.periodic_data import read,immutable,project_top512
    c=read(config_path);plan=outer_plan(outer);assert plan['actual_em_rerank']
    root=root_dir(c,outer);selection=root/'probe-score/pool-selection.json' if plan['probe_update'] else root/'probe-score/selection.json'
    scoped=root/'rerank-full-pool-config.json'
    immutable(scoped,dict(c,rollout_continuation_questions=plan['continuation_questions'],rollout_full_subdirectory='full-pool'))
    launch_phase(scoped,outer,'continue',selection)
    return project_top512(config_path,outer,actual_selection(config_path,outer))


def arguments(phase=False):
    p=argparse.ArgumentParser()
    if phase:p.add_argument('phase',choices=('complete-pool',))
    p.add_argument('--config',required=True);p.add_argument('--outer',type=int,required=True)
    return p.parse_args()


def complete_main():
    a=arguments(True);complete_pool(a.config,a.outer)


def main():
    a=arguments();complete_pool(a.config,a.outer)
