"""Compact per-outer facts from verified actual rank ledgers; no remote operations."""
import argparse
import json
from pathlib import Path
import shutil


def summarize(root):
    root=Path(root);out=root/'outputs/search-g1-i1i6-fresh200-20261017'
    facts=json.loads((out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json').read_text())
    rows=[]
    for outer in sorted(map(int,facts)):
        fact=facts[str(outer)];selection=fact['actor_selection'];probe=fact['probe_metrics']
        curve=probe['global_step_loss_curve'];points=curve['points']
        if (outer!=len(rows)+1 or len(points)!=20 or [v['step']for v in points]!=list(range(1,21))
                or curve['actual_adam_updates_per_rank']!=20
                or probe['actual_adam_steps_per_rank']!=[outer*20]*8
                or any(v['writes']!=20 for v in fact['actor']+fact['probe'])
                or sum(v['global_questions']for v in points)!=1024
                or sum(v['global_trajectories']for v in points)!=5120):
            raise ValueError('Per-outer summary must bind the complete natural1024G5/20-write clock')
        selected,continued=selection['actor512'],selection['continued1024']
        if selected['questions']!=512 or continued['questions']!=1024 or selection['uncontinued_candidate_questions']!=4096:
            raise ValueError('Do not merge selected512 and continued1024 metrics')
        rows.append(dict(outer=outer,run_id=selection['run_id'],actor_sampled_version=selection['actor_version'],
            actual_actor_adam_per_rank=outer*20,actual_probe_adam_per_rank=outer*20,
            selected512_mixed_fraction=selected['mixed_fraction'],selected512_mean_prediction=selected['mean_predicted_mixed_probability'],
            selected512_population_em_variance=selected['mean_population_em_variance'],
            continued1024_mixed_fraction=continued['mixed_fraction'],continued1024_mean_prediction=continued['mean_predicted_mixed_probability'],
            continued1024_population_em_variance=continued['mean_population_em_variance'],
            selected512_correct_count_histogram=selected['correct_count_histogram'],
            continued1024_correct_count_histogram=continued['correct_count_histogram'],
            post_update_actor_selected512_probe_accuracy=probe['actor_selected_train']['accuracy'],
            post_update_actor_selected512_probe_balanced_accuracy=probe['actor_selected_train']['balanced_accuracy'],
            post_update_continued1024_probe_accuracy=probe['train']['accuracy'],
            post_update_continued1024_probe_balanced_accuracy=probe['train']['balanced_accuracy'],
            post_update_fixed512_probe_accuracy=probe['test']['accuracy'],
            post_update_fixed512_probe_balanced_accuracy=probe['test']['balanced_accuracy'],
            probe_learning_rate=probe['actual_learning_rate'],probe_online_total_loss=probe['train_online_total_loss'],
            probe_online_group_bce=probe['train_online_group_bce'],probe_online_trajectory_bce=probe['train_online_trajectory_bce'],
            natural_step1_loss=points[0]['total_loss'],natural_step20_loss=points[-1]['total_loss'],
            natural20_step_losses=[{k:p[k]for k in ('step','global_questions','global_trajectories','total_loss','group_bce','trajectory_bce')}for p in points],
            same_batch_loss_comparison=False,monotonic_loss_decrease_claimed=False,
            phase_seconds={name:value['elapsed_seconds']for name,value in fact['phase_timings'].items()},
            normal_phase_total_seconds=sum(value['elapsed_seconds']for value in fact['phase_timings'].values()),
            all5120_candidate_accuracy_available=False,uncontinued_candidate_questions=4096,
            actual_joint_receipt_sha256=fact['joint_sha256']))
    value=dict(schema_version=1,complete_outer_count=len(rows),rows=rows,
        semantics='Selected512 and continued1024 are different populations. Fixed512 probe classification is not actor QA. Twenty natural batches use different questions and successive head states; step1-to20 difference is not same-batch fitting evidence. Uncontinued4096 have no outcome labels.')
    p=out/'MIXED_AND_PROBE_LOSS_SERIES.json';p.write_text(json.dumps(value,indent=2)+'\n')
    for dest in (root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002'/out.name,
                 Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')/out.name):
        dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest/p.name)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=summarize(a.root)
    print(json.dumps(dict(complete_outer_count=r['complete_outer_count'],latest=r['rows'][-1]if r['rows']else None)))
