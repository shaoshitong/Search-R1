"""Small committed metrics only; no tensor/rollout reads or training writes.

An explicit base and through-boundary keep a rolled-back execution's later
outers separate from the resumed lineage. This does not reaudit rank journals.
"""
import hashlib
import json
from pathlib import Path

MODE='b_cp50_prefix20_40_80_two_stage_v1'
METRICS=('count','accuracy','balanced_accuracy','mixed_recall','nonmixed_recall',
    'trajectory_bce','group_bce','group_weight','total_loss','mean_q','tp','tn','fp','fn')


def metric_row(config,joint,outer):
    if joint['run_id']!=config['run_id'] or joint['outer']!=outer:
        raise ValueError('Committed metric identity differs')
    selected=joint['selection_metrics']
    if (selected['questions']!=512 or sum(selected[k]for k in ('mixed','all_wrong','all_correct'))!=512
            or selected['mixed_fraction']!=selected['mixed']/512):
        raise ValueError('Selected G5 count/rate differs')
    progressive=config.get('b_progressive',{}).get('mode')==MODE
    prefix=(5 if outer<=50 else 20 if outer<=100 else 40 if outer<=150 else 80)if progressive else 5
    periodic=outer%10==0 and outer<200
    admitted=2560 if periodic else ({20:768,40:1024,80:1536}[prefix]if progressive and outer>50 else 512)
    update=joint['probe_metrics'];sampling=update.get('sampling',{})
    paired={phase:{kind:{split:{k:v for k,v in value.items()if k in METRICS}
        for split,value in splits.items()}for kind,splits in update.get(phase,{}).items()}
        for phase in ('before_update','after_update')}
    curve=update.get('global_step_loss_curve',[])
    curve_summary=dict(count=len(curve),first=curve[0]if curve else None,last=curve[-1]if curve else None)
    timings=joint.get('phase_timings',{})
    walls=[value for key,value in timings.items()if key.endswith('-training-wall')]
    return dict(outer=outer,prefix_tokens=prefix,periodic=periodic,
        candidate_questions=2560 if outer<=5 or periodic else 5120,
        continued_questions=admitted,selected=selected,
        final_actor_selection_used_outcomes=joint.get('actor_selection_labels_used'),
        actor_adam_per_rank=joint['actor']['actual_adam_steps_per_rank'],
        probe_adam_per_rank=joint['probe']['actual_adam_steps_per_rank'],
        probe_new_adam_per_rank=update['new_adam_steps_per_rank'],
        ema_updates=joint['probe'].get('ema_updates',0),
        sampling={k:sampling[k]for k in ('original_counts','sampled_counts','unique_questions','repeated_fraction','status')if k in sampling},
        paired_original_distribution=paired,update_loss_first_last=curve_summary,
        training_wall_seconds=walls[0]if len(walls)==1 else None,
        source_commit=joint.get('source_commit'),observed_unix=joint.get('observed_unix'))


def summarize(base,through):
    base=Path(base)
    config=json.loads((base/'preparation/launch-config.json').read_text())
    latest=json.loads((base/'latest-joint.json').read_text())
    if type(through)is not int or not 1<=through<=latest['outer']:
        raise ValueError('Report only an explicit actually committed boundary')
    rows=[]
    for outer in range(1,through+1):
        path=base/'outers'/('outer-%06d'%outer)/'complete.json'
        raw=path.read_bytes();joint=json.loads(raw);row=metric_row(config,joint,outer)
        row.update(committed_receipt=str(path),committed_receipt_sha256=hashlib.sha256(raw).hexdigest())
        rows.append(row)
    return dict(role='SMALL_COMMITTED_METRICS_NO_LEDGER_REAUDIT',base=str(base),run_id=config['run_id'],
        through=through,rows=rows,uncommitted_or_later_history_included=False,
        full_candidate_accuracy_claimed=False,tensors_or_rollouts_read=False,training_modified=False)
