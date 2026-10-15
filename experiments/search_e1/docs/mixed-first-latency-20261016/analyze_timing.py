"""Reconcile scalar timing rows with immutable selection and update receipts."""
import collections
import hashlib
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
capture = json.loads((HERE / 'remote-timing-scalars-first.json').read_text())
rows = []
for run in capture['runs']:
    selected = {int(x['step']):x for x in run['selection']}
    assert len(selected) == len(run['selection'])
    updates = {int(x['step']):x for x in run['updates']}
    metric_rows = [row for log in run['metric_logs'] for row in log['rows']]
    assert len({row['step'] for row in metric_rows}) == len(metric_rows)
    for row in metric_rows:
        step, m = row['step'], row['metrics']
        audit, update = selected[step], updates[step]
        assert abs(m['timing_s/candidate_fill'] - audit['candidate_fill_seconds']) <= 0.000501
        for key in ('attempted_candidates','selected_mixed','selected_all_correct_fallback',
                    'raw_generated_trajectories','raw_generated_policy_tokens','raw_actual_retrieval_calls'):
            assert int(m['selection/'+key]) == audit[key], (step,key)
        assert audit['selected_mixed'] + audit['selected_all_correct_fallback'] == 512
        assert audit['raw_generated_trajectories'] == audit['attempted_candidates'] * 5
        assert m['actor/optimizer_steps_this_update'] == 20
        assert m['actor/skipped_update_opportunities'] == 0
        # Old legacy updates do not export newer write-count fields.
        if update.get('actual_optimizer_writes') is not None:
            assert update['actual_optimizer_writes'] == 20
            assert update['optimizer_writes_after'] - update['optimizer_writes_before'] == 20
        phases = dict(rollout=audit['candidate_fill_seconds'],
                      reference=m['timing_s/ref'],reward_advantage=m['timing_s/adv'],
                      actor=m['timing_s/update_actor'],
                      testing=m.get('timing_s/testing',0),
                      checkpoint=m.get('timing_s/save_checkpoint',0))
        phases['other_step'] = m['timing_s/step'] - sum(v for k,v in phases.items() if k!='rollout')
        assert phases['other_step'] >= -0.005
        phases['post_sampling'] = m['timing_s/step']
        phases['outer_measured'] = phases['rollout'] + phases['post_sampling']
        rows.append(dict(step=step,base=run['base'],source_commit=run['run']['source_commit'],
                candidates=audit['attempted_candidates'],raw_trajectories=audit['raw_generated_trajectories'],
                raw_policy_tokens=audit['raw_generated_policy_tokens'],
                raw_retrieval_calls=audit['raw_actual_retrieval_calls'],
                selected_mixed=audit['selected_mixed'],fallback=audit['selected_all_correct_fallback'],
                selected_trajectories=2560,adam_updates=20,phases_seconds=phases,
                selection_source=audit['source'],metric_line_sha256=row['line_sha256']))
rows.sort(key=lambda r:r['step'])
assert [r['step'] for r in rows] == list(range(1,5)) + list(range(6,201))
assert [r['step'] for r in rows if r['candidates']!=2048] == [27,28,29]

def describe(values):
    values = sorted(values)
    def quantile(q):
        i=(len(values)-1)*q; left=int(i); right=min(left+1,len(values)-1)
        return values[left] + (values[right]-values[left])*(i-left)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p10=quantile(.1),p90=quantile(.9),min=min(values),max=max(values),total=sum(values))

def aggregate(group):
    phases={key:describe([r['phases_seconds'][key] for r in group])
            for key in group[0]['phases_seconds']}
    sums={key:sum(r[key] for r in group) for key in
          ('candidates','raw_trajectories','raw_policy_tokens','raw_retrieval_calls',
           'selected_trajectories','adam_updates')}
    rt=phases['rollout']['total']; at=phases['actor']['total']; total=phases['outer_measured']['total']
    return dict(n=len(group),iterations=[r['step'] for r in group],phases_seconds=phases,counts=sums,
            pooled_rates=dict(candidate_questions_per_second=sums['candidates']/rt,
                raw_trajectories_per_second=sums['raw_trajectories']/rt,
                raw_policy_tokens_per_second=sums['raw_policy_tokens']/rt,
                retrieval_calls_per_rollout_second=sums['raw_retrieval_calls']/rt,
                selected_trajectory_exposures_per_actor_second=sums['selected_trajectories']/at,
                seconds_per_adam=at/sums['adam_updates'],outer_per_hour=3600*len(group)/total,
                selected_trajectories_per_outer_second=sums['selected_trajectories']/total,
                rollout_to_actor_seconds=rt/at),
            phase_share={k:v['total']/total for k,v in phases.items()
                         if k not in ('post_sampling','outer_measured')})

ordinary=lambda r:r['candidates']==2048 and r['phases_seconds']['testing']==0 and r['phases_seconds']['checkpoint']==0
groups=dict(accepted6_200=[r for r in rows if r['step']>=6],
            ordinary2048_6_200=[r for r in rows if r['step']>=6 and ordinary(r)],
            ordinary2048_11_199=[r for r in rows if 11<=r['step']<=199 and ordinary(r)],
            original_fp32_1_4=[r for r in rows if r['step']<=4])
for lo,hi in ((6,10),(11,49),(51,99),(101,149),(151,199)):
    groups[f'ordinary2048_{lo}_{hi}']=[r for r in rows if lo<=r['step']<=hi and ordinary(r)]
report=dict(capture_sha256=hashlib.sha256((HERE/'remote-timing-scalars-first.json').read_bytes()).hexdigest(),
    reconciled_metric_rows=len(rows),scope='accepted parent1-4 + official continuation6-200; outer5 lacks console timing row',
    count_all200=dict(candidate_questions=sum(x['attempted_candidates'] for r in capture['runs'] for x in r['selection']),
                     raw_trajectories=sum(x['raw_generated_trajectories'] for r in capture['runs'] for x in r['selection'])),
    summary={key:aggregate(group) for key,group in groups.items()},rows=rows,
    limitations=['Console phase timers are rounded to .001s; candidate receipt retains full precision.',
                 'Measured outer is candidate_fill + step; out-of-timer logging and milestone probe refresh are excluded.',
                 'Candidate fill includes generation/search/EM/serialization, not pure decoder time.',
                 'Actor timer includes monitoring and distributed overhead, not isolated optimizer kernel time.',
                 'Ordinary group excludes recorded validation/checkpoint and 1536-question rows; no latency-based filtering.'])
(HERE/'timing-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for key in ('ordinary2048_6_200','ordinary2048_11_199','accepted6_200'):
    a=report['summary'][key]
    print(json.dumps(dict(scope=key,n=a['n'],phases={k:{x:v[x] for x in ('mean','median','p10','p90')} for k,v in a['phases_seconds'].items()},rates=a['pooled_rates'],share=a['phase_share'])))
