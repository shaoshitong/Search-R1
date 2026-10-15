"""Deduplicate console rows and reconcile Q-probe timing against actual writes."""
import hashlib
import json
import math
import pathlib
import statistics

HERE=pathlib.Path(__file__).resolve().parent
capture=json.loads((HERE/'remote-scalars-first.json').read_text())
mixed=json.loads((HERE.parent/'mixed-first-latency-20261016/timing-analysis.json').read_text())
result=dict(read_only=True,mixed_reference=mixed['summary']['ordinary2048_6_200'],runs=[])

def stats(v):
    return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),
                minimum=min(v),maximum=max(v),total=sum(v))

for run in capture['runs']:
    by_step={}
    for log in run['metric_logs']:
        for row in log['rows']:
            if row['step'] in by_step:
                assert row['metrics']==by_step[row['step']]['metrics'],row['step']
            by_step[row['step']]=row
    audits={x['step']:x for x in run['selection']}
    assert len(audits)==len(run['selection'])
    updates={x['step']:x for x in run['updates']}
    assert set(by_step)==set(updates)
    rows=[]
    for step,row in sorted(by_step.items()):
        m=row['metrics'];audit=audits[step];update=updates[step]
        assert abs(m['timing_s/candidate_fill']-audit['candidate_fill_seconds'])<=.000501
        for k in ('attempted_candidates','raw_generated_trajectories',
                  'raw_generated_policy_tokens','raw_actual_retrieval_calls',
                  'selected_mixed','selected_all_wrong','selected_all_correct'):
            assert m['selection/'+k]==audit[k],(step,k)
        assert audit['attempted_candidates']==5120 and audit['selected_groups']==512
        assert audit['raw_generated_trajectories']==2560
        actual=update['actual_optimizer_writes']
        assert actual==m['actor/optimizer_steps_this_update']
        assert actual==update['optimizer_writes_after']-update['optimizer_writes_before']
        assert update['opportunities_processed']==20
        assert actual+m['actor/skipped_update_opportunities']==20
        p=dict(collection=audit['candidate_fill_seconds'],
               agent_interaction=m['infra/rollout/phase_elapsed_seconds'],
               reward_advantage=m['timing_s/adv'],actor=m['timing_s/update_actor'],
               question_head_update=m['timing_s/question_probe_update'],
               head_internal_update=m['question_probe/elapsed_seconds'],
               testing=m.get('timing_s/testing',0),checkpoint=m.get('timing_s/save_checkpoint',0))
        p['collection_other']=p['collection']-p['agent_interaction']
        p['post_collection']=m['timing_s/step']
        p['other_step']=p['post_collection']-sum(p[k] for k in
                    ('reward_advantage','actor','question_head_update','testing','checkpoint'))
        p['outer_measured']=p['collection']+p['post_collection']
        assert all(math.isfinite(v) and v>=0 for v in p.values())
        rows.append(dict(step=step,phases_seconds=p,actual_adam=actual,
            head_updates=int(m['question_probe/head_optimizer_updates']),
            replay_groups=int(m['question_probe/replay_groups']),
            generated_trajectories=2560,selected_trajectories=2560,
            generated_policy_tokens=audit['raw_generated_policy_tokens'],
            source_selection=audit['source'],metric_line_sha256=row['line_sha256']))
    ordinary=[r for r in rows if r['phases_seconds']['testing']==0 and r['phases_seconds']['checkpoint']==0]
    def aggregate(items):
        phases={k:stats([r['phases_seconds'][k] for r in items]) for k in items[0]['phases_seconds']}
        total=phases['outer_measured']['total'];at=phases['actor']['total'];ct=phases['collection']['total']
        return dict(n=len(items),steps=[r['step'] for r in items],phases_seconds=phases,
            counts=dict(adam=sum(r['actual_adam'] for r in items),head_updates=sum(r['head_updates'] for r in items)),
            pooled_rates=dict(generated_trajectories_per_collection_second=2560*len(items)/ct,
                generated_trajectories_per_agent_second=2560*len(items)/phases['agent_interaction']['total'],
                selected_trajectories_per_actor_second=2560*len(items)/at,
                actor_seconds_per_actual_adam=at/sum(r['actual_adam'] for r in items),
                measured_outers_per_hour=3600*len(items)/total),
            shares={k:phases[k]['total']/total for k in ('collection','reward_advantage','actor','question_head_update','other_step','testing','checkpoint')})
    result['runs'].append(dict(base=run['base'],identity=run['run'],
        metrics_unique=len(by_step),raw_metric_rows=sum(len(x['rows']) for x in run['metric_logs']),
        excluded_partial_rollout_steps=sorted(set(audits)-set(updates)),
        all_complete=aggregate(rows),ordinary=aggregate(ordinary),rows=rows))
result['evidence_sha256']=hashlib.sha256((HERE/'remote-scalars-first.json').read_bytes()).hexdigest()
(HERE/'comparison-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
compact=dict(result);compact['runs']=[{k:v for k,v in r.items() if k!='rows'} for r in result['runs']]
(HERE/'summary.json').write_text(json.dumps(compact,indent=2)+'\n')
for r in result['runs']:
    a=r['ordinary'];print(json.dumps(dict(run=r['identity']['run_id'],n=a['n'],steps=a['steps'],
        mean={k:round(v['mean'],6) for k,v in a['phases_seconds'].items()},
        medians={k:round(v['median'],6) for k,v in a['phases_seconds'].items()},rates=a['pooled_rates'],shares=a['shares'],counts=a['counts'])))
