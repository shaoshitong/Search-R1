"""Build small plotting scalars from actual remote readback files; no remote actions."""
from pathlib import Path
import json,hashlib,datetime,copy,statistics
D=Path(__file__).resolve().parent
S=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2')
B=S/'outputs/tree-nested-branch-handoff-20261013'
curve_path=B/'LEARNING_CURVE_COMPLETE_THROUGH185.json';curve=json.loads(curve_path.read_text())
files=sorted(B.glob('NATIVE5_CONTINUE_POLL_[0-9]*.json'),key=lambda p:p.stat().st_mtime)
polls=[(p,json.loads(p.read_text())) for p in files[-2:]]
for p,x in polls:assert x['source_commit']=='778b2b43dad90fc0c1c5d95d77c2326286011b73' and not x['progress']['errors'] and x['run_result'] is None
latest_path,latest=polls[-1];completed=latest['progress']['actual_completed_outer_from_attempt_results']
def ref(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
old=json.loads((D/'base-all-seven-results.json').read_text())
oldruns=old['runs']
names=['Tree-GRPO original guarded200','Tree BRANPO fresh200 3360','Tree nested branching fresh200']
runs=[copy.deepcopy(next(r for r in oldruns if r['name']==name))for name in names]
r=runs[-1];series={}
for point in curve['training_points']:
 i=point['outer'];assert point.get('effective_record_present',True) and point.get('original_complete_metric_record_present',True)
 assert set(point['actual_outer_adam_calls_per_rank'].values())=={20}
 assert set(point['actual_cumulative_adam_calls_per_rank'].values())=={20*i}
 name=point.get('source_metric_file',point.get('source_receipt',''));source=(S/name if name.startswith('outputs/') else B/name);assert source.exists(),(i,str(source))
 expected=point.get('source_metric_file_sha256',point.get('source_receipt_sha256'))
 if expected:assert ref(source)['sha256']==expected,(i,source)
 series[i]=dict(iter=i,em=point['training_score_mean'],path=str(source),sha256=expected,source_selector=point.get('source_metric_selector'),run_directory=point['run_directory'],actual_adam_per_rank=point['actual_cumulative_adam_calls_per_rank'])
for p,x in polls:
 for m in x['iteration_metrics']:
  i=int(m['step'])
  if i>completed or 'critic/score/mean'not in m:continue
  assert m['actor/optimizer_steps']==20 and m['actor/scheduler_step_after_outer']==i and m['training/actual_outer_updates']==i
  if i in series:assert abs(series[i]['em']-m['critic/score/mean'])<1e-12
  series[i]=dict(iter=i,em=m['critic/score/mean'],path=str(p),sha256=ref(p)['sha256'],source_selector=f'iteration_metrics[step={i}]',run_directory=x['run'],optimizer_steps=20)
assert sorted(series)==list(range(1,completed+1))
oldseries={p['iter']:p['em']for p in r['train_series']};assert all(abs(series[i]['em']-v)<1e-12 for i,v in oldseries.items())
fixed={}
for f in curve['completed_dev512_points']:
 assert f['rows']==512 and f['status']=='VALIDATION_COST_COMPLETE'
 fixed[f['outer']]=dict(iter=f['outer'],em=f['em'],f1=f['f1'],n=512,correct=round(f['em']*512),path=str(B/f['source_receipt']),sha256=f.get('source_receipt_sha256'),ledger=f.get('ledger_path'),ledger_sha256=f.get('ledger_sha256'),nq_em=f.get('nq_em'),hotpotqa_em=f.get('hotpotqa_em'))
for f in latest['development']:
 if f['rows']!=512 or f['status']!='VALIDATION_COST_COMPLETE':continue
 i=f['outer'];m=f['metrics'];v=dict(iter=i,em=m['val/em/all'],f1=m['val/f1/all'],n=512,correct=round(m['val/em/all']*512),path=str(latest_path),sha256=ref(latest_path)['sha256'],source_selector=f'development[outer={i}]',ledger=f['path'],ledger_sha256=f.get('ledger_sha256'))
 if i in fixed:assert abs(fixed[i]['em']-v['em'])<1e-12
 fixed[i]=v
r.update(completed_iter=completed,train_series=[series[i]for i in sorted(series)],train_latest_em=series[completed]['em'],train_recent10_mean=statistics.mean(series[i]['em']for i in sorted(series)[-10:]),train_recent10_n=10,fixed512=[fixed.get(i,dict(iter=i,em=None,f1=None,n=None,path=None,reason='Not completed yet'))for i in [5,50,100,150,200]],additional_fixed512=[fixed[i]for i in sorted(fixed)if i not in [5,50,100,150,200]],status='training_running',note='Nested setting; same effective chain1,2–5,6 onward. Remote readback metadata and original scalar log records cross-bound locally, two actual observations retained. This turn direct SSH transport timed out; timestamps refer to existing actual remote output files, not conversation. No final51713 yet.',reader_started_epoch=latest['epoch'],observed_at_cst=datetime.datetime.fromtimestamp(latest['epoch'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),optimizer_per_rank=latest['progress']['completed_actual_optimizer_calls_per_rank'])
checks=[]
for p,x in polls:
 checks.append(dict(**ref(p),observed_at_cst=datetime.datetime.fromtimestamp(x['epoch'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),completed=x['progress']['actual_completed_outer_from_attempt_results'],systemd=x['systemd'],stage_log_bytes=x['own_output_files']['stage.log'],retriever_log_bytes=x['own_output_files']['retriever.log'],errors=x['progress']['errors'],terminal=x['run_result']))
checks.append(dict(name='full185_curve_sha_bound',**ref(curve_path),training_points=len(curve['training_points']),dev512_points=len(curve['completed_dev512_points'])))
obj=dict(observed_at_cst=r['observed_at_cst'],runs=runs,checks=checks,actual_remote_observations=checks[:2],scope='Read-only, no messages to training agents or remote mutation; fresh plotting suffix from original actual remote output readback files, not chat narrative.',latest_phase=f'Completed{completed}; outer{completed+1} sampling continues in two observations',remote_connection_note='Two read-only direct transport attempts made this turn. First timed out; second result recorded separately. Existing actual source poll files provide recent metadata, full original scalar source SHA bindings checked. Completed original Tree/BRANPO retained10/15 audited values and not advertised freshly refreshed.',overlap_exact_points=len(oldseries))
(D/'tree-updated-runs.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(output=str(D/'tree-updated-runs.json'),completed=completed,train_latest=r['train_latest_em'],train_recent10=r['train_recent10_mean'],fixed=[(v['iter'],v['em'])for v in r['fixed512']],latest_fixed=max(fixed),latest_fixed_em=fixed[max(fixed)]['em'],observed=r['observed_at_cst'],observations=[(c['observed_at_cst'],c['stage_log_bytes'],c['retriever_log_bytes'])for c in checks[:2]]),ensure_ascii=False))
