"""Bind final Tree-Nested scalar products to original output readback evidence."""
from pathlib import Path
import json,hashlib,datetime,copy,statistics
D=Path(__file__).resolve().parent
B=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013')
def read(p):return json.loads(p.read_text())
def ev(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
paths=[B/n for n in ['LEARNING_CURVE_COMPLETE_THROUGH200_ORIGINAL_RECORDS.json','FINAL200_FULL51713_METRICS.json','FINAL200_ROOT_FULL51713_DOUBLE_TERMINAL_PROOF.json','FINAL200_FAST_EVALUATION_COMPLETE_RUN_RESULT.json','final_original_costs200/SCALARS_OUTER0200.json','final_original_costs200/DEV_OUTER0200_ORIGINAL.json','FINAL200_INDEPENDENT_REQUIREMENTS_AUDIT_FINAL_DELTA.json']]
checks=[]
for i in range(2):checks.append(dict(pass_number=i+1,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=[ev(p) for p in paths]))
assert checks[0]['sources']==checks[1]['sources']
curve,met,proof,rr,scalar,dev,accept=[read(p) for p in paths]
assert curve['training_point_count']==200
assert [x['outer'] for x in curve['training_points']]==list(range(1,201))
assert len(curve['completed_dev512_points'])==40
assert met==proof['original_byte_descriptors']['final-metrics.json']['value']
assert rr==proof['original_byte_descriptors']['run-result.json']['value']
assert met['evaluated_count']==rr['evaluated_count']==51713
assert rr['status']=='COMPLETE' and rr['worker_exit_code']==0 and not rr['optimizer_constructed']
assert rr['final_metrics_sha256']==proof['original_byte_descriptors']['final-metrics.done.json']['value']['sha256']
assert len(proof['observations'])==2 and all(x['rows']==51713 and 'ExecMainStatus=0' in x['systemd'] for x in proof['observations'])
last=curve['training_points'][-1]
assert hashlib.sha256(paths[4].read_bytes()).hexdigest()==last['input_summary_sha256']
assert hashlib.sha256(paths[5].read_bytes()).hexdigest()==curve['completed_dev512_points'][-1]['source_original_dev_summary_sha256']
assert scalar['outer']==dev['actual_completed_outer']==200
assert dev['status']=='VALIDATION_COST_COMPLETE' and dev['evaluated_rows']==512
assert all(v==4000 for v in last['actual_cumulative_adam_calls_per_rank'].values())
series=[]
for p in curve['training_points']:
 series.append(dict(iter=p['outer'],em=p['training_score_mean'],path=str(paths[0]),source_metric_file=p.get('source_metric_file'),source_metric_file_sha256=p.get('source_metric_file_sha256'),original_metric_line_sha256=p.get('original_metric_line_sha256'),run_directory=p['run_directory']))
fixed=[]
for p in curve['completed_dev512_points']:
 assert p['rows']==512 and abs(p['em']*512-round(p['em']*512))<1e-8
 fixed.append(dict(iter=p['outer'],em=p['em'],f1=p['f1'],n=512,correct=round(p['em']*512),path=str(paths[0]),original_control_source=p.get('original_control_source'),source_original_dev_summary=p.get('source_original_dev_summary'),source_original_dev_summary_sha256=p.get('source_original_dev_summary_sha256'),ledger=p.get('ledger_path'),ledger_sha256=p.get('ledger_sha256')))
doc=read(D/'tree-updated-runs.json')
backup=D/'tree-updated-runs-before-final200.json'
if not backup.exists():backup.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
r=next(x for x in doc['runs'] if x['name']=='Tree nested branching fresh200')
old={x['iter']:x['em'] for x in r['train_series']}
assert all(abs(old[x['iter']]-x['em'])<1e-12 for x in series if x['iter'] in old)
obs=max(x['epoch'] for x in proof['observations'])
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
m=met['metrics'];ds=['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']
src={k:dict(count=m['val/qa_count/'+k],em=m['val/em/'+k],f1=m['val/f1/'+k]) for k in ds}
correct=sum(round(v['em']*v['count']) for v in src.values())
assert correct/51713==m['val/em/all']
r.update(completed_iter=200,train_latest_em=series[-1]['em'],train_recent10_mean=statistics.mean(x['em'] for x in series[-10:]),train_recent10_n=10,train_series=series,fixed512=[x for x in fixed if x['iter'] in [5,50,100,150,200]],additional_fixed512=[x for x in fixed if x['iter'] not in [5,50,100,150,200]],status='training200_and_final51713_complete',phase='complete',optimizer_per_rank=last['actual_cumulative_adam_calls_per_rank'],errors=[],observed_at_cst=datetime.datetime.fromtimestamp(obs,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),compiled_at_cst=now,runtime_observation_mode='Original output files and two retained remote terminal readbacks; compiled locally this turn without training mutation',runtime_observation_timestamps=[datetime.datetime.fromtimestamp(x['epoch'],datetime.timezone(datetime.timedelta(hours=8))).isoformat() for x in proof['observations']],note='Same effective fresh nested chain 1 / 2–5 / 6–200. Full200 original scalar records and all40 dev512 receipts verified. Final evaluation8 resident TP1 workers, local batch64; HF weights unchanged, altered greedy batching/kernel shapes not guaranteed bit-identical. Original training normal-exit not inferred from evaluation success.')
r['final51713']=dict(complete=True,n=51713,correct=correct,em=m['val/em/all'],f1=m['val/f1/all'],source_metrics=src,path=proof['original_byte_descriptors']['final-metrics.json']['path'],sha256=proof['original_byte_descriptors']['final-metrics.json']['sha256'],local_semantic_copy=ev(paths[1]),identity=met['identity'],execution_changes=met['execution_changes'],numerical_caveat=met['numerical_caveat'],worker_exit_code=0,terminal_proof=ev(paths[2]))
doc.update(observed_at_cst=r['observed_at_cst'],scope='Only latest Tree-Nested result refreshed; original Tree andBRANPO preserved as earlier snapshots.',latest_phase='training200_and_final51713_complete',latest_source_readbacks=checks,final_remote_terminal_observations=proof['observations'],remote_connection_note='This refresh uses original scalar products and retained double terminal remote output readbacks; no fresh SSH or training agent messages were issued.',latest_input_files=[ev(p) for p in paths],overlap_exact_points=len(old))
(D/'tree-updated-runs.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
receipt=dict(created_at_cst=now,completed_iter=200,train_latest_em=r['train_latest_em'],train_recent10_mean=r['train_recent10_mean'],fixed512=r['fixed512'],final51713=r['final51713'],source_checks=checks,old_training_overlap_preserved=len(old),original_remote_observations=proof['observations'],training_agent_messages_sent=False,remote_jobs_changed=False)
(D/'tree-final200-refresh-verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(completed=200,train_em=r['train_latest_em'],recent10=r['train_recent10_mean'],fixed200=r['fixed512'][-1]['em'],correct=correct,micro=m['val/em/all'],macro=sum(v['em'] for v in src.values())/7,source_em=src),ensure_ascii=False))
