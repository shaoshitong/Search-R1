"""Combine independent read-only audits; never merge different run identities."""
from pathlib import Path
import json,datetime,hashlib
D=Path(__file__).resolve().parent
names=['g1-audit.json','tree-e1-b-audit.json','gigpo-co-audit.json','topd-audit.json']
sources=[];runs=[]
for name in names:
 p=D/name;raw=p.read_bytes();a=json.loads(raw)
 sources.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
 for r in a['runs']:
  run=dict(r);run['audit_file']=str(p)
  if 'completed_iter' not in run:run['completed_iter']=run.get('completed_iterations')
  if 'train_latest_em' not in run:run['train_latest_em']=run.get('training_latest_em')
  if 'train_recent10_mean' not in run:run['train_recent10_mean']=run.get('training_recent10_mean')
  if 'fixed512' not in run:run['fixed512']=run.get('fixed512_series',[])
  if run['name'].startswith('TOP-D'):
   run.setdefault('session','01a0fcbe-15f8-7600-ae84-9487d924560a')
   run.setdefault('status','TRAIN200_FINAL51713_COMPLETE_AND_ACCEPTED')
   run.setdefault('target_iter',200)
   run.setdefault('output_dir',run.get('training_series',[{}])[-1].get('source','').rsplit('/',1)[0])
  final=run.get('final51713')
  if final and 'n' not in final:final['n']=final.get('rows')
  for e in run['fixed512']:
   if 'iter' not in e:e['iter']=e.get('iteration',e.get('step'))
   if e.get('em') is not None:
    assert e.get('n')==512,(run['name'],e)
    assert 0<=e['em']<=1
    if e.get('correct') is not None:assert abs(e['em']-e['correct']/512)<1e-12
  if final and final.get('complete'):
   assert final['n']==51713 and 0<=final['em']<=1 and 0<=final['f1']<=1
   if final.get('correct') is not None:assert abs(final['em']-final['correct']/51713)<1e-12
  runs.append(run)
report={'observed_report_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'sessions_requested':['01a0fb1b-d3ff-74b3-aa13-49ca57871f92','01a0fcbb-863f-7450-926d-db537f648265','01a0fcbc-0b46-7502-8b66-d8b21a0e032d','01a0fcbe-15f8-7600-ae84-9487d924560a','01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef','01a1028b-b626-7f91-93db-0099ff1623c5','01a12e27-764b-7432-a5f2-74d88ee4a8d4'],'read_only':True,'training_agent_messages_sent':0,'remote_jobs_changed':False,'large_local_payloads_downloaded':False,'definitions':{'training_em':'Each run native strict-EM from completed outer rollout; denominator/selection differs by setting; not fixed512 evaluation','fixed512':'Held-out fixed dev NQ256+Hotpot256; stage measurement, not final51713','final51713':'Saved iter200 checkpoint on complete seven-dataset51713 split; partial progress is not full EM','missing':'null/empty means not generated, not verified, or unavailable; never zero-filled','run_identity':'Different initialization/setting/recovery lineage remains separate'},'runs':runs,'sources':sources,'cosearch_final_observations':json.loads((D/'topd-audit.json').read_text())['cosearch_final_observations']}
(D/'all-seven-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for r in runs:
 print(json.dumps({k:r.get(k) for k in ['name','completed_iter','train_latest_em','train_recent10_mean','fixed512','final51713','status','note']},ensure_ascii=False))
