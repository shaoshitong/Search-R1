import json,statistics,datetime,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;old=D.parent/'session-refresh-20261015-current/all-seven-results.json'
run=next(x for x in json.loads(old.read_text())['runs'] if x['name']=='Recursive B four GPU')
v=json.loads((D/'b-observation-1.json').read_text());e=v['experiments'][0]
hashes={s['path']:s for s in json.loads((D/'b-source-hashes.json').read_text())['sources']}
series=[];fixed=[]
for idx,l in enumerate(e['logs']):
 lo,hi=[(1,3),(4,8),(9,1000)][idx]
 source=hashes[l['path']]
 for r in l['rows']:
  i=r['step'];m=r['metrics']
  if not lo<=i<=hi:continue
  if 'critic/score/mean' in m:series.append({'iter':i,'em':m['critic/score/mean'],'path':l['path'],'line':r['line']})
  if m.get('val/qa_count/all')==512 and 'val/em/all' in m:
   fixed.append({'iter':i,'em':m['val/em/all'],'f1':m.get('val/f1/all'),'n':512,'correct':round(m['val/em/all']*512),'path':l['path'],'line':r['line'],'sha256':source['sha256'],'source_hashed_prefix_bytes':source['hashed_prefix_bytes'],'sha256_scope':'log prefix of exact source_hashed_prefix_bytes','datasets':{'nq':{'n':int(m['val/qa_count/nq']),'em':m['val/em/nq'],'f1':m.get('val/f1/nq')},'hotpotqa':{'n':int(m['val/qa_count/hotpotqa']),'em':m['val/em/hotpotqa'],'f1':m.get('val/f1/hotpotqa')}}})
new={x['iter']:x for x in series};overlap=[]
for x in run['train_series']:
 if x['iter'] in new:
  assert abs(x['em']-new[x['iter']]['em'])<1e-10,(x,new[x['iter']])
  overlap.append(x['iter'])
assert len(series)==169 and sorted(new)==list(range(1,170))
for x in run['fixed512']:
 if x['em'] is not None:
  y=next(y for y in fixed if y['iter']==x['iter']);assert abs(x['em']-y['em'])<1e-10
r1=json.loads((D/'b-runtime-1.json').read_text());r2=json.loads((D/'b-runtime-2.json').read_text())
progress={'observed_seconds':r2['at']-r1['at'],'outer':r2['phase']['outer'],'generated_tokens_before':r1['phase']['generated_tokens'],'generated_tokens_after':r2['phase']['generated_tokens'],'generated_tokens_delta':r2['phase']['generated_tokens']-r1['phase']['generated_tokens'],'generation_calls_before':r1['phase']['generation_calls'],'generation_calls_after':r2['phase']['generation_calls'],'log_bytes_delta':r2['log']['bytes']-r1['log']['bytes'],'recent_errors':r2['errors']}
assert progress['generated_tokens_delta']>0
run.update({'completed_iter':169,'train_series':series,'train_latest_em':series[-1]['em'],'train_recent10_mean':statistics.mean(x['em'] for x in series[-10:]),'train_recent10_n':10,'fixed512':fixed+[{'iter':200,'em':None,'f1':None,'n':None,'path':None,'reason':'Training currently at complete169; no completed fixed512 outer200 result.'}],'additional_fixed512':[],'final51713':{'complete':False,'em':None,'f1':None,'n':0,'path':None,'reason':'Training has not reached200; no completed final51713 result.'},'status':'training_running','note':'HKUST5 GPUs0–3 RTX4090; canonical ancestors1–3,4–8,current9–169. Full state manifest169 committed, all ranks Adam3380/scheduler169 finite as emitted in manifest. Outer170 sampling has measured token growth. No training or filesystem remote mutations.','reader_started_epoch':v['at'],'log_evidence':list(hashes.values()),'audit_file':str(D/'b-audit-summary.json'),'runtime_progress':progress,'checkpoint_manifest':r2['checkpoint']})
cst=datetime.timezone(datetime.timedelta(hours=8));at=datetime.datetime.fromtimestamp(r2['at'],cst).isoformat()
(D/'b-updated-runs.json').write_text(json.dumps({'observed_at_cst':at,'runs':[run]},ensure_ascii=False,indent=2))
audit={'observed_at_cst':at,'session':run['session'],'read_only':True,'training_agent_messages_sent':False,'remote_mutations':False,'canonical_segments':[{'first':1,'last':3,'path':e['logs'][0]['path']},{'first':4,'last':8,'path':e['logs'][1]['path']},{'first':9,'last':169,'path':e['logs'][2]['path']}],'old_train_overlap_matched':len(overlap),'old_train_overlap_exact':True,'series_count':len(series),'missing_outer':[],'completed_outer':169,'target_outer':200,'train_latest_em':run['train_latest_em'],'train_recent10_mean':run['train_recent10_mean'],'fixed512':fixed,'final51713_complete':False,'runtime_progress':progress,'checkpoint_manifest':r2['checkpoint'],'source_hashes':list(hashes.values()),'observation_files':[str(D/n) for n in ['b-observation-1.json','b-runtime-1.json','b-runtime-2.json','b-source-hashes.json']]}
(D/'b-audit-summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps({k:audit[k] for k in ['observed_at_cst','completed_outer','series_count','old_train_overlap_matched','train_latest_em','train_recent10_mean','runtime_progress']}))
