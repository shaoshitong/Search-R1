import pathlib,sys,json,datetime
repo=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/qprobe-h1008-transformer3');sys.path.insert(0,str(repo))
from reproduction.h1008_qprobe.management_transport import remote
root=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work')
socket=json.loads((root/'qprobe3-management-current.json').read_text())['socket']
base='/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015'
script=f'''import sys,json,pathlib,re,hashlib
base=pathlib.Path({base!r});sys.path.insert(0,str(base/'repo'))
from reproduction.h1008_qprobe.observe_warmup_progress import snapshot,rows,read
s=snapshot(base,3402)
assert s['run_id']=='f77c3245dcfe49a88a5a99b1dec161e6',s['run_id']
assert s['source_commit']=='c5fc09603a10244655b98ec0ce8eada32668f83b'
driver_keys=('step','logical_update','attempt_id','attempt_index','complete_outer','completed_outer','status','optimizer_writes_before','optimizer_writes_after','optimizer_writes_this_attempt','opportunities_processed','scheduler_advances','source_consumed','data_cursor','cursor')
s['driver_actual_updates']=[{{k:r[k] for k in driver_keys if k in r}} for r in rows(base/'logs/actual-updates.jsonl')][-6:]
s['driver_stage_timings']=[dict(step=r.get('step'),metrics={{k:v for k,v in r.get('metrics',{{}}).items() if 'time' in k.lower() or 'seconds' in k.lower() or k in ('actor/optimizer_steps_this_update','actor/update_opportunities_this_attempt','actor/skipped_update_opportunities')}}) for r in rows(base/'logs/actual-updates.jsonl')][-6:]
s['driver_actual_attempts']=[{{k:r[k] for k in driver_keys if k in r}} for r in rows(base/'logs/actual-attempts.jsonl')][-6:]
s['rank_actual_schedulers']=[]
s['rank_last_attempt_write_counts']=[]
for rank in range(8):
 records=rows(base/f'logs/scheduler-rank-{{rank}}.jsonl')
 keys=('kind','logical_update','attempt_id','attempt_index','writes_before','writes_after','actual_optimizer_writes','completed_outer_before','completed_outer_after','scheduler_last_epoch_before','scheduler_last_epoch_after','scheduler_step_count_after','scheduler_advances','opportunities_processed')
 s['rank_actual_schedulers'].append([{{k:r[k] for k in keys if k in r}} for r in records[-2:]])
 writes=rows(base/f'logs/optimizer-rank-{{rank}}.jsonl')
 last=next((r.get('context',{{}}).get('attempt_id') for r in reversed(writes) if r.get('kind')=='OPTIMIZER_WRITE'),None)
 s['rank_last_attempt_write_counts'].append(dict(attempt_id=last,writes=sum(r.get('kind')=='OPTIMIZER_WRITE' and r.get('context',{{}}).get('attempt_id')==last for r in writes)))
s['console_timing_scalars']=[]
for log in (base/'logs').glob('*.log'):
 with log.open('rb') as stream:
  stream.seek(max(0,log.stat().st_size-1048576));tail=stream.read().decode(errors='replace')
 for line in tail.splitlines():
  if 'timing_s/' not in line:continue
  timing={{k:float(v) for k,v in re.findall(r'(timing_s/[A-Za-z_]+|question_probe/elapsed_seconds):\\s*([0-9.eE+-]+)',line)}}
  if timing:s['console_timing_scalars'].append(dict(log=log.name,step=re.findall(r'step[:= ]+(\\d+)',line)[:1],metrics=timing))
s['probe_online_updates']=[]
for path in (base/'artifacts/question-probe/completed').glob('*.json'):
 event=read(path);h=event.get('head_update',{{}})
 s['probe_online_updates'].append(dict(outer=event.get('outer'),head_version=event.get('head_version'),elapsed_seconds=h.get('elapsed_seconds'),head_optimizer_updates=h.get('head_optimizer_updates'),head_update_count=h.get('update_count')))
cp=base/'checkpoints/search-g1-common8/actor/global_step_5'
s['checkpoint5']=dict(path=str(cp),exists=cp.exists())
if cp.exists():
 manifest=read(cp/'complete.json');files=[path for path in cp.rglob('*') if path.is_file()]
 s['checkpoint5'].update(manifest_exists=(cp/'complete.json').exists(),manifest_step=manifest.get('step'),manifest_run_id=manifest.get('run_id'),manifest_status=manifest.get('status'),file_count=len(files),total_bytes=sum(path.stat().st_size for path in files),independent_full_sha_performed=False,semantic_reload_performed=False)
pair_path=base/'artifacts/question-probe/completed-pairs/global_step_5/source-receipt.json'
s['checkpoint5']['pair_receipt_exists']=pair_path.exists()
if pair_path.exists():
 pair=read(pair_path)
 s['checkpoint5']['paired_source_receipt']=dict(path=str(pair_path),receipt_sha256=hashlib.sha256(pair_path.read_bytes()).hexdigest(),status=pair.get('status'),run_id=pair.get('run_id'),step=pair.get('step'),world_size=pair.get('world_size'),actor_snapshot_id=pair.get('actor_snapshot_id'),checkpoint=pair.get('checkpoint'),checkpoint_receipt_files=len(pair.get('files',[])),boundary_worker_ranks=sorted(row['rank'] for row in pair.get('boundary_worker_files',[])),independent_all_file_sha_recheck_performed=False)
s['selection_actual_mixed']=[]
for path in sorted((base/'artifacts/question-probe/attempts').glob('*/selection.json')):
 selection=read(path);audit=read(path.parent/'rollout-audit.json')
 groups=audit.get('groups',[]);rates={{}}
 for source in ('predicted_mixed','random_exploration'):
  vectors=[g['strict_correctness'] for g in groups if g.get('source')==source]
  rates[source]=dict(count=len(vectors),actual_mixed=sum(1<=sum(v)<=4 for v in vectors),actual_mixed_rate=sum(1<=sum(v)<=4 for v in vectors)/len(vectors) if vectors else None)
 s['selection_actual_mixed'].append(dict(step=selection.get('step'),attempt_id=selection.get('attempt_id'),prediction_before_label=selection.get('prediction_before_label'),head_version=selection.get('head_version'),ranking=selection.get('ranking'),rollout_complete=bool(groups),source_consumed=audit.get('source_consumed'),candidate_fill_seconds=audit.get('candidate_fill_seconds'),generation_timings=audit.get('candidate_generation_timings'),rates=rates))
print(json.dumps(s))
'''
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
res=remote(script,root/('qprobe3402-scalar-transport-'+tag+'.json'),control_path=socket)
s=json.loads(res);p=root/('qprobe3402-scalar-'+tag+'.json');p.write_text(json.dumps(s,indent=2));compact={k:s[k] for k in ('utc','native','pipeline','pipeline_error','warmup_completed_epochs','warmup_adam_updates','warmup_ready_status','complete_actor_outer','actual_actor_adam_per_rank','errors','checkpoint5')};compact['latest_driver_update']=s['driver_actual_updates'][-1:];compact['latest_attempt']=s['driver_actual_attempts'][-1:];compact['latest_selection']=sorted(s['selection_actual_mixed'],key=lambda r:r.get('step',0))[-1:];compact['latest_probe_update']=sorted(s['probe_online_updates'],key=lambda r:r.get('outer',0))[-1:];compact['latest_console_timing']=sorted(s['console_timing_scalars'],key=lambda r:int(r['step'][0]) if r['step'] else 0)[-1:];print(json.dumps(dict(evidence=str(p),**compact)))
