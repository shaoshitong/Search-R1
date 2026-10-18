"""Record one existing bounded scalar receipt; never launch or schedule work."""
import pathlib,json,time,sys,datetime
root=pathlib.Path('/Users/shaoshitong/Documents/Search-G1');w=json.loads(pathlib.Path(sys.argv[1]).read_text());r=w['result'];last=r.get('latest')
p=root/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json';h=json.loads(p.read_text());c=h['current_authoritative']['current_a800_actor50_probe_balanced100'];old_epoch=c.get('complete_epochs',0);assert h['current_authoritative']['active_experiment_key']=='current_a800_actor50_probe_balanced100';assert r['run_id']==c['run_id'] and r['source_commit']==c['source_commit']
c.update(latest_actual=w['receipt'],latest_actual_checked_unix=r['checked_unix'],pipeline_identity=r['leader'],native_stage=r.get('status'),latest_failure=r.get('failure'),latest_cleanup=r.get('cleanup'),latest_exit=r.get('exit'))
if last:
 assert len(set(last['actual_adam_steps_per_rank']))==1 and last['actual_adam_steps_per_rank'][0]==last['ema_updates']
 c.update(complete_epochs=last['completed_epochs'],actual_adam_updates=last['actual_adam_steps_per_rank'][0],ema_updates=last['ema_updates'],latest_complete_checkpoint=last,latest_test_mixed_balanced_accuracy=last['metrics']['after']['test']['balanced_accuracy'])
 if last['completed_epochs']>c.get('resume_completed_epochs',0) and c.get('recovery_CPU_proof'):
  c['actual_CUDA_recovery_verified']=True
  c['actual_CUDA_recovery_receipt']=w['receipt']
 if last['cycle']==2:
  first=json.loads((pathlib.Path(__file__).resolve().parents[1]/'reports/actor50-probe100-cycle1-and-recovery-20261018.json').read_text())
  assert last['metrics']['before']['test']==first['metrics']['after']['test']
  c['same_fixed_test_preupdate_EMA_metrics_exact_match_after_recovery']=True
 c['status']='COMPLETE%d_OF100_ADAM%d_EMA%d_CURRENT_CYCLE%d_%s'%(last['completed_epochs'],last['ema_updates'],last['ema_updates'],(r.get('status') or {}).get('cycle',0),'HEALTHY' if not r.get('failure') else 'FAULT')
if r.get('complete'):c['status']='COMPLETE100_NATIVE_FINAL_AUDIT_PENDING'
if r.get('failure'):c['status']='FAULT_TERMINAL_OR_CLEANUP_IN_PROGRESS_KEEP_LATEST_COMPLETE_CYCLE'
if last and last['completed_epochs']>=c.get('resume_completed_epochs',0)+5 and (r.get('status') or {}).get('cycle',0)>last['cycle']:
 c.update(startup_recovery_followup_satisfied=True,active_launch_owner=False,recovery_followup_scope='real epochs3..8 complete and nextcycle5 or later started; continue target100')
p.write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n')
smallpath=root/'preparation/actor50-probe100-current-handoff.json';small=json.loads(smallpath.read_text());small.update(updated_unix=time.time(),**{k:c.get(k) for k in ['handle','base','execution_id','source_commit','pipeline_identity','latest_actual','status','complete_epochs','actual_adam_updates','ema_updates','actual_CUDA_recovery_verified','startup_recovery_followup_satisfied','active_launch_owner']});small['latest_actual_flow']=r.get('status');small['latest_failure']=r.get('failure');small['latest_metrics']=last['metrics'] if last else None;smallpath.write_text(json.dumps(small,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'epoch':None if not last else last['completed_epochs'],'Adam_EMA':None if not last else last['ema_updates'],'probe_CUDA_resume_actual':c.get('actual_CUDA_recovery_verified'),'current_stage':r.get('status'),'failure':r.get('failure'),'actual':w['receipt']}))

if last and (last['completed_epochs']!=old_epoch or r.get('failure')):
 stamp=datetime.datetime.fromtimestamp(r['checked_unix'],datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S Shanghai')
 test=last['metrics']['after']['test'];train=last['metrics']['after']['train']
 path=root/'run-status.md';old=path.read_text()
 header='## '+stamp+' — frozen actor50 probe warmup '+str(last['completed_epochs'])+'/100 epochs, Adam/EMA '+str(last['ema_updates'])+'; '+('FAULT' if r.get('failure') else 'same recovered execution advancing')+'\n\n'
 body='Currentkey current_a800_actor50_probe_balanced100; handle '+c['handle']+'; run81df904ad9584e29a35d703570b2bcef / exec'+c['execution_id']+' / source'+c['source_commit']+'. Latest actual '+w['receipt']+'. Full checkpoint '+last['sha256']+'; fourrank actual Adam '+str(last['actual_adam_steps_per_rank'])+', EMA '+str(last['ema_updates'])+', actorupdates0. Currentstage '+str(r.get('status'))+'. EMA fixedtest MixedBA '+str(test['balanced_accuracy'])+', ordinaryaccuracy '+str(test['accuracy'])+', loss '+str(test['total_loss'])+'; originalunsampled currenttrain512 BA '+str(train['balanced_accuracy'])+'. Training questions change each2epochs, test G5 events stayfixed. No currentfailure '+str(r.get('failure') is None)+'. Probe CUDA fullstate continuation actually verified '+str(c.get('actual_CUDA_recovery_verified'))+'; initialrecoveryfollowupsatisfied '+str(c.get('startup_recovery_followup_satisfied',False))+'. Actualfirst2 sourcef0 preserved, source d52 scheduling128/4096/cache.85 only; newrun fullraw/EMA/Adam/4rankRNG continue, no repeatedcompletedepochs. Fulltarget100 still pending; do not stop at startupacceptance. Originalformalactor119 remains USER_STOPPED; Appmonitor search-g1-a800-30 remains PAUSED, no new scheduledautomation. Historicalothergoals untouched; no subagents. Operationalhandoff '+str(root/'preparation/actor50-probe100-current-handoff.json')+'. All tensors/features/rollouts/checkpoints remote; local/publicGit code/config/scalar/hash only.\n\n'
 path.write_text(header+body+old)
