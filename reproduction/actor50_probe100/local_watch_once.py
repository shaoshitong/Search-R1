"""Foreground single metadata observation; no background loop or App automation."""
import pathlib,json,subprocess,time,sys
root=pathlib.Path('/Users/shaoshitong/Documents/Search-G1');w=root/'actor50-probe-balanced100-20261018'
h=json.loads((root/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json').read_text());c=h['current_authoritative']['current_a800_actor50_probe_balanced100'];assert h['current_authoritative']['active_experiment_key']=='current_a800_actor50_probe_balanced100';stamp=str(time.time_ns())
result=subprocess.run(['python3','-m','reproduction.actor50_probe100.control','--action','observe','--handle',c['handle']],cwd=w,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=75)
if result.returncode:
 print(json.dumps({'observer_exit':result.returncode,'stderr':result.stderr.decode(),'outcome':'unknown_not_zero_or_exit'}));sys.exit(result.returncode)
wrap=json.loads(result.stdout);out=pathlib.Path('/tmp')/('actor50-probe100-watch-'+stamp+'.json');out.write_bytes(result.stdout)
subprocess.run(['python3','-m','reproduction.actor50_probe100.local_record',str(out)],cwd=w,check=True,stdout=subprocess.DEVNULL)
r=wrap['result'];last=r.get('latest');progress=r.get('rank_progress') or []
summary={'wrapper':str(out),'receipt':wrap['receipt'],'at':r['checked_unix'],'leader_alive':r['leader'] is not None,'stage':r.get('status'),'epoch':None if not last else last['completed_epochs'],'Adam_EMA':None if not last else last['ema_updates'],'checkpoint_sha256':None if not last else last['sha256'],'test_BA':None if not last else last['metrics']['after']['test']['balanced_accuracy'],'train_BA':None if not last else last['metrics']['after']['train']['balanced_accuracy'],'test_accuracy':None if not last else last['metrics']['after']['test']['accuracy'],'test_loss':None if not last else last['metrics']['after']['test']['total_loss'],'current_rank_steps':[None if x is None else x.get('actual_returned_adam') for x in progress],'rollout_counts':{k:[None if x is None else x['completed_trajectories'] for x in v] for k,v in (r.get('rollout_progress') or {}).items()},'failure':r.get('failure'),'exit':r.get('exit'),'complete':r.get('complete') is not None,'cleanup':r.get('cleanup')}
print(json.dumps(summary,ensure_ascii=False))
