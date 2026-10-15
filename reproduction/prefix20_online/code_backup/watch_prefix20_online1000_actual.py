"""Read-only current run watcher; no submit, signal, or mutable state writer."""
import datetime,json,pathlib,sys,time,hashlib
WORK=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work')
REPO=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
sys.path.insert(0,str(REPO))
from reproduction.h1008_qprobe.management_transport import remote
control=json.loads((WORK/'prefix20-online1000-state.json').read_text());base=control['base']
while True:
 tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 script="import sys,json;sys.path.insert(0,"+repr(base+'/repo')+");from reproduction.prefix20_online.observe import observe;print(json.dumps(observe("+repr(base)+"),allow_nan=False))"
 try:
  result=json.loads(remote(script,WORK/('prefix20-online-observe-transport-'+tag+'.json')))
  path=WORK/('prefix20-online-observe-'+tag+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
  latest=result.get('latest') or {};stage=result.get('stage') or {};metrics=result.get('latest_metrics') or {}
  compact=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observed_unix=result['observed_unix'],native=result.get('native'),stage=stage,complete_epoch=latest.get('epoch',0),actual_adam_per_rank=latest.get('actual_adam_steps_per_rank'),metric_epoch=metrics.get('epoch'),train_accuracy=metrics.get('train_online_accuracy'),test_accuracy=metrics.get('raw',{}).get('accuracy'),test_balanced_accuracy=metrics.get('raw',{}).get('balanced_accuracy'),test_BCE=metrics.get('raw',{}).get('bce'),test_AUROC=metrics.get('auroc'),rollout_progress=result.get('current_rollout_progress'),current_rank_actual_updates=result.get('current_rank_actual_updates'),current_train_errors=result.get('current_train_errors'),owned_active_births=result.get('owned_active_births'))
  (WORK/'prefix20-online1000-latest-compact.json').write_text(json.dumps(compact,indent=2)+'\n')
  print(json.dumps(compact),flush=True)
  if result.get('native_terminal_verified') or stage.get('stage')=='FAILED':
   print('ACTUAL_TERMINAL_OR_FAILURE_REQUIRES_ROOT',flush=True);break
  progress=[x for x in (compact['rollout_progress'] or {}).values() if x]
  if latest.get('epoch',0)>=control.get('initial_acceptance_complete_epoch',2) and stage.get('epoch',0)>=control.get('initial_acceptance_next_epoch',3) and sum(x.get('completed_trajectories',0) for x in progress)>0:
   print('REQUIRED_FRESH_ROLLOUT_TRAIN_ACCEPTANCE_COMPLETE_NEXT_EPOCH_ADVANCING',flush=True);break
 except Exception as error:
  print(json.dumps(dict(observation_channel_error=type(error).__name__,message=str(error)[:500],training_exit_proven=False,tag=tag)),flush=True)
 time.sleep(45)
