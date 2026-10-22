"""Blind production observer: counters/process identities only, never EM or probe scores."""
import runpy,pathlib,json,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
module=runpy.run_path(str(base/'prepare_frozen_final4096_generation.py'))
x=module['status']();x['epoch']=time.time()
x['squeue']=subprocess.check_output(['squeue','-h','-u','mou176','-n','q7b-final-cA,q7b-final-cB,q7b-final-sA,q7b-final-sB','-o','%i|%T|%j|%N|%R'],universal_newlines=True).strip().splitlines()
x['sacct']=subprocess.check_output(['sacct','-n','-P','-j','749134,749136,749137,749138','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).strip().splitlines()
prepared=json.loads((base/'prepared.json').read_text())
for event,entry in zip(x['arrays'],prepared['arrays']):
 for shard in event['shards']:
  root=pathlib.Path(entry['original_event_root'])/'sampling'/('shard-%02d'%shard['shard'])
  if (root/'failure.json').exists():
   error=json.loads((root/'failure.json').read_text());shard['failure']={k:error[k]for k in ('error_type','error','epoch')if k in error}
  batches=sorted((root/'batches').glob('batch-*'))if(root/'batches').exists()else[]
  unsealed=[p for p in batches if not(p/'complete.json').exists()]
  if unsealed:
   active=unsealed[-1];shard['active_batch']=active.name;shard['phases']=[]
   for phase in ('prefix','full'):
    p=active/'outers/outer-000001'/phase/'rank-0-progress.json'
    if p.exists():
     progress=json.loads(p.read_text());shard['phases'].append(dict(phase=phase,**{k:progress[k]for k in ('stage','completed_trajectories','expected_trajectories','active_trajectories','actual_sampled_tokens','pending_generation','pending_retrieval','elapsed_seconds','utc')if k in progress}))
x['final_labels_or_predictions_read']=False
print(json.dumps(x))
