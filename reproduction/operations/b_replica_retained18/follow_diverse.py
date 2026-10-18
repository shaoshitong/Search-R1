"""Read-only foreground follow through restored51..55 and positive56 activity."""
import datetime,json,subprocess,sys,time
from pathlib import Path
from .control import REPO,OUT
ROOT=OUT.parents[1]

def main():
 while True:
  result=subprocess.run([sys.executable,'-m','reproduction.operations.b_replica_retained18.diverse_control','observe'],cwd=REPO,capture_output=True,text=True,timeout=260)
  if result.returncode:
   print(json.dumps(dict(status='THIS_OBSERVATION_UNKNOWN',error=result.stderr[-1000:])),flush=True);time.sleep(35);continue
  latest=json.loads((OUT/'DIVERSE_LATEST.json').read_text());v=latest['actual'];stage=v.get('stage')or{};facts=v.get('facts',[])
  count=v.get('joint_outer')or 50
  small=dict(status='ACTUAL_NATIVE_EXITED_NEEDS_REPAIR' if v.get('cleanup') else 'ACTUAL_NATIVE_CONTINUING',observed_unix=v['observed_unix'],node='h100-10',ip='10.33.0.18',world_size=8,
   parent_complete_outer=50,complete_outer=count,new_complete_outers=count-50,stage=stage,
   actual_facts=facts,progress={p:{k:r.get(k)for k in ('completed_trajectories','expected_trajectories','retained_prefixes','elapsed_seconds','stage','scored_trajectories')}for p,r in v.get('progress',{}).items()},
   latest_evidence=latest['evidence'],runtime_commit=v['source_commit'],active_actor=v.get('active_actor',[]),training_started=True,total_target_outer=200,
   no_optimizer_reset=True,exact_executor_resume_claimed=False)
  (OUT/'DIVERSE_STATUS.json').write_text(json.dumps(small,ensure_ascii=False,indent=2))
  old=json.loads((ROOT/'outputs/STATUS.json').read_text());old.update(status=small['status'],current_stage=stage.get('stage'),current_task='B_CP50_DIVERSE10TO5_EIGHT_GPU_FORK',world_size=8,native_leader=(v.get('training')or{}).get('receipt',{}).get('leader'),physical_base='/mnt/data/shaoshitong/'+Path(json.loads((OUT/'DIVERSE_CONFIG.json').read_text())['base']).name,actual_training_running=not bool(v.get('cleanup')),training_started=True,training_authorized=True,preparation_only=False,experiment_assignment_pending=False,base=json.loads((OUT/'DIVERSE_CONFIG.json').read_text())['base'],source_commit=v['source_commit'],actual_complete_outer=count,new_complete_outers=count-50,current_evidence=latest['evidence'],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(ROOT/'outputs/STATUS.json').write_text(json.dumps(old,ensure_ascii=False,indent=2))
  mirrors=[ROOT/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')]
  for target in mirrors:
   target.mkdir(parents=True,exist_ok=True);(target/'DIVERSE_STATUS.json').write_text(json.dumps(small,ensure_ascii=False,indent=2))
  print(json.dumps(dict(observed_unix=v['observed_unix'],complete=count,stage=stage.get('stage'),progress=small['progress'],latest_fact=facts[-1]if facts else None,cleanup=v.get('cleanup')),ensure_ascii=False),flush=True)
  training=v.get('training')or{};actual=training.get('actual')
  if v.get('cleanup')or not actual:
   raise RuntimeError('Actual native exited or cleanup recorded; preserve evidence and repair, never replay blindly')
  if count>=55 and stage.get('outer',0)>=56 and small['progress']:
   print('FIVE_RESTORED_OUTERS_AND_POSITIVE_NEXT_ACCEPTED',flush=True);return
  time.sleep(35)
if __name__=='__main__':main()
