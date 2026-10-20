"""Owned read-only follow-through: await two real advances and a sealed batch/phase."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parents[4]
LATEST=CONTROL/'outputs/qwen7b-warmup-20261019/extraG5-fork-live-observation.json'
# Calculate the control root from the repo path, never from a remote payload.
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
LATEST=CONTROL/'outputs/qwen7b-warmup-20261019/extraG5-fork-live-observation.json'

def summarize(v):
 result={}
 for offset,p in v['phases'].items():
  sampled=completed=sealed=0;failures=[]
  for shard,row in p['shards'].items():
   if row.get('failure.json'):failures.append(dict(shard=shard,**row['failure.json']))
   for batch in row['batches']:
    for phase,own in batch['phases'].items():
     sampled+=own.get('actual_sampled_tokens',0);completed+=own.get('completed_trajectories',0)
    sealed+=int('sealed' in batch)
  result[offset]=dict(actual_sampled_tokens=sampled,completed_phase_trajectories=completed,sealed_batches=sealed,failures=failures)
 return result

def main():
 source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();seen={'10':set(),'15':set()};advances={'10':0,'15':0};last={}
 for old in (HERE/'observations.jsonl').read_text().splitlines():
  try:v=json.loads(old)
  except ValueError:continue
  for o,row in summarize(v).items():
   value=(row['actual_sampled_tokens'],row['completed_phase_trajectories'])
   if value[0]>0 and value[1]>0:seen[o].add(value)
 started=time.monotonic()
 while True:
  p=subprocess.run([sys.executable,str(HERE/'observe_forks.py')],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=65)
  if p.returncode:
   report=dict(status='OBSERVER_READ_FAILURE_NOT_TRAINING_FAILURE',error=p.stderr[-1500:],operator_sha256=source)
   print(json.dumps(report),flush=True);return 2
  v=json.loads(LATEST.read_text());rows=summarize(v)
  for o,row in rows.items():
   value=(row['actual_sampled_tokens'],row['completed_phase_trajectories'])
   if value[0]>0 and value[1]>0:seen[o].add(value)
   row['distinct_real_counter_observations']=len(seen[o])
  result=dict(status='WAITING_FOR_REAL_COLLECTION_ADVANCE',observed_epoch=v['observed_epoch'],phases=rows,squeue=v['squeue'],sacct=v['sacct'],operator_sha256=source,elapsed=time.monotonic()-started)
  failed=any(row['failures'] for row in rows.values()) or any(x in v['sacct'] for x in ['|FAILED|','|OUT_OF_MEMORY|','|NODE_FAIL|','|TIMEOUT|','|CANCELLED|'])
  if failed:result['status']='OWNED_COLLECTION_REQUIRES_FAILURE_DIAGNOSIS'
  elif all(row['distinct_real_counter_observations']>=2 and row['sealed_batches']>=1 for row in rows.values()):result['status']='BOTH_PHASES_TWO_REAL_OBSERVATIONS_AND_ONE_SEALED_BATCH'
  (HERE/'watch-latest.json').write_text(json.dumps(result,indent=2)+'\n')
  if result['phases']!=last or result['status']!='WAITING_FOR_REAL_COLLECTION_ADVANCE':print(json.dumps(result),flush=True);last=result['phases']
  if result['status']!='WAITING_FOR_REAL_COLLECTION_ADVANCE':return 1 if failed else 0
  time.sleep(50)
if __name__=='__main__':sys.exit(main())
