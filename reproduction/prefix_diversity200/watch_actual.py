"""Bounded root read-only observer; never submits, stops or changes training."""
import argparse
import json
from pathlib import Path
import time
from reproduction.h1008_qprobe.management_transport import remote
from .poll_local import SCRIPT

EXTRA = r'''
facts=[]
for completed in range(VERIFIED+1,value['latest_joint']['outer']+1):
 root=base/'outers'/('outer-%06d'%completed);joint=json.loads((root/'complete.json').read_text());actor=joint['actor'];ar=[];pr=[]
 for rank in range(8):
  p=pathlib.Path(actor['rank_receipts'][rank]['ledger_path']);rows=[json.loads(x) for x in p.read_text().splitlines()];writes=[x for x in rows if x['event']=='ACTUAL_ADAM_WRITE'];restores=[x for x in rows if x['event']=='ACTUAL_RESTORE']
  assert len(writes)==20 and [x['actual_adam_steps'] for x in writes]==list(range(20*(completed-1)+1,20*completed+1))
  assert hashlib.sha256(p.read_bytes()).hexdigest()==actor['rank_receipts'][rank]['ledger_sha256']
  assert len(restores)==1 and restores[0]['restored_completed_outer']==completed-1
  ar.append(dict(rank=rank,path=str(p),writes=20,restore=restores[0],last=writes[-1],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  p=root/'probe-update'/('rank-%d-updates.jsonl'%rank);rows=[json.loads(x) for x in p.read_text().splitlines()]
  assert len(rows)==20 and [x['actual_adam_steps'] for x in rows]==list(range(20*(completed-1)+1,20*completed+1))
  pr.append(dict(rank=rank,path=str(p),writes=20,first=rows[0],last=rows[-1],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 facts.append(dict(outer=completed,actor=ar,probe=pr,joint_sha256=hashlib.sha256((root/'complete.json').read_bytes()).hexdigest(),observed_unix=joint['observed_unix'],phase_timings=joint['phase_timings'],actor_selection={k:v for k,v in joint['actor_selection'].items() if k!='selected'},probe_metrics=joint['probe_metrics']))
value['new_complete_facts']=facts
qa=base/'recovered-validation/complete.json';value['qa5']=json.loads(qa.read_text()) if qa.exists() else None
value['qa5_progress']=[json.loads(p.read_text()) for p in (base/'recovered-validation').glob('rank-*-progress.json')]
value['qa_pending']=json.loads((base/'logs/qa-pending.json').read_text()) if (base/'logs/qa-pending.json').exists() else None
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipts',required=True);p.add_argument('--through',type=int,default=10);args=p.parse_args()
 c=json.loads(Path(args.config).read_text());w=Path(args.receipts);verified=c['resume_joint']['completed_outer'];facts={};previous=None
 started=time.monotonic()
 while time.monotonic()-started<3600:
  tag=str(time.time_ns());path=w/('resume-watch-'+tag+'-transport.json')
  try:
   code=SCRIPT.replace('print(json.dumps(value,allow_nan=False))',EXTRA.replace('VERIFIED',str(verified))+'\nprint(json.dumps(value,allow_nan=False))').replace('BASE',repr(c['base']))
   value=json.loads(remote(code,path,target=c['node']));actual=w/('resume-watch-'+tag+'-actual.json');actual.write_text(json.dumps(value,indent=2)+'\n');(w/'LATEST_ACTUAL.json').write_text(actual.read_text())
   for row in value['new_complete_facts']:facts[str(row['outer'])]=row;verified=max(verified,row['outer'])
   stage=value.get('stage') or {};native=value.get('native') or {};n=value['latest_joint']['outer']
   next_generation=stage.get('outer',0)>args.through and any(any(k in('completed','completed_trajectories','completed_prefixes','generated','generated_trajectories','engine_steps') and isinstance(v,(int,float)) and v>0 for k,v in (row or {}).items()) for row in value.get('current_prefix',[]))
   summary=dict(job=native.get('JobId'),native=native.get('JobState','UNKNOWN'),complete_outer=n,stage=stage,new_verified_outers=sorted(facts,key=int),qa5=value.get('qa5'),qa5_progress=sum(x.get('completed_questions',0) for x in value['qa5_progress']),evidence=str(actual))
   key=(summary['native'],stage.get('stage'),n,summary['qa5_progress'])
   if key!=previous:print(json.dumps(summary),flush=True);previous=key
   (w/'RESUMED_FIVE_LATEST.json').write_text(json.dumps(dict(summary=summary,facts=facts,next_generation=next_generation),indent=2)+'\n')
   if verified>=args.through and all(str(i) in facts for i in range(c['resume_joint']['completed_outer']+1,args.through+1)) and next_generation:
    (w/'RESUMED_FIVE_ACCEPTANCE.json').write_text(json.dumps(dict(summary=summary,facts=facts,next_generation=True),indent=2)+'\n');return
   if native.get('JobState') in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY') or (value.get('cleanup') and not value.get('pipeline_complete')):raise SystemExit(3)
  except (RuntimeError,ValueError,KeyError,AssertionError) as error:
   print(json.dumps(dict(observation_error=repr(error),training_modified=False)),flush=True)
  time.sleep(40)
 raise SystemExit('Bounded read-only observer expired; training unchanged')


if __name__=='__main__':main()
