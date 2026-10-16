"""Root-only local observer: no submission, stop, restart, or GPU testing."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from reproduction.h1008_qprobe.management_transport import remote

REMOTE = r"""
import sys,json,pathlib,hashlib,time
b=pathlib.Path(BASE);sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_selection200.launch import observe
v=observe(b);j=v['latest_joint']or{};n=j.get('outer',0)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
facts=[]
for outer in range(VERIFIED+1,n+1):
 root=b/'outers'/('outer-%06d'%outer);joint=read(root/'complete.json');a=joint['actor'];pr=[];ar=[]
 for rank in range(8):
  path=pathlib.Path(a['rank_receipts'][rank]['ledger_path']);rows=[json.loads(s)for s in path.read_text().splitlines()];writes=[r for r in rows if r['event']=='ACTUAL_ADAM_WRITE'];restored=[r for r in rows if r['event']=='ACTUAL_RESTORE']
  assert len(writes)==20 and [r['actual_adam_steps']for r in writes]==list(range(20*(outer-1)+1,20*outer+1)) and sha(path)==a['rank_receipts'][rank]['ledger_sha256']
  ar.append(dict(rank=rank,writes=20,last_actual_adam=writes[-1]['actual_adam_steps'],restore=restored[0],sha256=sha(path),path=str(path)))
  path=root/'probe-update'/('rank-%d-updates.jsonl'%rank);rows=[json.loads(s)for s in path.read_text().splitlines()]
  assert len(rows)==10 and [r['actual_adam_steps']for r in rows]==list(range(10*(outer-1)+1,10*outer+1))
  pr.append(dict(rank=rank,writes=10,last_actual_adam=rows[-1]['actual_adam_steps'],sha256=sha(path),path=str(path)))
 facts.append(dict(outer=outer,actor=ar,probe=pr,phase_timings=joint['phase_timings'],probe_metrics=joint['probe_metrics']))
qa=b/'recovered-validation/complete.json';qa_value=read(qa)if qa.exists()else None
if qa_value:qa_value={k:val for k,val in qa_value.items()if k not in('output_shards','rank_receipts')}
stage=v.get('stage')or{};current=stage.get('outer',0);prefix=v.get('current_prefix')or[];continuation=v.get('current_continuation')or[]
print(json.dumps(dict(base=str(b),observed_unix=v['observed_unix'],native=v['native'],stage=stage,complete_outer=n,actor_adam_per_rank=j.get('actor',{}).get('actual_adam_steps_per_rank'),probe_adam_per_rank=j.get('probe',{}).get('actual_adam_steps_per_rank'),probe_metrics=j.get('probe_metrics'),facts=facts,qa5=qa_value,gpu_actual=v['gpu_actual'],prefix=prefix,continuation=continuation,cleanup=v['cleanup'],pipeline_complete=v['pipeline_complete'])))
"""

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--job',required=True);p.add_argument('--root',required=True);a=p.parse_args()
 root=Path(a.root);state=root/'search-g1-prefix-selection200-state.json';verified=5;last=None;proofs={}
 while True:
  tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
  try:
   script=REMOTE.replace('BASE',repr(a.base)).replace('VERIFIED',str(verified))
   value=json.loads(remote(script,root/('prefix-selection200-watch-'+tag+'-transport.json')))
   path=root/('prefix-selection200-watch-'+tag+'-actual.json');path.write_text(json.dumps(value,indent=2)+'\n')
   for fact in value['facts']:proofs[fact['outer']]=fact;verified=max(verified,fact['outer'])
   native=(value.get('native')or{}).get('JobState','UNKNOWN');stage=value['stage'];n=value['complete_outer']
   next_started=stage.get('outer',0)>=11 and any((r or{}).get('completed_trajectories',0)>0 for r in value['prefix'])
   # Prefix workers use a real generation token/trajectory counter; a phase
   # name alone is not the following iteration starting successfully.
   if stage.get('outer',0)>=11:
    next_started=any(any(isinstance(val,(int,float))and val>0 and k in('completed','completed_trajectories','completed_prefixes','generated','generated_trajectories','engine_steps')for k,val in(r or{}).items())for r in value['prefix'])
   accepted=n>=10 and set(range(6,11))<=set(proofs) and next_started
   s=json.loads(state.read_text())
   if str(s['current_job_id'])!=a.job:raise SystemExit('Observer superseded; preserve its evidence')
   s.update(actual_native_state=native,stage=stage.get('stage'),complete_outer=n,actor_adam_per_rank=value['actor_adam_per_rank'],probe_adam_per_rank=value['probe_adam_per_rank'],last_actual_observation=str(path),last_actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),resumed_outers_verified=sorted(proofs),resumed_five_accepted=accepted,root_readonly_watch_active=not accepted,recovery_in_progress=n<6)
   if value.get('qa5'):s['qa5']=value['qa5']
   state.write_text(json.dumps(s,indent=2)+'\n')
   key=(native,stage.get('stage'),n)
   if key!=last:print(json.dumps(dict(job=a.job,native=native,stage=stage,n=n,qa5=value.get('qa5'),new_verified_outers=sorted(proofs))),flush=True);last=key
   if accepted:
    summary=root/'prefix-selection200-resume5-five-actual-facts.json';summary.write_text(json.dumps(dict(job=a.job,base=a.base,facts=proofs,next_started_observation=str(path),next_started=True),indent=2)+'\n');print('FIVE_RESUMED_COMPLETE_AND_NEXT_GENERATION_ACTUAL',flush=True);return
   if native in('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')or(value.get('cleanup')and not value.get('pipeline_complete')):
    s['root_readonly_watch_active']=False;state.write_text(json.dumps(s,indent=2)+'\n');raise SystemExit(3)
  except (RuntimeError,ValueError,KeyError,AssertionError)as error:
   print(json.dumps(dict(observation_error=repr(error),training_modified=False,tag=tag)),flush=True)
  time.sleep(45)

if __name__=='__main__':main()
