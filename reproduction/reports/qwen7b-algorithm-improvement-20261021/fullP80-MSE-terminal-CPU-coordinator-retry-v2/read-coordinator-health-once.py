from pathlib import Path
import json,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
common=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock')
def read(p):
 return json.loads(p.read_bytes()) if p.exists() else None
workers=[]
for version,job,pid,tick in [('v1','750189',3336295,'1408487'),('v2','750190',29786,'1771045')]:
 out=A/('fullP80-MSE-terminal-CPU-coordinators-20261023-'+version)/('job-'+job)
 controller=read(out/'controller.json');assert controller['pid']==pid and controller['start_ticks']==tick and controller['job_id']==job
 proc=Path('/proc/%d'%pid);live=False;state=None;children=None
 try:
  stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split();live=stat[19]==tick;state=stat[0]
  if live:children=(proc/'task'/str(pid)/'children').read_text().strip()
 except(FileNotFoundError,ProcessLookupError):pass
 fds=[]
 if live:
  for f in(proc/'fd').iterdir():
   try:fds.append(dict(target=str(f.resolve()),fdinfo=(proc/'fdinfo'/f.name).read_text()))
   except(FileNotFoundError,PermissionError):pass
 own=[f for f in fds if f['target']==str(out/'coordinator.lock')]
 workers.append(dict(job_id=job,version=version,PID=pid,start_ticks=tick,actual_same_identity_live=live,actual_proc_state=state,actual_proc_children=children,own_coordinator_FLOCK_confirmed=bool(own)and'FLOCK'in own[0]['fdinfo'],common_CPU_mutex_FD_open=common in[f['target']for f in fds],checkpoint_tensor_FD_open=any(f['target'].endswith(('.pt','.safetensors'))for f in fds),progress=read(out/'progress.json'),active_child=read(out/'active-child.json'),complete=read(out/'complete.json'),failure=read(out/'failure.json')))
print(json.dumps(dict(status='TWO_EXISTING_COORDINATORS_BOUNDED_METADATA_HEALTH_READ',time=time.time(),workers=workers,CP_tensor_loaded=False,worker_training_Slurm_changed=False,storage_quota_repeated=False,final4096_read=False)))
