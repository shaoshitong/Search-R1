from pathlib import Path
import json,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');root=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v1';common=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock');observations=[]
for job in('750189','750190'):
 out=root/('job-'+job);controller=json.loads((out/'controller.json').read_bytes());progress=json.loads((out/'progress.json').read_bytes());launch=json.loads((out/'launch.json').read_bytes());pid=controller['pid'];p=Path('/proc/%d'%pid);ticks=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19];assert ticks==controller['start_ticks']==launch['start_ticks']and pid==launch['pid'];argv=[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x];assert controller['source']['path']in argv and controller['config']['path']in argv
 fds=[]
 for f in(p/'fd').iterdir():
  try:fds.append(dict(fd=f.name,target=str(f.resolve()),fdinfo=(p/'fdinfo'/f.name).read_text()))
  except(FileNotFoundError,PermissionError):pass
 own=str(out/'coordinator.lock');locks=[f for f in fds if f['target']==own];assert len(locks)==1 and'FLOCK'in locks[0]['fdinfo']and str(pid)in locks[0]['fdinfo'];assert common not in[f['target']for f in fds];assert not any(f['target'].endswith(('.pt','.safetensors'))for f in fds)
 assert progress['job_id']==job and progress['stage']=='WAITING_ACTUAL_TERMINAL_NO_CP_OR_COMMON_LOCK_HELD'and progress['CP_or_tensor_opened']is False and progress['shared_CPU_mutex_held']is False and progress['actual_complete']is None and'RUNNING'in progress['accounting'];assert not(out/'failure.json').exists()and not(out/'active-child.json').exists()
 observations.append(dict(job_id=job,actual_pid=pid,actual_start_ticks=ticks,actual_identity_live=True,argv=argv,own_coordinator_lock_FD=locks[0],common_CPU_mutex_FD_open=False,checkpoint_tensor_FD_open=False,progress=progress,no_failure=True,no_CPU_child_launched=True))
print(json.dumps(dict(status='TWO_ACTUAL_FULL_MSE_TERMINAL_WORKERS_METADATA_IDENTITY_AND_LOCKS_VERIFIED',time=time.time(),observations=observations,CP_or_tensor_read=False,common_CPU_lock_acquired_by_check=False,workers_started_or_changed=False,training_or_Slurm_mutated=False,final4096_read=False)))
