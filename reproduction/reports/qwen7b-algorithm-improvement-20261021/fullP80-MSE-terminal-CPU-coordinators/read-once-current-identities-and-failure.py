from pathlib import Path
import json,time,subprocess
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');root=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v1';expected={'750189':(3336295,'1408487',9137,2026102227),'750190':(3344791,'1410521',19137,2026102302)};common=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock');results=[]
def read(p):return json.loads(p.read_bytes())
for job,(pid,tick,seed,plan)in expected.items():
 out=root/('job-'+job);controller=read(out/'controller.json');assert controller['pid']==pid and controller['start_ticks']==tick and controller['job_id']==job and controller['source']['sha256']=='03f147a6405ee3ea44e62e1a3097d8202120956af444be4fb378583efa7afd38';p=Path('/proc/%d'%pid);live=False
 try:live=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==tick
 except(FileNotFoundError,ProcessLookupError):pass
 fds=[]
 if live:
  for f in(p/'fd').iterdir():
   try:fds.append(dict(fd=f.name,target=str(f.resolve()),fdinfo=(p/'fdinfo'/f.name).read_text()))
   except(FileNotFoundError,PermissionError):pass
  assert common not in[f['target']for f in fds]and not any(f['target'].endswith(('.pt','.safetensors'))for f in fds)
 snapshot={name:read(out/name)for name in('launch.json','controller.json','progress.json','active-child.json','complete.json','failure.json')if(out/name).is_file()};base=A/('full169028-L15P80-MSE-ownG5-physical-seed%d-plan%d-20261023-v3'%(seed,plan));training={}
 for rank in(0,1):
  path=base/('progress-rank-%d.json'%rank)
  try:r=read(path);training[str(rank)]={k:r.get(k)for k in('epoch','cursor','actual_Adam_updates','time')}
  except(FileNotFoundError,json.JSONDecodeError)as error:training[str(rank)]=dict(metadata_read_error=repr(error))
 results.append(dict(job_id=job,expected_PID=pid,expected_start_ticks=tick,actual_same_identity_live=live,actual_open_FDs=fds,common_CPU_mutex_FD_open=False if live else None,coordinator_snapshot=snapshot,fresh_training_progress=training,CPU_child_was_launched='active-child.json'in snapshot))
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j','750189,750190','--format=JobIDRaw,State,ExitCode,Elapsed'],universal_newlines=True,timeout=15);print(json.dumps(dict(status='ONCE_ONLY_CURRENT_COORDINATOR_METADATA_IDENTITY_CHECK_WITH_PRESERVED_FAILURE',time=time.time(),accounting=accounting.strip(),results=results,workers_started_or_restarted=False,worker_or_training_source_changed=False,CP_or_tensor_or_final_read=False,Slurm_mutated=False)))
