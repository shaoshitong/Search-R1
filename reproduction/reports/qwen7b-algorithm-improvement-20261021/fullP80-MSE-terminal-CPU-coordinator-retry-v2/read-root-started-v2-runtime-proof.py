from pathlib import Path
import hashlib,json,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');old=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v1';out=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v2/job-750190';base=A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3';common=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock')
def read(p):return json.loads(p.read_bytes())
def identity(pid,tick):
 p=Path('/proc/%d'%pid)
 try:return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==tick
 except(FileNotFoundError,ProcessLookupError):return False
def worker_snapshot(root,pid,tick,source_sha):
 c=read(root/'controller.json');assert c['pid']==pid and c['start_ticks']==tick and c['source']['sha256']==source_sha;p=Path('/proc/%d'%pid);live=identity(pid,tick);fds=[];argv=[]
 if live:
  argv=[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]
  for f in(p/'fd').iterdir():
   try:fds.append(dict(fd=f.name,target=str(f.resolve()),fdinfo=(p/'fdinfo'/f.name).read_text()))
   except(FileNotFoundError,PermissionError):pass
 own=[f for f in fds if f['target']==str(root/'coordinator.lock')];v=dict(PID=pid,start_ticks=tick,actual_same_identity_live=live,actual_argv=argv,own_coordinator_FLOCK_confirmed=bool(own)and'FLOCK'in own[0]['fdinfo']and str(pid)in own[0]['fdinfo'],own_lock_fd=own,common_CPU_lock_FD_open=common in[f['target']for f in fds],checkpoint_tensor_FD_open=any(f['target'].endswith(('.pt','.safetensors'))for f in fds),CPU_child_receipt_exists=(root/'active-child.json').is_file(),controller=c,progress=read(root/'progress.json'),failure=read(root/'failure.json')if(root/'failure.json').exists()else None);return v
v2=worker_snapshot(out,29786,'1771045','5304f9e9e290820fdde3123f0e34ab98b925f8ecd79a6cda011113ad56ca23ec');seed1=worker_snapshot(old/'job-750189',3336295,'1408487','03f147a6405ee3ea44e62e1a3097d8202120956af444be4fb378583efa7afd38');training={}
assert str(read(base/'root-submission.json')['job_id'])=='750190'
for rank in(0,1):
 try:r=read(base/('progress-rank-%d.json'%rank));training[str(rank)]={k:r.get(k)for k in('epoch','cursor','actual_Adam_updates','time')}
 except(OSError,json.JSONDecodeError)as error:training[str(rank)]=dict(metadata_read_error=repr(error))
f=old/'job-750190/failure.json';raw=f.read_bytes();proof=dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());assert proof['sha256']=='b10be9829e3355a6f31af022b8ae6be2d764920d6195c275e77c31686945d088';print(json.dumps(dict(status='ROOT_STARTED_V2_ONCE_RUNTIME_METADATA_IDENTITY_AND_LOCK_SNAPSHOT',time=time.time(),v2=v2,healthy_seed1_v1=seed1,actual_same_job_training_progress=training,old_seed2_v1_failure_preserved=proof,old_seed2_v1_same_identity_live=identity(3344791,'1410521'),CP_or_tensor_or_final_read=False,workers_started_or_changed=False,training_or_Slurm_changed=False)))
