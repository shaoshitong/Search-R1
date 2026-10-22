"""One bounded storage/stat/proc read; never traverses trees or opens CP data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,pwd,shutil,subprocess,time
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';assert B.is_dir();user=pwd.getpwuid(os.getuid()).pw_name
commands={name:shutil.which(name)for name in('quota','lfs','mmlsquota','beegfs-ctl','dpc','dpcstat')}
def query(argv):
 try:
  p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);return dict(argv=argv,timeout_seconds=10,returncode=p.returncode,stdout=p.stdout[-5000:],stderr=p.stderr[-1500:])
 except subprocess.TimeoutExpired as e:return dict(argv=argv,timeout_seconds=10,status='TIMEOUT_UNKNOWN',error=repr(e))
 except OSError as e:return dict(argv=argv,timeout_seconds=10,status='UNAVAILABLE_UNKNOWN',error=repr(e))
quota_argv=[commands['quota'],'-s','-u','mou176']if commands['quota']and user=='mou176'else([commands['lfs'],'quota','-u','mou176',str(B)]if commands['lfs']and user=='mou176'else None)
pool=ThreadPoolExecutor(max_workers=2);df_future=pool.submit(query,['df','-B1','--output=source,fstype,size,used,avail,pcent,target',str(B)]);quota_future=pool.submit(query,quota_argv)if quota_argv else None;v=os.statvfs(B);storage=dict(target=str(B),statvfs_block_size=v.f_frsize,total_bytes=v.f_blocks*v.f_frsize,available_to_current_user_bytes=v.f_bavail*v.f_frsize,free_bytes=v.f_bfree*v.f_frsize,current_uid=os.getuid(),current_user=user,available_bytes_are_filesystem_free_not_user_quota=True)
checkpoints=[]
for seed,plan,job in [(9137,2026102227,'750189'),(19137,2026102302,'750190')]:
 base=A/('full169028-L15P80-MSE-ownG5-physical-seed%d-plan%d-20261023-v3'%(seed,plan));published=[];temporary=[]
 for p in sorted(base.glob('checkpoint-epoch-*')):
  if p.suffix not in('.pt','.tmp'):continue
  s=p.stat();r=dict(name=p.name,apparent_bytes=s.st_size,allocated_bytes=s.st_blocks*512,mtime_ns=s.st_mtime_ns)
  (published if p.suffix=='.pt'else temporary).append(r)
 epochs=[int(r['name'].split('epoch-')[1].split('.')[0])for r in published];assert len(epochs)==len(set(epochs))and all(1<=e<=10 for e in epochs);reference=max([r['apparent_bytes']for r in published],default=None);remaining=[e for e in range(1,11)if e not in epochs];total=sum(r['apparent_bytes']for r in published);partial=sum(r['apparent_bytes']for r in temporary);checkpoints.append(dict(job_id=job,init_seed=seed,base=str(base),published_files=published,published_checkpoint_count=len(published),published_checkpoint_apparent_bytes=total,published_checkpoint_allocated_bytes=sum(r['allocated_bytes']for r in published),temporary_files=temporary,published_epochs=epochs,remaining_epochs_to10=remaining,reference_CP_bytes=reference,remaining_full_CP_save_bytes_conservative=len(remaining)*reference if reference else None,remaining_additional_bytes_after_existing_temporary_approx=max(0,len(remaining)*reference-partial)if reference else None,only_direct_CP_directory_entries_statted=True,checkpoint_content_not_opened=True))
common=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock');workers=[]
for version,job,pid,tick in [('v1','750189',3336295,'1408487'),('v2','750190',29786,'1771045')]:
 out=A/('fullP80-MSE-terminal-CPU-coordinators-20261023-'+version)/('job-'+job);controller=json.loads((out/'controller.json').read_bytes());assert controller['pid']==pid and controller['start_ticks']==tick and controller['job_id']==job;p=Path('/proc/%d'%pid);live=False
 try:live=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==tick
 except(FileNotFoundError,ProcessLookupError):pass
 fds=[]
 if live:
  for f in(p/'fd').iterdir():
   try:fds.append(dict(target=str(f.resolve()),fdinfo=(p/'fdinfo'/f.name).read_text()))
   except(FileNotFoundError,PermissionError):pass
 own=[f for f in fds if f['target']==str(out/'coordinator.lock')];progress=json.loads((out/'progress.json').read_bytes());workers.append(dict(job_id=job,version=version,PID=pid,start_ticks=tick,actual_same_identity_live=live,own_coordinator_FLOCK_confirmed=bool(own)and'FLOCK'in own[0]['fdinfo'],common_CPU_mutex_FD_open=common in[f['target']for f in fds],checkpoint_tensor_FD_open=any(f['target'].endswith(('.pt','.safetensors'))for f in fds),CPU_child_receipt_exists=(out/'active-child.json').exists(),progress=progress,failure=json.loads((out/'failure.json').read_bytes())if(out/'failure.json').exists()else None))
storage['df']=df_future.result();quota=quota_future.result()if quota_future else dict(status='UNKNOWN_NO_SUPPORTED_INSTALLED_CURRENT_USER_QUOTA_COMMAND',quota_bytes_or_limit_claimed=None);pool.shutdown();print(json.dumps(dict(status='BOUNDED_SHARED_STORAGE_CP_STAT_AND_TWO_COORDINATOR_METADATA_CHECK_COMPLETE',time=time.time(),storage=storage,quota_command_inventory=commands,quota_query=quota,checkpoints=checkpoints,combined_published_CP_apparent_bytes=sum(x['published_checkpoint_apparent_bytes']for x in checkpoints),combined_remaining_CP_save_bytes_conservative=sum(x['remaining_full_CP_save_bytes_conservative']for x in checkpoints)if all(x['remaining_full_CP_save_bytes_conservative']is not None for x in checkpoints)else None,workers=workers,du_or_recursive_tree_scan=False,CP_hash_or_tensor_load=False,files_deleted=False,workers_or_training_or_Slurm_changed=False,final4096_read=False)))
