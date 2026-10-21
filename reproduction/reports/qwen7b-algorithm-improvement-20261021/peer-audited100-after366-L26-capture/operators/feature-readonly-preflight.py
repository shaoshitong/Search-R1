import json,os,pathlib,shutil,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-audited100-after366-L26-replay-v1')
p=b if b.exists() else b.parent
helptext=subprocess.check_output(['srun','--help'],universal_newlines=True)
queue=subprocess.check_output(['squeue','-h','-u','mou176','-o','%i|%j|%T|%R|%b'],universal_newlines=True)
owned=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  if proc.stat().st_uid!=os.getuid():continue
  cmd=(proc/'cmdline').read_bytes()
  if int(proc.name)!=os.getpid() and any(str(b).encode() in arg for arg in cmd.split(b'\0')[1:] if len(arg)<2048) and b'python' in cmd.split(b'\0')[0]:
   owned.append(dict(pid=int(proc.name),start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
quota={};commands={}
for name in ['lfs','quota','mmlsquota','mmlsattr','findmnt']:
 commands[name]=shutil.which(name)
def query(label,args):
 try:
  v=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=8);quota[label]=dict(args=args,exit_code=v.returncode,output=v.stdout[-5000:])
 except subprocess.TimeoutExpired:quota[label]=dict(args=args,status='TIMEOUT_QUOTA_NOT_CONFIRMED')
uid=os.getuid();owner=p.stat().st_uid
if commands['findmnt']:query('mount', [commands['findmnt'],'-T',str(p),'-n','-o','SOURCE,FSTYPE,TARGET'])
if commands['lfs']:
 query('current_user_lustre',[commands['lfs'],'quota','-u',str(uid),str(p)])
 if owner!=uid:query('directory_owner_lustre',[commands['lfs'],'quota','-u',str(owner),str(p)])
 query('directory_project_metadata',[commands['lfs'],'project','-d',str(p)])
if commands['quota']:query('current_user_quota',[commands['quota'],'-s','-u',str(uid)])
if commands['mmlsquota']:query('current_user_gpfs',[commands['mmlsquota'],'-u',str(uid),'-Y',str(p)])
out=dict(status='READ_ONLY_NEW_DELTA_FEATURE_STORAGE_QUEUE_AND_LOCK_PREFLIGHT',epoch=time.time(),root=str(b),root_exists=b.exists(),current_uid=uid,directory_owner_uid=owner,quota_tools=commands,quota_observations=quota,user_or_directory_available_quota_confirmed=False,shared_df_is_not_user_quota=True,active_training_future_CP_reserve_bytes=400*1024**3,free_bytes=shutil.disk_usage(p).free,minimum_free_bytes=850*1024**3,other_current_account_jobs=queue.strip().splitlines(),new_namespace_python_processes=owned,existing_job_ids=[str(x) for x in b.glob('pair-*/job.id')],existing_capture_locks=[str(x) for x in b.glob('shard-*/capture/capture.lock')],srun_gpus_per_task_supported='--gpus-per-task' in helptext,srun_gpu_bind_supported='--gpu-bind' in helptext,slurm_version=subprocess.check_output(['srun','--version'],universal_newlines=True).strip())
print(json.dumps(out))
assert out['free_bytes']>out['minimum_free_bytes']
assert out['srun_gpus_per_task_supported'] and out['srun_gpu_bind_supported']
assert not owned and not out['existing_job_ids'] and not out['existing_capture_locks']
print(json.dumps(out))
