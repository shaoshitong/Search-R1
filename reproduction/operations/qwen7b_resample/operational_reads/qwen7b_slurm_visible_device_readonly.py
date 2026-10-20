import subprocess,json,time
from pathlib import Path
r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
rows=[]
for line in r.stdout.splitlines():
 uid,pid=[x.strip()for x in line.split(',')];p=Path('/proc')/pid
 try:
  env={}
  for entry in(p/'environ').read_bytes().split(b'\0'):
   key,sep,value=entry.partition(b'=')
   if key in [b'CUDA_VISIBLE_DEVICES',b'CUDA_DEVICE_ORDER',b'SLURM_JOB_ID',b'SLURM_JOB_GPUS']:env[key.decode()]=value.decode(errors='replace')
  if env.get('SLURM_JOB_ID'):rows.append(dict(gpu_uuid=uid,pid=int(pid),env=env,cgroup=(p/'cgroup').read_text()))
 except(OSError,ValueError):pass
 if len(rows)>=2:break
print(json.dumps(dict(epoch=time.time(),rows=rows)))
