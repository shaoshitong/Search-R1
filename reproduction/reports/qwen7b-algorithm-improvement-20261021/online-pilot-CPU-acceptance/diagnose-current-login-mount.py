import pathlib,json,socket,os,time
r=dict(time=time.time(),hostname=socket.gethostname(),uid=os.getuid(),paths=[])
for name in('/data','/data/user','/data/user/jxu957','/data/modulefiles','/etc/slurm','/etc/slurm/slurm.conf','/etc/slurm.conf'):
 p=pathlib.Path(name);v=dict(path=name,exists=p.exists())
 if p.exists():
  s=p.stat();v.update(device=s.st_dev,inode=s.st_ino,is_dir=p.is_dir())
 r['paths'].append(v)
r['slurm_environment']={k:v for k,v in os.environ.items()if k in('SLURM_CONF','SLURM_CONF_SERVER')}
mounts=pathlib.Path('/proc/mounts').read_text().splitlines();r['data_mounts']=[line for line in mounts if' /data 'in line or' /data/'in line]
print(json.dumps(r))
