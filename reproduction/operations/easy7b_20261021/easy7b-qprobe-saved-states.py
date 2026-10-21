import json,time,os,subprocess
from pathlib import Path
out=dict(epoch=time.time(),uid=os.getuid(),files=[],errors=[],bases=[])
root=Path('/data/user/jxu957/experiments')
for b in root.glob('qprobe-teacher-warmup200-*'):
 out['bases'].append(str(b))
 for top in ['artifacts','warmup','checkpoints','archives','reserve','recovery']:
  start=b/top
  if not start.is_dir()or start.is_symlink():continue
  for directory,children,files in os.walk(start,followlinks=False,onerror=lambda e:out['errors'].append(str(e))):
   children[:]=[n for n in children if n not in ['dumps','cache','__pycache__','raw-input','raw-trajectories','token-features','features','trajectory-cache'] and not Path(directory,n).is_symlink()]
   for name in files:
    p=Path(directory,name)
    if p.suffix not in ['.pt','.pth','.safetensors','.bin'] or p.is_symlink():continue
    try:
     s=p.stat();out['files'].append(dict(path=str(p),size=s.st_size,allocated=s.st_blocks*512,dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,uid=s.st_uid,mtime_ns=s.st_mtime_ns))
    except OSError as e:out['errors'].append(str(e))
print(json.dumps(out))
