import os,json,time,subprocess
from pathlib import Path
r=Path('/data/user/jxu957/experiments');out=dict(epoch=time.time(),bases=[],errors=[])
for b in r.glob('qprobe-teacher-warmup200-*'):
 if not b.is_dir():continue
 row=dict(base=str(b),subdirs=[],checkpoint_files=[],small={})
 for p in b.iterdir():
  if p.name in ('checkpoints','checkpoint-archive','archives','reserve','recovery') and p.is_dir():
   dirs=[]
   for directory,children,files in os.walk(p,followlinks=False):
    children[:]=[x for x in children if not Path(directory,x).is_symlink()]
    for name in files:
     f=Path(directory,name)
     if f.is_symlink():continue
     try:
      s=f.stat()
      if f.suffix in ('.pt','.bin','.safetensors'):
       row['checkpoint_files'].append(dict(path=str(f),size=s.st_size,allocated=s.st_blocks*512,dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,uid=s.st_uid,mtime_ns=s.st_mtime_ns))
     except OSError as e:out['errors'].append(str(e))
  elif p.is_dir() and not p.is_symlink():row['subdirs'].append(p.name)
 for n in ['job.id','logs/run-manifest.json','logs/cleanup-status.json','logs/pipeline-stage.json','logs/final-generation-complete.json']:
  p=b/n
  if p.is_file() and p.stat().st_size<100000:
   try:
    x=json.loads(p.read_text())if n.endswith('.json')else p.read_text().strip()
    if n.endswith('run-manifest.json'):x={k:x.get(k)for k in ['run_id','initial_model','full_state_restore','initial_reward_probe_source_base','target_updates','completed_updates']}
    row['small'][n]=x
   except Exception as e:row['small'][n]=repr(e)
 out['bases'].append(row)
x=subprocess.run(['squeue','-u','jxu957,mou176','-h','-o','%i|%j|%T|%R'],capture_output=True,text=True,timeout=8);out['queue']=dict(rc=x.returncode,stdout=x.stdout,stderr=x.stderr)
print(json.dumps(out))
