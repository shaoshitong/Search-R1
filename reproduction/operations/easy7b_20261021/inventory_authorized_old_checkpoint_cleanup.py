"""Read-only, exact-root checkpoint inventory for the 2026-10-21 human cleanup."""
import collections
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

ROOT = Path('/data/user/jxu957/experiments')
NAMES = (
 'dynrpo-prefix5-thread5-20261017',
 'dynrpo-prefix5-thread5-20261017-r2',
 'dynrpo-prefix5-thread5-20261017-r3',
 'dynrpo-prefix5-thread5-20261017-r4',
 'topd-20261008', 'cosearch-continue300-x203-20261008',
 'easy2560-corollout-joint4-micro8-after731063-20261018',
 'easy2560-corollout-joint4-micro16-after730080-20261018',
 'easy2560-corollout-joint4-recovery5-after730961-20261018',
 'easy2560-corollout-joint4-20261017',
 'easy2560-corollout-joint4-recovery5-after730912-20261018',
 'easy2560-corollout-joint4-recovery5-after730226-20261018',
)
PRUNE = {'.git','__pycache__','environment','env','venv','.venv','site-packages',
         'repo','source','node_modules','cache','hf-cache','pip-cache'}
TENSOR = {'.pt','.pth','.safetensors','.bin','.ckpt'}
TEXT = {'.json','.jsonl','.log','.txt','.out','.err','.csv','.tsv','.md','.yaml','.yml',
        '.py','.sh','.sha256','.html','.toml','.ini','.conf'}

def identity(path):
 s=path.lstat()
 return dict(path=str(path),bytes=s.st_size,allocated_bytes=s.st_blocks*512,device=s.st_dev,
             inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,uid=s.st_uid,
             regular=stat.S_ISREG(s.st_mode))

def checkpoint_context(path):
 rel=path.relative_to(ROOT)
 if any(x in {'trajectory','trajectories','rollouts','sampling','features','feature','candidate-pools','mixed-candidate-pools','evaluations','eval','final-eval'}for x in rel.parts[1:-1]):return False
 if any(x in path.name.lower()for x in ('trajectory','rollout','feature','candidate','prediction')):return False
 parts=rel.parts[1:-1]
 explicit=any(x in {'checkpoint','checkpoints','training_state','full-state','resume-state','checkpoint-archives','rolling-checkpoints'}for x in parts)
 dynrpo=(rel.parts[0].startswith('dynrpo-prefix5-thread5-20261017')and len(parts)>=3 and parts[:2]==('actor','outers')and re.fullmatch(r'outer-[0-9]+(?:-attempt-[a-f0-9]+)?',parts[2])is not None)
 dynrpo_probe=(rel.parts[0].startswith('dynrpo-prefix5-thread5-20261017')and path.suffix=='.pt'and (
  (parts==('probe',)and (path.name=='latest.pt'or re.fullmatch(r'head-outer-[0-9]+\.pt',path.name)))or
  (len(parts)==2 and parts[0]=='outers'and re.fullmatch(r'outer-[0-9]+',parts[1])and path.name=='probe-state-at-completion.pt')or
  (len(parts)==3 and parts[0]=='outers'and re.fullmatch(r'outer-[0-9]+',parts[1])and parts[2]=='probe-update'and re.fullmatch(r'epoch-[0-9]+\.pt',path.name))or
  (parts==('preparation','probe-audit-pins')and re.fullmatch(r'outer-[0-9]+-[a-f0-9]+\.pt',path.name))or
  (parts==('reference-assets',)and path.name=='warmup-epoch5-ema.pt')))
 cosearch_final=(rel.parts[0]=='cosearch-continue300-x203-20261008'and len(parts)==4 and parts[:3]==('runtime','cosearch-final200-four-gpu-20261015','models')and parts[3]in('main','reranker')and re.fullmatch(r'model-[0-9]+-of-[0-9]+\.safetensors',path.name)is not None)
 return explicit or dynrpo or dynrpo_probe or cosearch_final


def main():
 assert os.getuid() in (14090,21930)
 result=dict(epoch=time.time(),uid=os.getuid(),scope=list(NAMES),read_only=True,
             files=[],preserved_text=[],symlinks=[],errors=[],pruned=[],active_references=[])
 q=subprocess.run(['squeue','-h','-u','jxu957,mou176','-o','%i|%u|%T|%j'],
                  capture_output=True,text=True,timeout=15)
 assert q.returncode==0,q.stderr
 result['queue']=q.stdout;result['active_jobs']=[]
 for line in q.stdout.splitlines():
  job=line.split('|')[0];refs=set();errors=[]
  for cmd in (['scontrol','show','job','-o',job],['scontrol','write','batch_script',job,'-']):
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
   if r.returncode: errors.append(dict(command=cmd[:3],returncode=r.returncode,stderr=r.stderr[:300]));continue
   refs.update(x.rstrip(';,)]}') for x in re.findall(r'/data/user/[^\s\"\x27<>$]+',r.stdout))
  expanded=set(refs)
  # Inspect only small explicitly referenced configs/entrypoints in RAM. Store paths, never raw scripts or credentials.
  for name in refs:
   p=Path(name)
   try:
    if p.suffix in {'.json','.yaml','.yml','.sh'} and p.is_file() and p.stat().st_size<=1048576:
     text=p.read_text(errors='replace')
     expanded.update(x.rstrip(';,)]}') for x in re.findall(r'/data/user/[^\s\"\x27<>$]+',text))
   except OSError as e:errors.append(dict(path=name,error=str(e)))
  for name in list(expanded):
   p=Path(name)
   try:
    if p.exists()or p.is_symlink():expanded.add(str(p.resolve()))
   except OSError as e:errors.append(dict(path=name,error=str(e)))
  result['active_jobs'].append(dict(job=job,paths=sorted(expanded),errors=errors))
 for name in NAMES:
  base=ROOT/name
  if not base.is_dir(): result['errors'].append(dict(path=str(base),error='missing or inaccessible root'));continue
  def error(e):result['errors'].append(dict(path=e.filename,error=str(e)))
  for d,ds,fs in os.walk(base,followlinks=False,onerror=error):
   p=Path(d);keep=[]
   for n in ds:
    child=p/n
    if child.is_symlink():result['symlinks'].append(dict(path=str(child),target=os.readlink(child)));continue
    if n in PRUNE:result['pruned'].append(str(child));continue
    keep.append(n)
   ds[:]=keep
   for n in fs:
    path=p/n
    try:
     if path.is_symlink():result['symlinks'].append(dict(path=str(path),target=os.readlink(path)));continue
     suffix=path.suffix.lower()
     if suffix not in TENSOR and suffix not in TEXT and not n.endswith(('.tar','.tar.gz','.tgz')):continue
     row=identity(path)
     if not row['regular']:continue
     if suffix in TEXT:result['preserved_text'].append(row);continue
     row['checkpoint_context']=checkpoint_context(path)
     row['experiment']=name
     result['files'].append(row)
    except OSError as e:result['errors'].append(dict(path=str(path),error=str(e)))
 for row in result['files']:
  p=Path(row['path'])
  if not row['checkpoint_context']:continue
  for job in result['active_jobs']:
   for ref in job['paths']:
    rp=Path(ref)
    if p==rp or rp in p.parents or p.parent==rp.parent:
     result['active_references'].append(dict(job=job['job'],path=str(p),reference=ref))
 groups=collections.defaultdict(lambda:dict(payload_files=0,payload_allocated_bytes=0,checkpoint_files=0,checkpoint_allocated_bytes=0))
 for r in result['files']:
  g=groups[r['experiment']];g['payload_files']+=1;g['payload_allocated_bytes']+=r['allocated_bytes']
  if r['checkpoint_context']:g['checkpoint_files']+=1;g['checkpoint_allocated_bytes']+=r['allocated_bytes']
 result['groups']=dict(groups);result['finished_unix']=time.time()
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
