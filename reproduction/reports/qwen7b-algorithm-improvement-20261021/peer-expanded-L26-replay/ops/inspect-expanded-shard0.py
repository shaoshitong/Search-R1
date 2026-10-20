"""Owned read-only CPU step in existing allocation; no GPU or new writer."""
import json,subprocess
program=r'''import os,pathlib,json
out=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'peer_expanded_l26_replay' not in cmd or 'program=r' in cmd:continue
  status=(p/'status').read_text();io=(p/'io').read_text();files=[]
  for f in (p/'fd').iterdir():
   try:
    target=os.readlink(str(f))
    if '/data/user/' in target:files.append(target)
   except OSError:pass
  out.append(dict(pid=int(p.name),cmdline=cmd[:1600],status=[x for x in status.splitlines() if x.startswith(('State:','VmRSS:','VmHWM:','Threads:'))],io=io,open_shared_files=files[:10],wchan=(p/'wchan').read_text()))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(out))'''
result={}
for name,args in [('steps',['scontrol','show','step','743486']),('sstat',['sstat','-a','-j','743486','-n','-P','--format=JobID,AveRSS,MaxRSS,AveCPU']),('owned_process',['srun','--jobid=743486','--overlap','--ntasks=1','--cpus-per-task=1','--gres=gpu:0','--immediate=10','python3','-c',program])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 result[name]=dict(returncode=r.returncode,stdout=r.stdout[-6000:],stderr=r.stderr[-1500:])
print(json.dumps(result))
