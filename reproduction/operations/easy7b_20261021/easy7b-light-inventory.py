import os,json,time,subprocess
from pathlib import Path
r=Path('/data/user/jxu957/experiments');d=dict(epoch=time.time(),existing=[str(x)for x in r.glob('*easy*7b*')])
for b in r.glob('*qprobe*'):
 try:
  cps=[str(p)for p in (b/'checkpoints/search-g1-common8/actor').glob('global_step_*')]
  if cps:d.setdefault('old_qprobe_checkpoints',[]).append(dict(base=str(b),cps=cps,job=(b/'job.id').read_text().strip()if(b/'job.id').is_file()else None))
 except OSError:continue
for tag,cmd in [('nodes',['sinfo','-p','emergency_acd','-h','-o','%N|%T|%G']),('queue',['squeue','-u','jxu957,mou176','-h','-o','%i|%j|%T|%R'])]:
 try:
  x=subprocess.run(cmd,capture_output=True,text=True,timeout=8);d[tag]=dict(rc=x.returncode,stdout=x.stdout,stderr=x.stderr)
 except subprocess.TimeoutExpired:d[tag]='observation timed out'
print(json.dumps(d))
