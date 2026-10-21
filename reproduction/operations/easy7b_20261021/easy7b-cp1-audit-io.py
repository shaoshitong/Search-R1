import json,subprocess,time
from pathlib import Path
code='''import json,time,os
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  args=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if 'runpy.run_module("search_g1.easy2560_restore"' not in args or 'srun' in args[:100] or int(p.name)==os.getpid():continue
  status={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in (p/'status').read_text().splitlines()if ':'in x}
  io={x.split(':',1)[0]:int(x.split(':',1)[1]) for x in (p/'io').read_text().splitlines()}
  rows.append(dict(pid=int(p.name),state=status.get('State'),rss=status.get('VmRSS'),io=io,wchan=(p/'wchan').read_text().strip()))
 except (OSError,ValueError):pass
print(json.dumps(dict(epoch=time.time(),audit_processes=rows)))'''
s=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021');cmd=['srun','--jobid=745198','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=none','env','CUDA_VISIBLE_DEVICES=',str(s/'env/bin/python'),'-B','-c',code]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
