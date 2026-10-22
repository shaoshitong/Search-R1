import json,subprocess,sys,time
code="""import os,json,time
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  if p.stat().st_uid!=14090:continue
  args=(p/'cmdline').read_bytes().split(bytes([0]));cmd=' '.join(x.decode(errors='replace') for x in args)
  if b'reproduction.operations.easy7b_20261021.fa3_lean_transition' not in args:continue
  rows.append({'pid':int(p.name),'cmdline':cmd,'io':(p/'io').read_text(),'stat':(p/'stat').read_text(),'status':[(s) for s in (p/'status').read_text().splitlines() if s.startswith(('State:','VmRSS:','VmHWM:','Threads:'))]})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps({'epoch':time.time(),'processes':rows}),flush=True)
"""
cmd=['srun','--jobid=745198','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=none','/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
print(json.dumps({'epoch':time.time(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}),flush=True)
