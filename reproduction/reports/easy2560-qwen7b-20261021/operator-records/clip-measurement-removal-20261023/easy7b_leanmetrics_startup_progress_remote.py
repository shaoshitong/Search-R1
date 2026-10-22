import json,subprocess,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023')
r={'epoch':time.time(),'read_only':True,'job':'750029','documents':{},'log_metadata':[]}
for name in ['pipeline-status.json','native-allocation-admission.json','disk-admission.json','process-ledger.json']:
 p=base/'logs'/name
 if p.exists():
  try:r['documents'][name]=json.loads(p.read_text())
  except Exception as e:r['documents'][name]={'read_error':repr(e)}
for p in (base/'logs').glob('*'):
 if p.is_file():r['log_metadata'].append({'name':p.name,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime})
code="""import json,os,time
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  if p.stat().st_uid!=14090:continue
  cg=(p/'cgroup').read_text()
  if 'job_750029' not in cg:continue
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if not cmd:continue
  rows.append({'pid':int(p.name),'cmd':cmd[:300],'stat':(p/'stat').read_text(),'io':(p/'io').read_text(),'wchan':(p/'wchan').read_text()})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps({'epoch':time.time(),'processes':rows}))
"""
p=subprocess.run(['srun','--jobid=750029','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=none','/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code],capture_output=True,text=True,timeout=25)
r['actual_processes']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};print(json.dumps(r),flush=True)
