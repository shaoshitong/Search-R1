import json,time,subprocess
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/retrieval-batching-r2-singleton-e5-20261023')
r=dict(epoch=time.time(),root=str(root),read_only=True)
for name in ['submit-receipt.json','result.json']:
 p=root/name
 if p.exists():
  try:r[name]=json.loads(p.read_text())
  except json.JSONDecodeError:r[name]={'partial':True}
job=(root/'job.id').read_text().strip() if (root/'job.id').exists() else None
r['job']=job
if job:
 r['queue']=subprocess.run(['squeue','-h','-j',job,'-o','%i|%T|%N|%R'],capture_output=True,text=True,timeout=10).stdout
 r['accounting']=subprocess.run(['sacct','-n','-X','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList','-P'],capture_output=True,text=True,timeout=10).stdout
 p=root/('slurm-'+job+'.log')
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-10000));r['log_tail']=f.read().decode(errors='replace')
print(json.dumps(r),flush=True)
