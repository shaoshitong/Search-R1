import json,subprocess,time
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/node13-health-after749812-r1-20261023')
job=(root/'job.id').read_text().strip();r={'epoch':time.time(),'root':str(root),'job':job,'formal_job750029_modified':False,'ranks':{}}
for name,cmd in {'queue':['squeue','-h','-j',job,'-o','%i|%u|%T|%N|%R'], 'allocation':['scontrol','show','job',job,'-o'], 'accounting':['sacct','-n','-X','-j',job,'--format=JobIDRaw,User,State,ExitCode,Elapsed,NodeList','-P']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=12);r[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for rank in range(4):
 p=root/('rank-%d.json'%rank)
 if p.exists():r['ranks'][str(rank)]=json.loads(p.read_text())
p=root/'slurm.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-10000));r['log_tail']=f.read().decode(errors='replace')
print(json.dumps(r),flush=True)
