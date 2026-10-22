import sys,json,time,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r1-20261022')
r={'epoch':time.time(),'records':{}}
for name in ['job.id','preparation/fa3-transition-submit-intent.json','preparation/lean-transition-retrieval-launch-claim.json','preparation/lean-transition-retrieval-launched.json','preparation/lean-transition-retrieval-supervisor.json','preparation/lean-transition-retrieval-srun-process.json','preparation/lean-transition-retrieval-exit.json','preparation/fa3-transition-source-ready.json','preparation/fa3-transition-submitted.json']:
 p=b/name
 if p.is_file():r['records'][name]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text().strip()
job=r['records'].get('job.id');jobs='749255,745198'+(','+job if job else '')
for k,cmd in {'queue':['squeue','-h','-j',jobs,'-o','%i|%u|%T|%N|%R'],'accounting':['sacct','-n','-X','-j',jobs,'-o','JobIDRaw,User,State,ExitCode','-P']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15);r[k]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
p=b/'logs/lean-transition-retrieval-cpu-audit.log'
if p.is_file():r['audit_tail']=p.read_text()[-4000:]
r['cpu_audit_progress']={k:{'bytes':(b/k).stat().st_size,'mtime':(b/k).stat().st_mtime} for k in ['preparation/fa3-full-state-source-receipt.json','logs/lean-transition-retrieval-cpu-audit.log'] if (b/k).is_file()}
print(json.dumps(r),flush=True)
