# Historical exact operator executed on 2026-10-22; no credentials.
import subprocess,json,time
from pathlib import Path
p=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/rollout-batch-sweep64-cause-r2');before=subprocess.run(['scontrol','show','job','749530'],capture_output=True,text=True,timeout=10).stdout
r={'epoch':time.time(),'before':before,'reason':'Independent replay needs under15min; reduce2h reservation to allow backfill; no duplicate or new resource request.'}
if 'JobState=PENDING' in before:
 cmd=['scontrol','update','JobId=749530','TimeLimit=00:15:00'];v=subprocess.run(cmd,capture_output=True,text=True,timeout=10);r.update(command=cmd,returncode=v.returncode,stdout=v.stdout,stderr=v.stderr)
r['after']=subprocess.run(['scontrol','show','job','749530'],capture_output=True,text=True,timeout=10).stdout;(p/'backfill-time-limit-update.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
