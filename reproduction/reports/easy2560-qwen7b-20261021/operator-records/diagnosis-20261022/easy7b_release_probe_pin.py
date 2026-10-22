# Historical exact operator executed on 2026-10-22; no credentials.
import subprocess,json,time
from pathlib import Path
p=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/rollout-batch-sweep64-cause-r2')
a=subprocess.run(['scontrol','show','job','749530'],capture_output=True,text=True,timeout=10)
r={'epoch':time.time(),'before':a.stdout,'reason':'Known-tested nodes6 and55 are full8/8; allow scheduler choice within same single-GPU diagnostic, exclude failed13; no duplicate submission.'}
if 'JobState=PENDING' in a.stdout:
 cmd=['scontrol','update','JobId=749530','ReqNodeList=','ExcNodeList=ACD1-13'];v=subprocess.run(cmd,capture_output=True,text=True,timeout=10);r.update(command=cmd,returncode=v.returncode,stdout=v.stdout,stderr=v.stderr)
r['after']=subprocess.run(['scontrol','show','job','749530'],capture_output=True,text=True,timeout=10).stdout
(p/'scheduling-update.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
