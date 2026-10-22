import json,time,subprocess
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/search-action-stop-r1-20261023/cpu-replay-r1')
def read(name):
 p=root/name
 return json.loads(p.read_text()) if p.exists() else None
log=root/'replay.log'
tail=''
if log.exists():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000));tail=f.read().decode(errors='replace')
print(json.dumps(dict(epoch=time.time(),launch=read('launch.json'),result=read('result.json'),exit=read('exit.json'),accounting=subprocess.run(['sacct','-n','-P','-j','750432','-o','JobID,State,ExitCode,Elapsed'],capture_output=True,text=True,timeout=10).stdout,log_tail=tail)))
