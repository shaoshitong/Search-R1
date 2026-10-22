import json,subprocess,time
code=r'''
from pathlib import Path
import json,subprocess,time
roots=sorted(Path('/tmp').glob('e7bc-750398-*'))
result=dict(epoch=time.time(),roots=[str(p) for p in roots],logs=[])
for root in roots[-1:]:
 logs=root/'session_latest/logs'
 paths=list(logs.glob('*.err'))+list(logs.glob('*worker*.out'))+list(logs.glob('raylet.out'))+list(logs.glob('runtime_env_agent.log'))
 for p in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True)[:12]:
  if p.stat().st_size:
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));text=f.read().decode(errors='replace')
   result['logs'].append(dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=text))
print(json.dumps(result))
'''
argv=['srun','--jobid=750398','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--mem=256M','--gres=none','env','CUDA_VISIBLE_DEVICES=','/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023/env/bin/python','-B','-c',code]
r=subprocess.run(argv,capture_output=True,text=True,timeout=35)
print(json.dumps(dict(epoch=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
