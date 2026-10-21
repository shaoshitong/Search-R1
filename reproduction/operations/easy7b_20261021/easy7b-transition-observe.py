import json,time,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');s=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021');d=dict(epoch=time.time())
for root,names in [(b,['preparation/rolling-checkpoints/latest.json','checkpoints/search-g1-common8/actor/global_step_1/complete.json','logs/actual-updates.jsonl','logs/cleanup-status.json']),(s,['diagnostics/stage64-four-vs-two.json','diagnostics/actor-visible-parity/result.json','diagnostics/actor-visible-parity/exit.json'])]:
 for name in names:
  p=root/name
  if p.is_file():
   if name.endswith('jsonl'):value=[json.loads(x)for x in p.read_text().splitlines()if x][-2:]
   else:value=json.loads(p.read_text())
   d[str(p)]=value
cp=b/'checkpoints/search-g1-common8/actor/global_step_1'
d['checkpoint_members']=[dict(file=str(p.relative_to(cp)),bytes=p.stat().st_size,mtime=p.stat().st_mtime)for p in cp.rglob('*')if p.is_file()]if cp.exists()else[]
p=s/'diagnostics/actor-visible-parity/stdout.log'
if p.exists():d['parity_log_tail']=p.read_text()[-1800:]
r=subprocess.run(['squeue','-j','745022,745198','-h','-o','%i|%T|%N|%R'],capture_output=True,text=True,timeout=10);d['queue']=r.stdout
print(json.dumps(d))
