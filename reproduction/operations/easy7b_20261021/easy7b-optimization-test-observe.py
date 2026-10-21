import json,time,os
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/optimizations');d=dict(epoch=time.time())
for r in b.glob('selected-head-*'):
 for p in r.glob('*test*'):
  if p.is_file()and p.stat().st_size<80000:d[str(p)]=p.read_text()[-6500:]
print(json.dumps(d))
