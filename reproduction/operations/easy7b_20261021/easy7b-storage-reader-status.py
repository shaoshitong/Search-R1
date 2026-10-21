import json,time,os
from pathlib import Path
out=dict(epoch=time.time(),uid=os.getuid())
for c in [Path('/tmp/easy7b-storage-current-20261021-21930'),Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/control/storage-current')]:
 for p in c.glob('*'):
  try:
   if p.stat().st_size<40000:out[str(p)]=p.read_text()
  except OSError as e:out[str(p)]=str(e)
print(json.dumps(out))
