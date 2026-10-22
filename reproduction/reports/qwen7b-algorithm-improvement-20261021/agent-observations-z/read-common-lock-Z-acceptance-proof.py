from pathlib import Path
import json,hashlib,time
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-full169028-variance-CPU-acceptance-20261022-v1')
out=dict(time=time.time(),files={})
for name in('common-lock-final-Z-receipt.json','z_mse_anchor/accepted.json','run-complete.json'):
 p=b/name
 if p.is_file():
  raw=p.read_bytes();out['files'][name]=dict(proof=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),value=json.loads(raw))
print(json.dumps(out))
