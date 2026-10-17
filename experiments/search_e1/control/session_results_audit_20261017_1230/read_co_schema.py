import json,time,hashlib
from pathlib import Path
b=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017')
p=sorted(b.glob('artifacts/question-probe/attempts/*/rollout-audit.json'),key=lambda p:p.stat().st_mtime)[-1]
d=json.loads(p.read_text());print(json.dumps(dict(epoch=time.time(),path=str(p),data=d,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
