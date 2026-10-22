"""Read only small acceptance metadata, never checkpoint tensor payloads."""
import json
from pathlib import Path

base = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1')
reports = {}
for layer in (21, 27):
    path = base / ('layer%d-prefix20' % layer) / 'CPU-checkpoint-audit-epoch-03.json'
    if path.exists():
        assert path.stat().st_size < 20000
        reports[str(layer)] = json.loads(path.read_text())
print(json.dumps({'audits': reports, 'final4096_read': False}))
