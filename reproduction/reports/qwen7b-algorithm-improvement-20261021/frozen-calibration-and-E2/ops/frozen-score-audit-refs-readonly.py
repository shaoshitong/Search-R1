from pathlib import Path
import json,hashlib
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
refs={}
for name,path in [('G10_independent_audit',A/'peer8192-frozen-calibration-scores-v2/independent-score-audit.json'),('E2_independent_audit',A/'E2-same-HF-baseline-20261021/independent-audit.json')]:
 data=path.read_bytes();refs[name]=dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(refs))
