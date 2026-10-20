from pathlib import Path
import json
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
v=json.loads((A/'peer8192-frozen-calibration-scores-v2/CPU-compatible-inputs.json').read_text());e=json.loads((A/'E2-same-HF-baseline-20261021/config.json').read_text());c=json.loads((A/'peer8192-frozen-calibration-scores-v2/config.json').read_text())
print(json.dumps(dict(compatible_keys=list(v),sources=[dict(name=x.get('name'),keys=list(x)) for x in v.get('sources',[])],e2_keys=list(e),checkpoints=[dict(name=x['name'],sha256=x['sha256'],keys=list(x)) for x in c['checkpoints']])))
