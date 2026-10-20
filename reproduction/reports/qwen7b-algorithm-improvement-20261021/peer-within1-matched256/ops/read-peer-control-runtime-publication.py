from pathlib import Path
import json
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1');v=json.loads((b/'source-publication.json').read_text());assert v['commit']=='d8f74420aa09fa56e8ec59207c6d1453fa86f95e';print(json.dumps(v))
