import json
from pathlib import Path
base=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/env/lib/python3.10/site-packages/vllm')
print(json.dumps({n:(base/n).read_text()[:26000] for n in ['engine/output_processor/stop_checker.py','engine/output_processor/single_step.py']}))
