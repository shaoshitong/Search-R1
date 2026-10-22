import sys,json,time
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022')
p=root/'diagnostics/native4-legacy-r1/submit-claim.json';v=json.loads(p.read_text());print(json.dumps({'checkpoint':v['checkpoint'],'source_command':v['command']}),flush=True)
