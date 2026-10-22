import json,time
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/candidate-retrieval-pair-r1-20261023/attempt-r3')
candidate=root/'runs/startup-legacy/candidate'
workers=json.loads((candidate/'workers.json').read_text())
launch=json.loads((candidate/'launch.json').read_text())
result=dict(epoch=time.time(),job='750419',root=str(root),failure=json.loads((candidate/'failure.json').read_text()),runner_failure=json.loads((root/'runs/failure.json').read_text()),nproc=launch['native_nproc'],worker_import_path=launch['import_path_receipt'],actual_worker_count=len(workers),workers=workers,optimizer_writes=0,production_modified=False)
print(json.dumps(result))
