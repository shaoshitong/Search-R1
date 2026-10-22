import json,sys,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023')
a=base/'preparation/leanmetrics-qualified-authorization.json'
authorization=json.loads(a.read_text())
sys.path.insert(0,str(Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source')/authorization['execution_operator_commit']))
from reproduction.operations.easy7b_20261021.stop_for_fa3_boundary import stop
print(json.dumps({'epoch':time.time(),'result':stop(str(base),str(a))}),flush=True)
