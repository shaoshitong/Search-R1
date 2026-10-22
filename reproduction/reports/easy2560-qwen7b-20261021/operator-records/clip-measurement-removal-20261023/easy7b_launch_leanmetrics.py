import json,sys,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023')
a=base/'preparation/leanmetrics-qualified-authorization.json'
authorization=json.loads(a.read_text())
sys.path.insert(0,str(Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source')/authorization['execution_operator_commit']))
from reproduction.operations.easy7b_20261021.launch_lean_transition_on_retrieval import launch
from reproduction.operations.easy7b_20261021.fa3_lean_transition import source_binding
with source_binding(source_base=authorization['source_base'],source_job=authorization['source_job'],source_commit=authorization['source_commit'],authorization=authorization,minimum_outer=60) as bound:
 terminal=bound._terminal_source()
result=launch(str(base),authorization['target_runtime_commit'],str(a),allocation_job='745198',source_base=authorization['source_base'],source_job=authorization['source_job'],source_commit=authorization['source_commit'],minimum_outer=60)
print(json.dumps({'epoch':time.time(),'source_terminal':terminal,'result':result}),flush=True)
