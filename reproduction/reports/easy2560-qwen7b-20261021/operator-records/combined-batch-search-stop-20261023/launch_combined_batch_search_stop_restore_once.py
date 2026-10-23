import json,sys
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023/source/dde1452d2ec65fb3f4a0464d4beacf3b8136414c')
from reproduction.operations.easy7b_20261021.launch_lean_transition_on_retrieval import launch
value=launch('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023','dde1452d2ec65fb3f4a0464d4beacf3b8136414c','/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023/preparation/fastpath-transition-qualified-authorization.json',source_base='/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023',source_job='750029',source_commit='2d5183455551145c52d1783a38663e2fcf170ceb',minimum_outer=70,infra_profile='fa3-lean-batched-search-stop-v1')
print(json.dumps(value,sort_keys=True))
