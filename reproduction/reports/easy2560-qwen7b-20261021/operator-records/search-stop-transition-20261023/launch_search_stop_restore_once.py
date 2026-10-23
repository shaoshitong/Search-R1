import json,sys
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-searchstop-r1-20261023/source/06c5f91772d03358c703dbfc0c2ed935c1784153')
from reproduction.operations.easy7b_20261021.launch_lean_transition_on_retrieval import launch
value=launch('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-searchstop-r1-20261023','06c5f91772d03358c703dbfc0c2ed935c1784153','/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-searchstop-r1-20261023/preparation/fastpath-transition-qualified-authorization.json',source_base='/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023',source_job='750029',source_commit='2d5183455551145c52d1783a38663e2fcf170ceb',minimum_outer=69,infra_profile='fa3-lean-search-stop-v1')
print(json.dumps(value,sort_keys=True))
