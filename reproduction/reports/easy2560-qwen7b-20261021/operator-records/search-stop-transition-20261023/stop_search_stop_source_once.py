import json,sys
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-searchstop-r1-20261023/source/06c5f91772d03358c703dbfc0c2ed935c1784153')
from reproduction.operations.easy7b_20261021.stop_for_fa3_boundary import stop
value=stop('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-searchstop-r1-20261023','/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-searchstop-r1-20261023/preparation/fastpath-transition-qualified-authorization.json')
print(json.dumps({k:value.get(k) for k in ("status","source_job","returncode","stdout","stderr","source_boundary_observed","actual_source_terminal_not_yet_verified")},sort_keys=True))
