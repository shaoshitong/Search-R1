import sys,json
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/063942e9009a023a6e1b179e10d422a56db2c77f')
from reproduction.operations.easy7b_20261021.recover_native_prefix_admission import recover
r=recover('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r1-20261022','/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r2-20261023','c7976da3afc863a1c20b99daa7cc99de3bb61f15')
print(json.dumps(r),flush=True)
