import sys,io
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/2d5183455551145c52d1783a38663e2fcf170ceb')
from reproduction.operations.easy7b_20261021.submit_native4_fastpath_hpc import main
sys.stdin=io.StringIO('{"base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022", "source_base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022", "commit": "2d5183455551145c52d1783a38663e2fcf170ceb", "checkpoint": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022/checkpoints/search-g1-common8/actor/global_step_50", "modes": ["clip-native4"], "suffix": "paired-r1-20261023"}')
main()
