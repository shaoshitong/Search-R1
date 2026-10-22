import sys,io
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/d87d3b54c505b581250b152bccef0f01e9a5f071')
from reproduction.operations.easy7b_20261021.submit_rollback_snapshot_probe_hpc import main
sys.stdin=io.StringIO('{"base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022", "source_base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022", "commit": "d87d3b54c505b581250b152bccef0f01e9a5f071", "checkpoint": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022/checkpoints/search-g1-common8/actor/global_step_50"}')
main()
