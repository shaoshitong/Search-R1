import sys,io
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/c14bedd9e897e4bed51de9c693a1ef29302b436d')
from reproduction.operations.easy7b_20261021.submit_native4_fastpath_hpc import main
sys.stdin=io.StringIO('{"base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022", "source_base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022", "commit": "c14bedd9e897e4bed51de9c693a1ef29302b436d", "checkpoint": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022/checkpoints/search-g1-common8/actor/global_step_50", "geometry_report": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/native4-leancompact-r1/rank-0.json", "modes": ["clip-geometry"], "suffix": "remaining-measurements-r1-20261023"}')
main()
