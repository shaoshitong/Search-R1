import io,json,runpy,sys
request={'base': '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022', 'commit': 'a025a12b19c29238920b791675acde02d795a756', 'raw_pool_file': '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023/artifacts/mixed-candidate-pools/2b9e0ec5be244d33a495d0ceef86c62f-pool-0/0.pt', 'index': '/data/user/jxu957/experiments/topd-20261008/bundle/retrieval/wiki/e5_Flat.index', 'encoder': '/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021/assets/e5-base-v2'}
sys.stdin=io.StringIO(json.dumps(request))
runpy.run_path(request['base']+'/source/'+request['commit']+'/reproduction/operations/easy7b_20261021/submit_retrieval_batching_probe_hpc.py',run_name='__main__')
