import io,json,runpy,sys
request=json.loads("{\"base\":\"/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-batch-wait16-service-r1-20261023\",\"commit\":\"2303dd444e077ac28d65bef95e427ff7de9fcb38\",\"index\":\"/data/user/jxu957/experiments/topd-20261008/bundle/retrieval/wiki/e5_Flat.index\",\"corpus\":\"/data/user/jxu957/experiments/topd-20261008/bundle/retrieval/wiki/wiki-18.jsonl\",\"faiss_max_wait_ms\":16,\"port\":18024}")
sys.stdin=io.StringIO(json.dumps(request))
runpy.run_path(request['base']+'/source/'+request['commit']+'/reproduction/operations/easy7b_20261021/submit_private_retrieval_service_hpc.py',run_name='__main__')
