import json,time,urllib.request
url='http://ACD1-4:18026/health'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(url,timeout=12) as response:
 v=json.load(response)
keys=('ok','service_job_id','service_run_id','source_commit','request_rebatching_enabled','singleton_e5','faiss_batch_size','faiss_max_wait_ms','physical_gpu_count','logical_shards','observed_unix','statistics')
print(json.dumps({'observed_unix':time.time(),'health_url':url,'health':{k:v.get(k) for k in keys},'retrieval_queries_sent':0,'gpu_forward_triggered':False}))
