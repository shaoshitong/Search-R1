import json,pathlib,hashlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
x=json.loads((b/'prepared.json').read_text())
for e in x['arrays']:
 for k in ('entry','config'):
  r=e[k];p=pathlib.Path(r['path']);raw=p.read_bytes();assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
 e['active_same_name_jobs']=subprocess.check_output(['squeue','-h','-u','mou176','-n',e['job_name'],'-o','%i|%T|%j|%N'],universal_newlines=True).strip().splitlines()
 e['actual_sealed_shards']=[i for i in range(4) if (pathlib.Path(e['original_event_root'])/'sampling'/('shard-%02d'%i)/'complete.json').exists()]
print(json.dumps(dict(epoch=time.time(),prepared=x,final_labels_or_model_payload_read=False)))
