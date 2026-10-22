import json,time,os
from pathlib import Path
raw=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023/artifacts/mixed-candidate-pools/2b9e0ec5be244d33a495d0ceef86c62f-pool-0/0.pt')
service=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021')
ready=json.loads((service/'logs/service-ready.json').read_text())
paths={k:Path(ready[k]['path']) for k in ('index_file','corpus_file')}
paths['raw_pool_file']=raw
paths['encoder']=service/'assets/e5-base-v2'
print(json.dumps(dict(epoch=time.time(),read_only=True,paths={k:dict(path=str(p),exists=p.exists(),bytes=p.stat().st_size if p.exists() and p.is_file() else None) for k,p in paths.items()},service_environment=dict(path=str(service/'env/bin/python'),exists=(service/'env/bin/python').exists()),ready_source_commit=ready['source_commit'],faiss_library=str(service/'retriever-gpu-lib'))))
