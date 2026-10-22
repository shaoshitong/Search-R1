"""Small descriptor graph only; no trajectory payload/CP/final data."""
import json,hashlib,time,os
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');F=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture'
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p);r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
d=read(F/'full-training-descriptor.json');result=dict(time=time.time(),descriptor=ref(F/'full-training-descriptor.json'),declared_questions=d['questions'],declared_trajectories=d['trajectories'],groups=d['groups'],prefix_sources=len(d['prefix_sources']),prefix_compressed_bytes=sum(x['bytes']for x in d['prefix_sources']),source_metadata=[])
for x in d['source_metadata']:
 assert x['bytes']<3000000 and not any(k in x['path']for k in('final4096','confirmation-materialization'));assert ref(x['path'])==x;y=read(x['path']);result['source_metadata'].append(dict(ref=x,type=type(y).__name__,keys=list(y)if isinstance(y,dict)else['LIST',len(y)],scalar_fields={k:v for k,v in y.items()if isinstance(v,(str,int,float,bool))or v is None}if isinstance(y,dict)else{}))
v=os.statvfs(A);result.update(free_shared_bytes=v.f_bavail*v.f_frsize,large_payloads_opened=False,groups_payload_opened=False,final4096_read=False);print(json.dumps(result))
