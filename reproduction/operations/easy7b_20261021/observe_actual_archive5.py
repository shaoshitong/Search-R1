"""HK metadata-only byte progress and actual existing milestone5 receipts."""
import json
import os
from pathlib import Path
import socket
import time

assert os.getuid()==1002 and socket.gethostname()=='xielab3'
root=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
control=root/'archive-controller/job-745371/recovery-transport-r1-20261021'
target=root/'checkpoints/232f20e112ac4a978c5aba52a0367195/global_step_5'
result=dict(epoch=time.time(),read_only=True,target=str(target),target_exists=target.is_dir(),files=0,apparent_bytes=0,allocated_bytes=0)
for name in ['progress.json','failure.json','result.json']:
 p=control/name
 if p.is_file():result['controller/'+name]=json.loads(p.read_text())
if target.is_dir():
 for p in target.rglob('*'):
  if p.is_file()and not p.is_symlink():
   s=p.stat();result['files']+=1;result['apparent_bytes']+=s.st_size;result['allocated_bytes']+=s.st_blocks*512
 for name in ['archive-receipt.json','source-acceptance.json']:
  p=target/name
  if p.is_file():result[name]=json.loads(p.read_text())
 p=target/'source-snapshot.json'
 if p.is_file():
  d=json.loads(p.read_text());result['snapshot']=dict(run_id=d['run_id'],step=d['step'],source_commit=d['source_commit'],files=len(d['files']),total_bytes=d['total_bytes'])
print(json.dumps(result))
