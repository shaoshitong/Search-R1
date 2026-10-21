from pathlib import Path
import json,time
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/retriever-gpu-lib/faiss');out=[]
for name in ('gpu_wrappers.py','__init__.py'):
 p=root/name
 if p.is_file():
  s=p.read_text().splitlines()
  for i,z in enumerate(s):
   if z.startswith('def index_cpu_to_gpu_multiple_py')or z.startswith('def index_cpu_to_all_gpus'):
    out.append(dict(file=str(p),line=i+1,text='\n'.join(s[i:i+38])))
print(json.dumps(dict(epoch=time.time(),apis=out)))
