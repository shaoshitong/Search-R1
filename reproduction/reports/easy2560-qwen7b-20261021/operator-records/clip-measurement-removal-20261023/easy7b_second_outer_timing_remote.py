import json,time,hashlib
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r2-20261023')
value={'epoch':time.time(),'base':str(base),'tensor_files_read':False,'production_modified':False,'before_generation_files':[],'sealed':{}}
for p in (base/'artifacts/mixed-candidate-pools').glob('*/0-before-generation.pt'):
 progress=p.parent/'progress.json';row={'file':str(p),'mtime':p.stat().st_mtime,'bytes':p.stat().st_size}
 if progress.is_file():row['pool_step']=json.loads(progress.read_text()).get('step')
 value['before_generation_files'].append(row)
for step in (55,56):
 root=base/'preparation/rolling-checkpoints'/('global_step_'+str(step));p=root/'full-manifest.json'
 if p.is_file():value['sealed'][str(step)]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'manifest':json.loads(p.read_text())}
print(json.dumps(value),flush=True)
