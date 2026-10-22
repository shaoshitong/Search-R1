import json,time,hashlib
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r2-20261023')
value={'epoch':time.time(),'base':str(base),'tensor_files_read':False,'production_modified':False,'before_generation_files':[],'sealed':{}}
for p in (base/'artifacts/mixed-candidate-pools').glob('*/0-before-generation.pt'):
 progress=p.parent/'progress.json';row={'file':str(p),'mtime':p.stat().st_mtime,'bytes':p.stat().st_size}
 if progress.is_file():row['pool_step']=json.loads(progress.read_text()).get('step')
 value['before_generation_files'].append(row)
for step in (55,56,57,58,59):
 root=base/'preparation/rolling-checkpoints'/('global_step_'+str(step));p=root/'full-manifest.json'
 if p.is_file():value['sealed'][str(step)]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'manifest':json.loads(p.read_text())}
value['unsealed_checkpoint_files']={}
for step in (57,58,59):
 if str(step) in value['sealed']:continue
 root=base/'checkpoints/search-g1-common8/actor'/('global_step_'+str(step))
 if root.is_dir():
  files=[]
  for p in root.rglob('*'):
   if p.is_file():
    st=p.stat();files.append({'path':str(p.relative_to(root)),'bytes':st.st_size,'mtime':st.st_mtime})
  value['unsealed_checkpoint_files'][str(step)]={'files':files,'total_bytes':sum(row['bytes'] for row in files)}
print(json.dumps(value),flush=True)
