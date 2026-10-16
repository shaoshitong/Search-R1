import json,time,re,hashlib,subprocess
from pathlib import Path
root=Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
out=root/'output-speed-resume3-2e3a9029051d4771bc61c95b28b60e21'
ms=sorted((out/'checkpoints/training_state').glob('global_step_*/manifest.json'),key=lambda p:int(p.parent.name.split('_')[-1]))
p=ms[-1];raw=p.read_bytes();m=json.loads(raw);cur=m['driver']['next_step']
v={'at':time.time(),'checkpoint':{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'status':m['status'],'completed':m['identity']['completed_step'],'next_step':cur,'workers':[{'rank':w['identity']['rank'],'adam':w['actual_total_optimizer_calls'],'scheduler':w['scheduler_last_epoch'],'finite':w['full_worker_state_finite']} for w in m['worker_states']]}}
log=Path('/home/shaoshitong/rbb5-context2-ray/ray/session_latest/logs/worker-11344cfc20371f65c7cab4989b78e523fe63b02ac0ee8493b64d8895-01000000-1111494.out')
with log.open('rb') as f:
 f.seek(max(0,log.stat().st_size-20000000));lines=f.read().decode(errors='replace').splitlines()
v['log']={'path':str(log),'bytes':log.stat().st_size,'mtime':log.stat().st_mtime}
tokens=0;calls=0;stages=[]
for l in lines:
 if 'TREE_ACTION_TIMING=' in l:
  try:
   z=json.loads(l.split('TREE_ACTION_TIMING=',1)[1])
   if z.get('outer')==cur and z.get('successful'):
    calls+=int(z.get('cost_delta',{}).get('generation_rpc_calls',0));tokens+=int(z.get('cost_delta',{}).get('generated_tokens',0))
  except ValueError: pass
 if 'TREE_DRIVER_STAGE=' in l:
  try:
   z=json.loads(l.split('TREE_DRIVER_STAGE=',1)[1]);stages.append(z)
  except ValueError:pass
v['phase']={'outer':cur,'generated_tokens':tokens,'generation_calls':calls,'stages':stages[-6:]}
v['errors']=[l[:600] for l in lines[-1500:] if any(x in l for x in ['Traceback','CUDA out of memory','RayTaskError'])]
v['processes']=[]
for pid in [1109195,1111494,3807560]:
 pp=Path('/proc')/str(pid);v['processes'].append({'pid':pid,'live':pp.exists(),'command':(pp/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if pp.exists() else None})
v['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(v))
