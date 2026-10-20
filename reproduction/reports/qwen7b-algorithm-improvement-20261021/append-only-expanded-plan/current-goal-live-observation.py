"""One bounded read-only Slurm/real-progress snapshot; no model payloads."""
import json,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
jobs=[743466,743467,743485,743486,743488,743489,743490]
out=dict(epoch=time.time(),jobs=jobs)
out['unavailable_optional_observations']=[]
def optional_json(p):
 if not p.exists():return None
 for attempt in range(3):
  try:return json.loads(p.read_text())
  except FileNotFoundError:
   if attempt<2:time.sleep(.1)
 out['unavailable_optional_observations'].append(str(p))
 return None
for name,args in [('squeue',['squeue','-h','-j',','.join(map(str,jobs)),'-o','%i|%j|%T|%M|%N|%R']),('sacct',['sacct','-n','-X','-j',','.join(map(str,jobs)),'-o','JobIDRaw,State,ExitCode,Elapsed,NodeList','-P'])]:
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 out[name]=dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr[-600:])
out['runs']={}
for name in ('peer8192-inter1-matched256-v1','peer8192-bothrank1-matched256-v1','g20-exact-uniform15504-dp-v1'):
 b=A/name;v={}
 for n in ('progress.json','latest.json','failure.json','interrupted.json','complete.json'):
  p=b/n
  x=optional_json(p)
  if x is not None:v[n]={k:x[k] for k in ('status','time','cursor','steps','seconds','message','type','sha256','bytes','epoch_time') if k in x}
 out['runs'][name]=v
out['capture']={}
for i in range(4):
 b=A/'peer-expanded-L26-replay-v1'/('shard-%02d'%i)/'capture';v={}
 for n in ('progress.json','complete.json','failure.json'):
  p=b/n
  x=optional_json(p)
  if x is not None:v[n]={k:x[k] for k in ('status','time','trajectories','expected_trajectories','completed_chunks','elapsed_seconds','message','type') if k in x}
 out['capture'][str(i)]=v
print(json.dumps(out,sort_keys=True))
