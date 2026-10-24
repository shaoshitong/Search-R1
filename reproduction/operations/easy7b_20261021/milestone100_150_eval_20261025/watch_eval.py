import json,subprocess,time
from pathlib import Path
r=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-milestone100-150-native2-eval-20261025');v={'epoch':time.time(),'root':str(r),'steps':{}}
for n in ['job.id','submit-receipt.json']:
 p=r/n
 if p.is_file():v[n]=json.loads(p.read_text()) if n.endswith('.json') else p.read_text().strip()
job=v.get('job.id')
if job:
 v['queue']=subprocess.run(['squeue','-j',job,'-h','-o','%i|%T|%b|%R'],capture_output=True,text=True).stdout
 v['accounting']=subprocess.run(['sacct','-j',job,'-n','-P','-o','JobID,State,ExitCode,Elapsed,NodeList'],capture_output=True,text=True).stdout
 p=r/'logs'/('eval-'+job+'.log')
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-20000));t=f.read().decode(errors='replace')
  v['log_tail']=t[-4500:];v['errors']=[s for s in t.splitlines() if any(x in s for x in ['Traceback (most recent call last)','RuntimeError:','ValueError:','CUDA out of memory','ModuleNotFoundError:'])][-12:]
for step in [100,150]:
 x={};d=r/('eval-'+str(step))
 for n in ['progress.json','result.json','failure.json']:
  p=d/n
  if p.is_file():
   try:x[n]=json.loads(p.read_text())
   except json.JSONDecodeError:x[n]={'partial':True}
 p=d/'workers.json'
 if p.is_file():x['workers']=[{k:z.get(k) for k in ['rank','hostname','gpu_name','cuda_visible_devices','checkpoint','actor_model_loaded','optimizer_loaded']} for z in json.loads(p.read_text())]
 x['raw_batch_count']=len(list((d/'raw').glob('batch-*.pt'))) if (d/'raw').is_dir() else 0
 v['steps'][step]=x
print(json.dumps(v))
