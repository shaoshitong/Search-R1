import subprocess,json,time
script="import os,json,pathlib; p=pathlib.Path('/proc/3288134'); print(json.dumps({'pid':3288134,'stat':(p/'stat').read_text(),'io':(p/'io').read_text(),'open_metadata_paths':[os.readlink(x)for x in (p/'fd').iterdir()if x.name.isdigit() and x.name not in ('0','1','2') and ('index.jsonl' in os.readlink(x) or '.json' in os.readlink(x))]}))"
r=subprocess.run(['srun','--overlap','--jobid=747723','--nodes=1','--ntasks=1','--cpus-per-task=1','python3','-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
print(json.dumps({'observed_epoch':time.time(),'returncode':r.returncode,'stdout':r.stdout[-6000:],'stderr':r.stderr[-1000:]}))
