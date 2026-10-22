import json,subprocess,time
r={'epoch':time.time()}
for name,cmd in {'steps':['squeue','--steps','-h','-j','745198','-o','%i|%T|%N'], 'io':['sstat','-a','-n','-j','745198','--format=JobID,AveCPU,MaxRSS,MaxDiskRead,MaxDiskWrite','-P']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=20);r[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(r),flush=True)
