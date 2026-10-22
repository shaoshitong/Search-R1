import json,subprocess,time
r={'epoch':time.time(),'job':'750029','mutation':False}
for name,cmd in {'queue':['squeue','-h','-j','750029,745198','-o','%i|%u|%T|%N|%R'], 'allocation':['scontrol','show','job','750029','-o'], 'predicted_start':['squeue','--start','-h','-j','750029','-o','%i|%T|%S|%N|%R']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15);r[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(r),flush=True)
