import json,subprocess,time
q=subprocess.run(['squeue','-h','-j','749598,745198','-o','%i|%u|%T|%N|%R'],capture_output=True,text=True,timeout=12)
a=subprocess.run(['sacct','-n','-X','-j','749598','--format=JobIDRaw,User,State,ExitCode,Elapsed,NodeList','-P'],capture_output=True,text=True,timeout=12)
print(json.dumps({'epoch':time.time(),'queue':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'accounting':{'returncode':a.returncode,'stdout':a.stdout,'stderr':a.stderr},'mutation':False}),flush=True)
