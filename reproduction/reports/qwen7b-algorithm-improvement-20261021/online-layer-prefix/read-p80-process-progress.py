import subprocess,json,time
cmd=['sstat','-j','749079.batch,749079.0','--noheader','--parsable2','--format=JobID,AveCPU,MaxRSS,AveRSS,MaxDiskRead,MaxDiskWrite']
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0,r.stderr
print(json.dumps(dict(time=time.time(),job='749079',sstat=r.stdout.strip())))
