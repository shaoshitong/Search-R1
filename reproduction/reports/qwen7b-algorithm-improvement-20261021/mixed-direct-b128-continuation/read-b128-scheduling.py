import subprocess,json,time
out={'epoch':time.time()}
for name,args in [('queue',['squeue','-h','-j','746263','-o','%i|%T|%R|%S|%Q']),('job',['scontrol','show','job','-o','746263'])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);out[name]=r.stdout.strip()
print(json.dumps(out))
