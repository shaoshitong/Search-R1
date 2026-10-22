import json,subprocess,time
job='749369'
def run(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 assert r.returncode==0,r.stderr
 return r.stdout.strip()
print(json.dumps(dict(time=time.time(),job=job,identity=run(['scontrol','show','job','-o',job]),forecast=run(['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R']),forecast_is_not_guarantee=True)))
