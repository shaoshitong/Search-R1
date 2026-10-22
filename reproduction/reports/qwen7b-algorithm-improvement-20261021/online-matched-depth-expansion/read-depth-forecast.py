import json,subprocess,time
jobs='749398,749399,749400'
r=subprocess.run(['squeue','--start','-h','-j',jobs,'-o','%i|%j|%P|%T|%S|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
assert r.returncode==0,r.stderr
print(json.dumps(dict(time=time.time(),forecast=r.stdout.strip(),forecast_not_guaranteed=True)))
