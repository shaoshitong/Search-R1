import json,time,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021');job=(b/'job.id').read_text().strip();d=dict(epoch=time.time(),job=job)
for key,cmd in [('queue',['squeue','-j',job,'-h','-o','%i|%T|%N|%R|%S']),('accounting',['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,Elapsed'])]:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=8);d[key]=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
for name in ('status.json','logs/service-ready.json'):
 p=b/name
 if p.is_file():d[name]=json.loads(p.read_text())
p=b/('logs/service-'+job+'.log')
if p.is_file():
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-3500));tail=f.read().decode(errors='replace')
 d['log']=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=tail)
print(json.dumps(d))
