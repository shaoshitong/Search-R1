import json,time,subprocess,os
from pathlib import Path
out=dict(epoch=time.time())
for jid in ['734689','745022']:
 row={}
 for n,cmd in [('queue',['squeue','-j',jid,'-h','-o','%i|%T|%N|%R']),('accounting',['sacct','-j',jid,'-n','-P','-o','JobIDRaw,State,ExitCode,NodeList,Elapsed'])]:
  x=subprocess.run(cmd,capture_output=True,text=True,timeout=8);row[n]=dict(rc=x.returncode,stdout=x.stdout,stderr=x.stderr)
 out[jid]=row
for c in [Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/control/storage-current'),Path('/tmp/easy7b-storage-current-20261021-21930')]:
 for p in c.glob('result-*.json'):out[str(p)]=json.loads(p.read_text())
 for p in c.glob('claim-*.json'):
  x=json.loads(p.read_text());q=Path('/proc')/str(x['pid'])
  if q.exists():
   s=(q/'stat').read_text().rsplit(')',1)[1].split();x.update(actual_state=s[0],same_birth=s[19]==x['birth'],io=(q/'io').read_text() if os.access(q/'io',os.R_OK)else 'not readable')
  else:x['process_exists']=False
  out[str(p)]=x
print(json.dumps(out))
