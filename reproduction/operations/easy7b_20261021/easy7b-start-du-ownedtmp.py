import subprocess,os,time,json
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');c=Path('/tmp/easy7b-storage-current-20261021-'+str(os.getuid()));c.mkdir(parents=True,exist_ok=True)
uid=os.getuid();claim=c/('claim-'+str(uid)+'.json')
if claim.exists():
 print(json.dumps(dict(status='EXISTING_SCAN_NO_RESTART',claim=json.loads(claim.read_text()))))
else:
 out=c/('du-'+str(uid)+'.out');err=c/('du-'+str(uid)+'.err');code="""import subprocess,json,time,os
from pathlib import Path
c=Path(CONTROL);uid=os.getuid();r=subprocess.run(['du','-sx','-B1','/data/user/jxu957/experiments'],capture_output=True,text=True)
(c/('result-'+str(uid)+'.json')).write_text(json.dumps(dict(epoch=time.time(),uid=uid,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)))
""".replace('CONTROL',repr(str(c)))
 with out.open('x') as f,err.open('x') as e:
  p=subprocess.Popen(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code],stdin=subprocess.DEVNULL,stdout=f,stderr=e,start_new_session=True)
 birth=Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]
 value=dict(pid=p.pid,uid=uid,birth=birth,epoch=time.time(),root='/data/user/jxu957/experiments')
 with claim.open('x') as f:json.dump(value,f)
 print(json.dumps(value))
