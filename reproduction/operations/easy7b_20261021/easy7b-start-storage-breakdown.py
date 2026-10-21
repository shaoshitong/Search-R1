import subprocess,os,time,json,shutil
from pathlib import Path
c=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/control/storage-breakdown');c.mkdir(parents=True,exist_ok=True)
claim=c/'claim.json'
if claim.exists():print(json.dumps(dict(existing=json.loads(claim.read_text()),restarted=False)))
else:
 command=['du','-x','-B1','--max-depth=1','/data/user/jxu957/experiments']
 if shutil.which('ionice'):command=['ionice','-c3']+command
 if shutil.which('nice'):command=['nice','-n','15']+command
 code="""import subprocess,time,json
from pathlib import Path
c=Path(CONTROL)
with (c/'usage.tsv').open('x') as out,(c/'errors.txt').open('x') as err:
 r=subprocess.run(COMMAND,stdout=out,stderr=err)
(c/'result.json').write_text(json.dumps(dict(epoch=time.time(),rc=r.returncode)))
""".replace('CONTROL',repr(str(c))).replace('COMMAND',repr(command))
 with (c/'controller.out').open('x') as out,(c/'controller.err').open('x') as err:
  p=subprocess.Popen(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 birth=Path('/proc/%s/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]
 value=dict(pid=p.pid,birth=birth,uid=os.getuid(),epoch=time.time(),command=command,purpose='per-directory distribution; prior root-only scans completed')
 with claim.open('x') as f:json.dump(value,f)
 print(json.dumps(value))
