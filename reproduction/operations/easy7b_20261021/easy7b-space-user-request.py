import os,json,time,subprocess,shutil
from pathlib import Path
root=Path('/tmp/easy7b-space-user-request-20261021-1902-'+str(os.getuid()));root.mkdir(exist_ok=True);claim=root/'claim.json'
if claim.exists():print(json.dumps(dict(existing=json.loads(claim.read_text()),restarted=False,root=str(root))));raise SystemExit(0)
cmd=['du','-x','-B1','--max-depth=2','/data/user/jxu957/experiments']
if shutil.which('ionice'):cmd=['ionice','-c3']+cmd
cmd=['nice','-n','15']+cmd
code='import subprocess,json,time;from pathlib import Path;root=Path('+repr(str(root))+');start=time.time();o=(root/"usage.tsv").open("x");e=(root/"errors.txt").open("x");r=subprocess.run('+repr(cmd)+',stdout=o,stderr=e);o.close();e.close();(root/"result.json").write_text(json.dumps(dict(started_unix=start,completed_unix=time.time(),returncode=r.returncode,uid='+str(os.getuid())+'))+"\\n")'
with(root/'controller.log').open('xb')as out:p=subprocess.Popen(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code],stdout=out,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
v=dict(uid=os.getuid(),pid=p.pid,epoch=time.time(),command=cmd,scope='Read-only depth2 allocated storage inventory requested by user; NO deletion')
claim.write_text(json.dumps(v)+'\n');print(json.dumps(dict(launched=v,root=str(root))))
