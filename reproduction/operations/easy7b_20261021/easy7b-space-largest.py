import os,json,time,subprocess
from pathlib import Path
root=Path('/tmp/easy7b-space-user-request-largest-20261021');root.mkdir(exist_ok=True);claim=root/'claim.json'
if not claim.exists():
 cmd=['nice','-n','15','ionice','-c3','du','-x','-B1','--max-depth=1','/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019']
 code='import subprocess,json,time;from pathlib import Path;r=Path('+repr(str(root))+');start=time.time();o=(r/"usage.tsv").open("x");e=(r/"errors.txt").open("x");p=subprocess.run('+repr(cmd)+',stdout=o,stderr=e);o.close();e.close();(r/"result.json").write_text(json.dumps(dict(started_unix=start,completed_unix=time.time(),returncode=p.returncode))+"\\n")'
 with (root/'controller.log').open('xb')as f:p=subprocess.Popen(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 claim.write_text(json.dumps(dict(epoch=time.time(),pid=p.pid,command=cmd))+'\n')
d=dict(epoch=time.time(),claim=json.loads(claim.read_text()))
for name in ['usage.tsv','errors.txt','result.json']:
 p=root/name
 if p.exists():d[name]=p.read_text()
print(json.dumps(d))
