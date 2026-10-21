import os,time,json,subprocess
from pathlib import Path
out=dict(epoch=time.time(),uid=os.getuid(),groups=os.getgroups())
for name in ['qprobe-teacher-warmup200-sigma3-20261015','qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017']:
 p=Path('/data/user/jxu957/experiments')/name/'checkpoints/search-g1-common8/actor'
 try:out[name]=dict(stat=list(p.stat()[:]),children=[dict(name=x.name,symlink=x.is_symlink(),mode=oct(x.stat().st_mode),files=os.listdir(x))for x in p.iterdir()])
 except OSError as e:out[name]=repr(e)
print(json.dumps(out))
