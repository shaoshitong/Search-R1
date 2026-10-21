import os,time,json
from pathlib import Path
c=Path('/tmp/easy7b-storage-current-20261021-21930');d=dict(epoch=time.time(),uid=os.getuid())
for p in c.glob('result-*.json'):d[str(p)]=json.loads(p.read_text())
f=c/'claim-21930.json';x=json.loads(f.read_text());q=Path('/proc')/str(x['pid']);d['controller']=x
if q.exists():
 s=(q/'stat').read_text().rsplit(')',1)[1].split();d['controller'].update(state=s[0],same_birth=s[19]==x['birth'])
 kids=(q/'task'/str(x['pid'])/'children').read_text().split();d['children']=[]
 for pid in kids:
  a=Path('/proc')/pid
  try:
   s=(a/'stat').read_text().rsplit(')',1)[1].split();fds=[]
   for z in (a/'fd').iterdir():
    try:fds.append(os.readlink(z))
    except OSError:pass
   d['children'].append(dict(pid=pid,state=s[0],start_ticks=s[19],wchan=(a/'wchan').read_text(),fds=fds[-12:]))
  except OSError as e:d['children'].append(dict(pid=pid,error=str(e)))
print(json.dumps(d))
