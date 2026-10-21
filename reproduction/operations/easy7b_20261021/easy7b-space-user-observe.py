import os,time,json
from pathlib import Path
root=Path('/tmp/easy7b-space-user-request-20261021-1902-'+str(os.getuid()));d=dict(epoch=time.time(),uid=os.getuid(),root=str(root))
for n in ['claim.json','result.json']:
 p=root/n
 if p.exists():d[n]=json.loads(p.read_text())
p=root/'usage.tsv'
if p.exists():
 rows=[]
 for line in p.read_text().splitlines():
  try:a,b=line.split('\t',1);rows.append(dict(bytes=int(a),path=b))
  except ValueError:pass
 d['rows']=rows;d['output_bytes']=p.stat().st_size;d['output_mtime']=p.stat().st_mtime
p=root/'errors.txt'
if p.exists():d['errors']=dict(bytes=p.stat().st_size,lines=p.read_text().splitlines())
print(json.dumps(d))
