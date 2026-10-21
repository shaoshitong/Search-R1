import json,time
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/control/storage-breakdown');p=root/'usage.tsv';rows=[]
for line in p.read_text().splitlines():
 try:a,b=line.split('\t',1);rows.append(dict(bytes=int(a),path=b))
 except ValueError:pass
print(json.dumps(dict(epoch=time.time(),previous_inventory_result=json.loads((root/'result.json').read_text()),rows=rows)))
