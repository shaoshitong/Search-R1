import json,time,hashlib,re
from pathlib import Path
ps=[Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011/logs/train-r2-3da9045.log'),Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011/logs/isolated-resume-bc666df-a177507c.log'),Path('/home/shaoshitong/rbb5-context2-ray/ray/session_latest/logs/worker-11344cfc20371f65c7cab4989b78e523fe63b02ac0ee8493b64d8895-01000000-1111494.out')]
rows=[]
for p in ps:
 size=p.stat().st_size;h=hashlib.sha256();rem=size
 with p.open('rb') as f:
  while rem:
   b=f.read(min(8*1024*1024,rem));h.update(b);rem-=len(b)
 rows.append({'path':str(p),'hashed_prefix_bytes':size,'sha256':h.hexdigest(),'size_after':p.stat().st_size})
print(json.dumps({'at':time.time(),'sources':rows}))
