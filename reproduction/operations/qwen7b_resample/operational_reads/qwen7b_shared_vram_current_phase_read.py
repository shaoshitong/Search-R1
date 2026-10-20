from pathlib import Path
import time,json
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');rows=[]
for p in(b/'sampling').glob('shard-*/batches/batch-*/outers/outer-000001/prefix/rank-0-complete.json'):
 if p.stat().st_mtime<1792519363:continue
 root=p.parent.parent;files=[]
 for phase in ['prefix','full']:
  for q in(root/phase).glob('rank-0-*'):
   if q.is_file():s=q.stat();files.append(dict(path=str(q),bytes=s.st_size,mtime=s.st_mtime))
 rows.append(dict(root=str(root),prefix_complete=json.loads(p.read_text()),files=files))
print(json.dumps(dict(epoch=time.time(),recent_batches=rows)))
