"""Read existing small old5 audit receipts, never raw/model/features payload."""
import pathlib,json,hashlib,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
items=[]
for directory in('bank','preparation','audit'):
 d=B/directory
 if d.is_dir():
  for p in sorted(d.glob('*.json')):
   if p.stat().st_size<=512000:
    v=json.loads(p.read_bytes())
    if not isinstance(v,dict):continue
    items.append(dict(ref=proof(p),keys=sorted(v),status=v.get('status'),batches_count=len(v.get('batches',[]))if isinstance(v.get('batches'),list)else v.get('batches'),strict_EM_recomputed=v.get('strict_EM_recomputed'),own_strict_EM_recomputed=v.get('own_strict_EM_recomputed'),train_questions=v.get('train_questions'),train_trajectories=v.get('train_trajectories')))
print(json.dumps(dict(epoch=time.time(),small_existing_receipts=items,no_payload_read=True)))
