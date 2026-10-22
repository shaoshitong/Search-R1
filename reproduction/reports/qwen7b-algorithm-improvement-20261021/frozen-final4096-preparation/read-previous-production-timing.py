import pathlib,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
paths=[b/'selection-2048-20261020',b/'heldout2048-20261020',b]
rows=[]
for root in paths:
 for s in range(2):
  p=root/'sampling'/('shard-%02d'%s)/'progress.json'
  if p.exists():
   x=json.loads(p.read_text());rows.append(dict(path=str(p),progress={k:x[k]for k in ('questions','trajectories','completed_batches','total_batches','elapsed','epoch')if k in x}))
print(json.dumps(dict(epoch=time.time(),metadata_only=True,examples=rows)))
