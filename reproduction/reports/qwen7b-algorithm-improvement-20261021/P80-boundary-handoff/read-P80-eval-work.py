import pathlib,json,time
s=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={}
for layer in(15,18):
 b=s/('layer%d-prefix80'%layer);v={}
 for epoch in(0,1,2):
  p=b/('evaluation-epoch-%02d.json'%epoch)
  if p.exists():
   x=json.loads(p.read_bytes());v[p.name]={k:x[k]for k in('epoch','cursor','seconds','layer','prefix')if k in x}
 for role in('fixed_train2048','fixed_validation2048'):
  p=b/('epoch-02-%s-predictions.jsonl.gz'%role)
  v[p.name]=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime)if p.exists()else None
 out[str(layer)]=v
print(json.dumps(dict(time=time.time(),experiments=out,checkpoint_contents_opened=False,final4096_read=False,remote_mutation=False)))
