import pathlib,json,time,subprocess
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g10-matched6528-v1')
out=dict(epoch=time.time(),variants={})
for name in ('random','global-weighted-balanced'):
 root=b/name;own={}
 for file in ('controller.json','progress.json','latest.json','failure.json','next-update-replay.json','gpu-memory.json'):
  p=root/file
  if p.exists():
   v=json.loads(p.read_text());own[file]={k:v[k] for k in ('status','steps','cursor','time','epoch','pid','source_steps','candidate_steps','error','message','checkpoint','raw_model_bit_identical','EMA_bit_identical','Adam_bit_identical','RNG_bit_identical') if k in v}
 own['logs']=[]
 for p in list(root.glob('*.log'))+list((root/'logs').glob('*.log'))+list((b/'logs').glob('*'+('743226' if name=='random' else '743227')+'*')):
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));tail=f.read().decode(errors='replace')
  own['logs'].append(dict(name=str(p.relative_to(b)),bytes=p.stat().st_size,tail=tail))
 out['variants'][name]=own
print(json.dumps(out))
