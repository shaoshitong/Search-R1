import pathlib,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer21-prefix80');x=dict(time=time.time(),files=[])
for p in sorted(b.glob('epoch-*-predictions.jsonl.gz')):
 s=p.stat();x['files'].append(dict(name=p.name,bytes=s.st_size,mtime=s.st_mtime))
x['completed_evaluation_metadata']=[dict(name=p.name,mtime=p.stat().st_mtime)for p in sorted(b.glob('evaluation-epoch-*.json'))]
x['failure_files']=[p.name for p in b.glob('failure-rank-*.json')]
p=b/'train-749369.log'
if p.exists():
 with p.open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-1000));x['log_tail']=f.read().decode(errors='replace')
print(json.dumps(x))
