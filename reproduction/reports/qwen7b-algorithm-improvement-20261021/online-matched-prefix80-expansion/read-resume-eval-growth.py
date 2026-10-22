import pathlib,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer18-prefix80');out={'time':time.time(),'files':[]}
for p in b.glob('epoch-02-*'):
 st=p.stat();out['files'].append({'name':p.name,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
for name in('latest.json','complete.json','progress-rank-0.json','progress-rank-1.json'):
 if(b/name).exists():out[name]=json.loads((b/name).read_bytes())
print(json.dumps(out))
