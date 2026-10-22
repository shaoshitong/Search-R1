import pathlib,json,time,subprocess
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');b=S/'layer18-prefix80';out={'time':time.time(),'job_id':json.loads((b/'root-submission.json').read_bytes())['job_id'],'rank_progress':[json.loads((b/('progress-rank-%d.json'%k)).read_bytes())for k in(0,1)]}
p=b/'curve.jsonl'
with p.open('rb')as f:f.seek(max(0,p.stat().st_size-24000));data=f.read().decode()
rows=[]
for l in data.splitlines():
 try:
  d=json.loads(l)
  if d.get('epoch')==3 and d.get('cursor',0)>128:rows.append(d)
 except ValueError:pass
out['recent_E3_real_updates']=rows[-8:];out['failure_files']=[p.name for p in b.glob('failure-*.json')];out['eval2_seconds']=json.loads((b/'evaluation-epoch-02.json').read_bytes()).get('seconds');print(json.dumps(out))
