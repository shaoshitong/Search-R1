import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v1/window2018')
x=dict(time=time.time(),metadata_only=True,job_id='750023')
for name in('training.sbatch','train-750023.log','ready.json','root-submission.json'):
 p=b/name
 if p.exists():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-16000));x[name]=f.read().decode(errors='replace')
for p in b.glob('failure-rank-*.json'):x[p.name]=json.loads(p.read_bytes())
print(json.dumps(x))
