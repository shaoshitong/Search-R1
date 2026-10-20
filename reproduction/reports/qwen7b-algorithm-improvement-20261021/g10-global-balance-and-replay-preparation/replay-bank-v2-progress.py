import pathlib,json,time
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-v2-20261021')
out=dict(time=time.time())
for rel in ('prepared/progress.json','cpu-result.json','prepared/preparation.json'):
 p=root/rel
 if p.exists():
  value=json.loads(p.read_text())
  if rel=='prepared/preparation.json':value={k:value.get(k) for k in ('status','identity','per_shard_questions','per_shard_trajectories','jobs_submitted')}
  out[rel]=value
controller=json.loads((root/'cpu-controller.json').read_text())
pid=controller.get('child_pid')
if pid and pathlib.Path('/proc/%d/stat'%pid).exists():
 stat=pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
 assert stat[19]==controller['child_start_ticks']
 out['same_child_process']=True
 out['child_cpu_ticks']=int(stat[11])+int(stat[12])
 out['child_RSS_pages']=int(stat[21])
 out['child_io']={k.strip():int(v) for k,v in (line.split(':',1) for line in pathlib.Path('/proc/%d/io'%pid).read_text().splitlines())}
else:out['same_child_process']=False
print(json.dumps(out))
