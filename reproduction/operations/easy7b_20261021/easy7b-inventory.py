import os,json,time,subprocess
from pathlib import Path
m=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70')
d=dict(epoch=time.time(),uid=os.getuid(),model=str(m),files={},existing_easy7b=[])
for n in ['config.json','complete.json','official-inventory.json','model.safetensors.index.json']:
 p=m/n
 if p.is_file():d['files'][n]=json.loads(p.read_text())
d['shards']=[dict(name=p.name,bytes=p.stat().st_size,readable=os.access(p,os.R_OK))for p in m.glob('*.safetensors')]
r=Path('/data/user/jxu957/experiments')
d['existing_easy7b']=[str(p) for p in r.glob('*easy*7b*')]
for n,cmd in [('queue',['squeue','-u','jxu957,mou176','-h','-o','%i|%u|%j|%T|%P|%R|%b']),('quota',['df','-B1',str(r)])]:
 x=subprocess.run(cmd,capture_output=True,text=True);d[n]=dict(rc=x.returncode,stdout=x.stdout,stderr=x.stderr)
print(json.dumps(d))
