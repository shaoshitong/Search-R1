"""Read actual selected development/train source event seal metadata, no final."""
import json,pathlib,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):
 p=pathlib.Path(p);assert not any(s in str(p)for s in('final4096','confirmation','stability2048'));return json.loads(p.read_bytes())
v=read(A/'validation-g10-20261021/materialized/bank/source-audit.json');t=read(A/'peer-newseed8192-g10-train-v1/training-input.json');r=read(t['raw_complete']['path']);e=read(r['outputs']['source-events']['path']);out={'time':time.time(),'dev':{},'train':{},'final4096_read':False}
for k,bs in v['events'].items():
 b=bs[0];out['dev'][k]=b
for ev in e['events']:
 b=ev['batches'][0];out['train'][ev['event']]=b
print(json.dumps(out))
