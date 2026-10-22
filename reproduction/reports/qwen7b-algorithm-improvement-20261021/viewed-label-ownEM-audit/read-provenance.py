"""Read development and fixed-train metadata provenance only; no final holdouts."""
import hashlib,json,pathlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):
 p=pathlib.Path(p);assert not any(z in str(p)for z in('final4096','confirmation','stability2048'));return json.loads(p.read_bytes())
def proof(p):
 p=pathlib.Path(p);r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
def shape(v,depth=0):
 if isinstance(v,dict):return {k:shape(x,depth+1)for k,x in v.items()}if depth<3 else list(v)
 if isinstance(v,list):return dict(count=len(v),first=shape(v[0],depth+1)if v else None)
 return v
out={'time':time.time(),'final4096_read':False,'Slurm_mutated':False}
for name,p in [('pilot',A/'online-packed-matched-L15P80-8192-20261023-v1/fixture/fixture-complete.json'),('causal_fixture',A/'online-mixed-layer-prefix-20261022-v1/fixture-complete.json'),('dev_source_audit',A/'validation-g10-20261021/materialized/bank/source-audit.json'),('train_input',A/'peer-newseed8192-g10-train-v1/training-input.json')]:
 x=read(p);out[name]={'proof':proof(p),'shape':shape(x)}
 if name=='train_input':
  r=read(x['raw_complete']['path']);out['train_raw_complete']={'proof':proof(x['raw_complete']['path']),'shape':shape(r)}
  e=r['outputs']['source-events'];ev=read(e['path']);out['train_source_events']={'proof':proof(e['path']),'shape':shape(ev)}
print(json.dumps(out))
