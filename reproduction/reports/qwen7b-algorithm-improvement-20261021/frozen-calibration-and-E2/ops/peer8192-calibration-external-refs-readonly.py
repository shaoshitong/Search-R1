from pathlib import Path
import json,gzip,hashlib,time
T=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1')
def read(p):return json.loads(Path(p).read_text())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
refs={};baseline=read(T/'E4-same-HF-validation/complete.json');fit=read(T/'complete.json')
for name,r in [('E4',baseline['checkpoints']['E4']['predictions']),('peer256',fit['metrics']['selection2048FreshG5HF']['predictions'])]:
 own=proof(r['path']);assert own['sha256']==r['sha256']
 with gzip.open(r['path'],'rt') as f:keys=sorted(json.loads(f.readline()))
 refs[name]=dict(predictions=own,row_keys=keys)
result=dict(status='EXISTING_SAME_HF_EXTERNAL_PREDICTIONS_REFS_AND_ROWKEYS_ONLY',sources=refs,external_new_inference=False,final_pool_read=False,time=time.time());print(json.dumps(result))
