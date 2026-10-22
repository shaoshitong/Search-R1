import pathlib,json,hashlib
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
out={}
for rel in('joint169028-native-training-bank-v1/complete.json','joint-mixed-direct-official-169028-v1/config.json'):
 p=A/rel;v=read(p);out[rel]=dict(proof=proof(p),data=v if rel.endswith('complete.json')else{k:v[k]for k in v if any(s in k for s in('groups','complete','bank','source','features'))})
print(json.dumps(out))
