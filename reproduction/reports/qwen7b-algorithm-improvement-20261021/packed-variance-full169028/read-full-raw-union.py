import pathlib,json,hashlib
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
p=A/'hpc3-tail-full169028-qualified-raw-union-v1/complete.json';v=read(p);out=dict(proof=proof(p),data=v)
for key,ref in v.get('outputs',{}).items():
 if any(s in key for s in('source','event','refs')):
  q=pathlib.Path(ref['path']);z=read(q);out[key]=dict(proof=proof(q),type=type(z).__name__,keys=list(z)if isinstance(z,dict)else None,length=len(z),sample=z[:2]if isinstance(z,list)else{k:(str(type(a)),len(a)if hasattr(a,'__len__')else None,a[:1]if isinstance(a,list)else a)for k,a in z.items()})
print(json.dumps(out))
