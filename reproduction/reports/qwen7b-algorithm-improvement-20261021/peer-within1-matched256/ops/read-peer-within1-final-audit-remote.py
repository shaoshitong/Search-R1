import pathlib,json,hashlib
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-within1-matched256-v1/final-independent-audit.json');b=p.read_bytes();assert len(b)<65536;print(json.dumps(dict(audit=json.loads(b),ref=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))))
