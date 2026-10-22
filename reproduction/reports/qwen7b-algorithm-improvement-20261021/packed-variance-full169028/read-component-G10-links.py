import pathlib,json
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
paths=['peer-threebank58432-internal5120-plan-v1/complete.json','peer-joint124194-recovered1216-L26-v1/bank/complete.json','peer-joint124194-audited243-L26-v1/bank/complete.json','peer-joint124194-audited262-L26-v1/bank/complete.json']
print(json.dumps({p:json.loads((A/p).read_bytes())for p in paths}))
