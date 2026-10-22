import pathlib,json
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
paths=['peer-expanded-L26-replay-v1/bank/complete.json','peer-expanded-L26-replay-v1/preparation.json','peer-expanded-L26-replay-v1/bank/sources.json']
print(json.dumps({p:json.loads((A/p).read_bytes())for p in paths if(A/p).exists()}))
