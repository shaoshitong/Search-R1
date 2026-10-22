import pathlib,json
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
paths=['joint124194-native-training-bank-v1/complete.json','peer-newseed-ingest-audited138-after971-20261021-51057f37/complete.json','joint124194-native-training-bank-v1/source-refs.json']
out={}
for rel in paths:
 p=A/rel
 if p.exists():out[rel]=json.loads(p.read_bytes())
print(json.dumps(out))
