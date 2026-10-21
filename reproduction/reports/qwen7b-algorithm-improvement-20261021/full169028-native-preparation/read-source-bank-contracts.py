import json,pathlib,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
out={'epoch':time.time()}
for name,relative in [('main','joint124194-native-training-bank-v1/complete.json'),('HPC3_first9344','hpc3-tail9344-independent-L26-prepared-v2/bank/complete.json'),('first9344_audit','hpc3-tail9344-independent-L26-prepared-v2/bank/independent-source-bound-reader-audit.json')]:
 p=A/relative;raw=p.read_bytes();out[name]={'ref':{'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'value':json.loads(raw)}
print(json.dumps(out))
