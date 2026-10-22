"""Explicit root call to the previously staged submit-once controller."""
import hashlib,json,pathlib,subprocess

entry=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/selected-state-fixed-trainB2048-unified-20261023-v1/root-control-code-45bccf6e7046/root-bind-and-submit-once.sh')
assert hashlib.sha256(entry.read_bytes()).hexdigest()=='7446ed8166c4b24d48d61384f5adf9240c0fadbadc5e1f20288d437d7230f0e1'
result=subprocess.run(['bash',str(entry),'--action','submit'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
if result.returncode:
 raise RuntimeError(result.stdout[-3000:]+result.stderr[-3000:])
print(json.dumps(json.loads(result.stdout.strip().splitlines()[-1])))
