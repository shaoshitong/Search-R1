"""Explicit root call to the previously staged submit-once controller."""
import hashlib,json,pathlib,subprocess

entry=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/selected-state-fixed-trainB2048-unified-20261023-v2/root-control-code-1612ecb47f2e/root-bind-and-submit-once.sh')
assert hashlib.sha256(entry.read_bytes()).hexdigest()=='fc1db9f4bd064860486becacc056e9a18338095bb1fd89a845b03f2cdd391bd9'
result=subprocess.run(['bash',str(entry),'--action','submit'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
if result.returncode:
 raise RuntimeError(result.stdout[-3000:]+result.stderr[-3000:])
print(json.dumps(json.loads(result.stdout.strip().splitlines()[-1])))
