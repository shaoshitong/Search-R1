"""Root explicit submit-final invocation; existing durable controller owns prerequisites and reconciliation."""
import hashlib,json,pathlib,subprocess
p=pathlib.Path("/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fullP80-selected-root-control-20261023-v1/control.sh")
assert hashlib.sha256(p.read_bytes()).hexdigest()=="1d76063c65e7fdb82462fc627b4b098d21798860cd3f21d70aa9207b02e96a90"
r=subprocess.run(["bash",str(p),"--action","submit-final"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-1500:])
print(json.dumps(json.loads(r.stdout.strip().splitlines()[-1])))
