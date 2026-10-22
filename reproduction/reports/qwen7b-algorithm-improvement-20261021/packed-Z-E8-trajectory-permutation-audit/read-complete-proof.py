from pathlib import Path
import hashlib,json,time,subprocess
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-Z-E8-trajectory-permutation-audit-20261023-v1')
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
out=dict(time=time.time(),proofs={name:proof(b/name)for name in('complete.json','controller.json','root-submission.json','audit-config.json','source-publication.json','progress-rank-0.json','progress-rank-1.json')},complete=json.loads((b/'complete.json').read_bytes()))
out['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j','749929','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).strip()
assert out['accounting'].startswith('749929|COMPLETED|0:0|')
print(json.dumps(out))
