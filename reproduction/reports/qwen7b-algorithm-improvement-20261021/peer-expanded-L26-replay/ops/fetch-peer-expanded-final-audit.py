"""Only small aggregate acceptance/provenance JSON crosses control connection."""
from pathlib import Path
import json,subprocess,sys
C=Path(__file__).resolve().parents[2];O=C/'outputs/qwen7b-warmup-20261019';P=O/'reports'
program="""from pathlib import Path
import json,hashlib
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-L26-replay-v1')
def fp(p):
 data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
p=b/'bank/independent-input-audit.json';audit=json.loads(p.read_text());captures=[]
for i in range(4):
 q=b/('shard-%02d'%i)/'capture/complete.json';v=json.loads(q.read_text());captures.append(dict(complete=fp(q),questions=v['questions'],trajectories=v['trajectories'],chunks=len(v['chunks']),model_revision=v['model_revision'],job_id=v['job_id'],compute_layers=v['compute_layers'],persist_layers=v['persist_layers']))
print(json.dumps(dict(audit=audit,audit_ref=fp(p),captures=captures,capture_source='bea03d5f1d908f357c368684b9e8c086ca0e687b',cpu_merge_source='7623fb464e28c45caf2cf52e97b1306675304658',no_new_training_submitted=True)))
"""
script=O/'read-peer-expanded-final-audit-remote.py';script.write_text(program)
r=subprocess.run([sys.executable,str(O/'balanced-control.py'),str(script),'--timeout','55'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if r.returncode:raise RuntimeError(r.stderr[-1500:]+' '+r.stdout[-1500:])
v=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')][-1];(P/'peer-expanded-L26-final-delivery.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(audit_ref=v['audit_ref'],bank=v['audit']['bank_complete'],questions=v['audit']['questions'],trajectories=v['audit']['trajectories'],capture_job_ids=[x['job_id'] for x in v['captures']])))
