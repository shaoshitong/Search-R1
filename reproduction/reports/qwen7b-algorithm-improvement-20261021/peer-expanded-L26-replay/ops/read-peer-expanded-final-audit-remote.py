from pathlib import Path
import json,hashlib
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-L26-replay-v1')
def fp(p):
 data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
p=b/'bank/independent-input-audit.json';audit=json.loads(p.read_text());captures=[]
for i in range(4):
 q=b/('shard-%02d'%i)/'capture/complete.json';v=json.loads(q.read_text());captures.append(dict(complete=fp(q),questions=v['questions'],trajectories=v['trajectories'],chunks=len(v['chunks']),model_revision=v['model_revision'],job_id=v['job_id'],compute_layers=v['compute_layers'],persist_layers=v['persist_layers']))
print(json.dumps(dict(audit=audit,audit_ref=fp(p),captures=captures,capture_source='bea03d5f1d908f357c368684b9e8c086ca0e687b',cpu_merge_source='7623fb464e28c45caf2cf52e97b1306675304658',no_new_training_submitted=True)))
