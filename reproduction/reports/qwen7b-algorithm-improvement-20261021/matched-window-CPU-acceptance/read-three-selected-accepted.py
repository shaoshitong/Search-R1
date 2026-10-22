from pathlib import Path
import json,time,hashlib
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');O=A/'matched-window-terminal-CPU-acceptance-20261023-v1';C=A/'selected-checkpoint-CPU-coordinator-20261023-v1';out=dict(time=time.time(),receipts={})
for cell,e in [('current64',5),('mse-M5',7),('window2018',10)]:
 p=O/cell/('epoch-%02d'%e)/'accepted.json';raw=p.read_bytes();out['receipts'][cell]=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),accepted=json.loads(raw))
out['coordinator_complete']=json.loads((C/'complete.json').read_bytes());ctrl=json.loads((C/'controller.json').read_bytes());p=Path('/proc/%d/stat'%ctrl['pid']);out['coordinator_identity_live']=p.exists()and p.read_text().rsplit(')',1)[1].split()[19]==ctrl['start_ticks'];out['failure_exists']=(C/'failure.json').exists();print(json.dumps(out))
