# Historical exact operator executed on 2026-10-22; no credentials.
import json,hashlib,time,subprocess
from pathlib import Path
p=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/rollout-batch-sweep64-cause-r1')
r={'epoch':time.time(),'job':749520,'node':'ACD1-13','sacct':subprocess.run(['sacct','-j','749520','-n','-P','--format=JobID,State,ExitCode,Elapsed,NodeList'],capture_output=True,text=True,timeout=10).stdout,'files':{}}
for n in ('xformers.log','slurm-749520.log','failure.json','submit-claim.json','submit-receipt.json'):
 q=p/n
 if q.is_file():raw=q.read_bytes();r['files'][n]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'tail':raw.decode(errors='replace')[-3500:]}
(p/'failure-preserved.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
