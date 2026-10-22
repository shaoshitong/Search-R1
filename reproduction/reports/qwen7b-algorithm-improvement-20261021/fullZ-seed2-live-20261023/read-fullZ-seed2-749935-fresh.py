from pathlib import Path
from collections import deque
import json,time,subprocess
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1';out=dict(time=time.time(),base=str(b),Slurm_mutated=False,model_or_checkpoint_read=False,final4096_read=False)
for name in('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
 if(b/name).is_file():out[name]=json.loads((b/name).read_bytes())
job=str(out['root-submission.json']['job_id']);assert job=='749935';out['queue']=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%N|%M|%R'],universal_newlines=True,timeout=15);out['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed'],universal_newlines=True,timeout=15);out['failures']={p.name:json.loads(p.read_bytes())for p in b.glob('failure-rank-*.json')}
p=b/'curve.jsonl'
if p.is_file():
 with p.open()as f:out['last_six_real_updates']=[json.loads(x)for x in deque(f,maxlen=6)]
out['completed_evaluations']=[p.name for p in sorted(b.glob('evaluation-epoch-*.json'))];print(json.dumps(out))
