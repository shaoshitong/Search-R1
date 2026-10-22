"""Small read-only actual audit-job status; no CP, token or final data reads."""
from pathlib import Path
import json
import subprocess
import time

b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-Z-E8-trajectory-permutation-audit-20261023-v1')
out=dict(time=time.time(),expected_job='749929',checkpoint_contents_opened=False,Slurm_mutated=False,final4096_read=False)
for name in ('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','complete.json'):
    p=b/name
    if p.is_file(): out[name]=json.loads(p.read_bytes())
if 'controller.json' in out: assert str(out['controller.json']['job_id'])=='749929'
for label,args in [('accounting',['sacct','-X','-n','-P','-j','749929','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList']),('queue',['squeue','-h','-j','749929','-o','%i|%j|%T|%M|%R'])]:
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);out[label]=dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
out['failures']={p.name:json.loads(p.read_bytes()) for p in b.glob('failure-rank-*.json')}
p=b/'audit-749929.log'
if p.is_file():
    with p.open('rb') as stream: stream.seek(max(0,p.stat().st_size-2000));tail=stream.read().decode(errors='replace')
    out['error_tail']=tail if any(x in tail for x in ('Traceback (most recent','RuntimeError:','out of memory','NCCL error','DUE TO TIME LIMIT')) else None
print(json.dumps(out))
