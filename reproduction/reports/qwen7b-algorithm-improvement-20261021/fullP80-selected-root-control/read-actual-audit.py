"""Actual selected P80 audit metadata only; no checkpoint or final labels."""
import json,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
base=A/'full169028-L15P80-MSE-viewed-permutation-audit-20261023-v1'
control=A/'fullP80-selected-root-control-20261023-v1'
def read(p):return json.loads(p.read_bytes())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-700:])
out=dict(time=time.time(),read_only=True,checkpoint_tensors_loaded=False,final4096_read=False,files={})
for n in ['audit-submission.json','audit-submit-intent.json','audit-CUDA-retry-submission.json','audit-CUDA-retry-intent.json']:
 p=control/n
 if p.is_file():out['files'][n]=read(p)
for n in ['controller.json','progress-rank-0.json','progress-rank-1.json','failure-rank-0.json','failure-rank-1.json','complete.json','seed1/complete.json','seed2/complete.json']:
 p=base/n
 if p.is_file():out['files'][n]=read(p)
sub=out['files'].get('audit-CUDA-retry-submission.json')or out['files'].get('audit-submission.json')
if sub:
 job=str(sub['job_id'])
 out['squeue']=command(['squeue','-h','-j',job,'-o','%i|%T|%M|%R'])
 out['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])
 log=base/('audit-'+job+'.log')
 if log.is_file():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-5000));out['log_tail']=f.read().decode(errors='replace').splitlines()[-14:]
print(json.dumps(out))
