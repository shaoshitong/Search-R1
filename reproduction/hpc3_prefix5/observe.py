"""One bounded metadata-only check; never launch or change a running experiment."""
import json,pathlib,shlex,time
from .local_control import run
from .control import BASE,OUTPUT,PYTHON
QUERY=r'''
import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
def read(p):
 if not p.exists():return None
 if p.stat().st_size>1024**2:raise ValueError('Oversized scalar receipt')
 return json.loads(p.read_text())
s=read(b/'preparation/slurm-submission.json');job=s['job'];status=read(b/'logs/pipeline-status.json');a=read(b/'actor/latest.json');probe=read(b/'probe/latest.json');joint=read(b/'latest-joint.json')
result=dict(epoch=time.time(),job=job,queue=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%P|%R|%S'],text=True).strip(),accounting=subprocess.check_output(['sacct','-nP','-j',job,'--format=JobID,User,State,ExitCode,NodeList'],text=True).strip(),source_publication=read(b/'preparation/source-publication.json')['commit'],status=status,complete_outer=joint['outer'] if joint else 0,actor={k:a.get(k) for k in ['completed_outer','actual_adam_steps_per_rank','new_adam_steps_per_rank','scheduler_completed_outer','skipped_update_opportunities']} if a else None,probe={k:probe.get(k) for k in ['completed_outer','actual_adam_steps_per_rank']} if probe else None,allocation=read(b/'logs/slurm-allocation.json'),cleanup=read(b/'logs/cleanup-status.json'),pipeline_complete=bool((b/'logs/pipeline-complete.json').exists()),failed_jobs=[])
ledger=read(b/'logs/process-ledger.json')
if ledger:
 result['failed_jobs']=[dict(name=k,state=v.get('state'),exit_code=v.get('exit_code'),error=v.get('cleanup_error')) for k,v in ledger['jobs'].items() if v.get('exit_code') not in (None,0) or v.get('state')=='CLEANUP_FAILED']
if status and status.get('outer'):
 outer=status['outer'];r=b/'outers'/('outer-%06d'%outer)
 for phase in ['prefix','full']:
  result[phase]=[{k:v.get(k) for k in ['stage','rank','completed_trajectories','expected_trajectories','generation_requests','actual_sampled_tokens','elapsed_seconds']} if (v:=read(r/phase/('rank-%d-progress.json'%i))) else None for i in range(4)]
 attempts=sorted((b/'actor/outers').glob('outer-%04d-attempt-*'%outer),key=lambda p:p.stat().st_mtime_ns) if (b/'actor/outers').exists() else []
 if attempts:
  latest=attempts[-1];counts=[]
  for i in range(4):
   p=latest/('rank-%d-updates.jsonl'%i);value=None
   if p.exists():
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));lines=f.read().splitlines()
    for line in reversed(lines):
     try:v=json.loads(line)
     except ValueError:continue
     if 'actual_adam_steps' in v:value={k:v.get(k) for k in ['event','actual_adam_steps','opportunity','writes_this_attempt']};break
   counts.append(value)
  result['inflight_rank_updates']=counts
  result['rank_failures']=[read(latest/('rank-%d-failure.json'%i)) for i in range(4)]
log=b/'logs'/('slurm-'+job+'.log')
if log.exists():
 result['log_bytes']=log.stat().st_size
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode(errors='replace')
 result['error_lines']=[x[:400] for x in tail.splitlines() if any(w in x for w in ['Traceback','Error:','Exception:','AssertionError','ModuleNotFoundError'])][-12:]
print(json.dumps(result,allow_nan=False))
'''

def main():
 r=run(PYTHON+' -c '+shlex.quote(QUERY))
 if r.returncode:raise RuntimeError('Metadata read failed: '+r.stderr.decode()[:1000])
 v=json.loads(r.stdout);OUTPUT.mkdir(exist_ok=True,parents=True);(OUTPUT/'observation-latest.json').write_text(json.dumps(v,indent=2));
 with (OUTPUT/'observations.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v))
if __name__=='__main__':main()
