"""Read exact new full-data jobs; does not alter scheduling or launch work."""
import json,subprocess,time

def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr[-1000:])
out=dict(time=time.time(),Slurm_mutated=False,final4096_read=False,jobs={})
for job in ('750189','750190'):
 out['jobs'][job]=dict(
  actual=run(['scontrol','show','job',job,'-o']),
  projected_start=run(['squeue','--start','-h','-j',job,'-o','%i|%T|%S|%R|%q|%Q']),
  accounting=run(['sacct','-n','-X','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList','-P']),
 )
print(json.dumps(out))
