"""Bounded metadata/progress read of the one existing root-owned scorer."""
import json,pathlib,subprocess,time

base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/selected-state-fixed-trainB2048-unified-20261023-v2')
out=dict(time=time.time(),base=str(base),files={},read_only=True,checkpoint_or_tensor_loaded=False,final4096_read=False)
for name in ('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','failure-rank-0.json','failure-rank-1.json','complete.json'):
 if(base/name).is_file():out['files'][name]=json.loads((base/name).read_bytes())
if 'root-submission.json'in out['files']:
 job=str(out['files']['root-submission.json']['job_id'])
 for name,argv in [('queue',['squeue','-h','-j',job,'-o','%i|%T|%M|%R']),('accounting',['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])]:
  r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);out[name]=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-800:])
 log=base/('score-%s.log'%job)
 if log.is_file():
  with log.open('rb')as f:f.seek(max(0,log.stat().st_size-4000));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
