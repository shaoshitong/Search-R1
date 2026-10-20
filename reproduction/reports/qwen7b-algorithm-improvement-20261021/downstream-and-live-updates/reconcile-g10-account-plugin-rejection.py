import pathlib,json,subprocess,fcntl,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g10-pilot128-qualification-v1/random')
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 assert p.returncode==0,p.stdout+p.stderr
 return p.stdout.strip()
with (base/'submit-reconcile.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (base/'job.id').exists() and not (base/'submission.json').exists()
 path=base/'submission-intent.json';intent=json.loads(path.read_text())
 assert intent['status']=='SBATCH_REJECTED'
 assert 'can not get user mou176 default account' in intent['stderr'] and 'context deadline exceeded' in intent['stderr']
 name=intent['job_name'];assert name=='q7b-g10p128-random'
 queue=command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%j|%T|%R'])
 accounting=command(['sacct','-X','-n','-P','-u','mou176','--name='+name,'--starttime=2026-10-21T01:30:00','-o','JobID,JobName%60,State,ExitCode'])
 assert not queue and not accounting,'Actual job exists; do not retry'
 record=dict(status='KNOWN_ACCOUNT_PLUGIN_REJECTION_NO_JOB_CONFIRMED',intent=intent,squeue=queue,sacct=accounting,time=time.time(),training_modified=False)
 archive=base/('submission-intent-rejected-%d.json'%record['time']);path.rename(archive)
 (base/'submit-reconciliation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
