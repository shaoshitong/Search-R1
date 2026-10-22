"""Retain exact job handles; move only actual >30min normal waits one tier."""
import datetime,fcntl,json,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
RUNS=[('750189',A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3'),('750190',A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3')]
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stdout[-800:]+p.stderr[-1000:])
 return p.stdout.strip()
partition=run(['scontrol','show','partition','acd_ue','-o'])
assert 'PreemptMode=OFF'in partition and 'PriorityTier=200 'in partition
rows=[]
for job,base in RUNS:
 guard=(base/'priority-one-tier.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=json.loads((base/'root-submission.json').read_bytes());assert str(receipt['job_id'])==job
 before=run(['scontrol','show','job',job,'-o'])
 assert 'UserId=mou176('in before and 'Command='+str(base/'training.sbatch')in before
 tokens=dict(item.split('=',1)for item in before.split()if'='in item)
 row=dict(time=time.time(),job_id=job,before=before,policy='Try normal first; if actual projected wait exceeds30min, raise one tier on the same pending job',higher_tier='acd_ue',preemption='OFF',changed=False)
 if tokens.get('JobState')=='PENDING'and tokens.get('Partition')=='acd_u':
  try:wait=(datetime.datetime.strptime(tokens['StartTime'],'%Y-%m-%dT%H:%M:%S')-datetime.datetime.now()).total_seconds()
  except ValueError:wait=None
  row['actual_projected_wait_seconds']=wait
  if wait is not None and wait>1800:
   (base/'priority-one-tier-intent.json').write_text(json.dumps(row,sort_keys=True)+'\n')
   run(['scontrol','update','JobId='+job,'Partition=acd_ue']);row['changed']=True
 row['after']=run(['scontrol','show','job',job,'-o'])
 (base/'priority-one-tier-result.json').write_text(json.dumps(row,sort_keys=True)+'\n');rows.append(row)
 fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
print(json.dumps(dict(time=time.time(),jobs=rows,duplicate_jobs_submitted=False,training_budget_changed=False)))
