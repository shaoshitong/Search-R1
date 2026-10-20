import pathlib,json,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-downstream-comparison-20261021')
submission=json.loads((base/'submission.json').read_text())
j=(base/'job.id').read_text().strip()
assert j==submission['job_id']=='743011'
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 assert p.returncode==0,p.stdout+p.stderr
 return p.stdout.strip()
before=run(['scontrol','show','job',j,'-o'])
assert 'UserId=mou176(21930)' in before and 'JobName=q7b-replay-compare ' in before and 'Command='+str(base/'comparison.sbatch')+' ' in before
if 'JobState=PENDING ' in before and 'Partition=acd_u ' in before:
 run(['scontrol','update','JobId='+j,'Partition=acd_ue'])
 action='NORMAL_TO_MEDIUM_IN_PLACE'
else:action='ALREADY_STARTED_OR_ESCALATED_NO_CHANGE'
after=run(['scontrol','show','job',j,'-o'])
record=dict(job_id=j,action=action,before=before,after=after,reason='Normal start estimate02:10:51 at01:15; medium tier only, no emergency; same jobID/config/runtime',time=time.time())
path=base/'queue-adjustment.json'
assert not path.exists(),'Prior adjustment requires readback; do not repeat'
path.write_text(json.dumps(record,indent=2));print(json.dumps(record))
