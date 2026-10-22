"""Keep the original pending job; rightsize only its requested wall time."""
import json,pathlib,subprocess,time

A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
first=A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3'
second=A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3'
receipt=first/'root-pending-walltime-rightsize.json'
def read(p):return json.loads(p.read_bytes())
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr[-1000:])
if receipt.exists():
 print(json.dumps(read(receipt)))
else:
 intent=read(first/'root-submission.json');assert str(intent['job_id'])=='750189'and intent['epochs']==10 and intent['optimizer_update_budget']==13210
 evidence=read(second/'coverage-epoch-01.json');evaluation=read(second/'evaluation-epoch-01.json')
 assert evidence['questions']==evidence['unique_questions']==169028 and evaluation['actual_Adam_updates']==1321
 one_epoch=float(evidence['seconds']);estimate=10*(one_epoch+float(evaluation['seconds']))+3600
 before=run(['scontrol','show','job','750189','-o']);assert before['returncode']==0
 text=before['stdout'];assert 'JobId=750189 'in text and 'JobName=q7b-fullP80-MSE5-s9137 'in text and str(first/'training.sbatch')in text
 out=dict(time=time.time(),job_id='750189',before=before,observed_E1_training_seconds=one_epoch,observed_E1_evaluation_seconds=evaluation['seconds'],ten_epochs_plus_one_hour_estimate_seconds=estimate,requested_wall_seconds=21600,epochs_preserved=10,optimizer_update_budget_preserved=13210,training_source_changed=False,duplicate_submission=False,final4096_read=False,changed=False)
 if 'JobState=PENDING 'in text and 'TimeLimit=12:00:00 'in text and estimate<21600:
  update=run(['scontrol','update','JobId=750189','TimeLimit=06:00:00']);out['update']=update
  after=run(['scontrol','show','job','750189','-o']);out['after']=after
  out['changed']=update['returncode']==0 and after['returncode']==0 and 'TimeLimit=06:00:00 'in after['stdout']
 else:out['reason']='Already running, time already changed, or observed runtime does not fit six hours'
 tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(out,sort_keys=True)+'\n');tmp.replace(receipt)
 print(json.dumps(out))
