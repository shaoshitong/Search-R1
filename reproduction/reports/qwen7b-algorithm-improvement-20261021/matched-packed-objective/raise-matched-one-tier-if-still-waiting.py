"""Promote only this task's genuinely waiting matched cells by one tier."""
import pathlib,json,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-1500:]}
now=time.time();rows=[]
for cell in ('mse-Mrandom','mse-M5','mixed-bce-Mrandom','mixed-bce-M5'):
 d=A/'online-packed-matched-L15P80-8192-20261023-v1'/cell;r=json.loads((d/'root-submission.json').read_bytes());job=r['job_id'];before=run(['scontrol','show','job',job,'-o']);assert before['returncode']==0
 text=before['stdout'];assert 'UserId=mou176('in text and 'JobName='+r['name']+' 'in text and 'Command='+r['entry']['path']in text
 item={'time':now,'job_id':job,'cell':cell,'before':before,'elapsed_since_actual_submission_seconds':now-r['time'],'change':None}
 if 'JobState=PENDING'in text and 'Partition=acd_u 'in text and now-r['time']>=600:
  (d/'one-tier-priority-intent.json').write_text(json.dumps(item,sort_keys=True)+'\n')
  item['change']=run(['scontrol','update','JobId='+job,'Partition=acd_ue'])
 item['after']=run(['scontrol','show','job',job,'-o']);assert item['after']['returncode']==0
 if item['change']is not None and item['change']['returncode']:
  assert 'JobState=RUNNING'in item['after']['stdout'],'A scheduling update failed; leave job intact and inspect the same handle.'
 (d/'one-tier-priority-result.json').write_text(json.dumps(item,sort_keys=True)+'\n');rows.append(item)
print(json.dumps({'time':now,'policy':'Tried normal first, >=10min actual wait; raise one tier only; no emergency, no resubmission, no job dependencies.','jobs':rows}))
