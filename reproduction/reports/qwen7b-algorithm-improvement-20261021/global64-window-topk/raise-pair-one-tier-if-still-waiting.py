"""Same authorized normal-first policy for the two cached-window jobs."""
import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-global64-window2018-L15P80-8192-20261023-v1')
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-1500:]}
now=time.time();out=[]
for cell in ('current64','window2018'):
 d=b/cell;r=json.loads((d/'root-submission.json').read_bytes());job=r['job_id'];before=run(['scontrol','show','job',job,'-o']);assert before['returncode']==0
 text=before['stdout'];assert 'UserId=mou176('in text and 'JobName='+r['name']+' 'in text and 'Command='+r['entry']['path']in text
 item={'time':now,'job_id':job,'cell':cell,'before':before,'elapsed_since_actual_submission_seconds':now-r['time'],'change':None}
 if 'JobState=PENDING'in text and 'Partition=acd_u 'in text and now-r['time']>=600:
  (d/'one-tier-priority-intent.json').write_text(json.dumps(item,sort_keys=True)+'\n')
  item['change']=run(['scontrol','update','JobId='+job,'Partition=acd_ue'])
 item['after']=run(['scontrol','show','job',job,'-o']);assert item['after']['returncode']==0
 if item['change']is not None and item['change']['returncode']:
  assert 'JobState=RUNNING'in item['after']['stdout'],'Scheduling update failed; preserve existing job and inspect this same handle.'
 (d/'one-tier-priority-result.json').write_text(json.dumps(item,sort_keys=True)+'\n');out.append(item)
print(json.dumps({'time':now,'policy':'Actual normal wait >=10min with multi-hour observed normal forecast; one-tier only, no emergency, no duplicate submission.','jobs':out}))
