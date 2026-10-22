"""Root-only one-tier escalation of the existing independent P80 continuation."""
import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer15-prefix80');receipt=json.loads((b/'root-submission.json').read_bytes());job=receipt['job_id']
def run(c):
 p=subprocess.run(c,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if p.returncode:raise RuntimeError(p.stderr[-1000:])
 return p.stdout.strip()
before=run(['scontrol','show','job','-o',job]);assert 'JobName='+receipt['name']+' 'in before and 'UserId=mou176('in before and 'Command='+receipt['entry']['path']in before
result={'time':time.time(),'job_id':job,'before':before,'reason':'Tried normal first; latest fresh forecast2026-10-23T02:42:28 observed1792690164 was over70minutes away. Increase existingjob one tier; preserve checkpoint and no dependencies.','no_duplicate_submission':True,'no_emergency_partition':True}
(b/'continuation-priority-one-tier-intent.json').write_text(json.dumps(result,sort_keys=True)+'\n')
if 'JobState=PENDING'in before and 'Partition=acd_u 'in before:run(['scontrol','update','JobId='+job,'Partition=acd_ue'])
after=run(['scontrol','show','job','-o',job]);assert 'Partition=acd_ue 'in after or 'JobState=RUNNING'in after
result.update(after=after,changed_partition_to_medium='Partition=acd_ue 'in after);(b/'continuation-priority-one-tier-result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
print(json.dumps({'time':result['time'],'job_id':job,'medium':result['changed_partition_to_medium'],'running':'JobState=RUNNING'in after,'no_duplicate_submission':True,'no_emergency':True}))
