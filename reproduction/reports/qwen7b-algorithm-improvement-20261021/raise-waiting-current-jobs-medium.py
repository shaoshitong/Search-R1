import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def run(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout.strip()
items=[('749120','q7b-e1-uniform-g5',b/'mixed-direct-e1-uniform-g5-weighted-20261022-v1')]
for job,key,name in [('749134','confirmation2048-A','q7b-final-cA'),('749136','confirmation2048-B','q7b-final-cB'),('749137','stability2048-A','q7b-final-sA'),('749138','stability2048-B','q7b-final-sB')]:items.append((job,name,b/'frozen-final4096-generation-20261022-v1'/key))
result=dict(time=time.time(),normal_forecast='2026-10-24T08:48:32 at observed1792678678',reason='Tried normal first; actual forecast more than34hours, escalate one tier within user low-to-medium instructions',jobs=[])
for job,name,root in items:
 before=run(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName='+name+' 'in before
 row=dict(job=job,name=name,before=before);result['jobs'].append(row)
 (root/'priority-medium-intent.json').write_text(json.dumps(result,sort_keys=True)+'\n')
 if 'JobState=PENDING'in before and 'Partition=acd_u 'in before:run(['scontrol','update','JobId='+job,'Partition=acd_ue'])
 after=run(['scontrol','show','job','-o',job]);row['after']=after
 assert 'Partition=acd_ue 'in after or 'JobState=RUNNING'in after
 (root/'priority-medium-result.json').write_text(json.dumps(row,sort_keys=True)+'\n')
print(json.dumps(dict(time=result['time'],jobs=[dict(job=v['job'],medium_applied='Partition=acd_ue 'in v['after'],running='JobState=RUNNING'in v['after'])for v in result['jobs']])))
