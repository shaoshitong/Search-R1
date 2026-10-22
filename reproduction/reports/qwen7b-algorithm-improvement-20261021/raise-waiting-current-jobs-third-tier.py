import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def run(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout.strip()
partition=run(['scontrol','show','partition','emergency_acd','-o']);assert 'PreemptMode=OFF'in partition
items=[('749120','q7b-e1-uniform-g5',b/'mixed-direct-e1-uniform-g5-weighted-20261022-v1')]
for job,key,name in [('749134','confirmation2048-A','q7b-final-cA'),('749136','confirmation2048-B','q7b-final-cB'),('749137','stability2048-A','q7b-final-sA'),('749138','stability2048-B','q7b-final-sB')]:items.append((job,name,b/'frozen-final4096-generation-20261022-v1'/key))
rows=[]
for job,name,root in items:
 assert (root/'priority-medium-result.json').exists()
 before=run(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName='+name+' 'in before
 row=dict(time=time.time(),job=job,name=name,previous_normal_forecast='2026-10-24T08:48:32',medium_forecast_observed_epoch=1792678775,medium_forecasts=['2026-10-23T08:12:51','2026-10-23T13:04:28','2026-10-24T08:48:32'],reason='Normal then medium both actual forecasts >=9h; user permits gradual priority escalation; existing job handles retained, no preemption',before=before)
 (root/'priority-third-tier-intent.json').write_text(json.dumps(row,sort_keys=True)+'\n')
 if 'JobState=PENDING'in before and 'Partition=acd_ue 'in before:run(['scontrol','update','JobId='+job,'Partition=emergency_acd'])
 row['after']=run(['scontrol','show','job','-o',job]);assert 'Partition=emergency_acd 'in row['after']or'JobState=RUNNING'in row['after']
 (root/'priority-third-tier-result.json').write_text(json.dumps(row,sort_keys=True)+'\n');rows.append(dict(job=job,state=row['after'].split('JobState=')[1].split()[0],partition=row['after'].split('Partition=')[1].split()[0]))
print(json.dumps(dict(time=time.time(),jobs=rows,no_foreign_job_preemption=True)))
