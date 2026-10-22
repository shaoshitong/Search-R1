import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-M3to10-P20-variance-8192-20261022-v1');job=json.loads((b/'root-submission.json').read_bytes())['job_id'];assert job=='749515'
def run(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0,r.stderr
 return r.stdout.strip()
before=run(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName=q7b-packed-M3to10-var 'in before
row=dict(time=time.time(),job=job,before=before,normal_forecast_observed_epoch=1792683279.8899505,normal_forecast='2026-10-23T00:09:00',reason='Tried normal first; actual forecast>30min. User permits gradual priority increase; try medium only, keep same2GPU30min job.')
(b/'priority-medium-intent.json').write_text(json.dumps(row)+'\n')
if 'JobState=PENDING'in before and 'Partition=acd_u 'in before:run(['scontrol','update','JobId='+job,'Partition=acd_ue'])
row['after']=run(['scontrol','show','job','-o',job]);row['forecast']=run(['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R']);(b/'priority-medium-result.json').write_text(json.dumps(row)+'\n')
print(json.dumps(dict(time=row['time'],job=job,forecast=row['forecast'],same_handle=True,medium='Partition=acd_ue 'in row['after'],state=row['after'].split('JobState=')[1].split()[0])))
