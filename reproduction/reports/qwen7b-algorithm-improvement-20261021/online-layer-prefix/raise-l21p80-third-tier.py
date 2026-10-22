import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer21-prefix80')
job=json.loads((b/'root-submission.json').read_bytes())['job_id'];assert job=='749369'
def run(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 assert r.returncode==0,r.stderr
 return r.stdout.strip()
assert (b/'priority-medium-result.json').exists()
before=run(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName=q7b-online2-L21P80 'in before
assert 'PreemptMode=OFF'in run(['scontrol','show','partition','emergency_acd','-o'])
result=dict(time=time.time(),job=job,before=before,medium_forecast_observed_epoch=1792681074.2105172,reason='Normal and medium actual forecasts both2026-10-24T08:48:32 (>33h). Gradual authorized escalation, same job and no foreign preemption.')
(b/'priority-third-tier-intent.json').write_text(json.dumps(result,sort_keys=True)+'\n')
if 'JobState=PENDING'in before and 'Partition=acd_ue 'in before:run(['scontrol','update','JobId='+job,'Partition=emergency_acd'])
result['after']=run(['scontrol','show','job','-o',job]);result['forecast']=run(['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R'])
(b/'priority-third-tier-result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
print(json.dumps(dict(time=result['time'],job=job,forecast=result['forecast'],same_job_handle=True,no_foreign_preemption=True,state=result['after'].split('JobState=')[1].split()[0])))
