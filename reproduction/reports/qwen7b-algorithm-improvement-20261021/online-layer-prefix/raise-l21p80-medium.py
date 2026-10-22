import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer21-prefix80')
job=json.loads((b/'root-submission.json').read_bytes())['job_id'];assert job=='749369'
def run(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 assert r.returncode==0,r.stderr
 return r.stdout.strip()
before=run(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName=q7b-online2-L21P80 'in before
result=dict(time=time.time(),job=job,before=before,reason='Observed normal/acd_u forecast 2026-10-24T08:48:32 at epoch1792681037; more than33hours. Try next medium tier under existing user authorization.')
(b/'priority-medium-intent.json').write_text(json.dumps(result,sort_keys=True)+'\n')
if 'JobState=PENDING'in before and 'Partition=acd_u 'in before:run(['scontrol','update','JobId='+job,'Partition=acd_ue'])
result['after']=run(['scontrol','show','job','-o',job]);result['forecast']=run(['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R'])
(b/'priority-medium-result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
print(json.dumps(result))
