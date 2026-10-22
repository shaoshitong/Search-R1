import pathlib,subprocess,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-v1')
def command(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout.strip()
def save(x):(b/'priority-step-medium.json').write_text(json.dumps(x,sort_keys=True)+'\n')
x=dict(time=time.time(),reason='normal/acd_u backfill estimates nextday17:19 to18:45; gradual nextpartition acd_ue, no emergency',from_partition='acd_u',to_partition='acd_ue',jobs=[])
x['partition']=command(['scontrol','show','partition','acd_ue','-o'])
for layer,prefix in ((27,20),(21,20),(27,80)):
 d=b/('layer%d-prefix%d'%(layer,prefix));r=json.loads((d/'root-submission.json').read_text());job=r['job_id'];before=command(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName='+r['name']in before
 item=dict(job_id=job,before=before);x['jobs'].append(item);save(x)
 if 'JobState=PENDING'in before and 'Partition=acd_u 'in before:command(['scontrol','update','JobId='+job,'Partition=acd_ue'])
 item['after']=command(['scontrol','show','job','-o',job]);save(x)
print(json.dumps(dict(time=x['time'],partition=x['partition'],jobs=[dict(job_id=i['job_id'],after=i['after'])for i in x['jobs']])))
