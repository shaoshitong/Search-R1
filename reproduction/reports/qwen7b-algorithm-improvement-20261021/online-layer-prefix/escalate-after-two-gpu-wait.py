import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1')
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout.strip()
x=dict(time=time.time(),reason='normal4GPU nextday; medium4GPU then2GPU still repeatedly forecast8h wait; user authorized gradual priority increase and asks parallelexperiments fast; thirdtier only afterthese real attempts, no normaljob hotedit/no duplicate submission',from_partition='acd_ue',to_partition='emergency_acd',jobs=[])
x['partition']=command(['scontrol','show','partition','emergency_acd','-o'])
assert 'PreemptMode=OFF'in x['partition']
for layer,prefix in ((27,20),(21,20),(27,80)):
 d=b/('layer%d-prefix%d'%(layer,prefix));receipt=json.loads((d/'root-submission.json').read_text());job=receipt['job_id'];before=command(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName='+receipt['name']in before
 r=dict(job=job,before=before);x['jobs'].append(r);(b/'priority-third-tier.json').write_text(json.dumps(x)+'\n')
 if 'JobState=PENDING'in before and 'Partition=acd_ue 'in before:command(['scontrol','update','JobId='+job,'Partition=emergency_acd'])
 r['after']=command(['scontrol','show','job','-o',job]);(b/'priority-third-tier.json').write_text(json.dumps(x)+'\n')
print(json.dumps(dict(time=x['time'],jobs=[dict(job=r['job'],after=r['after'])for r in x['jobs']])))
