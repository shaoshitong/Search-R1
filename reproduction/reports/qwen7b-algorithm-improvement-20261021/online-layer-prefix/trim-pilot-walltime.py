import pathlib,subprocess,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-v1');out=dict(time=time.time(),walltime='03:00:00',reason='matched8192x3pilot not full169028 training; epoch checkpoint resumes available; shorten conservative8h request to improve backfill',jobs=[])
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout.strip()
for layer,prefix in ((27,20),(21,20),(27,80)):
 d=b/('layer%d-prefix%d'%(layer,prefix));r=json.loads((d/'root-submission.json').read_text());job=r['job_id'];before=command(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName='+r['name']in before
 if 'JobState=PENDING'in before:command(['scontrol','update','JobId='+job,'TimeLimit=03:00:00'])
 out['jobs'].append(dict(job_id=job,before=before,after=command(['scontrol','show','job','-o',job])))
(b/'walltime-backfill-change.json').write_text(json.dumps(out,sort_keys=True)+'\n');print(json.dumps(dict(time=out['time'],jobs=[dict(job_id=x['job_id'],after=x['after'])for x in out['jobs']])))
