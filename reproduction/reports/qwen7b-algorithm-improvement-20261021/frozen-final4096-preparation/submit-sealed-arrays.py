import pathlib,json,subprocess,fcntl,time,hashlib,re
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
def read(p):return json.loads(p.read_bytes())
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True)+'\n');q.replace(p)
def check(ref):
 raw=pathlib.Path(ref['path']).read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
def command(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if r.returncode:raise RuntimeError(r.stderr[-1500:])
 return r.stdout.strip()
lock=(b/'root-submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=read(b/'prepared.json');assert plan['questions']==4096 and plan['probe_training_or_scoring']is False and plan['max_simultaneous_GPUs']==8
out=[]
for item in plan['arrays']:
 root=b/item['key'];record=root/'root-submission.json';intent=root/'root-submission-intent.json'
 if record.exists():out.append(read(record));continue
 for key in('config','entry'):check(item[key])
 active=command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']);assert not active,'Do not duplicate owned array '+active
 assert not intent.exists(),'Unknown previous submit result: reconcile same-name array, no blindretry'
 atomic(intent,dict(time=time.time(),key=item['key'],entry=item['entry'],config=item['config'],array='0-3%1',max_active_GPUs=2,partition='acd_u',time_limit_override='02:00:00',purpose='Generate sealed final inputs only; no probe scoring or model selection',final_labels_read_for_selection=False))
 answer=command(['sbatch','--parsable','--time=02:00:00',item['entry']['path']]);assert re.fullmatch(r'[0-9]+(?:;[^ ]+)?',answer);job=answer.split(';')[0]
 receipt=dict(job_id=job,key=item['key'],name=item['job_name'],config=item['config'],entry=item['entry'],original_event_root=item['original_event_root'],time=time.time(),status='ARRAY_SUBMITTED_NOT_ROLLOUT_PROGRESS_VERIFIED',sealed_data_only=True,final_probe_scored=False,array='0-3%1',time_limit='02:00:00');atomic(record,receipt)
 identity=command(['scontrol','show','job','-o',job]);assert all(x in identity for x in ('UserId=mou176','JobName='+item['job_name'],'Partition=acd_u','Dependency=(null)','TimeLimit=02:00:00'));assert 'gres/gpu=2'in identity or'TresPerNode=gres/gpu:2'in identity;receipt['slurm_identity']=identity;atomic(record,receipt);out.append(receipt)
print(json.dumps(dict(jobs=[dict(job_id=x['job_id'],key=x['key'],array=x['array'])for x in out],max_concurrent_GPUs=8,no_probe_scoring=True)))
