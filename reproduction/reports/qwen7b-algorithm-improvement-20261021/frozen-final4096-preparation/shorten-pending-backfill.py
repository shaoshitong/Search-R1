import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
def run(cmd):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0,r.stderr[-1500:];return r.stdout.strip()
out=[]
for job,key,name in [('749134','confirmation2048-A','q7b-final-cA'),('749136','confirmation2048-B','q7b-final-cB'),('749137','stability2048-A','q7b-final-sA'),('749138','stability2048-B','q7b-final-sB')]:
 root=b/key;before=run(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName='+name+' 'in before
 record=dict(time=time.time(),job=job,source_entry_unmodified=True,actual_mature512q_previous_seconds=[358.886,373.544],reason='Pending2h unnecessarily exceeds actual6min task;45min still >=7x observed duration and improves backfill',before=before)
 (root/'backfill-walltime-intent.json').write_text(json.dumps(record,sort_keys=True)+'\n')
 if 'JobState=PENDING'in before:run(['scontrol','update','JobId='+job,'TimeLimit=00:45:00'])
 record['after']=run(['scontrol','show','job','-o',job]);(root/'backfill-walltime-result.json').write_text(json.dumps(record,sort_keys=True)+'\n');out.append(dict(job=job,state=record['after'].split('JobState=')[1].split()[0],TimeLimit=record['after'].split('TimeLimit=')[1].split()[0]))
print(json.dumps(dict(time=time.time(),arrays=out)))
