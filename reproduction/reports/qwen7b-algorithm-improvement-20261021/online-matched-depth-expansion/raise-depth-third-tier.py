import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1')
def run(a):
 r=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 assert r.returncode==0,r.stderr
 return r.stdout.strip()
assert 'PreemptMode=OFF'in run(['scontrol','show','partition','emergency_acd','-o'])
rows=[]
for layer,job in [(15,'749398'),(18,'749399'),(24,'749400')]:
 d=b/('layer%d-prefix20'%layer);assert json.loads((d/'root-submission.json').read_bytes())['job_id']==job and(d/'priority-medium-result.json').exists()
 before=run(['scontrol','show','job','-o',job]);assert 'UserId=mou176'in before and 'JobName=q7b-online2-L%dP20 '%layer in before
 row=dict(time=time.time(),job=job,before=before,medium_forecast_observed_epoch=1792681404.5661945,normal_and_medium_forecast='2026-10-24T08:48:32',reason='Both normal and medium actually forecast>33h; next authorized tier, same2GPU30min jobs; PreemptModeOFF.')
 (d/'priority-third-tier-intent.json').write_text(json.dumps(row)+'\n')
 if 'JobState=PENDING'in before and 'Partition=acd_ue 'in before:run(['scontrol','update','JobId='+job,'Partition=emergency_acd'])
 row['after']=run(['scontrol','show','job','-o',job]);(d/'priority-third-tier-result.json').write_text(json.dumps(row)+'\n');rows.append(dict(job=job,state=row['after'].split('JobState=')[1].split()[0],partition=row['after'].split('Partition=')[1].split()[0]))
print(json.dumps(dict(time=time.time(),jobs=rows,no_foreign_preemption=True)))
