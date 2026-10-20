"""Replace this owned pending job with one higher-tier job, never dual-run."""
import datetime,fcntl,json,os,subprocess,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020')
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
 assert r.returncode==0,(args,r.stdout,r.stderr)
 return r.stdout.strip()
def info(job):
 s=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R'])
 assert s and len(s.splitlines())==1,s
 return s.split('|')
old=(B/'job.id').read_text().strip();assert old=='741871',old
request=B/'priority-plus-one.json'
assert not request.exists(),'Inspect existing handoff outcome before any resubmission'
row=info(old);assert row[1:5]==['mou176','q7b-mixed-cp2-ft','PENDING','acd_ue'],row
publication=json.loads((B/'preparation/source-publication.json').read_text());root=publication['code_root']
with (B/'probe/fit.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record=dict(status='HANDOFF_INTENT',user_instruction='优先级+1，再排队一个看看',user_message_timestamp='not exposed; recorded receipt time only',recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Replace only owned pending741871; no running training cancellation',old_job=old,old_partition='acd_ue',new_partition='emergency_acd',gpus=4,source_commit=publication['commit'],code_root=root)
atomic(request,record)
command(['scontrol','hold',old]);row=info(old);assert row[3]=='PENDING' and 'JobHeldUser' in row[-1],row
record['status']='OLD_PENDING_HELD';atomic(request,record)
env=dict(os.environ,QWEN7B_CODE_ROOT=root)
r=subprocess.run(['sbatch','--hold','--parsable','--partition=emergency_acd','--job-name=q7b-mixed-cp2-high',root+'/reproduction/operations/qwen7b_warmup/mixed_fit.sbatch'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
assert r.returncode==0,r.stderr
new=r.stdout.strip().split(';')[0];assert new.isdigit(),r.stdout
record.update(new_job=new,status='NEW_HIGHER_PENDING_HELD');atomic(request,record)
newrow=info(new);assert newrow[1:5]==['mou176','q7b-mixed-cp2-high','PENDING','emergency_acd'] and 'JobHeldUser' in newrow[-1],newrow
oldrow=info(old);assert oldrow[3]=='PENDING' and 'JobHeldUser' in oldrow[-1],oldrow
command(['scancel',old]);record['status']='OLD_PENDING_CANCELLED_NEW_HELD';atomic(request,record)
previous=B/'submission.json'
if previous.exists():previous.rename(B/('submission-'+old+'.json'))
receipt=dict(job_id=new,time=time.time(),code_root=root,source_commit=publication['commit'],partition='emergency_acd',gpus=4,replaces_pending_job=old)
atomic(B/'submission.json',receipt)
t=B/'job.id.tmp';t.write_text(new+'\n');t.replace(B/'job.id')
command(['scontrol','release',new]);record.update(status='HIGHER_PRIORITY_REPLACEMENT_RELEASED',finished_at=time.time());atomic(request,record)
print(json.dumps(record))
print(command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%P|%R']))
print(command(['squeue','--start','-h','-j',new,'-o','%i|%S|%R']))
