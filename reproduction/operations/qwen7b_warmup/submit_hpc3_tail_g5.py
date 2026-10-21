"""Small guarded submit/reconcile control for exactly two prepared tail jobs.

Generation source/config is never edited. sbatch is held until actual identity
is verified and job.id is durable; unknown outcomes block a second sbatch.
"""
import argparse,hashlib,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3]
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail-newseed-preparation'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1'
GENERATION_COMMIT='2590d0b272013264b6d727a6a38493a790ee1305'
CONFIG_SHA='368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835'
REMOTE=r'''import pathlib,json,hashlib,subprocess,fcntl,os,time,re
b=pathlib.Path(v['base'])
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def verify(ref):
 actual=proof(ref['path']);assert actual['sha256']==ref['sha256']and actual['bytes']==ref['bytes'];return actual
def once(p,x):
 raw=(json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
 if p.exists():assert p.read_bytes()==raw,'Conflicting immutable submission receipt'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def call(args):return subprocess.check_output(args,universal_newlines=True,timeout=20).strip()
def ids(name):
 text=call(['squeue','-h','-u','mou176','-n',name,'-o','%i'])
 return [z for z in text.splitlines()if z.isdigit()]
def inspect(job,name,script):
 text=call(['scontrol','show','job',job,'--oneliner']);f=dict(re.findall(r'(\w+)=([^ ]+)',text))
 assert f['JobName']==name and f['Account']=='mou176'and f['UserId'].startswith('mou176(')
 assert f['Partition']=='acd_u'and f['QOS']=='normal'and f['Command']==script
 assert int(f['NumNodes'])==1 and int(f['NumCPUs'])==16 and int(f.get('NumTasks','1'))==1
 assert 'gres/gpu=2'in f.get('TRES',f.get('ReqTRES',''))or f.get('TresPerNode','')=='gres/gpu:2'
 assert f.get('Dependency','(null)')in ['(null)','None','none']
 return{k:f[k]for k in ['JobId','JobName','Account','UserId','Partition','QOS','JobState','Reason','NumNodes','NumCPUs','NumTasks','TRES','ReqTRES','TresPerNode','Dependency','Command','WorkDir','StdOut']if k in f}
guard=(b/'submission-control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
d=read(b/'prepared.json');verify(d['config']);assert d['config']['sha256']==v['config_sha']and d['source_commit']==v['generation_commit']
pub=read(b/'source-publication.json');assert pub['commit']==v['generation_commit']
for name,item in pub['files'].items():assert proof(pathlib.Path(pub['code_root'])/name)['sha256']==item['sha256']
c=read(d['config']['path']);assert c['potentialOverlap']and c['execution_host_cluster']=='HPC3'
selected=[0,1]if v['worker']is None else[v['worker']];out=[]
for w in selected:
 name='q7b-tail-newseed-w%d'%w;script=verify(d['scripts'][w]);jobfile=b/('worker-%d.job.id'%w);intentfile=b/('worker-%d.submission-intent.json'%w);receiptfile=b/('worker-%d.submission.json'%w)
 current=ids(name);controller=b/'workers'/('worker-%d'%w);lockfree=True
 if(controller/'worker.lock').exists():
  probe=(controller/'worker.lock').open('a+')
  try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lockfree=False
  probe.close()
 row=dict(worker=w,name=name,script=script,config=d['config'],current_same_name_jobs=current,worker_lock_free=lockfree,intent_exists=intentfile.exists(),job_id_exists=jobfile.exists())
 if v['action']=='preflight':out.append(row);continue
 if jobfile.exists():
  job=jobfile.read_text().strip();assert job.isdigit();row.update(status='EXISTING_OWN_JOB_PRESERVED_NO_SBATCH',job_id=job)
  if job in current:
   row['identity']=inspect(job,name,script['path'])
   if v['action']=='reconcile'and row['identity']['JobState']=='PENDING'and row['identity']['Reason']=='JobHeldUser':
    assert receiptfile.exists()and read(receiptfile)['job_id']==job,'Durable own-held receipt required'
    call(['scontrol','release',job]);row['status']='EXISTING_VERIFIED_HELD_JOB_RELEASE_RECONCILED_NO_SBATCH'
  else:row['sacct']=call(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode,Elapsed'])
  out.append(row);continue
 if v['action']=='status':out.append(row);continue
 assert lockfree and not(controller/'controller.json').exists()and not(controller/'complete.json').exists(),'Existing sampling state requires explicit continuation, not new submission'
 if v['action']=='reconcile':
  assert intentfile.exists(),'No unknown submission to reconcile'
  intent=read(intentfile);assert intent['config']==d['config']and intent['script']==script
  assert len(current)==1,'Cannot adopt zero/multiple candidates; preserve unknown intent and inspect sacct'
  job=current[0];identity=inspect(job,name,script['path']);assert identity['JobState']=='PENDING'and identity['Reason']=='JobHeldUser','Unknown existing job is not held; preserve without re-submit'
 else:
  assert v['action']=='submit'and not intentfile.exists()and not current,'Unknown prior intent/job; use reconcile, never blindly repeat sbatch'
  intent=dict(status='IMMUTABLE_TAIL_SUBMISSION_INTENT',worker=w,job_name=name,config=d['config'],script=script,source_publication=proof(b/'source-publication.json'),generation_commit=v['generation_commit'],submission_control_commit=v['control_commit'],submission_control_sha256=v['control_sha'],execution_run_id=c['execution_run_id'],potentialOverlap=True,resources=dict(partition='acd_u',qos='normal',nodes=1,GPUs=2,CPU=16,GiB=128,time='08:00:00',dependency=None),epoch=time.time())
  once(intentfile,intent)
  try:r=subprocess.run(['sbatch','--hold','--parsable','--chdir',str(b),script['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
  except subprocess.TimeoutExpired:
   once(b/('worker-%d.submit-outcome-unknown.json'%w),dict(status='UNKNOWN_DO_NOT_RESUBMIT',intent=proof(intentfile),epoch=time.time()));raise
  if r.returncode:
   once(b/('worker-%d.submit-rejected.json'%w),dict(status='SBATCH_REJECTED_DO_NOT_BLIND_RETRY',exit_code=r.returncode,error=r.stderr[-1200:],intent=proof(intentfile),epoch=time.time()));raise RuntimeError(r.stderr[-1200:])
  job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch response; reconcile before any new call'
  identity=inspect(job,name,script['path']);assert identity['JobState']=='PENDING'and identity['Reason']=='JobHeldUser'
 with jobfile.open('x')as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 once(receiptfile,dict(status='VERIFIED_OWN_HELD_JOB_ID_DURABLE',worker=w,job_id=job,identity=identity,intent=proof(intentfile),epoch=time.time()))
 call(['scontrol','release',job]);row.update(status='SINGLE_OWN_SUBMISSION_RELEASED',job_id=job,identity=identity,queue=call(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),intent=proof(intentfile),submission=proof(receiptfile));out.append(row)
print(json.dumps(dict(status='HPC3_TAIL_'+v['action'].upper(),epoch=time.time(),target=str(b),workers=out,actor_updates=0,source_generation_unchanged=True,frozen124194_unchanged=True),sort_keys=True))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['preflight','submit','status','reconcile'],default='preflight');p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action in ['submit','reconcile']and(not a.submit_authorized or a.worker is None):p.error('Explicit GO --submit-authorized and single --worker required')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
 if commit!=remote:raise ValueError('Exact control entry must be pushed first')
 own=str(Path(__file__).relative_to(R));raw=Path(__file__).read_bytes()
 if subprocess.check_output(['git','show',commit+':'+own],cwd=R)!=raw:raise ValueError('Control source differs from published version')
 spec=importlib.util.spec_from_file_location('tail_deploy',Path(__file__).with_name('deploy_hpc3_tail_g5.py'));transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
 values=dict(base=BASE,config_sha=CONFIG_SHA,generation_commit=GENERATION_COMMIT,control_commit=commit,control_sha=hashlib.sha256(raw).hexdigest(),worker=a.worker,action=a.action)
 source='v='+repr(values)+'\n'+REMOTE;compile(source,'tail-submit-control','exec');result=transport.remote(source,timeout=65)
 (REPORT/('actual-control-'+a.action+(''if a.worker is None else'-worker%d'%a.worker)+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
