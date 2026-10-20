"""Bounded peer8192 operator: actual-data gates, immutable code, normal1H100.

No import/self-test action touches a remote surface. `advance` is one finite
reconciliation step, not a watcher. It waits on missing complete receipts,
starts CPU preparation once, then may submit with explicit authorization.
Only exact saved-cursor timeout/preemption continuations are automatic. Unknown
submission outcomes, cancelled jobs, later unsaved updates, and code failures
are preserved for diagnosis instead of being retried/reset.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
TARGET=PARENT+'/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1'
JOB_NAME='q7b-peer8192-g10-hf-fit'
PACKAGE='reproduction.operations.qwen7b_warmup'
SEEDS=tuple('reproduction/operations/qwen7b_warmup/'+n for n in
    ('peer_newseed_fit_operation.py','peer_newseed_train.py','test_peer_newseed_train.py',
     'test_deploy_peer_newseed_train.py','runtime.sh'))


def source_closure(repo):
    result=set();pending=list(SEEDS)
    while pending:
        name=pending.pop()
        if name in result:continue
        path=repo/name
        if not path.is_file():raise ValueError('Missing actual local source dependency '+name)
        result.add(name)
        if path.suffix!='.py':continue
        package=name[:-3].split('/')[:-1]
        for node in ast.walk(ast.parse(path.read_text(),filename=name)):
            modules=[]
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    prefix=package[:len(package)-node.level+1]
                    modules=['.'.join(prefix+[node.module])] if node.module else ['.'.join(prefix+[v.name]) for v in node.names]
                elif node.module and node.module.startswith('reproduction.'):modules=[node.module]
            elif isinstance(node,ast.Import):modules=[v.name for v in node.names if v.name.startswith('reproduction.')]
            for module in modules:
                child=module.replace('.','/')+'.py'
                if not (repo/child).is_file():raise ValueError('Unresolved runtime dependency '+module)
                pending.append(child)
    return tuple(sorted(result))


def published_bundle(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
    if remote!=commit:raise RuntimeError('Current runtime must first be committed/pushed and remote SHA confirmed')
    data=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=data,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in source_closure(repo):
                payload=(repo/name).read_bytes()
                if payload!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):
                    raise RuntimeError('Unpublished dependency '+name)
                files[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
                member=tarfile.TarInfo(name);member.size=len(payload);member.mode=0o444;member.mtime=0
                archive.addfile(member,io.BytesIO(payload))
    return commit,data.getvalue(),files


COMMON=r'''
import pathlib,json,hashlib,base64,os,fcntl,time,subprocess,tarfile,io,sys
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);assert str(b)==inputs['target'] and b.resolve()==b and b.is_absolute()
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
 with temp.open('w') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 temp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def verify(r,bulk=True):
 p=pathlib.Path(r['path']);assert p.is_absolute() and p.resolve()==p and p.is_file()
 if 'bytes' in r:assert p.stat().st_size==r['bytes']
 if bulk:assert sha(p)==r['sha256'],str(p)
 return p
def lock():
 b.mkdir(parents=True,exist_ok=True);h=(b/'deployment.lock').open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def command(args,timeout=25):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 if p.returncode and args[0]=='squeue' and 'Invalid job id specified' in p.stderr:return ''
 assert p.returncode==0,(args,p.stdout[-500:],p.stderr[-1500:]);return p.stdout.strip()
def publication():
 p=read(b/'source-publication.json');root=pathlib.Path(p['code_root'])
 assert root.parent==b and root.name=='code-'+p['commit'][:12]
 for name,v in p['files'].items():verify(dict(path=str(root/name),**v))
 return p,root
def ready():
 paths=[inputs['validation_complete']] if inputs.get('baseline_only') else [inputs['raw_complete'],inputs['identity_plan']]+inputs['feature_completes']+[inputs['validation_complete'],inputs['diagnostic_complete']]
 missing=[p for p in paths if not pathlib.Path(p).is_file()]
 if missing:return dict(status='WAITING_ACTUAL_RAW_FOUR_SHARDS_AND_NEW_HF_VALIDATION',missing=missing,training_submitted=False)
 if not inputs.get('baseline_only'):
  assert len(inputs['feature_completes'])==4 and len(set(inputs['feature_completes']))==4
  assert read(inputs['raw_complete'])['status']=='RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'
  assert all(read(p)['status']=='PEER_RAW_G10_FROZEN_HF_MULTILAYER_CAUSAL_REPLAY_SHARD_COMPLETE' for p in inputs['feature_completes'])
 assert read(inputs['validation_complete'])['status']=='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE'
 return dict(status='ACTUAL_COMPLETION_FILES_PRESENT_FULL_METADATA_GATE_PENDING',training_submitted=False)
def prepared():
 pub,root=publication();v=read(b/'preparation.json')
 assert v['status']=='PEER_NQ8192_G10_HF_FIT_PREPARED_ACTUAL_CPU_TESTS_PASSED' and v['commit']==pub['commit'] and v['code_root']==str(root)
 for key in ['config','sbatch','training_input','source_plan','cpu_tests']:verify(v[key])
 return v,pub,root
def cpu_env(root):return dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
def runtime(root):return str(root/'reproduction/operations/qwen7b_warmup/runtime.sh')
'''

PROBE=r'''
v=ready();v['observed_epoch']=time.time();print(json.dumps(v))
'''

INIT=r'''
v=ready();assert not v.get('missing'),v
guard=lock()
if (b/'preparation.json').exists():
 p,pub,root=prepared();print(json.dumps(dict(status='EXISTING_PEER8192_PREPARATION_VERIFIED',commit=pub['commit'])))
else:
 own={k:inputs[k] for k in ['commit','bundle_sha256','bundle_bytes','chunks','files']}
 if (b/'upload-controller.json').exists():assert read(b/'upload-controller.json')==own,'Different immutable runtime already staged'
 else:atomic(b/'upload-controller.json',own)
 (b/('upload-'+own['bundle_sha256'][:16])).mkdir(exist_ok=True)
 print(json.dumps(dict(status='PEER8192_UPLOAD_BOUND',chunks=own['chunks'])))
'''

CHUNK=r'''
guard=lock();own=read(b/'upload-controller.json');assert own['commit']==inputs['commit'] and own['bundle_sha256']==inputs['bundle_sha256']
index=inputs['chunk_index'];assert 0<=index<own['chunks'];payload=base64.b64decode(inputs['chunk_data'])
assert len(payload)<=4096 and hashlib.sha256(payload).hexdigest()==inputs['chunk_sha256']
p=b/('upload-'+own['bundle_sha256'][:16])/('part-%05d'%index)
if p.exists():assert p.read_bytes()==payload
else:
 with p.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='PEER8192_CODE_CHUNK_SHA_PASSED',index=index,bytes=len(payload))))
'''

EXTRACT=r'''
guard=lock();own=read(b/'upload-controller.json');parts=sorted((b/('upload-'+own['bundle_sha256'][:16])).glob('part-*'))
assert len(parts)==own['chunks'];payload=b''.join(p.read_bytes() for p in parts)
assert len(payload)==own['bundle_bytes'] and hashlib.sha256(payload).hexdigest()==own['bundle_sha256']
code=b/('code-'+own['commit'][:12]);stage=code.with_name(code.name+'.staging')
if not code.exists():
 stage.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
  members=archive.getmembers();assert {m.name for m in members}==set(own['files'])
  for m in members:
   rel=pathlib.Path(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
   data=archive.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==own['files'][m.name]['sha256']
   p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==data
   else:
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   p.chmod(0o444)
 assert {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}==set(own['files'])
 stage.rename(code)
pub=dict(commit=own['commit'],remote_sha=own['commit'],code_root=str(code),files=own['files'],bundle_sha256=own['bundle_sha256'],active_sources_unchanged=True)
if (b/'source-publication.json').exists():assert read(b/'source-publication.json')==pub
else:atomic(b/'source-publication.json',pub)
publication();print(json.dumps(dict(status='PEER8192_IMMUTABLE_RUNTIME_READY',code_root=str(code),files=len(own['files']))))
'''

CPU_START=r'''
guard=lock();v=ready();assert not v.get('missing'),v
baseline=inputs.get('baseline_only',False);prepname='E4-baseline-preparation.json' if baseline else 'preparation.json'
if (b/prepname).exists():
 if baseline:
  p=read(b/prepname);verify(p['config']);verify(p['sbatch']);print(json.dumps(dict(status='PEER_E4_BASELINE_PREPARED',config=p['config'])))
 else:
  p,pub,root=prepared();print(json.dumps(dict(status='PEER8192_PREPARED',config=p['config'])))
else:
 pub,root=publication();prefix='E4-' if baseline else '';controller=b/(prefix+'cpu-prepare-controller.json')
 if controller.exists():
  old=read(controller);pid=old['pid'];active=False
  try:
   cmd=pathlib.Path('/proc/%d/cmdline'%pid).read_bytes().decode(errors='replace');active='peer_newseed_fit_operation' in cmd and '--action' in cmd and 'prepare' in cmd and str(b) in cmd
  except OSError:pass
  assert active,'Prior CPU preparation exited; inspect log/failure before any retry'
  print(json.dumps(dict(status='EXISTING_PEER8192_CPU_PREPARATION_RUNNING',pid=pid,baseline_only=baseline)))
 else:
  (b/'logs').mkdir(exist_ok=True)
  tests=['bash',runtime(root),'-m','unittest','reproduction.operations.qwen7b_warmup.test_peer_newseed_train','reproduction.operations.qwen7b_warmup.test_deploy_peer_newseed_train']
  with (b/'actual-cpu-tests.log').open('w') as f:r=subprocess.run(tests,env=cpu_env(root),stdout=f,stderr=subprocess.STDOUT,timeout=120)
  assert r.returncode==0,'Actual immutable CPU tests failed; inspect actual-cpu-tests.log'
  args=['bash',runtime(root),'-u','-m','reproduction.operations.qwen7b_warmup.peer_newseed_fit_operation','--action','prepare-baseline' if baseline else 'prepare','--base',str(b),'--validation-complete',inputs['validation_complete']]
  if not baseline:
   args+=['--raw-complete',inputs['raw_complete'],'--identity-plan',inputs['identity_plan'],'--diagnostic-complete',inputs['diagnostic_complete']]
   for p in inputs['feature_completes']:args+=['--feature-complete',p]
  with (b/(prefix+'cpu-prepare.log')).open('ab') as log:p=subprocess.Popen(args,env=cpu_env(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  atomic(controller,dict(status='ACTUAL_METADATA_PREPARATION_STARTED',pid=p.pid,commit=pub['commit'],args=args,time=time.time(),checkpoint_tensors_loaded=False,baseline_only=baseline))
  print(json.dumps(dict(status='PEER8192_CPU_PREPARATION_STARTED',pid=p.pid,baseline_only=baseline)))
'''

STATUS=r'''
result=dict(base=str(b),observed_epoch=time.time(),exists=b.exists())
if 'validation_complete' in inputs:result['inputs']=ready()
if (b/'source-publication.json').exists():
 pub,root=publication();result['publication']=dict(commit=pub['commit'],code_root=str(root))
for prefix in ('','E4-'):
 if (b/(prefix+'cpu-prepare-controller.json')).exists():
  own=read(b/(prefix+'cpu-prepare-controller.json'));pid=own['pid'];active=False
  try:
   cmd=pathlib.Path('/proc/%d/cmdline'%pid).read_bytes().decode(errors='replace');active='peer_newseed_fit_operation' in cmd and str(b) in cmd
  except OSError:pass
  done=(b/('E4-baseline-preparation.json' if prefix else 'preparation.json')).exists()
  result[prefix+'cpu_preparation']=dict(pid=pid,identity_alive=active,complete=done)
  if not active and not done:result[prefix+'cpu_prepare_error_tail']=(b/(prefix+'cpu-prepare.log')).read_text()[-2500:]
if (b/'E4-baseline-preparation.json').exists():
 p=read(b/'E4-baseline-preparation.json');verify(p['config']);verify(p['sbatch']);result['E4_baseline_preparation']=p
if (b/'preparation.json').exists():
 p,pub,root=prepared()
 result['preparation']={k:p[k] for k in ['status','commit','config','questions','updates','global_questions','source_adam_steps','domain_scope']}
for filename in ['controller.json','progress.json','next-update-replay.json','interrupted.json','failure.json','gpu-memory.json','latest.json']:
 if (b/filename).exists():
  v=read(b/filename);keys=['status','cursor','steps','time','message','type','source_steps','candidate_steps','expected','actual','peak_allocated_bytes','exit_normally','sha256','bytes','path','config_sha256']
  result[filename]={k:v[k] for k in keys if k in v}
for filename in ['E4-same-HF-validation/progress.json','E4-same-HF-validation/complete.json']:
 if (b/filename).exists():
  v=read(b/filename);keys=['status','completed_questions','expected_questions','elapsed_seconds','metrics','predictions']
  result[filename]={k:v[k] for k in keys if k in v}
if (b/'E4-baseline-failure.json').exists():result['E4_baseline_failure']=read(b/'E4-baseline-failure.json')
if (b/'complete.json').exists():
 v=read(b/'complete.json');verify(v['checkpoint'],bulk=False)
 result['complete']={k:v[k] for k in ['status','cursor','steps','metrics']}
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();assert job.isdigit();result['job_id']=job
 result['squeue']=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R'])
 result['estimated_start']=command(['squeue','--start','-h','-j',job,'-o','%i|%T|%S|%R'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName%40,State,ExitCode,Elapsed'])
if (b/'E4-baseline.job.id').exists():
 job=(b/'E4-baseline.job.id').read_text().strip();assert job.isdigit();result['E4_baseline_job_id']=job
 result['E4_baseline_squeue']=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R'])
 result['E4_baseline_sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed'])
result['same_name_jobs']=command(['squeue','-h','-u','mou176','-n',inputs['job_name'],'-o','%i|%j|%T|%R'])
print(json.dumps(result))
'''

SUBMIT=r'''
guard=lock();done,pub,root=prepared();assert inputs['submit_authorized'] is True
baseline=b/'E4-same-HF-validation/complete.json'
if not baseline.exists():
 print(json.dumps(dict(status='WAITING_SEPARATE_E4_SAME_HF_BASELINE',training_submitted=False)));sys.exit(0)
baseline=read(baseline);bp=read(b/'E4-baseline-preparation.json');assert baseline['config_sha256']==bp['config']['sha256'];verify(baseline['predictions'])
same=command(['squeue','-h','-u','mou176','-n',inputs['job_name'],'-o','%i|%j|%T|%R'])
if same:
 if 'JobHeldUser' in same and (b/'job.id').exists():
  job=(b/'job.id').read_text().strip();owns=[p for p in (b/'submissions').glob('attempt-*') if (p/'held.json').exists() and read(p/'held.json')['job_id']==job]
  if len(owns)==1 and same.split('|')[0]==job:
   held=read(owns[0]/'held.json');assert held['config']==done['config'];command(['scontrol','release',job]);atomic(owns[0]/'released.json',dict(held,status='PEER8192_KNOWN_HELD_SUBMISSION_RECONCILED',time=time.time()))
 print(json.dumps(dict(status='EXISTING_OWN_PEER8192_JOB_PRESERVED',queue=same,training_submitted=False)));sys.exit(0)
trainlock=(b/'peer8192.lock').open('a+');fcntl.flock(trainlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
attempts=b/'submissions';attempts.mkdir(exist_ok=True);prior=sorted(attempts.glob('attempt-*'))
assert all((p/'released.json').exists() for p in prior),'Unknown/held submission outcome; reconcile, never duplicate'
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();assert job.isdigit()
 state=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode'])
 rows=[x.split('|') for x in state.splitlines() if x.split('|')[0].strip()==job];assert len(rows)==1,state
 state=rows[0][1].strip().split()[0].rstrip('+')
 assert state in ('COMPLETED','TIMEOUT','PREEMPTED','NODE_FAIL'),'Prior cancellation/failure requires diagnosis, not blind auto-retry: '+state
 assert len(prior)<=inputs['max_successors'],'Bounded exact-resume limit reached'
else:
 assert not prior,'Prior submission exists without current job ID'
latest=read(b/'latest.json') if (b/'latest.json').exists() else None
progress=read(b/'progress.json') if (b/'progress.json').exists() else None
intent=read(b/'update-intent.json') if (b/'update-intent.json').exists() else None
complete=read(b/'complete.json') if (b/'complete.json').exists() else None
script="import json;from pathlib import Path;from reproduction.operations.qwen7b_warmup.peer_newseed_fit_operation import resume_decision; b=Path("+repr(str(b))+"); read=lambda n:json.loads((b/n).read_text()) if (b/n).exists() else None; print(json.dumps({'decision':resume_decision(read('latest.json'),read('progress.json'),read('update-intent.json'),read('complete.json'))}))"
r=subprocess.run(['bash',runtime(root),'-c',script],env=cpu_env(root),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
assert r.returncode==0,r.stderr[-2000:];decision=json.loads(r.stdout.splitlines()[-1])['decision']
if decision=='COMPLETE_NO_SUBMISSION':print(json.dumps(dict(status=decision,training_submitted=False)));sys.exit(0)
if latest:
 assert latest['config_sha256']==done['config']['sha256'];verify(latest)
 assert (b/'job.id').exists(),'Existing accepted state requires prior job identity'
if (b/'job.id').exists() and decision=='INITIAL_SOURCE_FORK':
 assert not progress and not intent,'Pre-fit timeout must contain no unresolved fitting update'
attempt=attempts/('attempt-%03d'%len(prior));attempt.mkdir()
v=dict(status='PEER8192_SUBMISSION_INTENT',attempt=len(prior),decision=decision,job_name=inputs['job_name'],config=done['config'],sbatch=done['sbatch'],
 resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='02:00:00'),dependency=None,priority_escalation=False,time=time.time())
atomic(attempt/'intent.json',v)
try:r=subprocess.run(['sbatch','--hold','--parsable',done['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
except subprocess.TimeoutExpired:
 atomic(attempt/'unknown-outcome.json',dict(status='SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_REPEAT',time=time.time()));raise
assert r.returncode==0,r.stderr[-2000:]
job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown submission reply; reconcile before retry'
held=dict(v,status='PEER8192_JOB_HELD',job_id=job);atomic(attempt/'held.json',held)
if (b/'job.id').exists():atomic(attempt/'previous-job.json',dict(job_id=(b/'job.id').read_text().strip()))
temp=b/'job.id.tmp'
with temp.open('w') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
temp.replace(b/'job.id')
queue=command(['squeue','-h','-j',job,'-o','%j|%T|%R']);assert queue.startswith(inputs['job_name']+'|PENDING|') and 'JobHeldUser' in queue,queue
command(['scontrol','release',job]);released=dict(held,status='PEER8192_NORMAL_JOB_RELEASED',time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
atomic(attempt/'released.json',released);print(json.dumps(released))
'''


BASELINE_SUBMIT=r'''
guard=lock();assert inputs['submit_authorized'] is True
pub,root=publication();done=read(b/'E4-baseline-preparation.json');verify(done['config']);verify(done['sbatch'])
assert done['commit']==pub['commit']
if (b/'E4-same-HF-validation/complete.json').exists():
 v=read(b/'E4-same-HF-validation/complete.json');assert v['config_sha256']==done['config']['sha256'];verify(v['predictions'])
 print(json.dumps(dict(status='SAME_HF_E4_BASELINE_ALREADY_COMPLETE',baseline=v)));sys.exit(0)
name='q7b-peer-E4-same-HF-val';same=command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%j|%T|%R'])
if same:
 if 'JobHeldUser' in same and (b/'E4-baseline.job.id').exists():
  job=(b/'E4-baseline.job.id').read_text().strip();owns=[p for p in (b/'E4-baseline-submissions').glob('attempt-*') if (p/'held.json').exists() and read(p/'held.json')['job_id']==job]
  if len(owns)==1 and same.split('|')[0]==job:
   held=read(owns[0]/'held.json');assert held['config']==done['config'];command(['scontrol','release',job]);atomic(owns[0]/'released.json',dict(held,status='KNOWN_E4_BASELINE_HELD_JOB_RECONCILED',time=time.time()))
 print(json.dumps(dict(status='EXISTING_E4_BASELINE_JOB_PRESERVED',queue=same)));sys.exit(0)
guard2=(b/'E4-baseline.lock').open('a+');fcntl.flock(guard2,fcntl.LOCK_EX|fcntl.LOCK_NB)
attempts=b/'E4-baseline-submissions';attempts.mkdir(exist_ok=True);prior=sorted(attempts.glob('attempt-*'))
assert all((p/'released.json').exists() for p in prior),'Unknown E4 submit outcome; reconcile without duplicate'
if (b/'E4-baseline.job.id').exists():
 job=(b/'E4-baseline.job.id').read_text().strip();assert job.isdigit()
 state=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode'])
 rows=[x.split('|') for x in state.splitlines() if x.split('|')[0].strip()==job];assert len(rows)==1
 assert rows[0][1].strip().split()[0].rstrip('+') in ('TIMEOUT','PREEMPTED','NODE_FAIL'),'E4 inference failure/cancellation requires diagnosis'
 assert len(prior)<=inputs['max_successors'],'Bounded E4 evaluation retry limit reached'
else:assert not prior,'Previous E4 submit without job identity'
attempt=attempts/('attempt-%03d'%len(prior));attempt.mkdir()
v=dict(status='SAME_HF_E4_BASELINE_SUBMISSION_INTENT',config=done['config'],sbatch=done['sbatch'],job_name=name,dependency=None,priority_escalation=False,time=time.time())
atomic(attempt/'intent.json',v)
try:r=subprocess.run(['sbatch','--hold','--parsable',done['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
except subprocess.TimeoutExpired:
 atomic(attempt/'unknown-outcome.json',dict(status='OUTCOME_UNKNOWN_NEVER_REPEAT',time=time.time()));raise
assert r.returncode==0,r.stderr[-1500:];job=r.stdout.strip().split(';')[0];assert job.isdigit()
held=dict(v,status='SAME_HF_E4_BASELINE_JOB_HELD',job_id=job);atomic(attempt/'held.json',held)
temp=b/'E4-baseline.job.id.tmp'
with temp.open('w') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
temp.replace(b/'E4-baseline.job.id')
queue=command(['squeue','-h','-j',job,'-o','%j|%T|%R']);assert queue.startswith(name+'|PENDING|') and 'JobHeldUser' in queue,queue
command(['scontrol','release',job]);released=dict(held,status='SAME_HF_E4_BASELINE_NORMAL_JOB_RELEASED',time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%T|%R']))
atomic(attempt/'released.json',released);print(json.dumps(released))
'''


def remote_source(inputs,action):
    return 'inputs='+repr(inputs)+'\n'+COMMON+'\n'+dict(probe=PROBE,init=INIT,chunk=CHUNK,extract=EXTRACT,cpu=CPU_START,status=STATUS,submit=SUBMIT,baseline_submit=BASELINE_SUBMIT)[action]


def self_test():
    repo=Path(__file__).resolve().parents[3];names=source_closure(repo)
    required={'peer_newseed_train.py','peer_newseed_fit_operation.py','probe_g10_train.py','replay_feature_reader.py','replay_probe_layers.py','model.py','runtime.sh','evaluate_frozen_validation.py'}
    if not required<={Path(n).name for n in names}:raise ValueError('Incomplete actual runner import closure')
    values=dict(base=TARGET,target=TARGET,job_name=JOB_NAME,raw_complete='/data/raw.json',identity_plan='/data/identity.json',
        feature_completes=['/data/shard%d/complete.json'%i for i in range(4)],validation_complete='/data/new-fixed-g5/complete.json',diagnostic_complete='/data/diag/complete.json',
        commit='a'*40,bundle_sha256='b'*64,bundle_bytes=123456,chunks=31,files={n:dict(bytes=10000,sha256='c'*64) for n in names},max_successors=2,submit_authorized=True)
    sizes={}
    for action in ('probe','init','chunk','extract','cpu','status','submit','baseline_submit'):
        own=dict(values)
        if action!='init':own.pop('files')
        if action=='chunk':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64)
        source=remote_source(own,action);compile(source,'remote-'+action,'exec')
        sizes[action]=len(('python3 -c '+shlex.quote(source)).encode())
        if sizes[action]>=16000:raise ValueError('Control command must stay below16KB: '+action)
    return dict(status='PEER8192_DEPLOYMENT_LOCAL_SMOKE_PASSED',files=len(names),command_bytes=sizes,
                remote_actions_executed=False,GPU_submitted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-root',type=Path)
    p.add_argument('--action',choices=('prepare','status','submit','advance','prepare-baseline','submit-baseline','advance-baseline'),default='status');p.add_argument('--submit-authorized',action='store_true')
    p.add_argument('--raw-complete');p.add_argument('--identity-plan');p.add_argument('--feature-complete',action='append')
    p.add_argument('--validation-complete');p.add_argument('--diagnostic-complete',default=PARENT+'/replay-matched9088-bank-v2-20261021/prepared/bank-complete.json')
    p.add_argument('--max-successors',type=int,default=2);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test(),sort_keys=True));return
    if a.control_root is None:p.error('--control-root is required')
    if not 0<=a.max_successors<=2:p.error('--max-successors must be0..2; no unbounded automatic retries')
    baseline_only=a.action in ('prepare-baseline','submit-baseline','advance-baseline')
    if a.action in ('submit','advance','submit-baseline','advance-baseline') and not a.submit_authorized:p.error('Explicit --submit-authorized required')
    need_inputs=a.action in ('prepare','advance')
    if need_inputs and (not a.raw_complete or not a.identity_plan or len(a.feature_complete or [])!=4 or not a.validation_complete):
        p.error('Actual --raw-complete, --identity-plan, four --feature-complete, --validation-complete required')
    if baseline_only and not a.validation_complete:p.error('--validation-complete actual path required')
    values=dict(base=TARGET,target=TARGET,job_name=JOB_NAME,submit_authorized=a.submit_authorized,max_successors=a.max_successors,baseline_only=baseline_only)
    if baseline_only:values['validation_complete']=a.validation_complete
    if a.raw_complete:values.update(raw_complete=a.raw_complete,identity_plan=a.identity_plan,feature_completes=a.feature_complete,
        validation_complete=a.validation_complete,diagnostic_complete=a.diagnostic_complete)
    spec=importlib.util.spec_from_file_location('peer8192_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        own=dict(values);own.update(extra or {});command='python3 -c '+shlex.quote(remote_source(own,action))
        if len(command.encode())>=16000:raise ValueError('Control payload exceeds16KB')
        r=control.run(command,account='mou176',source_address=address,timeout=180 if action=='cpu' else 55)
        rows=[json.loads(line) for line in r.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Peer8192 action unconfirmed; inspect actual state before another attempt: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-3000:])
        return rows[-1]
    def prepare():
        probe=call('probe')
        if probe.get('missing'):return probe
        state=call('status')
        if state.get('publication'):return call('cpu')
        commit,bundle,files=published_bundle(Path(__file__).resolve().parents[3]);chunks=[bundle[i:i+4096] for i in range(0,len(bundle),4096)]
        common=dict(commit=commit,bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_bytes=len(bundle),chunks=len(chunks))
        result=call('init',dict(common,files=files))
        if result['status']!='EXISTING_PEER8192_PREPARATION_VERIFIED':
            for i,part in enumerate(chunks):call('chunk',dict(common,chunk_index=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=hashlib.sha256(part).hexdigest()))
            call('extract',common)
        return call('cpu')
    if a.action in ('prepare','prepare-baseline'):result=prepare()
    elif a.action in ('advance','advance-baseline'):
        state=call('status')
        if state.get('inputs',{}).get('missing'):result=state
        elif not state.get('E4_baseline_preparation' if baseline_only else 'preparation'):result=prepare()
        else:result=call('baseline_submit' if baseline_only else 'submit')
    else:result=call('baseline_submit' if a.action=='submit-baseline' else a.action)
    print(json.dumps(result,sort_keys=True),flush=True)
    out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('peer8192-fit-'+a.action+'-latest.json')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
