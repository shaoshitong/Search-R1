"""Stage, gracefully stop, seal and once-resume the same question5 probe.

The user-authorized global batch changes to 1024 trajectories on the next
complete epoch boundary. The same run, full
raw/EMA/Adam/four-rank RNG and exact sampler cursor continue in a new namespace.
Stage the published archive before stopping. Stop signals the exact native rank0
through a pidfd, never torchrun/a process group/foreign tasks. Seal requires all
native writers naturally gone and validates existing producer receipts plus the
full checkpoint SHA; it never loads a tensor. The accepted parent cache remains
read-only. No download, encoder, teacher, evaluation or extra GPU test is run.

Handle adds resume_parent_handle, resume_parent_handle_sha256, parent_binding
and parent_binding_sha256 to the accepted-cache handle fields. The new base is
<original prefix><same run>-resume-<new execution>. Stage payload has the published
source archive and exactly source-plan.json / resume-parent-deployment.json.
CLI actions are stage, stop, seal, launch, observe; only stage uses --payload.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
PYTHON = '/data/search-g1/env/bin/python'
CONTROL = 'reproduction/operations/a800_question5_global1024_resume_control.py'


def route(code):
    path = Path(__file__).with_name('a800_qprobe_prefix_scale20.py')
    spec = importlib.util.spec_from_file_location('existing_question5_global1024_route', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.route().ssh_command(code)


BINDING = r'''
import csv,hashlib,io,json,os,pathlib,re,signal,stat,subprocess,sys,time
PREFIX='/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-'
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def canonical_path(path):
 path=pathlib.Path(path)
 assert path.is_absolute() and path.resolve()==path,'Noncanonical path: '+str(path)
 return path
assert re.fullmatch('[0-9a-f]{32}',h['run_id']) and re.fullmatch('[0-9a-f]{32}',h['execution_id'])
assert str(b)==PREFIX+h['run_id']+'-resume-'+h['execution_id']
canonical_path(b)
parent_h=h['resume_parent_handle'];parent=canonical_path(parent_h['base'])
assert str(parent)==PREFIX+h['run_id']+'-resume-'+parent_h['execution_id'] and parent_h['run_id']==h['run_id']
assert parent_h['execution_id']!=h['execution_id'] and parent_h['source_commit']!=h['source_commit']
assert re.fullmatch('[0-9a-f]{40}',parent_h['source_commit'])
assert parent_h['fit_config']['global_group_pattern']==[51,51,51,51,52]
assert parent_h['fit_config']['micro_groups']==13 and parent_h['fit_config']['eval_micro_groups']==64
assert h['parent_binding']['global_group_pattern']==[51,51,51,51,52]
assert h['parent_binding']['updates_per_epoch']==8700 and h['parent_binding']['total_actual_adam_updates']==1740000
assert h['parent_binding']['run_id']==h['run_id'] and h['parent_binding']['source_commit']==parent_h['source_commit']
assert digest(h['parent_binding'])==h['parent_binding_sha256']
assert parent_h['parent_cache_base']==h['parent_cache_base'] and parent_h['model_dir']==h['model_dir']
assert parent_h['source_plan_sha256']==h['source_plan_sha256']
assert h['fit_config']==dict(parent_h['fit_config'],global_group_pattern=[204,204,204,204,208],micro_groups=52,eval_micro_groups=64)
for key in ('parent_data_manifest_sha256','parent_manifest_sha256','parent_prepare_complete_sha256'):
 assert h[key]==parent_h[key]
def regular(path,limit=None):
 path=canonical_path(path);s=path.lstat()
 assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid(),'Not an owned regular file: '+str(path)
 assert limit is None or s.st_size<=limit,'Oversized metadata: '+str(path)
 return s
def read(path):
 regular(path,64*1024**2)
 return json.loads(pathlib.Path(path).read_bytes())
def sha(path):
 regular(path);d=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):d.update(x)
 return d.hexdigest()
def write(path,value,exclusive=False):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 target=path if exclusive else path.with_name(path.name+'.partial')
 fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|(os.O_EXCL if exclusive else os.O_TRUNC),0o600)
 with os.fdopen(fd,'wb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
 if not exclusive:os.replace(target,path)
 directory=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(directory)
 finally:os.close(directory)
def birth(pid):
 q=pathlib.Path('/proc')/str(pid);s=(q/'stat').read_text();v=s[s.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=q.stat().st_uid)
def alive(x):
 try:
  s=(pathlib.Path('/proc')/str(x['pid'])/'stat').read_text()
  return birth(x['pid'])==x and s[s.rfind(')')+2:].split()[0] not in ('Z','X')
 except (FileNotFoundError,ProcessLookupError):return False
def native_identity(value):
 return dict(pid=value['pid'],start_ticks=value['birth'],sid=value['sid'],uid=value['uid'])
def bound_processes(base,exclude=()):
 root=str(base);found=[]
 for q in pathlib.Path('/proc').iterdir():
  if not q.name.isdigit() or int(q.name) in exclude:continue
  try:
   if q.stat().st_uid!=os.getuid():continue
   s=(q/'stat').read_text()
   if s[s.rfind(')')+2:].split()[0] in ('Z','X'):continue
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  args=[];cwd=None
  try:args=(q/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
  try:cwd=str((q/'cwd').resolve())
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
  matched=(cwd is not None and (cwd==root or cwd.startswith(root+'/'))) or any(a==root or a.startswith(root+'/') or ('='+root+'/') in a for a in args)
  try:
   if matched:found.append(birth(int(q.name)))
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 return found
def project_trainers():
 found=[]
 for q in pathlib.Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:
   if q.stat().st_uid!=os.getuid():continue
   args=(q/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if 'reproduction.question5_ema_warmup.train' in args:
    item=birth(int(q.name))
    if alive(item):found.append(item)
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 return found
def source():
 assert read(b/'preparation/deployment.json')==h
 expected=set(h['source_files']);actual=set()
 for q in repo.rglob('*'):
  if q.is_dir():assert not q.is_symlink() and q.resolve()==q
  else:regular(q);actual.add(str(q.relative_to(repo)))
 assert actual==expected,'Archive source inventory differs from published map'
 for name,d in h['source_files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
  assert sha(repo/name)==d,'Deployed source mismatch: '+name
 assert sha(b/'preparation/source-plan.json')==h['source_plan_sha256']
 assert sha(b/'preparation/resume-parent-deployment.json')==h['resume_parent_handle_sha256']
 assert read(b/'preparation/resume-parent-deployment.json')==parent_h
 assert sha(parent/'preparation/deployment.json')==h['resume_parent_handle_sha256']
 assert read(parent/'preparation/deployment.json')==parent_h
 assert read(parent/'train/run-claim.json')['binding']==h['parent_binding']
 assert read(parent/'train/run-claim.json')['binding_sha256']==h['parent_binding_sha256']
 cache=canonical_path(h['parent_cache_base'])
 for name,key in (('data/composite-manifest.json','parent_data_manifest_sha256'),('features/manifest.json','parent_manifest_sha256'),('data/PREPARE-COMPLETE.json','parent_prepare_complete_sha256')):
  assert sha(cache/name)==h[key]
def counter(value):
 keys=('optimizer_steps','ema_updates','complete_epochs','global_groups_seen','global_trajectories_seen','rejected_nonfinite_updates')
 assert isinstance(value,dict) and all(type(value.get(key)) is int and value[key]>=0 for key in keys)
 assert value['optimizer_steps']==value['ema_updates'] and value['global_trajectories_seen']==5*value['global_groups_seen']
 return {key:value[key] for key in keys}
def parent_native(rank,name):
 path=parent/'train/executions'/parent_h['execution_id']/(name+'-rank-%d.json'%rank)
 value=read(path);identity=value.get('identity',value)
 assert identity['rank']==rank and identity['local_rank']==rank and identity['world_size']==4
 assert all(identity[key]==parent_h[key] for key in ('run_id','execution_id','source_commit'))
 assert identity['uid']==os.getuid() and identity['sid']==identity['pid']
 return value,dict(path=str(path),sha256=sha(path))
def graceful_signal(identity):
 assert identity['uid']==os.getuid() and identity['sid']==identity['pid'] and alive(identity)
 # The deployed Python can predate the standard-library pidfd wrappers.
 # Linux x86_64/aarch64 expose the same stable pidfd syscall numbers; keep
 # descriptor-based signalling and the post-open birth check in that case.
 import ctypes
 libc=ctypes.CDLL(None,use_errno=True)
 def pidfd_syscall(number,*arguments):
  assert sys.platform.startswith('linux') and os.uname().machine in ('x86_64','aarch64')
  result=libc.syscall(ctypes.c_long(number),*arguments)
  if result<0:
   error=ctypes.get_errno();raise OSError(error,os.strerror(error))
  return result
 descriptor=(os.pidfd_open(identity['pid']) if hasattr(os,'pidfd_open') else
  pidfd_syscall(434,ctypes.c_int(identity['pid']),ctypes.c_uint(0)))
 try:
  assert alive(identity),'Rank0 birth changed before signal'
  if hasattr(signal,'pidfd_send_signal'):signal.pidfd_send_signal(descriptor,signal.SIGTERM)
  else:pidfd_syscall(424,ctypes.c_int(descriptor),ctypes.c_int(signal.SIGTERM),ctypes.c_void_p(),ctypes.c_uint(0))
 finally:os.close(descriptor)
def qualification():
 path=b/'preparation/resume-qualification.json';q=read(path)
 assert q['schema_version']==1 and q['role']=='question5_native_stopped_global_batch_resume_v1'
 assert all(q[key]==h[key] for key in ('run_id',))
 assert q['parent_execution_id']==parent_h['execution_id'] and q['parent_source_commit']==parent_h['source_commit']
 assert q['target_execution_id']==h['execution_id'] and q['target_source_commit']==h['source_commit']
 assert q['target_output_dir']==str(b/'train') and q['parent_binding_sha256']==h['parent_binding_sha256']
 assert q['legacy_through_epoch']==(q['retained_counters']['complete_epochs'] if q['completed_epoch'] else q['retained_epoch'])
 assert q['target_global_group_pattern']==[204,204,204,204,208] and q['target_updates_per_epoch']==2175
 assert q['target_total_actual_adam_updates']==q['legacy_through_epoch']*8700+(200-q['legacy_through_epoch'])*2175
 assert q['all_owned_training_processes_gone'] is True and q['checkpoint_writer_gone'] is True and q['old_output_preserved'] is True
 assert q['checkpoint']['path']==str(parent/'train/latest.pt')
 assert not bound_processes(parent) and not project_trainers(),'An owned old trainer/writer remains live'
 latest=parent/'train/latest.json'
 assert sha(latest)==q['parent_latest_receipt_sha256'] and read(latest)==q['parent_latest_receipt']
 for row in q['rank_terminal_receipts']:
  for key in ('exit','stopped'):
   if key in row:
    proof=row[key];assert sha(proof['path'])==proof['sha256']
    value=read(proof['path']);assert counter(value['counters'])==q['retained_counters']
    assert not alive(native_identity(value['identity']))
 path=pathlib.Path(q['checkpoint']['path']);before=regular(path)
 assert before.st_size==q['checkpoint']['bytes'] and sha(path)==q['checkpoint']['sha256']
 after=regular(path)
 assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 return q
'''

COMMON = "import json,pathlib,sys\np=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'\n" + BINDING

STAGE = COMMON + r'''
import base64,tarfile
assert not b.exists(),'Fresh performance namespace already exists'
assert set(p['side_files'])=={'source-plan.json','resume-parent-deployment.json'}
archive=base64.b64decode(p['source_archive_b64'],validate=True)
assert hashlib.sha256(archive).hexdigest()==h['source_archive_sha256']
b.mkdir(mode=0o700);(b/'logs').mkdir();(b/'preparation').mkdir();repo.mkdir()
seen=set()
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:*') as tar:
 for member in tar:
  rel=pathlib.PurePosixPath(member.name);path=repo/member.name
  assert not rel.is_absolute() and '..' not in rel.parts and path.resolve().is_relative_to(repo)
  assert member.name not in seen and not member.issym() and not member.islnk();seen.add(member.name)
  if member.isdir():path.mkdir(parents=True,exist_ok=True)
  else:
   assert member.isfile() and member.size<=64*1024**2 and member.name in h['source_files']
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as f:f.write(tar.extractfile(member).read());f.flush();os.fsync(f.fileno())
for name,value in p['side_files'].items():
 raw=base64.b64decode(value,validate=True);assert len(raw)<=64*1024**2
 with (b/'preparation'/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
write(b/'preparation/deployment.json',h,True);source()
v=dict(stage='QUESTION5_EXACT_SOURCE_ARCHIVE_STAGED',checked_unix=time.time(),source_commit=h['source_commit'],
 source_tree=h['source_tree'],source_files_verified=len(h['source_files']),source_archive_sha256=h['source_archive_sha256'],
 base=str(b),teacher_or_actor_started=False,training_started=False,parent_stopped=False)
write(b/'preparation/source-receipt.json',v,True);print(json.dumps(v))
'''

STOP = COMMON + r'''
source();assert not (b/'preparation/parent-stop-request.json').exists(),'Native stop was already requested; inspect its receipt'
initialization,init_proof=parent_native(0,'initialization')
identity=native_identity(initialization);assert alive(identity),'Rank0 not live; do not infer terminal state'
progress,progress_proof=parent_native(0,'progress')
assert native_identity(progress['identity'])==identity
v=dict(role='question5_exact_native_rank0_graceful_stop',run_id=h['run_id'],target_execution_id=h['execution_id'],
 parent_execution_id=parent_h['execution_id'],parent_source_commit=parent_h['source_commit'],process=identity,
 initialization=init_proof,progress=progress_proof,signals_requested=1,signal='SIGTERM',requested_unix=time.time(),
 process_groups_signalled=False,foreign_tasks_signalled=False,partial_updates_will_be_saved=True)
write(b/'preparation/parent-stop-request.json',v,True)
graceful_signal(identity)
v.update(signals_sent=1,delivered_unix=time.time(),native_checkpoint_not_yet_proven=True)
write(b/'preparation/parent-stop-delivered.json',v,True);print(json.dumps(v))
'''

SEAL = COMMON + r'''
source();assert (b/'preparation/parent-stop-delivered.json').exists(),'Native graceful stop has not been delivered'
assert not (b/'preparation/resume-qualification.json').exists(),'Full-state boundary already sealed'
remaining=bound_processes(parent);trainers=project_trainers()
assert not remaining and not trainers,'Old native writer or owned training process still lives'
status=read(parent/'logs/status.json');wrapper=read(parent/'logs/launch.json');exit=read(parent/'logs/exit.json')
assert not alive(wrapper['process']) and exit['exit_code']==0
assert all(not alive(job['process']) and job['exit_code']==0 for job in status['jobs'])
latest=read(parent/'train/latest.json');assert all(latest[key]==parent_h[key] for key in ('run_id','execution_id','source_commit'))
assert latest['binding_sha256']==h['parent_binding_sha256'] and latest['path']==str(parent/'train/latest.pt')
counters=counter(latest['counters']);n=counters['optimizer_steps']
assert 0<n<1740000 and latest['actual_adam_steps_per_rank']==[n]*4 and latest['ema_updates']==n
assert counters['complete_epochs']<200 and latest['actor_updates']==0
cursor=latest['next_update_in_epoch'];epoch=latest['epoch']
assert type(cursor) is int and 0<=cursor<=8700
if latest['completed_epoch']:
 assert cursor==0 and counters['complete_epochs']==epoch and n==epoch*8700
 directory=parent/'train/epochs'/('epoch-%04d'%epoch)
 metrics=read(directory/'metrics.json');cp=read(directory/'checkpoint.json');complete=read(directory/'complete.json')
 assert all(metrics[key]==parent_h[key] for key in ('run_id','execution_id','source_commit'))
 assert metrics['epoch']==epoch and counter(metrics['counters'])==counters
 assert digest(metrics)==latest['epoch_metrics_sha256'] and cp==latest
 assert complete['status']=='FULL_MERGED_EPOCH_COMPLETE' and complete['metrics']==metrics and complete['checkpoint']==latest
else:
 assert counters['complete_epochs']==epoch-1 and n==(epoch-1)*8700+cursor
rank_receipts=[]
for rank in range(4):
 value,proof=parent_native(rank,'exit');assert not alive(native_identity(value['identity']))
 assert counter(value['counters'])==counters and value['epoch']==epoch
 terminal='STOPPED_AT_COMPLETE_EPOCH_BOUNDARY' if latest['completed_epoch'] else 'STOPPED_AT_COMPLETE_ADAM_BOUNDARY'
 assert value['terminal']==terminal
 row=dict(rank=rank,exit=proof,terminal=value['terminal'])
 if not latest['completed_epoch']:
  stopped,stopped_proof=parent_native(rank,'stopped')
  assert counter(stopped['counters'])==counters and stopped['checkpoint']==latest
  row['stopped']=stopped_proof
 rank_receipts.append(row)
checkpoint=parent/'train/latest.pt';before=regular(checkpoint)
assert before.st_size==latest['bytes'] and not checkpoint.with_name(checkpoint.name+'.partial').exists()
fullsha=sha(checkpoint);assert fullsha==latest['sha256']
after=regular(checkpoint)
assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
assert not bound_processes(parent) and not project_trainers()
v=dict(schema_version=1,role='question5_native_stopped_global_batch_resume_v1',run_id=h['run_id'],
 parent_execution_id=parent_h['execution_id'],parent_source_commit=parent_h['source_commit'],target_execution_id=h['execution_id'],
 target_source_commit=h['source_commit'],target_output_dir=str(b/'train'),parent_base=str(parent),
 parent_binding_sha256=h['parent_binding_sha256'],checkpoint=dict(path=str(checkpoint),sha256=fullsha,bytes=before.st_size),
 parent_latest_receipt=latest,parent_latest_receipt_sha256=sha(parent/'train/latest.json'),rank_terminal_receipts=rank_receipts,
 retained_counters=counters,retained_epoch=epoch,retained_next_update_in_epoch=cursor,completed_epoch=latest['completed_epoch'],
 legacy_through_epoch=(counters['complete_epochs'] if latest['completed_epoch'] else epoch),
 target_global_group_pattern=[204,204,204,204,208],target_updates_per_epoch=2175,
 target_total_actual_adam_updates=(counters['complete_epochs'] if latest['completed_epoch'] else epoch)*8700+(200-(counters['complete_epochs'] if latest['completed_epoch'] else epoch))*2175,
 all_owned_training_processes_gone=True,checkpoint_writer_gone=True,old_output_preserved=True,
 actual_native_stop_exit_zero=True,checkpoint_tensor_payload_read=False,raw_ema_adam_rng_cursor_resume_required=True,
 parent_cache_base=h['parent_cache_base'],parent_feature_manifest_sha256=h['parent_manifest_sha256'],checked_unix=time.time())
write(b/'preparation/resume-qualification.json',v,True)
print(json.dumps(dict(qualification=v,qualification_sha256=sha(b/'preparation/resume-qualification.json'))))
'''

LAUNCH = COMMON + r'''
source();q=qualification();assert not (b/'logs/launch-once.json').exists() and not (b/'logs/launch.json').exists()
assert not bound_processes(b),'New namespace already has a process'
write(b/'logs/full-state-resume-acceptance.json',q,True)
write(b/'logs/launch-once.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],claimed_unix=time.time(),
 state_resume=True,parent_execution_id=parent_h['execution_id'],checkpoint_sha256=q['checkpoint']['sha256']),True)
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
 OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(b/'cache/hf'),XDG_CACHE_HOME=str(b/'cache'),
 TMPDIR=str(b/'tmp'),TRITON_CACHE_DIR=str(b/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(b/'cache/inductor'))
for name in ('tmp','cache/hf','cache/triton','cache/inductor'):(b/name).mkdir(parents=True,exist_ok=True)
for key in tuple(env):
 if key.startswith('SLURM_') or key in ('RAY_ADDRESS','PYTHONHOME','LD_LIBRARY_PATH'):env.pop(key,None)
command=['/data/search-g1/env/bin/python',str(repo/'reproduction/operations/a800_question5_global1024_resume_control.py'),
 '--worker','--handle',str(b/'preparation/deployment.json')]
with (b/'logs/wrapper.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(stage='QUESTION5_FULL_STATE_GLOBAL1024_RESUME_LAUNCHED_ONCE',process=birth(child.pid),run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],checked_unix=time.time(),training_progress_proven=False,state_resume=True,
 inherited_counters=q['retained_counters'],checkpoint_sha256=q['checkpoint']['sha256'])
write(b/'logs/launch.json',v,True);print(json.dumps(v))
'''

OBSERVE = COMMON + r'''
source();docs={}
for name in ('logs/launch.json','logs/status.json','logs/exit.json','logs/failure.json','logs/gpu-binding.json',
 'logs/full-state-resume-acceptance.json','preparation/resume-qualification.json','preparation/parent-stop-delivered.json',
 'train/status.json','train/complete.json','train/failure.json','train/latest-metrics.json','train/latest.json','train/run-claim.json'):
 q=b/name
 if q.exists():regular(q,256*1024);docs[name]=read(q)
launch=docs.get('logs/launch.json');docs['wrapper_alive']=alive(launch['process']) if launch else None
jobs=[dict(job,alive=alive(job['process'])) for job in docs.get('logs/status.json',{}).get('jobs',[])]
progress=[]
for q in sorted((b/'train').glob('**/progress-rank-*.json'))[-8:]:
 regular(q,64*1024);progress.append(dict(path=str(q.relative_to(b)),value=read(q)))
errors=[]
for q in sorted((b/'logs').glob('*.log')):
 regular(q)
 with q.open('rb') as f:f.seek(max(0,q.stat().st_size-16384));tail=f.read(16384).decode(errors='replace')
 classes=re.findall(r'^\s*((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception))(?::|$)',tail,re.M)
 frames=[dict(file=pathlib.Path(name).name,line=int(line)) for name,line in re.findall(r'File "([^"\n]+)", line (\d+)',tail)]
 if classes:errors.append(dict(log=q.name,error_classes=classes[-3:],frames=frames[-5:]))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
v=dict(stage='QUESTION5_BOUNDED_EXISTING_METADATA_OBSERVATION',checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],metadata=docs,jobs=jobs,rank_progress=progress,gpu_samples=gpu,error_metadata=errors,
 teacher_or_actor_updates=0,read_only=True,state_resume=True)
raw=json.dumps(v,allow_nan=False);assert len(raw.encode())<=128*1024;print(raw)
'''


def worker(handle):
    import csv
    import fcntl
    import io
    import signal
    import stat
    import traceback

    h = json.loads(handle.read_text())
    b = Path(h['base'])
    repo = b / 'repo'
    binding = dict(h=h, b=b, repo=repo)
    exec(BINDING, binding)
    birth, alive, write = binding['birth'], binding['alive'], binding['write']
    jobs, leases = [], []
    child = None
    requested = False
    native_signalled = False

    def status(stage, **extra):
        write(b / 'logs/status.json', dict(stage=stage, checked_unix=time.time(), run_id=h['run_id'],
              execution_id=h['execution_id'], source_commit=h['source_commit'], jobs=jobs, state_resume=True, **extra))

    def native_stop(_signum=None, _frame=None):
        nonlocal requested, native_signalled
        requested = True
        if native_signalled:
            return
        # Never signal torchrun or kill a process group while a state writer may
        # be active. Native rank0 reduces its stop flag and saves all four RNGs.
        q = b / 'train/executions' / h['execution_id'] / 'initialization-rank-0.json'
        if q.exists():
            row = binding['read'](q)
            identity = binding['native_identity'](row)
            if alive(identity):
                binding['graceful_signal'](identity)
                native_signalled = True

    def wait_native():
        while child.poll() is None:
            if requested:
                native_stop()
            time.sleep(.5)
        return child.returncode

    def inventory():
        rows = list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=15))))
        apps = list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=15))))
        busy = {row[0].strip() for row in apps if row and row[0].strip().startswith('GPU-')}
        eligible = []
        for index, uuid, memory, utilization in rows:
            if int(index) not in (5, 6) and int(memory) <= 64 and int(utilization) <= 5 and uuid.strip() not in busy:
                eligible.append((int(index), uuid.strip()))
        return rows, sorted(eligible)

    signal.signal(signal.SIGTERM, native_stop)
    signal.signal(signal.SIGINT, native_stop)
    try:
        binding['source']()
        q = binding['qualification']()
        assert not (b / 'train').exists(), 'New resume output namespace already exists'
        status('ACCEPTED_FULL_STATE_AND_READONLY_PARENT_CACHE', inherited_counters=q['retained_counters'],
               parent_cache_base=h['parent_cache_base'], data_or_features_copied=False)
        root = Path('/data/shaoshitong/Search-G1-qprobe-shared/token-feature-gpu-locks')
        root.mkdir(parents=True, exist_ok=True)
        assert root.resolve() == root and not root.is_symlink()
        while True:
            if requested:
                raise InterruptedError('Graceful stop requested before native trainer launch')
            rows, eligible = inventory()
            selected = []
            for physical, uuid in eligible:
                lockpath = root / (uuid + '.lock')
                descriptor = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                lease = os.fdopen(descriptor, 'a+b')
                try:
                    st = os.fstat(descriptor)
                    assert stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid()
                    assert (st.st_dev, st.st_ino) == (lockpath.stat().st_dev, lockpath.stat().st_ino)
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                except BaseException:
                    lease.close()
                    raise
                leases.append(lease)
                selected.append((physical, uuid))
                if len(selected) == 4:
                    break
            if len(selected) == 4:
                second_rows, second_eligible = inventory()
                if all(value in second_eligible for value in selected):
                    rows = second_rows
                    break
            for lease in leases:
                lease.close()
            leases = []
            status('WAITING_FOUR_CURRENTLY_IDLE_TRAINING_GPUS', idle_count=len(eligible),
                   excluded_physical_indices=[5, 6], foreign_tasks_signalled=False)
            time.sleep(60)
        fds = tuple(lease.fileno() for lease in leases)
        lease_metadata = [dict(uuid=uuid, device=os.fstat(lease.fileno()).st_dev,
                              inode=os.fstat(lease.fileno()).st_ino, inherited_fd=lease.fileno())
                          for (_, uuid), lease in zip(selected, leases)]
        write(b / 'logs/gpu-binding.json', dict(run_id=h['run_id'], execution_id=h['execution_id'],
            checked_unix=time.time(), indices=[value[0] for value in selected], uuids=[value[1] for value in selected],
            inventory=rows, auxiliary_gpu_used=False, foreign_tasks_signalled=False,
            inherited_open_file_description_leases=lease_metadata), True)
        # Recheck immutable checkpoint/fullSHA after admission, including when
        # admission had to wait. Parent files are never moved or rewritten.
        q = binding['qualification']()
        assert not binding['bound_processes'](b, exclude=(os.getpid(),))
        checkpoint = q['checkpoint']
        qualification_path = b / 'preparation/resume-qualification.json'
        resume = dict(kind='authenticated_global_batch_transition', checkpoint=checkpoint['path'],
            sha256=checkpoint['sha256'], parent_binding=h['parent_binding'],
            parent_binding_sha256=h['parent_binding_sha256'], parent_execution_id=h['resume_parent_handle']['execution_id'],
            qualification_receipt=str(qualification_path), qualification_receipt_sha256=binding['sha'](qualification_path))
        fit = dict(h['fit_config'], feature_manifest=str(Path(h['parent_cache_base']) / 'features/manifest.json'),
            feature_manifest_sha256=h['parent_manifest_sha256'], model_dir=h['model_dir'], output_dir=str(b / 'train'),
            run_id=h['run_id'], execution_id=h['execution_id'], source_commit=h['source_commit'],
            physical_gpu_uuids=[value[1] for value in selected], expected_train_groups=445440, expected_test_groups=512,
            deployment_handle_path=str(b / 'preparation/deployment.json'), resume=resume,
            batch_transition=dict(legacy_through_epoch=q['legacy_through_epoch']))
        write(b / 'preparation/fit-config.json', fit, True)
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            CUDA_VISIBLE_DEVICES=','.join(value[1] for value in selected), OMP_NUM_THREADS='8',
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', NCCL_NVLS_ENABLE='0',
            QUESTION5_GPU_LEASE_FDS=json.dumps(list(fds)), QUESTION5_GPU_LEASE_UUIDS=json.dumps([value[1] for value in selected]))
        status('NATIVE_FULL_STATE_GLOBAL1024_RESUME_STARTING', inherited_counters=q['retained_counters'])
        command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=4',
                   '-m', 'reproduction.question5_ema_warmup.train', '--config', str(b / 'preparation/fit-config.json')]
        with (b / 'logs/probe-train.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=fds)
        row = dict(name='probe-train', process=birth(child.pid), exit_code=None)
        jobs.append(row)
        status('CHILD_STARTED', current_job='probe-train')
        row['exit_code'] = wait_native()
        status('CHILD_EXITED', current_job='probe-train')
        if row['exit_code']:
            raise RuntimeError('Native resumed child exited: ' + str(row['exit_code']))
        status('NATIVE_RESUMED_CHILD_RETURNED_ZERO')
        write(b / 'logs/exit.json', dict(exit_code=0, checked_unix=time.time(), run_id=h['run_id'],
            execution_id=h['execution_id'], state_resume=True,
            training_complete_requires_native_200_metrics_and_checkpoint_audit=True))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            # Preserve the writer and inherited lease. Native graceful stop is
            # the only shutdown path; no timeout ever promotes it to SIGKILL.
            native_stop()
            wait_native()
            if jobs:
                jobs[-1]['exit_code'] = child.returncode
        write(b / 'logs/failure.json', dict(error_class=type(exc).__name__, checked_unix=time.time(),
            run_id=h['run_id'], execution_id=h['execution_id'], state_resume=True, writer_force_killed=False))
        write(b / 'logs/exit.json', dict(exit_code=1, checked_unix=time.time(), run_id=h['run_id'],
            execution_id=h['execution_id'], state_resume=True))
        traceback.print_exc()
        raise
    finally:
        for lease in leases:
            lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['stage', 'stop', 'seal', 'launch', 'observe'])
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--payload', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.handle)
        return
    if args.action is None or (args.action == 'stage' and args.payload is None):
        parser.error('--action is required, and stage requires --payload')
    h = json.loads(args.handle.read_text())
    payload = dict(handle=h)
    if args.action == 'stage':
        staged = json.loads(args.payload.read_text())
        assert 'handle' not in staged, 'Payload cannot replace the selected handle'
        payload.update(staged)
    code = {'stage': STAGE, 'stop': STOP, 'seal': SEAL, 'launch': LAUNCH, 'observe': OBSERVE}[args.action]
    started = time.time()
    result = subprocess.run(route(code), input=json.dumps(payload).encode(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=65)
    record = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time() - started,
        exit_code=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode(),
        source_commit=h['source_commit'], run_id=h['run_id'], execution_id=h['execution_id'])
    out = ROOT / 'preparation' / ('question5-global1024-resume-' + args.action + '-' + str(time.time_ns()) + '.json')
    out.write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError('Action failed; retained receipt ' + str(out) + ': ' + record['stderr'][:1000])
    assert len(result.stdout) <= 128 * 1024
    print(json.dumps(dict(receipt=str(out), result=json.loads(result.stdout))))


if __name__ == '__main__':
    main()
