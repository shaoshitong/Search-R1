"""Bounded A800 orchestration for a new four-rank actor/native-prefix5 run.

Stage the exact published source and qualify the chosen complete EMA model
before an explicit graceful stop of the previous warmup. The stop targets only
its authenticated rank0 PID through a pidfd. Launch requires every old writer
gone; submitting a wrapper is never reported as training acceptance. Observe
reads existing metadata only. This module does not run tests or evaluations.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
PYTHON = '/data/search-g1/env/bin/python'
CONTROL = 'reproduction/operations/a800_actorhidden_prefix5_control.py'
MAX_OUTPUT = 128 * 1024


def route(code):
    """Reuse the currently authorized physical transport, without SSH changes."""
    path = ROOT / 'repo/reproduction/operations/a800_question5_global1024_resume_control.py'
    spec = importlib.util.spec_from_file_location('existing_actorhidden_prefix5_route', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.route(code)


BINDING = r'''
import base64,ctypes,hashlib,io,json,os,pathlib,re,signal,stat,subprocess,sys,tarfile,time
MAX_METADATA=64*1024**2
metadata_bytes=0
PREFIX='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def canonical_path(path):
 path=pathlib.Path(path)
 assert path.is_absolute() and path.resolve()==path,'Noncanonical path'
 return path
assert re.fullmatch('[0-9a-f]{32}',h['run_id']) and re.fullmatch('[0-9a-f]{32}',h['execution_id'])
assert str(b)==PREFIX+h['run_id'] and h.get('newbase',str(b))==str(b)
canonical_path(b)
assert re.fullmatch('[0-9a-f]{40}',h['source_commit']) and re.fullmatch('[0-9a-f]{40}',h['source_tree'])
assert re.fullmatch('[0-9a-f]{64}',h['source_archive_sha256'])
assert isinstance(h['source_files'],dict) and h['source_files']
assert all(re.fullmatch('[0-9a-f]{64}',value) for value in h['source_files'].values())
old_h=h['old_warmup_handle'];old=canonical_path(old_h['base'])
assert old_h['run_id']=='8af926a49630428097b94027864da68b'
assert old_h['execution_id']=='bcfa1de0a90a4fcda70263a75153fcee'
assert old_h['source_commit']=='ddf36cca14ac7aafb0b5395ffa88d57ed8c35e74'
assert old!=b and str(old).startswith('/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-')
assert h['template_config_path']==str(b/'preparation/launch-config-template.json')
def regular(path,limit=None):
 path=canonical_path(path);s=path.lstat()
 assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid(),'Not an owned regular file'
 assert limit is None or s.st_size<=limit,'Oversized metadata'
 return s
def read(path):
 global metadata_bytes
 regular(path,MAX_METADATA)
 with pathlib.Path(path).open('rb') as stream:raw=stream.read(MAX_METADATA+1)
 metadata_bytes+=len(raw)
 assert len(raw)<=MAX_METADATA and metadata_bytes<=MAX_METADATA,'Metadata read budget exceeded'
 return json.loads(raw)
def sha(path):
 regular(path);value=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):value.update(chunk)
 return value.hexdigest()
def write(path,value,exclusive=False):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 target=path if exclusive else path.with_name(path.name+'.partial')
 descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|(os.O_EXCL if exclusive else os.O_TRUNC),0o600)
 with os.fdopen(descriptor,'wb') as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
 if not exclusive:os.replace(target,path)
 directory=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(directory)
 finally:os.close(directory)
def birth(pid):
 path=pathlib.Path('/proc')/str(pid);raw=(path/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=path.stat().st_uid)
def native_identity(value):
 value=value.get('identity',value)
 return dict(pid=value['pid'],start_ticks=value.get('birth',value.get('start_ticks')),sid=value['sid'],uid=value['uid'])
def alive(identity):
 try:
  raw=(pathlib.Path('/proc')/str(identity['pid'])/'stat').read_text()
  return birth(identity['pid'])==identity and raw[raw.rfind(')')+2:].split()[0] not in ('Z','X')
 except (FileNotFoundError,ProcessLookupError):return False
def bound_processes(base,exclude=()):
 root=str(base);found=[]
 for path in pathlib.Path('/proc').iterdir():
  if not path.name.isdigit() or int(path.name) in exclude:continue
  try:
   if path.stat().st_uid!=os.getuid():continue
   raw=(path/'stat').read_text()
   if raw[raw.rfind(')')+2:].split()[0] in ('Z','X'):continue
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  args=[];cwd=None
  try:args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
  try:cwd=str((path/'cwd').resolve())
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
  matched=(cwd is not None and (cwd==root or cwd.startswith(root+'/'))) or any(
   a==root or a.startswith(root+'/') or ('='+root+'/') in a for a in args)
  if matched:
   try:found.append(birth(int(path.name)))
   except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 return found
def verify_old():
 path=b/'preparation/old-warmup-deployment.json'
 assert sha(path)==h['old_warmup_handle_sha256'] and read(path)==old_h
 assert sha(old/'preparation/deployment.json')==h['old_warmup_handle_sha256']
 assert read(old/'preparation/deployment.json')==old_h
def source(require_receipt=True):
 assert read(b/'preparation/deployment.json')==h
 actual=set()
 for path in repo.rglob('*'):
  if path.is_dir():assert not path.is_symlink() and path.resolve()==path
  else:regular(path);actual.add(str(path.relative_to(repo)))
 assert actual==set(h['source_files']),'Exact deployed source inventory differs'
 for name,digest in h['source_files'].items():
  relative=pathlib.PurePosixPath(name)
  assert not relative.is_absolute() and '..' not in relative.parts
  assert sha(repo/name)==digest,'Published source file mismatch: '+name
 verify_old()
 if require_receipt:
  receipt=read(b/'preparation/source-receipt.json')
  assert receipt['source_commit']==h['source_commit'] and receipt['source_files_verified']==len(h['source_files'])
def qualification():
 ended=read(b/'preparation/prepare-exit.json')
 assert ended['exit_code']==0 and ended['run_id']==h['run_id'] and ended['source_commit']==h['source_commit']
 assert not alive(ended['identity']),'CPU qualification writer still live'
 q=read(b/'preparation/source-qualification.json')
 assert q.get('finite') is True and q.get('model_only_import') is True
 assert q.get('model_kind')=='ema' and type(q.get('source_epoch')) is int and q['source_epoch']>0
 assert re.fullmatch('[0-9a-f]{64}',q['source_sha256'])
 assert sha(b/'preparation/source-qualification.json')==ended['qualification_sha256']
 c=read(b/'preparation/launch-config.json')
 assert c['run_id']==h['run_id'] and c['base']==str(b) and c['source_commit']==h['source_commit']
 assert c['world_size']==4 and c['prefix_policy_tokens']==5 and c['outers']==200
 assert c['probe_source_epoch']==q['source_epoch'] and c['probe_source_sha256']==q['source_sha256']
 assert c['probe_source_model_kind']=='ema' and c['probe_source_checkpoint']==q['source_checkpoint']
 assert sha(b/'preparation/launch-config.json')==ended['config_sha256']
 return q,c
def old_native(rank,name):
 path=old/'train/executions'/old_h['execution_id']/(name+'-rank-%d.json'%rank)
 value=read(path);identity=value.get('identity',value)
 assert all(identity[key]==old_h[key] for key in ('run_id','execution_id','source_commit'))
 assert identity['rank']==rank and identity['local_rank']==rank and identity['world_size']==4
 assert identity['uid']==os.getuid() and identity['sid']==identity['pid']
 return value,dict(path=str(path),sha256=sha(path))
def graceful_signal(identity):
 assert identity['uid']==os.getuid() and identity['sid']==identity['pid'] and alive(identity)
 libc=ctypes.CDLL(None,use_errno=True)
 def call(number,*args):
  assert sys.platform.startswith('linux') and os.uname().machine in ('x86_64','aarch64')
  result=libc.syscall(ctypes.c_long(number),*args)
  if result<0:
   error=ctypes.get_errno();raise OSError(error,os.strerror(error))
  return result
 descriptor=os.pidfd_open(identity['pid']) if hasattr(os,'pidfd_open') else call(434,ctypes.c_int(identity['pid']),ctypes.c_uint(0))
 try:
  assert alive(identity),'Rank0 birth changed after pidfd_open'
  if hasattr(signal,'pidfd_send_signal'):signal.pidfd_send_signal(descriptor,signal.SIGTERM)
  else:call(424,ctypes.c_int(descriptor),ctypes.c_int(signal.SIGTERM),ctypes.c_void_p(),ctypes.c_uint(0))
 finally:os.close(descriptor)
def old_terminal():
 verify_old();ranks=[]
 for rank in range(4):
  initialized,_=old_native(rank,'initialization');who=native_identity(initialized)
  assert not alive(who),'An original warmup rank remains live'
  ended,proof=old_native(rank,'exit')
  assert native_identity(ended)==who
  assert ended.get('terminal') in ('STOPPED_AT_COMPLETE_EPOCH_BOUNDARY','STOPPED_AT_COMPLETE_ADAM_BOUNDARY')
  ranks.append(dict(rank=rank,identity=who,exit=proof,terminal=ended['terminal'],counters=ended.get('counters')))
 latest=read(old/'train/latest.json')
 assert all(latest[key]==old_h[key] for key in ('run_id','execution_id','source_commit'))
 assert all(row['counters']==latest['counters'] for row in ranks),'Stopped rank counters differ from saved final boundary'
 assert latest['actual_adam_steps_per_rank']==[latest['counters']['optimizer_steps']]*4
 assert latest['ema_updates']==latest['counters']['optimizer_steps'] and latest['path']==str(old/'train/latest.pt')
 regular(pathlib.Path(latest['path']))
 wrapper=read(old/'logs/launch.json');wrapper_identity=native_identity(wrapper['process'])
 assert not alive(wrapper_identity),'Original warmup pipeline wrapper remains live'
 wrapper_exit=read(old/'logs/exit.json')
 assert wrapper_exit['run_id']==old_h['run_id'] and wrapper_exit['execution_id']==old_h['execution_id']
 remaining=bound_processes(old)
 assert not remaining,'Original warmup pipeline, torchrun or owned writer remains live'
 status=read(old/'logs/status.json')
 for job in status.get('jobs',[]):
  process=job.get('process')
  if process:assert not alive(native_identity(process)),'Original warmup supervisor child remains live'
 process_ledger=old/'logs/process-ledger.json'
 if process_ledger.exists():
  ledger=read(process_ledger);assert ledger['run_id']==old_h['run_id']
  for job in ledger.get('jobs',{}).values():
   for who in job.get('ledger',{}).values():
    assert not alive(native_identity(who)),'A recorded old owned birth remains live'
 return dict(all_old_owned_processes_gone=True,typed_rank_terminal=ranks,
  wrapper=wrapper_identity,wrapper_exit_code=wrapper_exit.get('exit_code'),
  retained_final_counters=latest['counters'],retained_checkpoint_producer=project_latest(latest),
  process_ledger_preserved=True,old_output_preserved=True)
def project_latest(value):
 return {key:value[key] for key in ('path','sha256','bytes','epoch','next_update_in_epoch','completed_epoch','counters') if key in value}
def emit(value):
 raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<=128*1024,'Output budget exceeded'
 print(raw)
'''

COMMON = "import json,pathlib,sys\nsys.dont_write_bytecode=True\np=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'\n" + BINDING

STAGE = COMMON + r'''
assert not b.exists(),'Target namespace exists; inspect, never repeat stage'
assert isinstance(p.get('side_files'),dict)
assert {'launch-config-template.json','old-warmup-deployment.json'}<=set(p['side_files'])
assert all(pathlib.PurePosixPath(name).name==name and name not in ('deployment.json','source-receipt.json') for name in p['side_files'])
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
   assert member.isfile() and member.size<=MAX_METADATA and member.name in h['source_files']
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as stream:stream.write(tar.extractfile(member).read());stream.flush();os.fsync(stream.fileno())
for name,value in p['side_files'].items():
 raw=base64.b64decode(value,validate=True);assert len(raw)<=MAX_METADATA
 with (b/'preparation'/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
write(b/'preparation/deployment.json',h,True)
v=dict(stage='A800_ACTORHIDDEN_PREFIX5_EXACT_SOURCE_STAGED',run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],source_tree=h['source_tree'],source_files_verified=len(h['source_files']),
 source_archive_sha256=h['source_archive_sha256'],checked_unix=time.time(),base=str(b),old_warmup_stopped=False,
 training_started=False)
source(require_receipt=False);write(b/'preparation/source-receipt.json',v,True);emit(v)
'''

PREPARE = COMMON + r'''
source();assert not (b/'preparation/prepare-intent.json').exists(),'CPU prepare already requested; inspect its original receipt'
assert not (b/'preparation/prepare-exit.json').exists()
write(b/'preparation/prepare-intent.json',dict(run_id=h['run_id'],source_commit=h['source_commit'],
 checked_unix=time.time(),old_warmup_stopped=False),True)
command=['/data/search-g1/env/bin/python','-m','reproduction.operations.a800_actorhidden_prefix5_control',
 '--prepare-worker','--handle',str(b/'preparation/deployment.json')]
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
with (b/'logs/cpu-prepare.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
who=birth(child.pid);assert who['sid']==who['pid'] and who['uid']==os.getuid()
v=dict(status='CPU_PREPARE_ONCE_LAUNCHED_NOT_ACCEPTED',run_id=h['run_id'],source_commit=h['source_commit'],
 identity=who,started_unix=time.time(),old_warmup_stopped=False)
write(b/'preparation/prepare-launch.json',v,True);emit(v)
'''

STOP = COMMON + r'''
source();qualified,config=qualification()
assert not (b/'preparation/warmup-stop-intent.json').exists(),'A stop intent exists; reconcile original result without replay'
initialized,init_proof=old_native(0,'initialization');identity=native_identity(initialized)
progress,progress_proof=old_native(0,'progress');assert native_identity(progress)==identity
assert alive(identity),'Old rank0 is not proven live; no signal sent'
v=dict(role='actorhidden_prefix5_exact_old_rank0_graceful_stop',new_run_id=h['run_id'],new_execution_id=h['execution_id'],
 old_run_id=old_h['run_id'],old_execution_id=old_h['execution_id'],old_source_commit=old_h['source_commit'],
 process=identity,initialization=init_proof,progress=progress_proof,signals_requested=1,signal='SIGTERM',
 requested_unix=time.time(),qualification_sha256=sha(b/'preparation/source-qualification.json'),
 process_groups_signalled=False,foreign_tasks_signalled=False,partial_writes_must_remain_preserved=True)
write(b/'preparation/warmup-stop-intent.json',v,True);graceful_signal(identity)
v.update(signals_sent=1,delivered_unix=time.time(),old_terminal_not_yet_proven=True)
write(b/'preparation/warmup-stop-delivered.json',v,True);emit(v)
'''

LAUNCH = COMMON + r'''
source();qualified,config=qualification()
assert (b/'preparation/warmup-stop-delivered.json').exists(),'No authenticated old-warmup stop receipt'
terminal=old_terminal()
assert not (b/'preparation/actorhidden-launch-intent.json').exists(),'Launch already requested; observe original result without replay'
assert not (b/'preparation/submit-intent.json').exists(),'Native submit claim already exists'
assert not bound_processes(b,exclude=(os.getpid(),)),'A target writer already exists'
write(b/'preparation/old-warmup-terminal.json',terminal,True)
write(b/'preparation/actorhidden-launch-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],config_sha256=sha(b/'preparation/launch-config.json'),requested_unix=time.time(),
 new_actor_experiment=True,formal_start_outer=0,old_probe_fullstate_resume=False),True)
sys.path.insert(0,str(repo))
from reproduction.prefix_selection200.launch import submit
value=submit(b)
emit(dict(status='ACTORHIDDEN_NATIVE_SUBMIT_RETURNED_NOT_TRAINING_ACCEPTANCE',submit=value,old_terminal=terminal,
 checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit']))
'''

OBSERVE = COMMON + r'''
assert read(b/'preparation/deployment.json')==h
def project(value,keys):return {key:value[key] for key in keys if key in value}
def document(root,name):
 path=root/name
 if not path.exists():return None
 value=read(path);assert isinstance(value,dict),'Expected existing metadata object'
 return value
def proof(name):
 path=b/name
 if not path.exists():return None
 value=read(path)
 return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),value=value)
def process_projection(value):
 if not value:return None
 who=native_identity(value);return dict(who,alive=alive(who))
v=dict(base=str(b),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
 observed_unix=time.time(),metadata_only=True,models_or_tensors_read=False)
for name,key in (('preparation/source-receipt.json','source'),('preparation/prepare-launch.json','prepare_launch'),
 ('preparation/prepare-exit.json','prepare_exit'),('preparation/source-cpu-prepare-exit.json','source_cpu_prepare_exit'),
 ('preparation/source-qualification.json','source_qualification'),
 ('preparation/warmup-stop-delivered.json','old_stop'),('preparation/old-warmup-terminal.json','old_terminal'),
 ('preparation/submit-receipt.json','submit'),('logs/pipeline-status.json','stage'),
 ('logs/cleanup-status.json','cleanup'),('logs/qa-pending.json','qa_pending')):
 value=document(b,name);v[key]=value
 if key=='prepare_launch' and value:v['prepare_process']=process_projection(value['identity'])
 if key=='submit' and value:v['pipeline_process']=process_projection(value.get('leader'))
ledger=document(b,'logs/process-ledger.json');jobs=[]
if ledger:
 assert ledger['run_id']==h['run_id']
 for name,job in ledger.get('jobs',{}).items():
  item=dict(name=name,**project(job,('state','exit_code','started_unix','stopped_unix')),
   leader=process_projection(job.get('leader')))
  births=[]
  for recorded in job.get('ledger',{}).values():
   if all(key in recorded for key in ('pid','sid','uid')) and ('birth' in recorded or 'start_ticks' in recorded):
    births.append(process_projection(recorded))
  item['live_recorded_births']=[row for row in births if row['alive']]
  assert len(item['live_recorded_births'])<=256,'Live ownership projection oversized'
  item['birth_count']=len(births)
  jobs.append(item)
 assert len(jobs)<=4096,'Process ledger oversized'
v['jobs']=jobs[-24:]
v['job_history_count']=len(jobs)
v['job_history_projection']='last24_jobs_plus_all_live_jobs'
v['live_jobs']=[row for row in jobs if row['live_recorded_births'] or (row['leader'] or {}).get('alive')]
assert len(v['live_jobs'])<=64,'Active job projection oversized'
actor_latest=document(b,'actor/latest.json')
v['actor_latest']=None if actor_latest is None else project(actor_latest,('run_id','actor_version','completed_outer',
 'actual_adam_steps_per_rank','new_adam_steps_per_rank','scheduler_completed_outer','learning_rate_used',
 'next_learning_rate','actual_global_micro','actual_local_micro','gradient_accumulation'))
probe_latest=document(b,'probe/latest.json')
v['probe_latest']=None if probe_latest is None else project(probe_latest,('run_id','completed_outer',
 'actual_adam_steps_per_rank','bytes','sha256','binding_sha256'))
joint=document(b,'latest-joint.json');v['latest_joint']=None
if joint:
 assert joint['run_id']==h['run_id'] and joint['source_commit']==h['source_commit']
 actor,probe=joint['actor'],joint['probe'];n=joint['outer']
 assert actor['actual_adam_steps_per_rank']==[20*n]*4 and actor['scheduler_completed_outer']==n
 assert len(probe['actual_adam_steps_per_rank'])==4
 v['latest_joint']=dict(outer=n,actor=project(actor,('actor_version','completed_outer','actual_adam_steps_per_rank',
  'new_adam_steps_per_rank','scheduler_completed_outer','learning_rate_used','next_learning_rate',
  'requested_global_micro','actual_global_micro','actual_local_micro','gradient_accumulation','reward')),
  probe=project(probe,('completed_outer','actual_adam_steps_per_rank','sha256','bytes','binding_sha256')),
  probe_metrics=joint.get('probe_metrics'),phase_timings=joint.get('phase_timings'),
  selected=project(joint.get('rollout',{}),('actual_selected_mixed_fraction','questions','trajectories')),
  proof=dict(path=str(b/'latest-joint.json'),sha256=sha(b/'latest-joint.json')))
stage=v.get('stage') or {};current=stage.get('outer',0)
v['current_outer']=current;v['prefix_ranks']=[];v['continuation_ranks']=[];v['actor_attempts']=[]
if type(current) is int and 1<=current<=200:
 root=b/'outers'/('outer-%06d'%current)
 for phase,key in (('prefix','prefix_ranks'),('full','continuation_ranks')):
  for rank in range(4):
   value=document(root,phase+'/rank-%d-progress.json'%rank)
   v[key].append(None if value is None else project(value,('rank','outer','phase','stage','actor_version','utc',
    'completed_trajectories','expected_trajectories','active_trajectories','pending_generation','pending_retrieval',
    'generation_requests','actual_sampled_tokens','elapsed_seconds','device_binding')))
 attempts=list((b/'actor/outers').glob('outer-%04d-attempt-*'%current)) if (b/'actor/outers').exists() else []
 assert len(attempts)<=32,'Too many attempt directories; retain unknown rather than truncate accounting'
 for directory in sorted(attempts,key=lambda x:x.stat().st_mtime,reverse=True)[:4]:
  counts=[];ranks=[]
  for rank in range(4):
   path=directory/('rank-%d-updates.jsonl'%rank)
   if not path.exists():counts.append(None);ranks.append(None);continue
   regular(path,MAX_METADATA)
   raw=path.read_bytes();metadata_bytes+=len(raw);assert metadata_bytes<=MAX_METADATA
   rows=[];incomplete=False
   for ordinal,line in enumerate(raw.splitlines()):
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:
     assert ordinal==len(raw.splitlines())-1;incomplete=True
   writes=[row for row in rows if row.get('event')=='ACTUAL_ADAM_WRITE']
   assert all(row['rank']==rank and row['outer']==current and row['opportunity']==index for index,row in enumerate(writes))
   counts.append(len(writes));ranks.append(dict(rank=rank,confirmed_returned_adam=len(writes),
    last_actual_adam=writes[-1]['actual_adam_steps'] if writes else None,incomplete_final_line=incomplete,
    ledger_sha256=hashlib.sha256(raw).hexdigest(),failure=document(directory,'rank-%d-failure.json'%rank)))
  v['actor_attempts'].append(dict(attempt_dir=str(directory),confirmed_returned_per_rank=counts,
   four_ranks_agree=(len(set(counts))==1 and None not in counts),ranks=ranks))
completed=document(b,'logs/pipeline-complete.json')
v['pipeline_complete']=None if completed is None else project(completed,('run_id','complete_outers'))
v['metadata_bytes_read']=metadata_bytes
emit(v)
'''


def _write_once(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _identity():
    raw = Path('/proc/self/stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=os.getpid(), start_ticks=int(fields[19]), sid=os.getsid(0), uid=os.getuid())


def prepare_worker(handle_path):
    """Owned CPU-only writer; its exit receipt is distinct from submission."""
    h = json.loads(Path(handle_path).read_text())
    base = Path(h['base'])
    identity = _identity()
    receipt = dict(run_id=h['run_id'], execution_id=h['execution_id'], source_commit=h['source_commit'],
                   identity=identity, checked_unix=time.time(), old_warmup_stopped=False)
    try:
        from reproduction.prefix_selection200.a800_prepare import prepare
        result = prepare(h['template_config_path'], str(base / 'preparation/old-warmup-deployment.json'))
        qualification = base / 'preparation/source-qualification.json'
        config = base / 'preparation/launch-config.json'
        if not qualification.is_file() or not config.is_file():
            raise ValueError('CPU preparation returned without qualified configuration and source receipt')
        receipt.update(exit_code=0, result=result,
            qualification_sha256=hashlib.sha256(qualification.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), completed_unix=time.time())
    except BaseException as error:
        receipt.update(exit_code=1, error_class=type(error).__name__, completed_unix=time.time())
        _write_once(base / 'preparation/prepare-exit.json', receipt)
        traceback.print_exc()
        raise
    _write_once(base / 'preparation/prepare-exit.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('stage', 'prepare', 'stop-warmup', 'launch', 'observe'))
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--prepare-worker', action='store_true')
    args = parser.parse_args()
    if not args.handle.is_absolute():
        parser.error('--handle must be an absolute local or remote deployment JSON path')
    if args.prepare_worker:
        prepare_worker(args.handle)
        return
    if args.action is None or (args.action == 'stage' and args.payload is None):
        parser.error('--action is required; stage also requires --payload')
    h = json.loads(args.handle.read_text())
    payload = dict(handle=h)
    if args.action == 'stage':
        staged = json.loads(args.payload.read_text())
        if 'handle' in staged:
            raise ValueError('Stage payload cannot replace the explicitly selected handle')
        payload.update(staged)
    code = dict(stage=STAGE, prepare=PREPARE, **{'stop-warmup': STOP}, launch=LAUNCH, observe=OBSERVE)[args.action]
    started = time.time()
    try:
        result = subprocess.run(route(code), input=json.dumps(payload).encode(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=65)
        record = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time() - started,
            exit_code=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode(),
            source_commit=h['source_commit'], run_id=h['run_id'], execution_id=h['execution_id'])
    except subprocess.TimeoutExpired as error:
        record = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time() - started,
            exit_code=None, timed_out=True, remote_action_outcome_unknown=True,
            mutation_must_not_be_replayed=True, stdout=(error.stdout or b'').decode(errors='replace'),
            stderr=(error.stderr or b'').decode(errors='replace'), source_commit=h['source_commit'],
            run_id=h['run_id'], execution_id=h['execution_id'])
    out = ROOT / 'preparation' / ('a800-actorhidden-prefix5-' + args.action + '-' + str(time.time_ns()) + '.json')
    out.write_text(json.dumps(record, indent=2) + '\n')
    if record.get('timed_out'):
        raise RuntimeError('SSH observation failed; remote action remains unknown, do not replay. Receipt ' + str(out))
    if result.returncode:
        raise RuntimeError('Action failed; retained original receipt ' + str(out) + ': ' + record['stderr'][:1000])
    if len(result.stdout) > MAX_OUTPUT:
        raise ValueError('Remote output exceeded128KiB; preserve unknown and inspect retained receipt')
    print(json.dumps(dict(receipt=str(out), result=json.loads(result.stdout))))


if __name__ == '__main__': main()
