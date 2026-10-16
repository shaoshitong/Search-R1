"""Stage and launch one fresh question+5 joint200 trainer on an accepted cache.

This is a startup repair after the parent failed before head/Adam/RNG
initialization, never an optimizer-state resume. It uses the existing physical
A800 SSH route, copies only the complete source archive and bounded metadata,
and reads the parent's data/features in place. It never downloads, prepares,
encodes, copies feature payloads, changes the parent, or signals foreign jobs.

In addition to the existing source/base/run/execution/model_dir/fit_config
handle fields, require parent_cache_base, parent_handle (the complete original
handle), parent_handle_sha256, parent_data_manifest_sha256,
parent_manifest_sha256 (packed feature manifest), parent_prepare_complete_sha256,
parent_terminal_receipt_sha256, and parent_acceptance_receipt_sha256. Stage
payload side_files must contain the exact base64-encoded bytes of source-plan.json,
parent-deployment.json, parent-terminal-receipt.json and
parent-cache-acceptance-receipt.json. The latter two are the complete local
actual-observation receipts, whose stdout contains the remote JSON observation.
The source plan remains the exact original parent plan for source observers.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
PYTHON = '/data/search-g1/env/bin/python'
CONTROL = 'reproduction/operations/a800_question5_cached_train_control.py'


def route(code):
    path = Path(__file__).with_name('a800_qprobe_prefix_scale20.py')
    spec = importlib.util.spec_from_file_location('existing_question5_route', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.route().ssh_command(code)


BINDING = r'''
import csv,hashlib,io,json,os,pathlib,re,signal,stat,subprocess,sys,time
PREFIX='/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-'
PARENT=PREFIX+'d7dd61bb36454e1e80c2306b5fb7d23a'
EXPECTED_DATA='8ac391ba26466e9fdbe9220b157bec81294ead69a09a0b7c2c4994cc7873a2c7'
EXPECTED_FEATURES='e9ed944ed244513ab55bdd1b13aae745c5b7902228294fd62ef70ea349b6f246'
EXPECTED_PREPARE='7db1b571728cb39baf77a67e4bdaaa0cf5efc1f4bca187fd701a29231d48cc1b'
def canonical_path(path):
 path=pathlib.Path(path)
 assert path.is_absolute() and path.resolve()==path,'Noncanonical path: '+str(path)
 return path
assert str(b)==PREFIX+h['run_id'] and re.fullmatch('[0-9a-f]{32}',h['run_id'])
assert re.fullmatch('[0-9a-f]{32}',h['execution_id']) and b!=pathlib.Path(PARENT)
canonical_path(b)
def regular(path,limit=None):
 path=canonical_path(path);s=path.lstat()
 assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid(),'Not an owned regular file: '+str(path)
 assert limit is None or s.st_size<=limit,'Oversized metadata: '+str(path)
 return s
def read(path):
 regular(path,16*1024**2)
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
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 if not exclusive:os.replace(target,path)
def birth(pid):
 q=pathlib.Path('/proc')/str(pid);s=(q/'stat').read_text();v=s[s.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=q.stat().st_uid)
def alive(x):
 try:
  s=(pathlib.Path('/proc')/str(x['pid'])/'stat').read_text()
  return birth(x['pid'])==x and s[s.rfind(')')+2:].split()[0] not in ('Z','X')
 except (FileNotFoundError,ProcessLookupError):return False
def bound_processes(base,exclude=()):
 # Match exact path boundaries in argv/cwd; never signal any discovered PID.
 root=str(base);found=[]
 for q in pathlib.Path('/proc').iterdir():
  if not q.name.isdigit() or int(q.name) in exclude:continue
  try:
   if q.stat().st_uid!=os.getuid():continue
   s=(q/'stat').read_text()
   if s[s.rfind(')')+2:].split()[0] in ('Z','X'):continue
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  args=[];cwd=None
  try:
   args=(q/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
  try:
   cwd=str((q/'cwd').resolve())
  except (PermissionError,FileNotFoundError,ProcessLookupError):pass
  matched=(cwd is not None and (cwd==root or cwd.startswith(root+'/'))) or any(a==root or a.startswith(root+'/') or ('='+root+'/') in a for a in args)
  try:
   if matched:found.append(birth(int(q.name)))
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 return found
def source():
 assert read(b/'preparation/deployment.json')==h
 expected=set(h['source_files']);actual=set()
 for q in repo.rglob('*'):
  if q.is_dir():
   assert not q.is_symlink() and q.resolve()==q
  else:
   regular(q);actual.add(str(q.relative_to(repo)))
 assert actual==expected,'Archive source inventory differs from the published map'
 for name,digest in h['source_files'].items():
  rel=pathlib.PurePosixPath(name)
  assert not rel.is_absolute() and '..' not in rel.parts
  assert sha(repo/name)==digest,'Deployed source mismatch: '+name
 assert sha(b/'preparation/source-plan.json')==h['source_plan_sha256']
def receipt(name,digest):
 path=b/'preparation'/name
 assert sha(path)==digest,'Pinned local receipt bytes changed: '+name
 outer=read(path);assert outer['exit_code']==0
 return json.loads(outer['stdout'])
def parent_acceptance():
 parent=canonical_path(h['parent_cache_base']);assert str(parent)==PARENT
 old=h['parent_handle'];assert old['base']==str(parent)
 assert old['run_id']!=h['run_id'] and old['execution_id']!=h['execution_id']
 assert old['run_id']=='d7dd61bb36454e1e80c2306b5fb7d23a'
 assert old['execution_id']=='694d67a5f1a04849b9f254c4a6408a03'
 assert old['source_commit']=='0fde8e8bb59fc3bac0c6247704b2fd1bd063abf6'
 assert h['model_dir']==old['model_dir'];canonical_path(h['model_dir'])
 assert sha(b/'preparation/parent-deployment.json')==h['parent_handle_sha256']
 assert read(b/'preparation/parent-deployment.json')==old
 assert sha(parent/'preparation/deployment.json')==h['parent_handle_sha256']
 assert read(parent/'preparation/deployment.json')==old
 assert h['source_plan_sha256']==old['source_plan_sha256']
 assert sha(parent/'preparation/source-plan.json')==old['source_plan_sha256']
 terminal=receipt('parent-terminal-receipt.json',h['parent_terminal_receipt_sha256'])
 accepted=receipt('parent-cache-acceptance-receipt.json',h['parent_acceptance_receipt_sha256'])
 for proof in (terminal,accepted):
  for key in ('run_id','execution_id','source_commit'):assert proof[key]==old[key]
 assert terminal['role']=='question5_actual_startup_terminal_diagnosis'
 assert terminal['remaining_processes_bound_to_base']==[] and terminal['signals_sent']==0
 assert terminal['git_metadata_exists'] is False
 assert accepted['base']==str(parent) and accepted['checkpoint_tensors_read'] is False and accepted['feature_tensors_read'] is False
 assert not (parent/'repo/.git').exists()
 for name,digest in old['source_files'].items():
  assert sha(parent/'repo'/name)==digest,'Parent source changed: '+name
 for name,item in terminal['files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
  q=parent/name;assert sha(q)==item['sha256'] and regular(q).st_size==item['bytes'] and read(q)==item['value']
 data=parent/'data/composite-manifest.json';cache=parent/'features/manifest.json';prepared=parent/'data/PREPARE-COMPLETE.json'
 assert h['parent_data_manifest_sha256']==EXPECTED_DATA and sha(data)==EXPECTED_DATA
 assert h['parent_manifest_sha256']==EXPECTED_FEATURES and sha(cache)==EXPECTED_FEATURES
 assert h['parent_prepare_complete_sha256']==EXPECTED_PREPARE and sha(prepared)==EXPECTED_PREPARE
 dm=read(data);cm=read(cache);pc=read(prepared)
 assert dm['source_plan_sha256']==old['source_plan_sha256'] and dm['source_plan_path']==str(parent/'preparation/source-plan.json')
 assert dm['expected_train_groups']==445440 and dm['expected_test_groups']==512 and dm['total_trajectory_events']==2229760
 assert pc['stage']=='PINNED_PREFIX5_COMPOSITE_PREPARED' and pc['composite_manifest_sha256']==EXPECTED_DATA
 assert (pc['train_groups'],pc['test_groups'],pc['train_trajectories'],pc['test_trajectories'])==(445440,512,2227200,2560)
 assert cm['stage']=='QUESTION5_PACKED_FEATURE_CACHE_COMPLETE' and cm['role']=='question5_packed_bf16_features_v1'
 assert cm['source_manifest_path']==str(data) and cm['source_manifest_sha256']==EXPECTED_DATA
 assert cm['groups_by_split']==dict(train=445440,test=512) and cm['trajectories_by_split']==dict(train=2227200,test=2560)
 assert cm['newly_encoded_trajectories']==1715200 and cm['reused_trajectories']==514560
 assert cm['encoder_optimizer_updates']==0 and cm['padding_stored'] is False and cm['prefix_tokens']==5
 assert cm['question_max_tokens']==273 and cm['max_tokens']==278 and cm['feature_dim']==2048
 assert cm['feature_dtype']=='bfloat16_uint16_words'
 for name,digest in cm['source_sha256'].items():
  assert digest==old['source_files']['reproduction/question5_ema_warmup/'+name]
  assert digest==h['source_files']['reproduction/question5_ema_warmup/'+name],'Cache reader/encoder source unexpectedly changed'
 assert accepted['cache_manifest']['sha256']==EXPECTED_FEATURES
 assert accepted['cpu_prepare']['complete']['sha256']==EXPECTED_PREPARE
 for key,value in accepted['cache_manifest']['value'].items():assert cm[key]==value
 status=read(parent/'logs/status.json')
 assert status==accepted['metadata']['logs/status.json']
 expected_jobs={'prepare':0,'cache-0':0,'cache-1':0,'cache-2':0,'cache-3':0,'cache-finalize':0,'probe-train':1}
 assert {job['name']:job['exit_code'] for job in status['jobs']}==expected_jobs
 assert all(not alive(job['process']) for job in status['jobs'])
 assert not alive(read(parent/'logs/launch.json')['process'])
 assert read(parent/'logs/exit.json')['exit_code']==1
 workers=[]
 for index in range(4):
  path=parent/('features/worker-%03d/complete.json'%index);value=read(path)
  proof=accepted['cache_workers'][index]['complete'];assert sha(path)==proof['sha256'] and regular(path).st_size==proof['bytes']
  assert value['stage']=='QUESTION5_PACKED_CACHE_WORKER_COMPLETE' and value['worker_index']==index and value['world_size']==4
  assert value['source_manifest_path']==str(data) and value['source_manifest_sha256']==EXPECTED_DATA
  assert value['assigned_groups']==111488 and value['newly_encoded_trajectories']==428800 and value['reused_trajectories']==128640
  assert value['encoder_optimizer_updates']==0
  workers.append(dict(worker_index=index,complete_sha256=proof['sha256'],exit_code=0))
 # Inspect only cache metadata and sealed file identities, never tensor contents.
 for item in cm['shards']:
  shard=canonical_path(item['path']);assert shard.is_relative_to(parent/'features')
  assert sha(shard/'manifest.json')==item['manifest_sha256']
  sm=read(shard/'manifest.json');assert sm['source_manifest_sha256']==EXPECTED_DATA
  if 'packed_features.npy' in sm['files']:
   q=shard/'packed_features.npy';s=regular(q);seal=sm['files']['packed_features.npy']['file_identity']
   assert dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns)==seal
 output=parent/'train';execution=output/'executions'/old['execution_id']
 for name in ('latest.pt','latest.json','run-claim.json','complete.json'):assert not (output/name).exists(),'Parent is no longer a zero-update startup failure'
 for pattern in ('initialization-rank-*.json','progress-rank-*.json','updates-rank-*.jsonl','binding.json'):
  assert not list(output.glob('**/'+pattern)),'Parent initialization/update metadata exists: '+pattern
 errors=list(execution.glob('error-rank-*.json'));assert errors
 for path in errors:
  value=read(path)
  assert value['counters'] is None and value['epoch']==0 and value['next_update_in_epoch']==0
  assert value['error_type']=='CalledProcessError' and "['git', 'rev-parse', 'HEAD']" in value['error'] and '128' in value['error']
 remaining=bound_processes(parent);assert not remaining,'A parent-owned process is still alive'
 return dict(stage='QUESTION5_PARENT_CACHE_ACCEPTED_FOR_FRESH_STARTUP_REPAIR',checked_unix=time.time(),
  run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
  parent_cache_base=str(parent),parent_run_id=old['run_id'],parent_execution_id=old['execution_id'],parent_source_commit=old['source_commit'],
  parent_handle_sha256=h['parent_handle_sha256'],parent_source_plan_sha256=old['source_plan_sha256'],
  parent_data_manifest_path=str(data),parent_data_manifest_sha256=EXPECTED_DATA,parent_feature_manifest_path=str(cache),
  parent_feature_manifest_sha256=EXPECTED_FEATURES,parent_prepare_complete_sha256=EXPECTED_PREPARE,
  parent_terminal_receipt_sha256=h['parent_terminal_receipt_sha256'],parent_acceptance_receipt_sha256=h['parent_acceptance_receipt_sha256'],
  parent_exit_code=1,parent_epoch=0,parent_counters=None,parent_adam_updates=0,parent_ema_updates=0,parent_owned_processes=remaining,
  cache_workers=workers,cache_finalizer_exit_code=0,groups_by_split=cm['groups_by_split'],trajectories_by_split=cm['trajectories_by_split'],
  packed_features_reused=2229760,newly_encoded_in_parent=1715200,reused_by_parent=514560,
  data_or_features_copied=False,parent_modified=False,feature_tensors_read=False,model_tensors_read=False,
  startup_repair=True,state_resume=False,fresh_head=True,fresh_adam=True,fresh_rng=True,encoder_optimizer_updates=0)
'''

COMMON = "import json,pathlib,sys\np=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'\n" + BINDING

STAGE = COMMON + r'''
import base64,tarfile
assert not b.exists(),'Fresh recovery namespace must not already exist'
assert set(p['side_files'])=={'source-plan.json','parent-deployment.json','parent-terminal-receipt.json','parent-cache-acceptance-receipt.json'}
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
 raw=base64.b64decode(value,validate=True);assert len(raw)<=16*1024**2
 with (b/'preparation'/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
write(b/'preparation/deployment.json',h,True);source()
v=dict(stage='QUESTION5_EXACT_SOURCE_ARCHIVE_STAGED',checked_unix=time.time(),source_commit=h['source_commit'],
 source_tree=h['source_tree'],source_files_verified=len(h['source_files']),source_archive_sha256=h['source_archive_sha256'],
 base=str(b),teacher_or_actor_started=False,training_started=False)
write(b/'preparation/source-receipt.json',v,True);print(json.dumps(v))
'''

LAUNCH = COMMON + r'''
source();assert not (b/'logs/launch.json').exists() and not (b/'logs/launch-once.json').exists()
assert not bound_processes(b),'A process already belongs to the fresh recovery namespace'
acceptance=parent_acceptance();write(b/'logs/cached-parent-acceptance.json',acceptance,True)
write(b/'logs/launch-once.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],claimed_unix=time.time(),
 startup_repair=True,state_resume=False,parent_cache_base=h['parent_cache_base']),True)
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
 OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(b/'cache/hf'),
 XDG_CACHE_HOME=str(b/'cache'),TMPDIR=str(b/'tmp'),TRITON_CACHE_DIR=str(b/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(b/'cache/inductor'))
for name in ('tmp','cache/hf','cache/triton','cache/inductor'):(b/name).mkdir(parents=True,exist_ok=True)
for key in tuple(env):
 if key.startswith('SLURM_') or key in ('RAY_ADDRESS','PYTHONHOME','LD_LIBRARY_PATH'):env.pop(key,None)
command=['/data/search-g1/env/bin/python',str(repo/'reproduction/operations/a800_question5_cached_train_control.py'),
 '--worker','--handle',str(b/'preparation/deployment.json')]
with (b/'logs/wrapper.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(stage='QUESTION5_CACHED_TRAINING_LAUNCHED_ONCE',process=birth(child.pid),run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],checked_unix=time.time(),training_progress_proven=False,startup_repair=True,state_resume=False)
write(b/'logs/launch.json',v,True);print(json.dumps(v))
'''

OBSERVE = COMMON + r'''
source();docs={}
for name in ('logs/launch.json','logs/status.json','logs/exit.json','logs/failure.json','logs/gpu-binding.json',
 'logs/cached-parent-acceptance.json','train/status.json','train/complete.json','train/failure.json','train/latest-metrics.json','train/latest.json','train/run-claim.json'):
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
 teacher_or_actor_updates=0,read_only=True,startup_repair=True,state_resume=False)
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
    birth, alive = binding['birth'], binding['alive']
    write, source = binding['write'], binding['source']
    jobs, children, leases = [], [], []

    def status(stage, **extra):
        write(b / 'logs/status.json', dict(stage=stage, checked_unix=time.time(), run_id=h['run_id'],
              execution_id=h['execution_id'], source_commit=h['source_commit'], jobs=jobs, **extra))

    def stop(signum=None, _frame=None):
        for child, row in zip(children, jobs):
            if child.poll() is None and alive(row['process']) and row['process']['sid'] == child.pid:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if signum:
            raise InterruptedError('Human/native signal ' + str(signum))

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

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        source()
        acceptance = binding['parent_acceptance']()
        assert not (b / 'train').exists(), 'Training-only repair requires a fresh output directory'
        assert h['fit_config'].get('resume') is None, 'Startup repair cannot resume a checkpoint'
        assert h['fit_config'] == h['parent_handle']['fit_config'], 'The scientific 200-epoch recipe changed'
        status('ACCEPTED_COMPLETE_PARENT_CACHE_FOR_FRESH_TRAINING', parent_cache_base=h['parent_cache_base'],
               packed_features_reused=2229760, data_or_features_copied=False)
        root = Path('/data/shaoshitong/Search-G1-qprobe-shared/token-feature-gpu-locks')
        root.mkdir(parents=True, exist_ok=True)
        assert root.resolve() == root and not root.is_symlink()
        while True:
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
        # The same open file descriptions are inherited by the torchrun parent
        # and held by this worker for its complete lifetime, including cleanup.
        fds = tuple(lease.fileno() for lease in leases)
        lease_metadata = [dict(uuid=uuid, device=os.fstat(lease.fileno()).st_dev,
                              inode=os.fstat(lease.fileno()).st_ino, inherited_fd=lease.fileno())
                          for (_, uuid), lease in zip(selected, leases)]
        gpu_binding = dict(run_id=h['run_id'], execution_id=h['execution_id'], checked_unix=time.time(),
            indices=[value[0] for value in selected], uuids=[value[1] for value in selected], inventory=rows,
            auxiliary_gpu_used=False, foreign_tasks_signalled=False, inherited_open_file_description_leases=lease_metadata)
        write(b / 'logs/gpu-binding.json', gpu_binding, True)
        # Recheck terminal parent ownership immediately before creating the one
        # trainer, even if this worker waited for GPU admission.
        assert not binding['bound_processes'](Path(h['parent_cache_base']))
        assert binding['sha'](Path(h['parent_cache_base']) / 'features/manifest.json') == h['parent_manifest_sha256']
        assert not binding['bound_processes'](b, exclude=(os.getpid(),))
        fit = dict(h['fit_config'], feature_manifest=acceptance['parent_feature_manifest_path'],
            feature_manifest_sha256=h['parent_manifest_sha256'], model_dir=h['model_dir'], output_dir=str(b / 'train'),
            run_id=h['run_id'], execution_id=h['execution_id'], source_commit=h['source_commit'],
            physical_gpu_uuids=[value[1] for value in selected], expected_train_groups=445440, expected_test_groups=512,
            deployment_handle_path=str(b / 'preparation/deployment.json'), resume=None)
        write(b / 'preparation/fit-config.json', fit, True)
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            CUDA_VISIBLE_DEVICES=','.join(value[1] for value in selected), OMP_NUM_THREADS='8',
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', NCCL_NVLS_ENABLE='0',
            QUESTION5_GPU_LEASE_FDS=json.dumps(list(fds)), QUESTION5_GPU_LEASE_UUIDS=json.dumps([value[1] for value in selected]))
        status('FORMAL_PROBE_WARMUP200_STARTING', startup_repair=True, state_resume=False)
        command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=4',
                   '-m', 'reproduction.question5_ema_warmup.train', '--config', str(b / 'preparation/fit-config.json')]
        with (b / 'logs/probe-train.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=fds)
        children.append(child)
        row = dict(name='probe-train', process=birth(child.pid), exit_code=None)
        jobs.append(row)
        status('CHILD_STARTED', current_job='probe-train')
        row['exit_code'] = child.wait()
        status('CHILD_EXITED', current_job='probe-train')
        if row['exit_code']:
            raise RuntimeError('Owned child exited: probe-train code=' + str(row['exit_code']))
        status('PROBE_WARMUP200_CHILD_RETURNED_ZERO')
        write(b / 'logs/exit.json', dict(exit_code=0, checked_unix=time.time(), run_id=h['run_id'],
            execution_id=h['execution_id'], training_complete_requires_native_200_metrics_and_checkpoint_audit=True,
            startup_repair=True, state_resume=False))
    except BaseException as exc:
        stop()
        for child, row in zip(children, jobs):
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if alive(row['process']) and row['process']['sid'] == child.pid:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                child.wait(timeout=30)
            row['exit_code'] = child.returncode
        write(b / 'logs/failure.json', dict(error_class=type(exc).__name__, checked_unix=time.time(),
            run_id=h['run_id'], execution_id=h['execution_id'], parent_cache_base=h['parent_cache_base']))
        write(b / 'logs/exit.json', dict(exit_code=1, checked_unix=time.time(), run_id=h['run_id'], execution_id=h['execution_id']))
        traceback.print_exc()
        raise
    finally:
        for lease in leases:
            lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['stage', 'launch', 'observe'])
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
    code = {'stage': STAGE, 'launch': LAUNCH, 'observe': OBSERVE}[args.action]
    started = time.time()
    result = subprocess.run(route(code), input=json.dumps(payload).encode(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=65)
    record = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time() - started,
        exit_code=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode(),
        source_commit=h['source_commit'], run_id=h['run_id'], execution_id=h['execution_id'])
    out = ROOT / 'preparation' / ('question5-cached-train-' + args.action + '-' + str(time.time_ns()) + '.json')
    out.write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError('Action failed; retained receipt ' + str(out) + ': ' + record['stderr'][:1000])
    assert len(result.stdout) <= 128 * 1024
    print(json.dumps(dict(receipt=str(out), result=json.loads(result.stdout))))


if __name__ == '__main__':
    main()
