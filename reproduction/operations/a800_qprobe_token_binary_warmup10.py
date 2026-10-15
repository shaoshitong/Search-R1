"""Isolated, one-shot token/binary warmup operations; never launches the actor.

Create the handle only after publishing the exact new runtime. Then use stage,
prepare, observe, stop-once, audit-old-cleanup, and run-once in that order.
repair-old-cleanup-once is a separate recovery for the pinned old native cleanup
race, after its pipeline and trainer births have exited. An uncertain mutation
is observed through the same handle, never automatically retried.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
RUNTIME = ROOT/'qprobe-a800-token-binary-warmup10'
OLD_RUNTIME = ROOT/'qprobe-a800-sigma3-warmup10'
OLD_HANDLE = ROOT/'preparation/qprobe-a800-sigma3-warmup10-deployment.json'
HUMAN = ROOT/'preparation/qprobe-token-binary-warmup10-human-intent.json'
OLD_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-4a017644c8054de38b05314879527303'
OLD_SOURCE = 'ab8f8ca07f19419d7558f916d98513055cc3f71c'
OLD_RUN = 'f4fb19b1fede4dffaac823b8937f35cf'
OLD_EXECUTION = 'fa8986ebec0e4be78f15e79dd05fb7fa'
OLD_PID, OLD_TICKS = 98452, 614134225
OLD_TRAINER, OLD_TRAINER_TICKS = 105205, 614180120
LIBRARY = ROOT/'actor-probes-ten'
LIBRARY_SOURCE = 'c4d156de124de7a94652d75858bcbabc8c56083c'
REMOTE_LIBRARY = '/data/shaoshitong/Search-G1-a800-fresh200-6047f7f9674b49fdad718a256ab3db95/repo'
ROLE = 'a800_token_binary_warmup10_only_intent'
PREFIX = '/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-'
TEACHER = '/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter'
PYTHON = '/data/search-g1/env/bin/python'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve() == path and not path.is_symlink()
    assert path.stat().st_size <= 16*1024**2
    return json.loads(path.read_bytes())


def once(path, value):
    path = Path(path)
    assert path.parent.is_dir() and path.parent.resolve() == path.parent
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(canonical(value)+b'\n'); out.flush(); os.fsync(out.fileno())


def runtime_module():
    assert subprocess.check_output(['git', '-C', str(OLD_RUNTIME), 'rev-parse', 'HEAD'], text=True).strip() == OLD_SOURCE
    path = OLD_RUNTIME/'reproduction/a800_common/deploy.py'
    old = document(OLD_HANDLE)
    assert old['base'] == OLD_BASE and old['source_commit'] == OLD_SOURCE
    assert digest(path) == old['source_sha256']['reproduction/a800_common/deploy.py']
    spec = importlib.util.spec_from_file_location('token_binary_existing_ssh_route', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def operator_commit():
    commit = subprocess.check_output(['git', '-C', str(OPS), 'rev-parse', 'HEAD'], text=True).strip()
    relative = str(Path(__file__).resolve().relative_to(OPS))
    raw = subprocess.check_output(['git', '-C', str(OPS), 'show', commit+':'+relative])
    assert raw == Path(__file__).read_bytes(), 'Publish this exact operations entry before use'
    return commit


def authority(intent):
    human = document(HUMAN)
    assert digest(HUMAN) == intent['human_intent_sha256'], 'Human intent changed; reassess the same handle'
    assert human['role'] == 'latest_human_token_binary_probe_warmup10_override'
    expected = dict(same_training_groups=5120, warmup_head_epochs=10, head_batch_size=256,
                    expected_head_adam_updates=200, selected_questions=512, exploration_questions=0,
                    auxiliary_weight=.3, independent_validation=False, no_new_teacher_trajectories=True,
                    count6_loss=False, sigma3_loss=False, old_runtime_hotedit_forbidden=True)
    assert all(human.get(key) == value for key, value in expected.items())
    assert human['old_run_during_preparation'] == OLD_RUN and human['old_execution_during_preparation'] == OLD_EXECUTION
    assert human['selection_score'] == 'binary_mixed_probability_descending'
    assert human['main_loss'] == 'BCEWithLogits' and human['auxiliary_regression_target'] == 'sample_std'
    assert intent['role'] == ROLE and intent['host'] == 'baai-a800'
    assert re.fullmatch('[0-9a-f]{32}', intent['deployment_id'])
    assert intent['base'] == PREFIX+intent['deployment_id']
    assert intent['old_intent'] == document(OLD_HANDLE)
    assert intent['old_intent']['base'] == OLD_BASE and intent['old_intent']['source_commit'] == OLD_SOURCE


COMMON = r'''
import csv,hashlib,io,json,math,os,re,signal,socket,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sealed(path,limit=16*1024**2):
 path=Path(path);assert path.resolve()==path and not path.is_symlink()
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd);assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_size<=limit
  chunks=[];remaining=before.st_size
  while remaining:
   part=os.read(fd,min(1024**2,remaining));assert part;chunks.append(part);remaining-=len(part)
  raw=b''.join(chunks);after=os.fstat(fd)
  assert len(raw)==before.st_size and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert path.stat().st_ino==before.st_ino
 finally:os.close(fd)
 return raw
def document(path,limit=16*1024**2):return json.loads(sealed(path,limit))
def sha(path,limit=16*1024**2):return hashlib.sha256(sealed(path,limit)).hexdigest()
def once(path,value):
 raw=canonical(value)+b'\n';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
def source(repo,commit,mapping,allow_hydra=False):
 repo=Path(repo);assert repo.resolve()==repo and repo.stat().st_uid==os.getuid()
 def git(*args,**kwargs):return subprocess.check_output(['git','-C',str(repo),*args],timeout=30,**kwargs)
 assert git('rev-parse','HEAD').decode().strip()==commit and not git('status','--porcelain').strip()
 def selected(name):return (Path(name).suffix in ('.py','.sh','.yaml','.yml','.json') or name in ('verl/version/version','verl/single_controller/version/version')) and not {'.git','__pycache__','evidence'}.intersection(Path(name).parts)
 names=sorted(x.decode() for x in git('ls-tree','-rz','--name-only',commit).split(b'\0') if x and selected(x.decode()))
 assert names==sorted(mapping)
 blobs=io.BytesIO(git('cat-file','--batch',input=''.join(commit+':'+name+'\n' for name in names).encode()))
 for name in names:
  _,kind,size=blobs.readline().decode().strip().split();assert kind=='blob'
  raw=blobs.read(int(size));assert blobs.read(1)==b'\n'
  assert hashlib.sha256(raw).hexdigest()==mapping[name]==sha(repo/name)
 assert not blobs.read()
 for path in repo.rglob('*'):
  if path.is_file() and selected(str(path.relative_to(repo))) and str(path.relative_to(repo)) not in mapping:
   assert allow_hydra and re.fullmatch(r'outputs/\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}/\.hydra/(config|hydra|overrides)\.yaml',str(path.relative_to(repo)))
def process(record):
 try:
  p=Path('/proc')/str(record['pid']);text=(p/'stat').read_text();fields=text[text.rfind(')')+2:].split()
  matches=(int(fields[19]),int(fields[3]),p.stat().st_uid)==(record['start_ticks'],record['sid'],record['uid'])
  return dict(birth_matches=matches,alive=matches and fields[0] not in ('Z','X'),state=fields[0] if matches else None)
 except (FileNotFoundError,ProcessLookupError):return dict(birth_matches=False,alive=False,state=None)
def birth(pid):
 p=Path('/proc')/str(pid);text=(p/'stat').read_text();fields=text[text.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=p.stat().st_uid)
def emit(value):print('TOKEN_WARMUP_JSON:'+json.dumps(value,sort_keys=True,allow_nan=False),flush=True)
def bind(payload):
 global p,intent,base,repo,output
 p=payload;intent=p['intent'];base=Path(intent['base']);repo=base/'repo';output=base/'warmup'
 assert intent['role']=='a800_token_binary_warmup10_only_intent' and intent['host']=='baai-a800'
 assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id']) and str(base)=='/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-'+intent['deployment_id']
 assert base.resolve()==base and base.stat().st_uid==os.getuid() and document(base/'preparation/intent.json')==intent
 assert sha(base/'preparation/source-publication.json')==intent['source_publication_sha256']
 assert document(base/'preparation/source-publication.json')['source_commit']==intent['source_commit']
 source(repo,intent['source_commit'],intent['source_sha256'])
def environment(gpu=''):
 env=dict(os.environ)
 for key in tuple(env):
  if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
 env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+env.get('PATH','/usr/bin:/bin'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8')
 env.update({name:str(base/path) for name,path in dict(TMPDIR='tmp',HF_HOME='cache/hf',XDG_CACHE_HOME='cache',TRITON_CACHE_DIR='cache/triton',TORCHINDUCTOR_CACHE_DIR='cache/inductor').items()})
 return env
def runner(gpu=None):
 args=['/data/search-g1/env/bin/python','-m','reproduction.a800_common.qprobe_token_warmup','--source-run-base',intent['old_intent']['base'],'--output-dir',str(output),'--source-commit',intent['source_commit'],'--source-publication',str(base/'preparation/source-publication.json'),'--teacher-identity-sha256',intent['teacher_identity_sha256'],'--auxiliary-weight','0.3','--target-statistic','sample_std']
 return args+(['--prepare-only'] if gpu is None else ['--physical-index',str(gpu)])
'''

STAGE = COMMON+r'''
import shutil,tarfile
archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|');item=archive.next()
assert item and item.name=='request.json' and item.isfile() and item.size<=4*1024**2
p=json.load(archive.extractfile(item));intent=p['intent'];base=Path(intent['base']);repo=base/'repo'
assert intent['role']=='a800_token_binary_warmup10_only_intent' and re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
assert str(base)=='/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-'+intent['deployment_id'] and base.parent.resolve()==base.parent
base.mkdir(mode=0o700,exist_ok=False)
for relative in ('logs','preparation','tmp','cache/hf','cache/triton','cache/inductor'):(base/relative).mkdir(parents=True,exist_ok=True)
once(base/'preparation/intent.json',intent)
expected={'preparation/published-source.bundle':(intent['bundle_bytes'],intent['bundle_sha256']),'preparation/source-publication.json':(intent['source_publication_bytes'],intent['source_publication_sha256'])};seen=set()
for item in archive:
 assert item.isfile() and item.name in expected and item.name not in seen
 size,digest=expected[item.name];assert item.size==size
 path=base/item.name;raw=archive.extractfile(item).read(size+1);assert len(raw)==size and hashlib.sha256(raw).hexdigest()==digest
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
 seen.add(item.name)
assert seen==set(expected)
env=dict(os.environ,GIT_TERMINAL_PROMPT='0');direct=False
with (base/'preparation/github-clone.log').open('xb') as log:
 try:
  direct=subprocess.run(['git','clone','--no-checkout','--single-branch','--branch',intent['published_ref'].removeprefix('refs/heads/'),'https://github.com/shaoshitong/Search-R1.git',str(repo)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=90).returncode==0
 except subprocess.TimeoutExpired:pass
if not direct:
 if repo.exists():assert repo.resolve()==repo;shutil.rmtree(repo)
 with (base/'preparation/bundle-clone.log').open('xb') as log:
  subprocess.run(['git','clone','--no-checkout',str(base/'preparation/published-source.bundle'),str(repo)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=90,check=True)
subprocess.run(['git','-C',str(repo),'checkout','--detach',intent['source_commit']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,timeout=30,check=True)
source(repo,intent['source_commit'],intent['source_sha256'])
receipt=dict(stage='PUBLISHED_SOURCE_STAGED_CPU_ONLY',source_commit=intent['source_commit'],source_map_sha256=hashlib.sha256(canonical(intent['source_sha256'])).hexdigest(),source_transport='github_direct' if direct else 'published_git_bundle_relay',actor_trainer_started=False,gpu_used=False,base=str(base),checked_unix=time.time())
once(base/'logs/staged.json',receipt);emit(receipt)
'''

WORKER = r'''
import json,os,subprocess,sys,time
from pathlib import Path
request=json.loads(sys.argv[1]);base=Path(request['base']);action=request['action'];status=dict(action=action,started_unix=time.time())
try:status['exit_code']=subprocess.call(request['args'],cwd=request['repo'],stdin=subprocess.DEVNULL)
except BaseException as error:status.update(exit_code=None,failure_type=type(error).__name__);raise
finally:
 status['finished_unix']=time.time();fd=os.open(base/'logs'/(action+'-exit.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as out:json.dump(status,out,allow_nan=False);out.flush();os.fsync(out.fileno())
'''

OLD = r'''
oldbase=Path(intent['old_intent']['base']);oldrepo=oldbase/'repo';old=document(oldbase/'logs/run-manifest.json')
assert str(oldbase)=='/data/shaoshitong/Search-G1-a800-fresh200-4a017644c8054de38b05314879527303'
assert document(oldbase/'preparation/qprobe-intent.json')==intent['old_intent']
assert (old['run_id'],old['execution_id'],old['source_commit'])==('f4fb19b1fede4dffaac823b8937f35cf','fa8986ebec0e4be78f15e79dd05fb7fa','ab8f8ca07f19419d7558f916d98513055cc3f71c')
assert old['execution_backend']=='ssh_local' and old['slurm_job_id'] is None and old['execution_host']==socket.gethostname()
source(oldrepo,old['source_commit'],intent['old_intent']['source_sha256'],allow_hydra=True)
claim=document(oldbase/'logs/pipeline-launch-claim.json');launch=document(oldbase/'logs/deployment-launch-receipt.json')
assert claim['run_id']==launch['run_id']==old['run_id'] and claim['execution_id']==launch['execution_id']==old['execution_id']
producer={key:old['pipeline_identity'][key] for key in ('pid','start_ticks','sid','uid')}
assert producer=={key:claim['pipeline_identity'][key] for key in producer} and producer==dict(pid=98452,start_ticks=614134225,sid=98452,uid=os.getuid())
assert (launch['pid'],launch['start_ticks'],launch['sid'])==(98452,614134225,98452)
ledger=document(oldbase/'logs/process-ledger.json',256*1024**2);assert ledger['run_id']==old['run_id']
trainer={key:ledger['jobs']['trainer']['leader'][key] for key in ('pid','start_ticks','sid','uid')}
assert trainer==dict(pid=105205,start_ticks=614180120,sid=105205,uid=os.getuid())
def isolated_library():
 library=Path(intent['cleanup_library_repo']);source(library,intent['cleanup_library_source'],intent['cleanup_library_source_sha256'],allow_hydra=True)
 # Keep the C4 auditor and all its package imports out of old-runtime modules.
 for name in tuple(sys.modules):
  if name in ('reproduction','search_g1') or name.startswith(('reproduction.','search_g1.')):del sys.modules[name]
 sys.path[:]=[str(library)]+[x for x in sys.path if x not in (str(repo),str(oldrepo),str(library),'')]
 from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup,birth_inventory
 from reproduction.a800.owned_processes import owned_session_survivors,remaining_cuda_contexts,identity
 for name,module in tuple(sys.modules.items()):
  if not (name in ('reproduction','search_g1') or name.startswith(('reproduction.','search_g1.'))):continue
  path=getattr(module,'__file__',None)
  if path is None:continue
  path=Path(path).resolve();relative=str(path.relative_to(library));assert sha(path)==intent['cleanup_library_source_sha256'][relative]
 return verify_cleanup,birth_inventory,owned_session_survivors,remaining_cuda_contexts,identity
def audit():
 verify,inventory,_,_,_=isolated_library();saved=document(base/'logs/old-stop-intent.json')
 assert saved['run_id']==old['run_id'] and saved['execution_id']==old['execution_id']
 actual=verify(oldbase,old,preserved_births=saved['preserved_births'])
 assert actual['accepted'] is True
 return actual
'''

PREPARE = COMMON+r'''
bind(json.load(sys.stdin));assert document(base/'logs/staged.json')['source_commit']==intent['source_commit']
assert not (base/'logs/prepare-launch.json').exists() and not (base/'logs/prepare-exit.json').exists()
args=runner();once(base/'logs/prepare-once.json',dict(source_commit=intent['source_commit'],human_intent_sha256=intent['human_intent_sha256'],no_automatic_retry=True,claimed_unix=time.time()))
with (base/'logs/prepare.log').open('xb') as log:
 child=subprocess.Popen(['/data/search-g1/env/bin/python','-c',p['worker'],json.dumps(dict(base=str(base),repo=str(repo),action='prepare',args=args))],cwd=repo,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(action='prepare',source_commit=intent['source_commit'],process=birth(child.pid),gpu_used=False,actor_trainer_started=False,launched_unix=time.time())
 once(base/'logs/prepare-launch.json',record)
emit(record)
'''

STOP = COMMON+r'''
bind(json.load(sys.stdin))
ready=p['ready'];assert ready['role']=='token_binary_warmup10_reviewed_prepared_ready'
assert ready['source_commit']==intent['source_commit'] and ready['source_map_sha256']==hashlib.sha256(canonical(intent['source_sha256'])).hexdigest()
assert ready['independent_code_review_approved'] is True and ready['human_intent_sha256']==intent['human_intent_sha256']
assert ready['source_publication_sha256']==intent['source_publication_sha256']
assert document(base/'logs/prepare-exit.json')['exit_code']==0 and sha(output/'prepared.json')==ready['remote_prepared_sha256']
assert not (base/'logs/run-once.json').exists() and not (output/'run-claim.json').exists()
'''+OLD+r'''
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
_,inventory,_,_,_=isolated_library();preserved=inventory(oldbase,old)
snapshot=dict(run_id=old['run_id'],execution_id=old['execution_id'],source_commit=old['source_commit'],pipeline_identity=producer,trainer_identity=trainer,preserved_births=preserved,human_intent_sha256=intent['human_intent_sha256'],ready_sha256=hashlib.sha256(canonical(ready)).hexdigest(),checked_unix=time.time())
once(base/'logs/old-stop-intent.json',snapshot)
fd=os.pidfd_open(producer['pid'])
try:
 actual=process(producer);assert actual['alive'] is True
 command=(Path('/proc')/str(producer['pid'])/'cmdline').read_bytes()
 assert b'reproduction.a800_common.run_pipeline' in command and str(oldbase).encode() in command
 signal.pidfd_send_signal(fd,signal.SIGTERM)
finally:os.close(fd)
receipt=dict(stage='OLD_PINNED_PIPELINE_SIGTERM_SENT_ONCE',pipeline_identity=producer,trainer_identity=trainer,stop_intent_sha256=sha(base/'logs/old-stop-intent.json'),all_old_files_preserved=True,cleanup_accepted=False,no_automatic_retry=True,checked_unix=time.time())
once(base/'logs/old-stop-signal.json',receipt);emit(receipt)
'''

AUDIT = COMMON+r'''
bind(json.load(sys.stdin))
'''+OLD+r'''
actual=audit();receipt=dict(stage='OLD_ACTUAL_OWNED_CLEANUP_ACCEPTED',accepted=True,verification=actual,old_run_id=old['run_id'],old_execution_id=old['execution_id'],source_commit=intent['source_commit'],checked_unix=time.time())
emit(receipt)
'''

REPAIR = COMMON+r'''
bind(json.load(sys.stdin))
'''+OLD+r'''
assert not process(producer)['alive'] and not process(trainer)['alive'], 'Pinned pipeline/trainer must have exited before cleanup recovery'
verify,inventory,survivors,contexts,identity=isolated_library()
saved=document(base/'logs/old-stop-intent.json');current=inventory(oldbase,old)
from reproduction.common8.actor_probe_owned_cleanup import verify_preserved_births
verify_preserved_births(saved['preserved_births'],current)
leaders=[job['leader'] for job in current['jobs'].values()]+[producer]
records=[producer]+[row for job in current['jobs'].values() for row in job['births']]
live,reused=survivors(leaders);assert not reused, 'Reused private session; refuse cleanup'
known={tuple(row[key] for key in ('pid','start_ticks','sid','uid')) for row in records}
assert all(tuple(row[key] for key in ('pid','start_ticks','sid','uid')) in known for row in live), 'Unbound private-session process; refuse cleanup'
for row in records:
 actual=process(row)
 assert not actual['alive'], 'Real owned process remains live; this repair only resolves the terminal native cleanup race'
status=document(oldbase/'logs/pipeline-status.json');assert status['stage']=='OWNED_CLEANUP_FAILED'
assert 'Owned component cleanup incomplete' in status.get('error','') or any('Owned component cleanup incomplete' in job.get('cleanup_error','') for job in ledger['jobs'].values())
archive=base/'logs/old-cleanup-original';archive.mkdir(mode=0o700,exist_ok=False);archived=[]
for relative,limit in (('logs/process-ledger.json',256*1024**2),('logs/pipeline-status.json',16*1024**2),('logs/cleanup-status.json',16*1024**2),('logs/corpus-cache-claim.json',16*1024**2),('logs/corpus-cache-cleanup.json',16*1024**2)):
 path=oldbase/relative
 if not path.exists():continue
 raw=sealed(path,limit);target=archive/Path(relative).name;fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
 archived.append(dict(original_path=str(path),archive_path=str(target),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
once(archive/'manifest.json',dict(files=archived,original_failure_preserved=True,checked_unix=time.time()))
once(base/'logs/old-cleanup-repair-once.json',dict(no_automatic_retry=True,archived_manifest_sha256=sha(archive/'manifest.json'),checked_unix=time.time()))
# Use the original ab8 cleanup implementation only after terminal birth proof.
for name in tuple(sys.modules):
 if name in ('reproduction','search_g1') or name.startswith(('reproduction.','search_g1.')):del sys.modules[name]
sys.path[:]=[str(oldrepo)]+[x for x in sys.path if x not in (str(repo),str(oldrepo),intent['cleanup_library_repo'],'')]
from reproduction.a800.owned_processes import OwnedProcesses,publish
from reproduction.a800_common.cache import cleanup
owned=OwnedProcesses(oldbase/'logs/process-ledger.json',old['run_id']);assert owned.recovery_only is True
owned.stop_all();cleanup(oldbase)
# Retained C4 functions keep their original module globals after sys.modules
# isolation; ab8 intentionally supplies only its native recovery implementation.
actual,reused=survivors(leaders);assert not actual and not reused
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20))))
remaining,reused_pids=contexts(records,rows);assert not remaining
assert all(Path(path).exists() for path in old['retrieval_asset_paths'].values()) and Path(old['initial_model']).is_dir() and Path(old['reference_model']).is_dir()
publish(oldbase/'logs/cleanup-status.json',dict(run_id=old['run_id'],execution_id=old['execution_id'],execution_backend='ssh_local',slurm_job_id=None,cleaned=True,scope='only recorded project PID births and private local corpus cache',checked_unix=time.time(),terminal_native_cleanup_repair_archive=str(archive)))
actual=audit();receipt=dict(stage='OLD_TERMINAL_NATIVE_CLEANUP_REPAIRED_AND_ACTUALLY_VERIFIED',accepted=True,verification=actual,original_failure_preserved=True,original_archive_manifest_sha256=sha(archive/'manifest.json'),checked_unix=time.time())
once(base/'logs/old-cleanup-repair-result.json',receipt);emit(receipt)
'''

RUN = COMMON+r'''
bind(json.load(sys.stdin));assert document(base/'logs/prepare-exit.json')['exit_code']==0 and (output/'prepared.json').is_file()
'''+OLD+r'''
actual=audit();assert actual['accepted'] is True
index=p['physical_index'];assert type(index) is int and index>=0 and index not in (5,6)
rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20))))
inventory={int(row[0]):(row[1].strip(),float(row[2])) for row in rows};assert index in inventory
gpu,used=inventory[index];assert used<=256, 'Requested physical GPU is occupied'
apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20))))
assert not any(row and row[0].strip()==gpu for row in apps), 'Requested physical GPU has an existing compute process'
assert not (base/'logs/run-launch.json').exists() and not (output/'run-claim.json').exists()
once(base/'logs/run-once.json',dict(source_commit=intent['source_commit'],human_intent_sha256=intent['human_intent_sha256'],physical_index=index,gpu_uuid=gpu,old_actual_cleanup=actual,no_automatic_retry=True,claimed_unix=time.time()))
with (base/'logs/run.log').open('xb') as log:
 child=subprocess.Popen(['/data/search-g1/env/bin/python','-c',p['worker'],json.dumps(dict(base=str(base),repo=str(repo),action='run',args=runner(index)))],cwd=repo,env=environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(stage='ONE_REAL_TOKEN_BINARY_WARMUP_LAUNCHED',process=birth(child.pid),physical_index=index,gpu_uuid=gpu,source_commit=intent['source_commit'],actor_trainer_started=False,formal_training_started=False,no_new_teacher_rollouts=True,launched_unix=time.time())
 once(base/'logs/run-launch.json',record)
emit(record)
'''

OBSERVE = COMMON+r'''
bind(json.load(sys.stdin));result=dict(stage='EXISTING_METADATA_ONLY_OBSERVATION',base=str(base),source_commit=intent['source_commit'],metadata_only=True,remote_mutation=False,files={},checked_unix=time.time())
def scalars(value,depth=0):
 assert depth<12
 if value is None or type(value) in (bool,int):return value
 if type(value) is float:return value if math.isfinite(value) else None
 if type(value) is str:
  return value if len(value)<=512 and (re.fullmatch(r'[A-Za-z0-9_./:=+@ -]{0,512}',value) is not None) else '[text omitted]'
 if type(value) is list:return [scalars(x,depth+1) for x in value[:256]]
 if type(value) is dict:
  return {str(k):scalars(v,depth+1) for k,v in value.items() if str(k).endswith(('_sha256','_sha')) or not any(part in str(k).lower() for part in ('prompt','answer','question_text','token_ids','question_ids','trajectory_ids','label_vector','example','traceback'))}
 raise AssertionError('Unexpected metadata value')
for relative in ('logs/staged.json','logs/prepare-launch.json','logs/prepare-exit.json','logs/run-launch.json','logs/run-exit.json','logs/old-stop-signal.json','logs/old-cleanup-repair-result.json','warmup/prepare-claim.json','warmup/run-claim.json','warmup/prepared.json','warmup/progress.json','warmup/global-scan.json','warmup/ready.json','warmup/prepare-failure.json','warmup/run-failure.json','warmup/warmup-metrics.json'):
 path=base/relative
 if not path.exists():continue
 raw=sealed(path,2*1024**2);value=json.loads(raw);item=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),metadata=scalars(value))
 if relative in ('logs/prepare-launch.json','logs/run-launch.json'):item['current_process']=process(value['process'])
 result['files'][relative]=item
for action in ('prepare','run'):
 path=base/'logs'/(action+'.log')
 if path.exists():
  info=path.stat();result[action+'_log']=dict(bytes=info.st_size,modified_unix=info.st_mtime)
  with path.open('rb') as stream:stream.seek(max(0,info.st_size-8192));tail=stream.read()
  result[action+'_log']['exception_types']=sorted(set(x.decode() for x in re.findall(rb'(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',tail)))
emit(result)
'''


def create(args):
    module = runtime_module()
    repo = args.source_repo.resolve(); commit = args.source_commit
    assert re.fullmatch('[0-9a-f]{40}', commit)
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
    mapping = module.published_source(repo, commit)
    assert all(digest(repo/name) == value for name, value in mapping.items())
    publication = document(args.source_publication)
    assert publication['source_commit'] == publication['remote_commit'] == commit
    assert publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['source_map_sha256'] == hashlib.sha256(canonical(mapping)).hexdigest()
    assert re.fullmatch('refs/heads/[A-Za-z0-9_./-]+', publication['published_ref'])
    assert re.fullmatch('[0-9a-f]{64}', args.teacher_identity_sha256)
    bundle = args.bundle.resolve()
    subprocess.run(['git', '-C', str(repo), 'bundle', 'verify', str(bundle)], capture_output=True, check=True)
    assert subprocess.check_output(['git', '-C', str(repo), 'bundle', 'list-heads', str(bundle)], text=True).split()[0] == commit
    identifier = uuid.uuid4().hex
    intent = dict(role=ROLE, deployment_id=identifier, host='baai-a800', base=PREFIX+identifier,
                  created_unix=time.time(), source_commit=commit, source_sha256=mapping,
                  published_ref=publication['published_ref'], source_repo=str(repo),
                  source_publication=str(args.source_publication.resolve()), source_publication_bytes=args.source_publication.stat().st_size,
                  source_publication_sha256=digest(args.source_publication), bundle=str(bundle),
                  bundle_bytes=bundle.stat().st_size, bundle_sha256=digest(bundle),
                  human_intent_sha256=digest(HUMAN), old_intent=document(OLD_HANDLE),
                  teacher_cache=TEACHER, teacher_identity_sha256=args.teacher_identity_sha256,
                  cleanup_library_repo=REMOTE_LIBRARY, cleanup_library_source=LIBRARY_SOURCE,
                  cleanup_library_source_sha256=module.published_source(LIBRARY, LIBRARY_SOURCE),
                  warmup_only=True, actor_trainer_launch_authorized=False,
                  expected_schedule=dict(questions=5120, epochs=10, batch_size=256, adam_updates=200),
                  loss=dict(binary_bce_weight=1.0, normalized_sample_std_mse_weight=.3),
                  selection=dict(top_k=512, exploration=0, score='binary_mixed_probability_descending'),
                  local_storage='code_config_scalar_hash_only', no_automatic_retry=True)
    authority(intent)
    handle = args.handle.resolve() if args.handle else ROOT/'preparation'/('qprobe-token-binary-warmup10-'+identifier+'.json')
    once(handle, intent)
    print(json.dumps(dict(handle=str(handle), base=intent['base'], source_commit=commit)))


def perform(args):
    commit = operator_commit(); handle = args.handle.resolve(); intent = document(handle); authority(intent)
    payload = dict(intent=intent, operator_commit=commit, worker=WORKER)
    action = args.action
    if action == 'stop-once':
        assert args.ready is not None
        payload['ready'] = document(args.ready)
    if action == 'run-once':
        assert args.physical_index is not None
        payload['physical_index'] = args.physical_index
    codes = dict(stage=STAGE, prepare=PREPARE, **{'stop-once':STOP, 'audit-old-cleanup':AUDIT,
                 'repair-old-cleanup-once':REPAIR, 'run-once':RUN, 'observe':OBSERVE})
    code = codes[action]; ast.parse(code)
    if action not in ('observe', 'audit-old-cleanup'):
        once(handle.with_name(handle.stem+'.'+action+'.claim.json'), dict(action=action, operator_commit=commit,
             source_commit=intent['source_commit'], human_intent_sha256=intent['human_intent_sha256'],
             claimed_unix=time.time(), no_automatic_retry=True))
    if action == 'stage':
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w') as archive:
            raw = canonical(payload); item=tarfile.TarInfo('request.json'); item.size=len(raw); archive.addfile(item,io.BytesIO(raw))
            for source, name, expected in ((Path(intent['bundle']), 'preparation/published-source.bundle', intent['bundle_sha256']),
                                           (Path(intent['source_publication']), 'preparation/source-publication.json', intent['source_publication_sha256'])):
                assert digest(source) == expected
                with source.open('rb') as original:
                    item=tarfile.TarInfo(name); item.size=source.stat().st_size; archive.addfile(item,original)
        data=stream.getvalue()
    else:data=canonical(payload)
    route=runtime_module().ssh_command(code); assert route[-1].startswith('python3 -c ')
    route[-1]=PYTHON+' -c '+route[-1][len('python3 -c '):]
    receipt=args.receipt.resolve() if args.receipt else handle.with_name(handle.stem+'.'+action+'-'+str(time.time_ns())+'.json')
    result=dict(action=action,operator_commit=commit,source_commit=intent['source_commit'],base=intent['base'],no_automatic_retry=True,observation=None)
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            done=subprocess.run(route,input=data,stdout=out,stderr=err,timeout=400 if action=='stage' else 180 if action=='repair-old-cleanup-once' else 90)
            result['exit_code']=done.returncode
        except (OSError,subprocess.TimeoutExpired) as failure:result.update(exit_code=None,failure_type=type(failure).__name__,status='TRANSPORT_UNKNOWN_OBSERVE_SAME_HANDLE')
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(512*1024+1);err.seek(0)
        errors=err.read(256*1024);result['bounded_stderr_sha256']=hashlib.sha256(errors).hexdigest()
        result['remote_exception_types']=sorted(set(value.decode() for value in re.findall(rb'(?m)^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',errors)))
        result['remote_frames']=[dict(file=Path(name.decode(errors='replace')).name,line=int(line)) for name,line in re.findall(rb'File "([^"]+)", line ([0-9]+)',errors)[-8:]]
    if len(raw)<=512*1024:
        rows=[json.loads(row.split(':',1)[1]) for row in raw.decode(errors='strict').splitlines() if row.startswith('TOKEN_WARMUP_JSON:')]
        assert len(rows)<=1
        if rows:result['observation']=rows[0]
    result['checked_unix']=time.time();once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),**result),allow_nan=False))
    return 0 if result['exit_code']==0 and result['observation'] is not None else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('create','stage','prepare','stop-once','audit-old-cleanup','repair-old-cleanup-once','run-once','observe'))
    parser.add_argument('--handle',type=Path);parser.add_argument('--receipt',type=Path)
    parser.add_argument('--source-repo',type=Path,default=RUNTIME);parser.add_argument('--source-commit')
    parser.add_argument('--source-publication',type=Path);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--teacher-identity-sha256');parser.add_argument('--ready',type=Path);parser.add_argument('--physical-index',type=int)
    args=parser.parse_args()
    if args.action=='create':
        assert all(value is not None for value in (args.source_commit,args.source_publication,args.bundle,args.teacher_identity_sha256))
        return create(args)
    assert args.handle is not None
    return perform(args)


if __name__=='__main__':raise SystemExit(main())
