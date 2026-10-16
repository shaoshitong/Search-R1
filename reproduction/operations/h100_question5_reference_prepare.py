"""Stage/once-launch/observe a CPU-only slim prefix5 derivative on H100.

Uses the existing strict retained18 management transport unchanged. The eight
immutable reference archive parts remain on H100; only small published source
and scalar/hash metadata pass through the Mac. No model, GPU, old runtime or
network/auth setting is modified.
"""

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
import uuid

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO = ROOT / 'repo'
TRANSPORT = ROOT / 'qprobe-prefix20-online1000-h1008-20261016/reproduction/h1008_qprobe/management_transport.py'
READY = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-github-export-20261016T044748Z-111a2d/UPLOAD_READY.json')
PREFIX = '/mnt/data/shaoshitong/search-g1-question5-reference5-20261016-'
CPU_PYTHON = '/dataset/shaoshitong/Search-G1-common8/env/bin/python'
ARCHIVE = 'dataset-through-epoch-0068.tar.gz'
CONCAT_SHA = '085b039051359f341bcc123a952e2ac9500fd890331dc4f7cd2e367c81d39ced'
OPERATOR_REL = 'reproduction/operations/h100_question5_reference_prepare.py'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_base(value):
    if re.fullmatch(re.escape(PREFIX) + r'[0-9a-f]{32}', str(value)) is None:
        raise ValueError('Require this operation\'s independent exact UUID namespace')
    return str(value)


def source_config(ready_path):
    ready_path = Path(ready_path)
    value = json.loads(ready_path.read_text())
    entries = sorted((entry for entry in value['assets'] if entry['archive'] == ARCHIVE), key=lambda entry: entry['name'])
    if len(entries) != 8 or sum(entry['bytes'] for entry in entries) != 8389298946:
        raise ValueError('Require all eight original epoch68 dataset-only parts')
    parts = []
    for number, entry in enumerate(entries):
        if (entry['host'] != 'h100-8' or entry['name'] != ARCHIVE + '.part%03d' % number
                or entry['sha256'] != entry['independent_readback_sha256']
                or not entry['remote_path'].startswith('/mnt/data/shaoshitong/search-g1-github-exports/')
                or '..' in Path(entry['remote_path']).parts):
            raise ValueError('Original reference source identity/readback differs')
        parts.append(dict(path=entry['remote_path'], bytes=entry['bytes'], sha256=entry['sha256']))
    config = dict(schema_version=1, prefix_tokens=5,
                  reference_archive=dict(parts=parts, concatenated_sha256=CONCAT_SHA))
    return config, sha_bytes(ready_path.read_bytes())


def published_archive(commit):
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('Require an exact published source commit')
    local_operator = Path(__file__).read_bytes()
    source_operator = subprocess.check_output(['git', '-C', str(REPO), 'show', commit + ':' + OPERATOR_REL])
    if local_operator != source_operator:
        raise ValueError('Execute the exact published operator source')
    remote = subprocess.check_output(['git', '-C', str(REPO), 'ls-remote', 'origin', 'refs/heads/search-g1'], text=True)
    if remote.split()[0] != commit:
        raise ValueError('Source commit must be the current published search-g1 branch')
    blob = subprocess.check_output(['git', '-C', str(REPO), 'archive', '--format=tar.gz', commit,
                                    'reproduction/question5_ema_warmup', OPERATOR_REL])
    if len(blob) > 8 * 1024**2:
        raise ValueError('Only the small source package may be staged from Mac')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for member in archive:
            if member.isdir():
                continue
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or not (member.name.startswith('reproduction/question5_ema_warmup/') or member.name == OPERATOR_REL)):
                raise ValueError('Only regular explicitly scoped source files may be staged')
            if path.suffix not in ('.py', '.md', '.json', '.txt'):
                raise ValueError('Unexpected non-source asset in the source archive')
            content = archive.extractfile(member).read()
            files[member.name] = dict(bytes=len(content), sha256=sha_bytes(content))
    if 'reproduction/question5_ema_warmup/prepare_reference.py' not in files:
        raise ValueError('Publish the CPU prefix5 preparation module before staging')
    return blob, files


def atomic_local(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if exclusive else path.with_name(path.name + '.partial-%d' % os.getpid())
    with target.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(target, 0o600)
    if not exclusive:
        os.replace(target, path)


COMMON = r'''
import hashlib,json,os,pathlib,re,sys,time
B=pathlib.Path(BASE)
assert re.fullmatch('/mnt/data/shaoshitong/search-g1-question5-reference5-20261016-[0-9a-f]{32}',str(B))
assert B.is_absolute() and '..' not in B.parts
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb')as f:
  for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def once(path,value):
 with pathlib.Path(path).open('x')as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def ident(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  tail=(p/'stat').read_text().rsplit(') ',1)[1].split();uid=p.stat().st_uid
  return dict(pid=pid,birth=int(tail[19]),sid=int(tail[3]),uid=uid,state=tail[0])
 except FileNotFoundError:return None
def alive(owner):
 actual=ident(owner['pid']) if owner else None
 return bool(actual and all(actual[k]==owner[k] for k in('pid','birth','sid','uid')) and actual['state']!='Z')
def check_stage():
 assert B.resolve(strict=True)==B and B.stat().st_uid==os.getuid()
 p=read(B/'preparation/source-receipt.json')
 assert sha(B/'source-archive.tar.gz')==p['source_archive_sha256']
 assert sha(B/'preparation/source-config.json')==p['source_config_sha256']
 assert sha(B/'supervisor.py')==p['supervisor_sha256']
 for name,proof in p['source_files'].items():
  path=B/'repo'/name;assert path.is_file()and not path.is_symlink()and path.resolve(strict=True)==path
  assert path.stat().st_size==proof['bytes']and sha(path)==proof['sha256']
 return p
'''


SUPERVISOR = r'''import hashlib,json,os,pathlib,subprocess,sys,time
B=pathlib.Path(__file__).resolve().parent
def once(path,value):
 with path.open('x')as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def ident(pid):
 p=pathlib.Path('/proc')/str(pid)
 tail=(p/'stat').read_text().rsplit(') ',1)[1].split()
 return dict(pid=pid,birth=int(tail[19]),sid=int(tail[3]),uid=p.stat().st_uid,state=tail[0])
started=time.time();owner=ident(os.getpid());exitcode=1;child=None
try:
 claim=json.loads((B/'preparation/launch-claim.json').read_text())
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.environ.get('OMP_NUM_THREADS')=='2'
 assert os.getpriority(os.PRIO_PROCESS,0)==19
 once(B/'preparation/supervisor-start.json',dict(identity=owner,started_unix=started,nice=19,
      cuda_visible_devices='',omp_threads=2,source_commit=claim['source_commit'],probe_training=False))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
          MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
          HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
          TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(B/'repo'))
 command=[claim['cpu_python'],'-u','-m','reproduction.question5_ema_warmup.prepare_reference',
          '--config',str(B/'preparation/source-config.json'),'--output',str(B/'derived')]
 with(B/'logs/prepare-reference.log').open('xb')as log:
  child=subprocess.Popen(command,cwd=str(B/'repo'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
  identity=ident(child.pid)
  once(B/'preparation/producer-start.json',dict(identity=identity,started_unix=time.time(),
       supervisor_identity=owner,source_commit=claim['source_commit'],cpu_only=True,actor_updates=0))
  exitcode=child.wait()
 complete=B/'derived/REFERENCE-PREPARE-COMPLETE.json'
 if exitcode==0 and not complete.is_file():raise RuntimeError('Successful CPU child lacks completion receipt')
 if exitcode!=0:once(B/'preparation/failure.json',dict(error_type='CPU_PREPARE_NONZERO_EXIT',
       child_exit_code=exitcode,supervisor_identity=owner,observed_unix=time.time()))
except BaseException as error:
 if not(B/'preparation/failure.json').exists():once(B/'preparation/failure.json',dict(error_type=type(error).__name__,
      supervisor_identity=owner,child_exit_code=exitcode,observed_unix=time.time()))
 exitcode=exitcode or 1
finally:
 once(B/'preparation/exit.json',dict(exit_code=exitcode,supervisor_identity=owner,
      child_pid=child.pid if child is not None else None,started_unix=started,finished_unix=time.time(),
      elapsed_seconds=time.time()-started,cpu_only=True,actor_updates=0))
raise SystemExit(exitcode)
'''


def stage_script(payload):
    return COMMON.replace('BASE', repr(payload['base'])) + r'''
import base64,io,tarfile
p=PAYLOAD
if B.exists():
 existing=check_stage()
 assert existing['source_commit']==p['source_commit']and existing['source_archive_sha256']==p['source_archive_sha256']
 assert existing['source_config_sha256']==hashlib.sha256(p['source_config_bytes'].encode()).hexdigest()
 print(json.dumps(dict(stage='ALREADY_EXACT_SOURCE_STAGED_NO_REPLAY',receipt=existing)));raise SystemExit
assert pathlib.Path(p['cpu_python']).is_file()
for entry in p['source_config']['reference_archive']['parts']:
 path=pathlib.Path(entry['path']);assert path.is_file()and not path.is_symlink()and path.resolve(strict=True)==path
 assert path.stat().st_size==entry['bytes']
B.mkdir(mode=0o700);(B/'repo').mkdir(mode=0o700);(B/'preparation').mkdir(mode=0o700);(B/'logs').mkdir(mode=0o700)
archive=base64.b64decode(p['archive_base64']);assert hashlib.sha256(archive).hexdigest()==p['source_archive_sha256']
with(B/'source-archive.tar.gz').open('xb')as f:f.write(archive);f.flush();os.fsync(f.fileno())
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as handle:
 for member in handle:
  relative=pathlib.PurePosixPath(member.name);assert not relative.is_absolute()and'..'not in relative.parts
  target=B/'repo'/member.name
  if member.isdir():target.mkdir(mode=0o700,parents=True,exist_ok=True)
  else:
   assert member.isfile()and member.name in p['source_files']
   target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
   with target.open('xb')as f:f.write(handle.extractfile(member).read());f.flush();os.fsync(f.fileno())
   target.chmod(0o444)
with(B/'preparation/source-config.json').open('x')as f:f.write(p['source_config_bytes']);f.flush();os.fsync(f.fileno())
(B/'preparation/source-config.json').chmod(0o444)
with(B/'supervisor.py').open('x')as f:f.write(p['supervisor_source']);f.flush();os.fsync(f.fileno())
(B/'supervisor.py').chmod(0o444);(B/'source-archive.tar.gz').chmod(0o444)
receipt=dict(stage='EXACT_PUBLISHED_CPU_SOURCE_STAGED',base=str(B),source_commit=p['source_commit'],
 source_archive_sha256=p['source_archive_sha256'],source_archive_bytes=len(archive),source_files=p['source_files'],
 source_config_sha256=sha(B/'preparation/source-config.json'),supervisor_sha256=sha(B/'supervisor.py'),
 upload_ready_sha256=p['upload_ready_sha256'],cpu_python=p['cpu_python'],staged_unix=time.time(),
 archive_sources_stat_match=True,archive_full_sha_this_stage=False,
 archive_full_sha_required_during_prepare=True,old_reference_runtime_modified=False,
 models_or_checkpoints_staged=False,probe_training=False,actor_updates=0)
once(B/'preparation/source-receipt.json',receipt);check_stage();print(json.dumps(dict(stage='STAGED_NOT_LAUNCHED',receipt=receipt)))
'''.replace('PAYLOAD', repr(payload))


def launch_script(handle):
    return COMMON.replace('BASE', repr(handle['base'])) + r'''
import subprocess
receipt=check_stage();claim=B/'preparation/launch-claim.json'
if claim.exists():
 print(json.dumps(dict(stage='LAUNCH_ALREADY_CLAIMED_NO_REPLAY',claim=read(claim),
       launch_receipt=read(B/'preparation/launch.json')if(B/'preparation/launch.json').exists()else None)));raise SystemExit
once(claim,dict(source_commit=receipt['source_commit'],source_config_sha256=receipt['source_config_sha256'],
 cpu_python=receipt['cpu_python'],claimed_unix=time.time(),cpu_only=True,actor_updates=0))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
 NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
with(B/'logs/supervisor.log').open('xb')as log:
 child=subprocess.Popen(['/usr/bin/nice','-n','19','/usr/bin/python3',str(B/'supervisor.py')],
  cwd=str(B),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
owner=ident(child.pid)
result=dict(stage='ONE_CPU_PREPARE_LAUNCHED',identity=owner,base=str(B),source_commit=receipt['source_commit'],
 launch_unix=time.time(),nice=19,omp_threads=2,cuda_visible_devices='',cpu_only=True,actor_updates=0,
 launch_is_not_completion=True)
once(B/'preparation/launch.json',result);print(json.dumps(result))
'''


def observe_script(handle):
    return COMMON.replace('BASE', repr(handle['base'])) + r'''
receipt=check_stage()
def optional(relative):
 path=B/relative
 if not path.exists():return None
 assert path.is_file()and not path.is_symlink()and path.stat().st_size<=2*1024**2
 return read(path)
launch=optional('preparation/launch.json');supervisor=optional('preparation/supervisor-start.json')
producer=optional('preparation/producer-start.json');exited=optional('preparation/exit.json');failure=optional('preparation/failure.json')
progress=optional('derived/PROGRESS.json');complete=optional('derived/REFERENCE-PREPARE-COMPLETE.json')
safe_keys={'stage','status','source_epoch','epoch','prefix_tokens','converted_train','converted_test',
 'train_questions','train_groups','train_trajectories','test_questions','test_groups','test_trajectories',
 'train_question_instances','train_trajectory_instances','fixed_test_questions','fixed_test_trajectories',
 'converted_train_trajectories','converted_test_trajectories','completed_sources','source_count','counts',
 'part_bytes','part_count','parts','completed_parts','compressed_bytes','total_bytes','bytes','sha256','transfer_manifest_sha256',
 'source_archive_sha256','archive_sha256','reference_manifest_sha256','started_unix','updated_unix',
 'observed_unix','checked_unix','finished_unix','elapsed_seconds','new_rollouts','new_em_scoring',
 'actor_updates','encoder_calls','gpu_used'}
def scalar_view(value):
 if value is None:return None
 result={}
 for key,item in value.items():
  if key in safe_keys and isinstance(item,(str,int,float,bool,type(None))):result[key]=item
  elif key=='counts'and isinstance(item,dict):
   result[key]={k:v for k,v in item.items()if isinstance(v,(int,float,bool,type(None)))and len(k)<80}
 return result
logs={}
for relative in('logs/prepare-reference.log','logs/supervisor.log'):
 path=B/relative
 if path.exists():
  info=path.stat();logs[relative]=dict(bytes=info.st_size,mtime_unix=info.st_mtime)
  with path.open('rb')as f:f.seek(max(0,info.st_size-8192));tail=f.read(8192).decode('utf-8','replace')
  logs[relative]['error_classes']=sorted(set(re.findall(r'(?m)^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):',tail)))
transfer=B/'derived/transfer-manifest.json';transfer_proof=None
if transfer.exists():
 assert transfer.is_file()and not transfer.is_symlink()and transfer.stat().st_size<=2*1024**2
 transfer_proof=dict(bytes=transfer.stat().st_size,sha256=sha(transfer))
result=dict(stage='CPU_REFERENCE_PREPARE_OBSERVATION',base=str(B),source_commit=receipt['source_commit'],
 checked_unix=time.time(),launch=launch,supervisor=supervisor,producer=producer,exit=exited,failure=failure,
 supervisor_alive=alive(supervisor['identity']if supervisor else launch['identity']if launch else None),
 producer_alive=alive(producer['identity']if producer else None),progress=scalar_view(progress),
 complete=scalar_view(complete),transfer_manifest=transfer_proof,logs=logs,
 complete_transfer_manifest_sha_match=bool(complete and transfer_proof and
       complete.get('transfer_manifest_sha256')==transfer_proof['sha256']),
 complete_source_archive_sha_match=bool(complete and complete.get('source_archive_sha256')==
       read(B/'preparation/source-config.json')['reference_archive']['concatenated_sha256']),
 cpu_only=True,actor_updates=0,model_or_original_qa_readback=False)
print(json.dumps(result,allow_nan=False))
'''


def call_remote(script, handle_path, phase):
    path = Path(handle_path)
    evidence = path.with_name(path.stem + '-' + phase + '-transport-' + str(time.time_ns()) + '.json')
    spec = importlib.util.spec_from_file_location('question5_existing_retained18_transport', TRANSPORT)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    output = transport.remote(script, str(evidence), control_path=None, compat_handshake=False, target='h100-8')
    if len(output.encode()) > 128 * 1024:
        raise ValueError('Bounded metadata response exceeded128KiB')
    value = json.loads(output)
    result = evidence.with_name(evidence.stem + '-result.json')
    atomic_local(result, value, exclusive=True)
    return value, dict(path=str(result), sha256=sha_bytes(result.read_bytes()), transport=str(evidence))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('stage', 'launch', 'observe'))
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--base')
    parser.add_argument('--cpu-python', default=CPU_PYTHON)
    parser.add_argument('--upload-ready', type=Path, default=READY)
    args = parser.parse_args(argv)
    if args.phase == 'stage':
        if not args.source_commit:
            parser.error('stage requires the already-published --source-commit')
        if args.handle.exists():
            raise FileExistsError('Existing handle is retained; use observe or a new independent stage handle')
        base = validate_base(args.base or PREFIX + uuid.uuid4().hex)
        if not args.cpu_python.startswith(('/dataset/shaoshitong/', '/mnt/data/shaoshitong/')):
            raise ValueError('Use the existing task environment Python; do not install an environment')
        config, ready_sha = source_config(args.upload_ready)
        archive, files = published_archive(args.source_commit)
        payload = dict(base=base, source_commit=args.source_commit, cpu_python=args.cpu_python,
                       source_config=config, source_config_bytes=canonical(config).decode() + '\n',
                       upload_ready_sha256=ready_sha, source_archive_sha256=sha_bytes(archive),
                       archive_base64=base64.b64encode(archive).decode(), source_files=files,
                       supervisor_source=SUPERVISOR)
        handle = dict(schema_version=1, role='h100_reference5_cpu_prepare', base=base, target='h100-8',
                      source_commit=args.source_commit, source_config_sha256=sha_bytes(payload['source_config_bytes'].encode()),
                      source_archive_sha256=payload['source_archive_sha256'], upload_ready_sha256=ready_sha,
                      created_unix=time.time(), phase='STAGE_REQUESTED_OUTCOME_UNKNOWN')
        atomic_local(args.handle, handle, exclusive=True)
        script = stage_script(payload)
    else:
        handle = json.loads(args.handle.read_text())
        validate_base(handle['base'])
        if handle.get('role') != 'h100_reference5_cpu_prepare' or handle.get('target') != 'h100-8':
            raise ValueError('Handle belongs to another operation or host')
        script = launch_script(handle) if args.phase == 'launch' else observe_script(handle)
    result, proof = call_remote(script, args.handle, args.phase)
    handle.update(last_phase=args.phase, last_receipt=proof, last_checked_unix=time.time())
    if args.phase == 'stage':
        handle['phase'] = 'STAGED_NOT_LAUNCHED'
    elif args.phase == 'launch':
        handle['phase'] = 'ONCE_CPU_LAUNCH_RECORDED_NOT_COMPLETION'
    else:
        handle['phase'] = 'OBSERVED'
    atomic_local(args.handle, handle)
    print(json.dumps(dict(handle=str(args.handle), receipt=proof, result=result), allow_nan=False))


if __name__ == '__main__':
    main()
