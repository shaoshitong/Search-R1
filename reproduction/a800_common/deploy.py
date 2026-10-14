"""Persist one fresh A800 deployment intent, execute it once, then observe it.

Create: python -m reproduction.a800_common.deploy --source-commit FULL_SHA
Launch: python -m reproduction.a800_common.deploy --handle INTENT --execute
Read:   python -m reproduction.a800_common.deploy --handle INTENT --observe
Prepared only: add --launch-prepared to the original handle after explicit recovery authorization.
Unclaimed GPU-occupancy failure only: --retry-unclaimed permits one explicit retry of that same prepared run.

Only the existing controller's baai-a800 SSH route is used. Credentials and SSH
configuration are never copied or changed. An uncertain launch is observed, not
retried. The optional bundle is an exact published Git snapshot fallback before
any pipeline launch; canonical data and official model metadata stay byte exact.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import time
import uuid
from pathlib import Path


PROJECT = Path('/Users/shaoshitong/Documents/Search-G1')
CONTROL = Path('/Users/shaoshitong/Documents/remote_codex_control/control.py')
OFFICIAL = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-300/official-model-manifest-original.json')
OFFICIAL_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
SHARED_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
DATA_NAMES = ('manifest.json', 'train.parquet', 'validation.parquet', 'stage.parquet', 'final.parquet', 'exclusions-audit.json')
VERSION_FILES = ('verl/version/version', 'verl/single_controller/version/version')
RUNTIME_SUFFIXES = ('.py', '.sh', '.yaml', '.yml', '.json')
PLACEMENT_POLICY = 'launch_time_four_idle_excluding_auxiliary_v1'


def launch_placement(run):
    """Only an actual launch-time binding may become CUDA ENV or launch evidence."""
    assert run['gpu_placement_policy']==PLACEMENT_POLICY and run['gpu_placement_bound'] is True
    from reproduction.common8.native_allocation import local_placement
    training,auxiliary,_,_=local_placement(run)
    assert all(type(index) is int and index>=0 for index in training) and not set(training).intersection(auxiliary)
    physical={int(row[0]):row[1].strip() for row in run['physical_gpu_inventory']}
    assert len(physical)==len(run['physical_gpu_inventory']) and len(set(physical.values()))==len(physical)
    assert set(training+auxiliary)<=set(physical)
    assigned=run['assigned_gpu_uuids'];assert assigned=={str(index):physical[index] for index in training+auxiliary}
    assert run['training_gpu_uuids']==[assigned[str(index)] for index in training]
    assert run['excluded_physical_indices']==sorted(set(physical)-set(training+auxiliary))
    assert run['gpu_placement_binding_receipt']=='logs/local-gpu-placement-binding.json'
    assert re.fullmatch('[0-9a-f]{64}',run['gpu_placement_binding_sha256'])
    return {key:run[key] for key in ('gpu_placement_policy','gpu_placement_bound','training_physical_indices',
        'training_gpu_uuids','auxiliary_physical_indices','assigned_gpu_uuids','excluded_physical_indices',
        'gpu_placement_binding_receipt','gpu_placement_binding_sha256','gpu_placement_bound_unix')} | {
        'cuda_visible_devices':','.join(map(str,training))}


REMOTE_EXECUTE = r'''
import hashlib,io,json,os,re,shutil,socket,subprocess,sys,tarfile,time
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def write_once(path,value):
    data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def emit(value):
    try:print('DEPLOYMENT_JSON:'+json.dumps(value),flush=True)
    except BrokenPipeError:sys.stdout=open(os.devnull,'w')

def state(stage,**extra):
    value=dict(deployment_id=intent['deployment_id'],base=str(base),source_commit=intent['source_commit'],
               stage=stage,observed_unix=time.time(),**extra)
    temp=base/'logs'/('deployment-status-'+intent['deployment_id']+'.tmp')
    with temp.open('w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(base/'logs/deployment-status.json')
    fd=os.open(base/'logs',os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    emit(value)

def command(args,label,timeout=180,env=None):
    global last_command
    last_command=label
    with (base/'preparation'/(label+'.log')).open('xb') as out:
        result=subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                              env=env,timeout=timeout)
        out.flush();os.fsync(out.fileno())
    return result.returncode

base=None;intent=None;created=False;last_command=None
try:
    archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|')
    first=archive.next();assert first is not None and first.name=='deployment-intent.json' and first.isfile()
    intent=json.load(archive.extractfile(first))
    assert intent['role']=='fresh_a800_common_deployment_intent' and intent['host']=='baai-a800'
    assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
    assert re.fullmatch('[0-9a-f]{40}',intent['source_commit'])
    base=Path(intent['base'])
    assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
    assert base.parent.is_dir() and base.parent.resolve()==base.parent
    base.mkdir(mode=0o700,exist_ok=False)
    created=True
    for name in ('logs','preparation','data','tmp','ray','cache/triton','cache/inductor','cache/hf','artifacts','checkpoints'):
        (base/name).mkdir(parents=True,exist_ok=True)
    write_once(base/'preparation/deployment-intent.json',intent)
    state('RECEIVING_EXACT_LOCAL_INPUTS')
    expected={row['target']:row for row in intent['inputs']};seen=set()
    while True:
        item=archive.next()
        if item is None:break
        assert item.isfile() and item.name in expected and item.name not in seen
        row=expected[item.name];assert item.size==row['bytes']
        target=base/item.name;assert target.parent.resolve()==target.parent and not target.exists()
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        h=hashlib.sha256();count=0
        with os.fdopen(fd,'wb') as out:
            source=archive.extractfile(item)
            for part in iter(lambda:source.read(1024*1024),b''):out.write(part);h.update(part);count+=len(part)
            out.flush();os.fsync(out.fileno())
        assert count==row['bytes'] and h.hexdigest()==row['sha256'];seen.add(item.name)
    assert seen==set(expected)
    write_once(base/'logs/deployment-input-receipt.json',dict(deployment_id=intent['deployment_id'],
        inputs=[{k:row[k] for k in ('target','bytes','sha256')} for row in intent['inputs']],
        exact_original_local_bytes=True,training_data_partitioned=False))
    state('CLONING_PINNED_PUBLISHED_GITHUB_SOURCE')
    github='https://github.com/shaoshitong/Search-R1.git'
    existing=Path('/data/search-g1/repo')
    if (existing/'.git').exists():
        result=subprocess.run(['git','-C',str(existing),'remote','get-url','origin'],capture_output=True,text=True,timeout=10)
        candidate=result.stdout.strip()
        if result.returncode==0 and candidate in ('https://github.com/shaoshitong/Search-R1.git','git@github.com:shaoshitong/Search-R1.git'):
            github=candidate
    git_env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    repo=base/'repo';direct=False
    previous=intent.get('prior_git_direct_failure_base')
    if previous is not None:
        prior=Path(previous)
        assert re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',str(prior)) and prior.resolve()==prior
        receipt=prior/'logs/deployment-source-receipt.json'
        assert receipt.is_file() and not receipt.is_symlink() and receipt.stat().st_uid==os.getuid()
        original=json.loads(receipt.read_text())
        assert original['source_transport']=='published_git_bundle_relay' and original['source_is_exact_published_commit'] is True
        assert (prior/'preparation/github-clone.log').is_file()
        state('USING_PUBLISHED_BUNDLE_AFTER_ACTUAL_PRIOR_DIRECT_FAILURE',prior_source_receipt_sha256=digest(receipt))
    else:
        try:
            direct=command(['git','clone','--branch','search-g1','--single-branch',github,str(repo)],'github-clone',90,git_env)==0
            if direct:direct=command(['git','-C',str(repo),'checkout','--detach',intent['source_commit']],'github-checkout',30,git_env)==0
        except subprocess.TimeoutExpired:direct=False
    transport='github_direct'
    if not direct:
        bundle=base/'preparation/published-source.bundle'
        assert bundle.is_file(), 'Direct GitHub clone unavailable; no explicit published-source bundle was provided'
        assert not (base/'logs/pipeline-launch-claim.json').exists()
        if repo.exists():assert repo.resolve()==repo;shutil.rmtree(repo)
        assert command(['git','clone',str(bundle),str(repo)],'published-bundle-clone',90,git_env)==0
        assert command(['git','-C',str(repo),'checkout','--detach',intent['source_commit']],'published-bundle-checkout',30,git_env)==0
        transport='published_git_bundle_relay'
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()
    assert actual==intent['source_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=10).strip()
    sys.path.insert(0,str(repo))
    from reproduction.common8.source_files import runtime_files
    source={str(path.relative_to(repo)):digest(path) for path in runtime_files(repo)}
    assert source==intent['source_sha256'], 'Pinned runtime source bytes differ from the published Git snapshot'
    write_once(base/'logs/deployment-source-receipt.json',dict(deployment_id=intent['deployment_id'],
        source_commit=actual,source_sha256=source,source_transport=transport,
        source_is_exact_published_commit=True,uncommitted_source_copied=False))
    state('PREPARING_FRESH_OFFICIAL_RUN',source_transport=transport)
    # Preserve the original approved method, rebinding only fresh source/time.
    from reproduction.common8.formal_inline_safety import verify_method
    from reproduction.common8.policy_calibration_plan import document_sha256
    method=json.loads((base/'preparation/formal-inline-method-template.json').read_text())
    method.update(source_commit=actual,source_map_sha256=document_sha256(source),created_unix=time.time())
    write_once(base/'preparation/formal-inline-method.json',method)
    environment=dict(os.environ)
    for key in tuple(environment):
        if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
            environment.pop(key,None)
    environment.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',
        CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+environment.get('PATH','/usr/bin:/bin'),
        C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include',
        TMPDIR=str(base/'tmp'),RAY_TMPDIR=str(base/'ray'),HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',
        NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost',RAY_DEDUP_LOGS='0')
    os.chdir(repo)
    python='/data/search-g1/env/bin/python'
    prepare=[python,'-m','reproduction.a800_common.prepare_run','--base',str(base),
        '--model','/data/search-g1/assets/qwen2.5-3b-instruct',
        '--official-manifest',str(base/'preparation/official-model-manifest-original.json'),
        '--source-commit',actual]
    if intent.get('initial_probes_source_base') is not None:
        prepare.extend(['--initial-probes-source-base',intent['initial_probes_source_base']])
    assert command(prepare,'prepare-run',600,environment)==0
    run=json.loads((base/'logs/run-manifest.json').read_text());verify_method(method,run)
    environment['RAY_TMPDIR']=run['ray_tmp_dir']
    assert run['source_sha256']==source and run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None
    assert not (base/'logs/pipeline-launch-claim.json').exists() and not (base/'logs/process-ledger.json').exists()
    from reproduction.common8.native_allocation import bind_idle_local_placement
    from reproduction.a800_common.deploy import launch_placement
    run=bind_idle_local_placement(base);placement=launch_placement(run)
    environment['CUDA_VISIBLE_DEVICES']=placement['cuda_visible_devices']
    state('STARTING_ONE_ACTUAL_LOCAL_PIPELINE',run_id=run['run_id'],execution_id=run['execution_id'])
    args=[python,'-m','reproduction.a800_common.run_pipeline','--base',str(base)]
    log=base/'logs/deployment-pipeline.log'
    with log.open('xb') as out:
        process=subprocess.Popen(args,cwd=repo,env=environment,stdin=subprocess.DEVNULL,
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        text=(Path('/proc')/str(process.pid)/'stat').read_text()
        fields=text[text.rfind(')')+2:].split();ticks=int(fields[19]);sid=int(fields[3])
        assert ticks>0 and sid==process.pid
        receipt=dict(deployment_id=intent['deployment_id'],run_id=run['run_id'],execution_backend='ssh_local',
            execution_id=run['execution_id'],slurm_job_id=None,hostname=socket.gethostname(),
            pid=process.pid,start_ticks=ticks,sid=sid,stdout_log=str(log),source_commit=actual,
            source_transport=transport,launched_unix=time.time(),training_world_size=4,
            same_process_after_five=True,no_automatic_retry=True,**placement)
        write_once(base/'logs/deployment-launch-receipt.json',receipt)
    state('LAUNCHED',**{k:receipt[k] for k in ('run_id','execution_id','pid','start_ticks','stdout_log','source_transport')})
except BaseException as failure:
    if created and base is not None and (base/'logs').is_dir() and intent is not None:
        # Never turn an uncertain Popen/receipt result into a second launch.
        state('DEPLOYMENT_FAILED_OR_UNCERTAIN',failure_type=type(failure).__name__,
              last_command=last_command,no_automatic_retry=True)
    else:
        emit(dict(stage='REMOTE_INTENT_REJECTED',failure_type=type(failure).__name__))
    sys.exit(1)
'''


REMOTE_LAUNCH_PREPARED = r'''
import hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
payload=json.load(sys.stdin);intent=payload['intent'];base=Path(intent['base']);repo=base/'repo'
preclaim_retry=payload.get('preclaim_retry',False);assert type(preclaim_retry) is bool
assert intent['host']=='baai-a800' and re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id'] and base.resolve()==base
def read(name):
    path=base/name;assert path.is_file() and path.resolve()==path and path.stat().st_uid==os.getuid()
    return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def write_once(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
assert read('preparation/deployment-intent.json')==intent
inputs=read('logs/deployment-input-receipt.json');source=read('logs/deployment-source-receipt.json')
assert inputs['deployment_id']==source['deployment_id']==intent['deployment_id']
assert inputs['inputs']==[{k:row[k] for k in ('target','bytes','sha256')} for row in intent['inputs']]
for row in intent['inputs']:assert (base/row['target']).stat().st_size==row['bytes'] and digest(base/row['target'])==row['sha256']
assert source['source_commit']==intent['source_commit'] and source['source_sha256']==intent['source_sha256']
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==intent['source_commit']
for name,sha in intent['source_sha256'].items():assert digest(repo/name)==sha
run=read('logs/run-manifest.json');method=read('preparation/formal-inline-method.json');template=read('preparation/formal-inline-method-template.json')
assert run['source_commit']==intent['source_commit'] and run['source_sha256']==intent['source_sha256']
assert run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None and run['execution_host']==socket.gethostname()
assert run['completed_updates']==0 and run['training_world_size']==4
restore=run.get('full_state_restore')
if restore:
    sys.path.insert(0,str(repo))
    from search_g1.qprobe_checkpoint import read_source,SAVED_CHECKPOINT_STATUS
    path,qualification,_,_,parent=read_source(restore['source_receipt'])
    assert qualification['status']==SAVED_CHECKPOINT_STATUS and restore['kind']=='question_probe_saved_checkpoint'
    assert qualification['step']==restore['completed_outer']==5 and restore['actual_optimizer_writes_per_rank']==100
    assert intent['saved_checkpoint_restore']==dict(source_receipt=str(path),source_receipt_sha256=digest(path))
    assert run['run_id']==qualification['run_id']==parent['run_id'] and run['initial_probe_version']==qualification['reward_probe_version']
else:assert run['initial_probe_version'] is None and not intent.get('saved_checkpoint_restore')
for row in intent['inputs']:
    if row['target'].startswith('data/'):assert run['input_sha256'][Path(row['target']).name]==row['sha256']
official=read('preparation/official-model-manifest-original.json');model_records={row['file']:row['sha256'] for row in official['files']}
assert all(model_records[name]==sha for name,sha in run['initial_model_sha256'].items())
expected_template=dict(template,fresh_states=not bool(restore))
assert {k:v for k,v in method.items() if k not in ('source_commit','source_map_sha256','created_unix')}=={k:v for k,v in expected_template.items() if k not in ('source_commit','source_map_sha256','created_unix')}
sys.path.insert(0,str(repo))
from reproduction.common8.formal_inline_safety import verify_method
from reproduction.common8.formal_contract import assert_formal_run
assert_formal_run(run)
verify_method(method,run)
original=None;retry_provenance={}
if preclaim_retry:
    for name in ('logs/preclaim-retry-claim.json','logs/deployment-retry-receipt.json','logs/deployment-preclaim-retry.log',
                 'logs/pipeline-launch-claim.json','logs/training-launch-claim.json','logs/process-ledger.json',
                 'logs/pipeline-status.json','logs/native-allocation-admission.json','logs/actual-updates.jsonl'):
        path=base/name;assert not path.exists() and not path.is_symlink(), 'Preclaim retry requires an untouched, unclaimed runtime'
    assert not any((base/'logs').glob('optimizer-rank-*.jsonl')) and not any((base/'logs').glob('scheduler-rank-*.jsonl'))
    checkpoints=base/'checkpoints'
    assert not checkpoints.is_symlink() and (not checkpoints.exists() or not any(checkpoints.iterdir())), 'Any checkpoint evidence forbids this retry'
    assert not Path(run['checkpoint_dir']).exists() and not Path(run['checkpoint_dir']).is_symlink()
    assert not Path(run['train_log']).exists() and not Path(run['train_log']).is_symlink(), 'An existing trainer log forbids preclaim retry'
    original=read('logs/deployment-launch-receipt.json')
    assert all(original[k]==value for k,value in dict(deployment_id=intent['deployment_id'],
        run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
        execution_backend='ssh_local',slurm_job_id=None,hostname=socket.gethostname()).items())
    assert type(original['pid']) is int and original['pid']>0 and type(original['start_ticks']) is int and original['start_ticks']>0
    old_proc=Path('/proc')/str(original['pid']);birth_matches=False;old_state=None
    try:
        text=(old_proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        birth_matches=int(fields[19])==original['start_ticks'] and int(fields[3])==original['sid'] and old_proc.stat().st_uid==os.getuid()
        old_state=fields[0] if birth_matches else None
    except (FileNotFoundError,ProcessLookupError):pass
    assert not birth_matches or old_state in ('Z','X'), 'Original pipeline process is still alive'
    old_log=base/'logs/deployment-pipeline.log'
    assert original['stdout_log']==str(old_log) and old_log.is_file() and old_log.resolve()==old_log and old_log.stat().st_uid==os.getuid()
    assert old_log.stat().st_size<=1024*1024, 'Unexpectedly large preclaim failure log requires observation'
    with old_log.open('rb') as f:failure_log=f.read(1024*1024+1)
    assert any(line.startswith(b'AssertionError: Authorized local GPUs are occupied:') for line in failure_log.splitlines()), 'Only the actual native GPU-occupancy failure permits this retry'
    assert b'reproduction/common8/native_allocation.py' in failure_log and b'in admit_local' in failure_log
    retry_provenance=dict(preclaim_retry=True,original_launch_receipt=str(base/'logs/deployment-launch-receipt.json'),
        original_launch_receipt_sha256=digest(base/'logs/deployment-launch-receipt.json'),
        original_stdout_log=str(old_log),original_stdout_log_sha256=digest(old_log),
        original_pid=original['pid'],original_start_ticks=original['start_ticks'],
        original_pipeline_birth_matches=birth_matches,original_pipeline_process_state=old_state,
        retry_reason='actual native GPU occupancy failure before any pipeline claim or training',same_prepared_run=True)
else:
    for name in ('logs/deployment-launch-receipt.json','logs/pipeline-launch-claim.json','logs/training-launch-claim.json','logs/process-ledger.json','logs/deployment-pipeline.log'):
        assert not (base/name).exists(), 'Existing launch evidence requires observation, never another launch'
for proc in Path('/proc').iterdir():
    if proc.name.isdigit():
        try:
            cwd=(proc/'cwd').resolve()
            assert not (proc.stat().st_uid==os.getuid() and (cwd==repo or preclaim_retry and (cwd==base or base in cwd.parents))), 'Existing live process in the prepared runtime'
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
for key in ('launcher_source_commit','launcher_source_sha256','embedded_launcher_sha256'):
    assert re.fullmatch('[0-9a-f]{40}' if key=='launcher_source_commit' else '[0-9a-f]{64}',payload[key])
from reproduction.common8.native_allocation import bind_idle_local_placement
from reproduction.a800_common.deploy import launch_placement
run=bind_idle_local_placement(base);placement=launch_placement(run)
claim=dict(deployment_id=intent['deployment_id'],run_id=run['run_id'],execution_id=run['execution_id'],
    runtime_source_commit=run['source_commit'],claimed_unix=time.time(),no_automatic_retry=True,
    **{k:payload[k] for k in ('launcher_source_commit','launcher_source_sha256','embedded_launcher_sha256')},**retry_provenance)
write_once(base/'logs'/('preclaim-retry-claim.json' if preclaim_retry else 'prepared-launch-claim.json'),claim)
env=dict(os.environ)
for key in tuple(env):
    if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
env.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',CUDA_VISIBLE_DEVICES=placement['cuda_visible_devices'],PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+env.get('PATH','/usr/bin:/bin'),C_INCLUDE_PATH='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',NO_PROXY='127.0.0.1,localhost',RAY_DEDUP_LOGS='0')
env.update({name:str(base/path) for name,path in dict(TMPDIR='tmp',HF_HOME='cache/hf',XDG_CACHE_HOME='cache',TRITON_CACHE_DIR='cache/triton',TORCHINDUCTOR_CACHE_DIR='cache/inductor').items()})
env['RAY_TMPDIR']=run.get('ray_tmp_dir',str(base/'ray'))
log=base/'logs'/('deployment-preclaim-retry.log' if preclaim_retry else 'deployment-pipeline.log')
with log.open('xb') as out:
    p=subprocess.Popen(['/data/search-g1/env/bin/python','-m','reproduction.a800_common.run_pipeline','--base',str(base)],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    text=(Path('/proc')/str(p.pid)/'stat').read_text();fields=text[text.rfind(')')+2:].split();ticks=int(fields[19]);sid=int(fields[3]);assert ticks>0 and sid==p.pid
    receipt=dict(claim,execution_backend='ssh_local',slurm_job_id=None,hostname=socket.gethostname(),pid=p.pid,start_ticks=ticks,sid=sid,stdout_log=str(log),source_commit=run['source_commit'],source_transport=source['source_transport'],launched_unix=time.time(),training_world_size=4,same_process_after_five=True,**placement)
    write_once(base/'logs'/('deployment-retry-receipt.json' if preclaim_retry else 'deployment-launch-receipt.json'),receipt)
try:print('DEPLOYMENT_JSON:'+json.dumps(dict(stage='UNCLAIMED_PIPELINE_RETRY_LAUNCHED' if preclaim_retry else 'PREPARED_PIPELINE_LAUNCHED',**receipt)),flush=True)
except BrokenPipeError:sys.stdout=open(os.devnull,'w')
'''


REMOTE_OBSERVE = r'''
import hashlib,json,math,os,re,time
from pathlib import Path
intent=json.loads(__import__('sys').stdin.read());base=Path(intent['base'])
assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id'])
read_limits=[]
def owned_file(path):
    path=Path(path);path.relative_to(base)
    if not path.is_file():return None
    assert not path.is_symlink() and path.resolve()==path and path.stat().st_uid==os.getuid()
    return path.stat()
def read(name):
    path=base/name;stat=owned_file(path)
    if stat is None:return None
    if stat.st_size>8*1024*1024:read_limits.append(name);return None
    with path.open('rb') as f:return json.loads(f.read(8*1024*1024+1))
def number(value):return type(value) in (int,float) and math.isfinite(value)
def process(record):
    value=dict(pid=record['pid'],start_ticks=record['start_ticks'],birth_matches=False,state=None,alive=False)
    try:
        proc=Path('/proc')/str(record['pid']);text=(proc/'stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        matches=int(fields[19])==record['start_ticks']
        if 'sid' in record:matches=matches and int(fields[3])==record['sid']
        if 'uid' in record:matches=matches and proc.stat().st_uid==record['uid']==os.getuid()
        value.update(birth_matches=matches,state=fields[0] if matches else None,
                     alive=matches and fields[0] not in ('Z','X'))
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return value
def complete_rows(path,info):
    # Snapshot and stream at most32MiB; never load an entire long-running ledger.
    stat=owned_file(path)
    info.update(path=str(path),exists=stat is not None,malformed_complete_lines=0,
                oversized_lines=0,partial_final_line=False)
    if stat is None:return
    end=stat.st_size;start=max(0,end-32*1024*1024);line_limit=1024*1024
    info.update(file_bytes=end,scan_from_start=start==0,scan_start_byte=start)
    with path.open('rb') as f:
        f.seek(start)
        if start:
            # A tail begins inside an unknown record; discard its bounded chunks.
            while f.tell()<end:
                part=f.readline(min(line_limit+1,end-f.tell()))
                if part.endswith(b'\n'):break
                if f.tell()==end:info['partial_final_line']=True
        while f.tell()<end:
            raw=f.readline(min(line_limit+1,end-f.tell()))
            if not raw.endswith(b'\n'):
                if f.tell()==end:info['partial_final_line']=True;break
                info['oversized_lines']+=1
                while f.tell()<end:
                    part=f.readline(min(line_limit+1,end-f.tell()))
                    if part.endswith(b'\n'):break
                    if f.tell()==end:info['partial_final_line']=True
                continue
            if len(raw)>line_limit:info['oversized_lines']+=1;continue
            try:
                row=json.loads(raw)
                if not isinstance(row,dict):raise ValueError('Non-object receipt')
            except (ValueError,UnicodeDecodeError):info['malformed_complete_lines']+=1;continue
            yield row
def cumulative_available(info):
    return info.get('scan_from_start',False) and not info['malformed_complete_lines'] and not info['oversized_lines']
def diagnostic(path):
    if owned_file(path) is None:return []
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-8192));lines=f.read().decode(errors='replace').splitlines()
    selected=[]
    for line in lines:
        text=line.strip()
        # Retain traceback locations and bounded exception messages only.
        # No clone/auth output, model examples, JSON records, or QA log rows.
        if not (text.startswith('File "') or re.match(r'^(AssertionError|ValueError|RuntimeError|OSError|FileNotFoundError|PermissionError|ModuleNotFoundError|ImportError|KeyError|TypeError)(:|$)',text)):
            continue
        if any(word in text.lower() for word in ('password','authorization','bearer','github_pat','ghp_','prompt','question','answer','response')):
            text=text.split(':',1)[0] if not text.startswith('File "') else text.split(', in ',1)[0]
        if not text.startswith('File "'):
            text=re.split(r'[\{\[]',text,1)[0]
        selected.append(text[:240])
    return selected[-6:]
result=dict(deployment_id=intent['deployment_id'],base=str(base),base_exists=base.is_dir(),observed_unix=time.time())
owned=read('preparation/deployment-intent.json')
if owned is not None:
    assert owned==intent
    status=read('logs/deployment-status.json')
    if status:
        result['deployment_stage']=status['stage'];result['failure_type']=status.get('failure_type')
        result['last_command']=status.get('last_command')
        result['deployment_status_unix']=status.get('observed_unix')
        if status['stage']=='DEPLOYMENT_FAILED_OR_UNCERTAIN' and status.get('last_command')=='prepare-run':
            result['failure_diagnostic']=diagnostic(base/'preparation/prepare-run.log')
    original_receipt=read('logs/deployment-launch-receipt.json')
    retry_receipt=read('logs/deployment-retry-receipt.json')
    if retry_receipt:
        assert original_receipt is not None and retry_receipt.get('preclaim_retry') is True
        assert all(retry_receipt[k]==original_receipt[k] for k in ('deployment_id','run_id','execution_id','source_commit'))
        assert retry_receipt['stdout_log']==str(base/'logs/deployment-preclaim-retry.log')
    receipt=retry_receipt or original_receipt
    result['preclaim_retry_claimed']=(base/'logs/preclaim-retry-claim.json').is_file()
    result['using_preclaim_retry_receipt']=retry_receipt is not None
    if receipt:
        result.update({k:receipt[k] for k in ('run_id','execution_id','pid','start_ticks','stdout_log','source_commit','source_transport')})
        result.update({k:receipt[k] for k in ('launcher_source_commit','launcher_source_sha256','embedded_launcher_sha256') if k in receipt})
        identity=process(receipt)
        result.update(pipeline_birth_matches=identity['birth_matches'],pipeline_process_state=identity['state'],
                      pipeline_alive=identity['alive'])
        if not identity['alive']:result['startup_diagnostic']=diagnostic(Path(receipt['stdout_log']))
    run=read('logs/run-manifest.json')
    if run:
        assert run['source_commit']==intent['source_commit']
        if receipt:assert run['run_id']==receipt['run_id'] and run['execution_id']==receipt['execution_id']
        result['run_manifest']={k:run[k] for k in ('run_id','execution_id','source_commit',
            'training_world_size','completed_updates','train_log','training_physical_indices','auxiliary_physical_indices')}
        training_log=Path(run['train_log']);training_log.relative_to(base/'logs')
        stat=owned_file(training_log)
        result['training_log']=dict(path=str(training_log),exists=stat is not None,
            bytes=stat.st_size if stat else None,modified_unix=stat.st_mtime if stat else None,
            exception_diagnostic=diagnostic(training_log))
        ledger=read('logs/process-ledger.json')
        if ledger:
            assert ledger['run_id']==run['run_id']
            jobs=[]
            for name,job in list(ledger['jobs'].items())[:32]:
                assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,64}',name)
                path=Path(job['log']);path.relative_to(base/'logs');stat=owned_file(path)
                live_registered=[process(record) for record in job['ledger'].values()]
                live_registered=[record for record in live_registered if record['alive']]
                jobs.append(dict(name=name,ledger_state=job['state'],leader=process(job['leader']),
                    live_registered_count=len(live_registered),live_registered_identities=live_registered[:16],
                    exit_code=job.get('exit_code'),log_path=str(path),log_bytes=stat.st_size if stat else None,
                    log_modified_unix=stat.st_mtime if stat else None,exception_diagnostic=diagnostic(path)))
            result.update(owned_jobs=jobs,owned_jobs_truncated=len(ledger['jobs'])>32)
        cleanup=read('logs/cleanup-status.json')
        if cleanup:
            assert cleanup['run_id']==run['run_id'] and cleanup['execution_id']==run['execution_id']
            result['cleanup_status']={k:cleanup[k] for k in ('cleaned','checked_unix','scope')}
    for name,key in (('logs/pipeline-launch-claim.json','pipeline_claimed'),('logs/training-launch-claim.json','trainer_claimed')):
        result[key]=(base/name).is_file()
    pipeline=read('logs/pipeline-status.json')
    if pipeline:
        if run:assert pipeline['run_id']==run['run_id']
        result['pipeline_stage']=pipeline['stage']
        result['pipeline_status_unix']=pipeline.get('unix')
        if pipeline['stage'] in ('FAILED','OWNED_CLEANUP_FAILED'):
            result['failure_diagnostic']=diagnostic(Path(receipt['stdout_log']) if receipt else base/'logs/deployment-pipeline.log')
    if run:
        updates={};count=0;latest=None;foreign=0
        for row in complete_rows(base/'logs/actual-updates.jsonl',updates):
            if row.get('run_id')!=run['run_id'] or type(row.get('step')) is not int or row['step']<1:
                foreign+=1;continue
            count+=1;latest=row
        updates.update(matching_complete_rows=count,identity_or_step_rejected_rows=foreign)
        result['actual_updates_ledger']=updates
        exact=cumulative_available(updates) and foreign==0
        result.update(recorded_completed_outers=count if updates['exists'] and exact else None,
                      last_recorded_outer=latest['step'] if latest else None)
        if latest:
            metric_keys=('critic/rewards/mean','actor/pg_loss','actor/entropy_loss','actor/kl_loss',
                'actor/ppo_kl','actor/grad_norm','actor/optimizer_steps_this_update',
                'actor/update_opportunities_this_attempt','actor/skipped_update_opportunities')
            result['latest_actual_update']=dict(step=latest['step'],metrics={k:latest['metrics'][k]
                for k in metric_keys if number(latest.get('metrics',{}).get(k))},
                **{k:latest[k] for k in ('opportunities_processed','actual_optimizer_writes',
                    'optimizer_writes_before','optimizer_writes_after') if number(latest.get(k))})
        opportunities=[]
        assert type(run['training_world_size']) is int and 1<=run['training_world_size']<=8
        for rank in range(run['training_world_size']):
            info={};processed=0;written=0;last=None;foreign=0
            for row in complete_rows(base/'logs'/('optimizer-rank-%d.jsonl'%rank),info):
                if row.get('run_id')!=run['run_id'] or row.get('rank')!=rank:foreign+=1;continue
                if row.get('kind')=='OPPORTUNITY' and row.get('processed') is True:
                    processed+=1;written+=int(row.get('optimizer_write_succeeded') is True);last=row
            if info['exists']:
                exact=cumulative_available(info) and foreign==0
                info.update(rank=rank,identity_rejected_rows=foreign,scanned_processed_opportunities=processed,
                    processed_opportunities_per_rank=processed if exact else None,
                    scanned_successful_writes=written,successful_writes_per_rank=written if exact else None,
                    cumulative_optimizer_writes=last['optimizer_writes_after'] if last and number(last.get('optimizer_writes_after')) else None)
                opportunities.append(info)
        result['opportunity_ledgers']=opportunities
        gpu={};latest=None;count=0;foreign=0;previous=None;first=None;max_gap=None;peaks={}
        source_sha=hashlib.sha256(json.dumps(run['source_sha256'],sort_keys=True,separators=(',',':'),
            ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        producer=run.get('pipeline_identity')
        if producer is not None:producer={k:producer[k] for k in ('pid','start_ticks','sid','uid')}
        if producer is None and receipt:
            producer={k:receipt[k] for k in ('pid','start_ticks','sid')};producer['uid']=os.getuid()
        physical=run['training_physical_indices']+run['auxiliary_physical_indices']
        for row in complete_rows(base/'logs'/('gpu-samples-'+run['run_id']+'.jsonl'),gpu):
            valid=(row.get('run_id')==run['run_id'] and row.get('execution_id')==run['execution_id']
                and row.get('execution_backend')=='ssh_local' and row.get('role')=='actual_common8_gpu_occupancy'
                and row.get('source_map_sha256')==source_sha and row.get('sampler_identity')==producer
                and number(row.get('unix')) and number(row.get('query_started_unix'))
                and row['query_started_unix']<=row['unix'] and isinstance(row.get('gpus'),list))
            devices=row.get('gpus',[]) if valid else []
            valid=valid and len(devices)==len(physical) and all(isinstance(d,dict) and
                type(d.get('index')) is int and all(number(d.get(k)) for k in ('used_mib','total_mib','util_percent'))
                and d['total_mib']>0 and 0<=d['used_mib']<=d['total_mib'] and 0<=d['util_percent']<=100 for d in devices)
            valid=valid and sorted(d['index'] for d in devices)==sorted(physical)
            if not valid or (previous is not None and row['unix']<previous):foreign+=1;continue
            if previous is not None:max_gap=max(max_gap or 0,row['unix']-previous)
            if first is None:first=row['unix']
            previous=row['unix'];count+=1;latest=row
            for device in devices:peaks[device['index']]=max(peaks.get(device['index'],0),device['used_mib'])
        exact=cumulative_available(gpu) and foreign==0
        gpu.update(matching_sample_count=count,identity_or_scalar_rejected_rows=foreign,
            cumulative_sample_count=count if gpu['exists'] and exact else None,first_scanned_sample_unix=first,
            latest_sample_unix=previous,latest_sample_age_seconds=time.time()-previous if previous is not None else None,
            maximum_scanned_sampling_gap_seconds=max_gap,sampled_run_peak_mib=peaks if exact and count else None,
            sampled_scanned_peak_mib=peaks if count else None,
            scope='pipeline lifetime including service startup/calibration; sampled occupancy only',
            training_claimed=result['trainer_claimed'],pretraining_history_only=not result['trainer_claimed'])
        if latest:
            gpu['sampler']=process(producer)
            gpu['latest_devices']=[dict(physical_index=d['index'],assignment='training' if d['index'] in run['training_physical_indices'] else 'auxiliary',
                used_mib=d['used_mib'],total_mib=d['total_mib'],util_percent=d['util_percent'],
                sampled_run_peak_mib=peaks[d['index']] if exact else None) for d in latest['gpus']]
        result['gpu_samples']=gpu
    log=Path(receipt['stdout_log']) if receipt else base/'logs/deployment-pipeline.log'
    if log.is_file():result.update(stdout_log_bytes=log.stat().st_size,stdout_log_modified_unix=log.stat().st_mtime)
if read_limits:result['oversized_json_documents_skipped']=read_limits
print('DEPLOYMENT_JSON:'+json.dumps(result),flush=True)
'''


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda: source.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'w') as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write('\n'); target.flush(); os.fsync(target.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def published_source(repo, commit):
    """Hash Git blobs at the pinned commit, never mutable working-tree bytes."""
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', commit + '^{commit}'], text=True).strip()
    assert actual == commit
    raw = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', commit])
    selected = []
    for item in raw.split(b'\0'):
        if not item:
            continue
        metadata, name = item.split(b'\t', 1)
        mode, kind, identity = metadata.decode().split()
        name = name.decode(); relative = Path(name)
        if (relative.suffix in RUNTIME_SUFFIXES or name in VERSION_FILES) and not {'.git', '__pycache__', 'evidence'}.intersection(relative.parts):
            assert kind == 'blob' and mode in ('100644', '100755')
            selected.append((name, identity))
    assert set(VERSION_FILES).issubset(name for name, _ in selected)
    payload = ('\n'.join(identity for _, identity in selected) + '\n').encode()
    process = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'], input=payload, capture_output=True, check=True)
    blobs = io.BytesIO(process.stdout); result = {}
    for name, expected in selected:
        identity, kind, size = blobs.readline().decode().strip().split()
        assert identity == expected and kind == 'blob'
        result[name] = hashlib.sha256(blobs.read(int(size))).hexdigest()
        assert blobs.read(1) == b'\n'
    assert not blobs.read()
    return result


def create_intent(args):
    assert args.source_commit is not None and re.fullmatch('[0-9a-f]{40}', args.source_commit)
    repo = Path(__file__).resolve().parents[2]
    shared = args.shared_data.resolve(); official = args.official_manifest.resolve(); method = args.method.resolve()
    assert digest(shared / 'manifest.json') == SHARED_SHA and digest(official) == OFFICIAL_SHA
    manifest = json.loads((shared / 'manifest.json').read_text())
    inputs = []
    for name in DATA_NAMES:
        path = shared / name
        if name != 'manifest.json':
            expected = (manifest['exclusion_audit'] if name == 'exclusions-audit.json' else manifest['splits'][name.removesuffix('.parquet')])
            assert expected['path'] == name and digest(path) == expected['sha256']
        inputs.append(dict(source=str(path), target='data/' + name, bytes=path.stat().st_size, sha256=digest(path)))
    for source, target in ((official, 'preparation/official-model-manifest-original.json'),
                           (method, 'preparation/formal-inline-method-template.json')):
        inputs.append(dict(source=str(source), target=target, bytes=source.stat().st_size, sha256=digest(source)))
    if args.bundle is not None:
        bundle = args.bundle.resolve()
        subprocess.run(['git', '-C', str(repo), 'bundle', 'verify', str(bundle)], check=True, capture_output=True)
        inputs.append(dict(source=str(bundle), target='preparation/published-source.bundle', bytes=bundle.stat().st_size, sha256=digest(bundle)))
    identifier = uuid.uuid4().hex
    value = dict(schema_version=1, role='fresh_a800_common_deployment_intent', deployment_id=identifier,
        created_unix=time.time(), host='baai-a800', base='/data/shaoshitong/Search-G1-a800-fresh200-' + identifier,
        source_commit=args.source_commit, source_sha256=published_source(repo, args.source_commit),
        inputs=inputs, training_world_size=4, training_physical_indices=None,
        gpu_placement_policy=PLACEMENT_POLICY,gpu_placement_bound=False,
        auxiliary_physical_indices=[5, 6], target_updates=200, same_process_after_five=True,
        execution_policy='one explicit execution; inspect the same handle after failure or uncertainty; no automatic retry')
    if args.initial_probes_source_base is not None:
        source_base=args.initial_probes_source_base
        assert re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',source_base)
        assert source_base != value['base']
        value['initial_probes_source_base']=source_base
    if args.reuse_published_bundle:
        assert args.bundle is not None and args.initial_probes_source_base is not None, 'Prior real source receipt and a published bundle are required'
        value['prior_git_direct_failure_base']=args.initial_probes_source_base
    handle = args.handle.resolve() if args.handle is not None else PROJECT / 'preparation' / ('a800-common-deployment-' + identifier + '.json')
    write_once(handle, value)
    print(json.dumps(dict(handle=str(handle), base=value['base'], source_commit=value['source_commit'], intent_created=True)))


def ssh_command(code):
    spec = importlib.util.spec_from_file_location('a800_existing_control', CONTROL)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # The existing physical-route netcat has a ten-second idle timeout.
    # Keep this invocation alive sooner, without editing its route or config.
    route = [argument.replace('ServerAliveInterval=15', 'ServerAliveInterval=5')
             for argument in module.ssh('baai-a800')]
    return route + ['python3 -c ' + shlex.quote(code)]


def perform(args):
    assert args.handle is not None, 'Execution/observation requires the original persisted handle'
    handle = args.handle.resolve(); intent = json.loads(handle.read_text())
    assert intent['role'] == 'fresh_a800_common_deployment_intent' and intent['host'] == 'baai-a800'
    assert re.fullmatch('[0-9a-f]{32}', intent['deployment_id']) and re.fullmatch('[0-9a-f]{40}', intent['source_commit'])
    assert intent['base'] == '/data/shaoshitong/Search-G1-a800-fresh200-' + intent['deployment_id']
    if args.execute:
        # An exclusive claim precedes even the SSH attempt. A timeout or lost
        # receipt cannot make a later invocation launch a duplicate process.
        claim = handle.with_name(handle.stem + '.execute-once.json')
        write_once(claim, dict(deployment_id=intent['deployment_id'], claimed_unix=time.time(),
                              local_pid=os.getpid(), no_automatic_retry=True))
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as archive:
            raw = json.dumps(intent, sort_keys=True, allow_nan=False).encode()
            member = tarfile.TarInfo('deployment-intent.json'); member.size = len(raw); archive.addfile(member, io.BytesIO(raw))
            for row in intent['inputs']:
                source = Path(row['source'])
                assert source.is_file() and source.stat().st_size == row['bytes'] and digest(source) == row['sha256']
                with source.open('rb') as original:
                    member = tarfile.TarInfo(row['target']); member.size = row['bytes']; member.mode = 0o600
                    archive.addfile(member, original)
        payload = data.getvalue(); code = REMOTE_EXECUTE; timeout = 1000
    elif args.launch_prepared or args.retry_unclaimed:
        repo = Path(__file__).resolve().parents[2]
        launcher_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        launcher_bytes = subprocess.check_output(['git', '-C', str(repo), 'show', launcher_commit + ':reproduction/a800_common/deploy.py'])
        assert hashlib.sha256(launcher_bytes).hexdigest() == digest(Path(__file__).resolve()), 'Prepared launcher must be the published committed tool'
        write_once(handle.with_name(handle.stem + ('.retry-unclaimed-once.json' if args.retry_unclaimed else '.launch-prepared-once.json')),
                   dict(deployment_id=intent['deployment_id'], claimed_unix=time.time(), launcher_source_commit=launcher_commit, no_automatic_retry=True))
        payload = json.dumps(dict(intent=intent, launcher_source_commit=launcher_commit,
            launcher_source_sha256=hashlib.sha256(launcher_bytes).hexdigest(),
            embedded_launcher_sha256=hashlib.sha256(REMOTE_LAUNCH_PREPARED.encode()).hexdigest(),
            preclaim_retry=args.retry_unclaimed), sort_keys=True, allow_nan=False).encode()
        code = REMOTE_LAUNCH_PREPARED; timeout = 60
    else:
        payload = json.dumps(intent, sort_keys=True, allow_nan=False).encode(); code = REMOTE_OBSERVE; timeout = 60
    action = 'execute' if args.execute else 'retry-unclaimed' if args.retry_unclaimed else 'launch-prepared' if args.launch_prepared else 'observe'
    stderr_path = handle.with_name(handle.stem + '.' + action + '-' + str(time.time_ns()) + '.stderr')
    output = b''; exit_code = None; failure_type = None
    with stderr_path.open('xb') as errors:
        try:
            result = subprocess.run(ssh_command(code), input=payload, stdout=subprocess.PIPE, stderr=errors, timeout=timeout)
            output, exit_code = result.stdout, result.returncode
        except (subprocess.TimeoutExpired, OSError) as failure:
            failure_type = type(failure).__name__
            output = getattr(failure, 'stdout', None) or b''
        errors.flush(); os.fsync(errors.fileno())
    rows = [json.loads(line[len('DEPLOYMENT_JSON:'):]) for line in output.decode(errors='replace').splitlines()
            if line.startswith('DEPLOYMENT_JSON:')]
    receipt_path = handle.with_name(handle.stem + '.' + action + '-' + str(time.time_ns()) + '.json')
    write_once(receipt_path, dict(deployment_id=intent['deployment_id'], action=action, exit_code=exit_code, failure_type=failure_type,
                                 observations=rows, transport_stderr=str(stderr_path), observed_unix=time.time()))
    print(json.dumps(dict(handle=str(handle), exit_code=exit_code, failure_type=failure_type, receipt=str(receipt_path),
                          observation=rows[-1] if rows else None, next_action='observe the same handle')))
    return exit_code if exit_code is not None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle', type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', action='store_true'); actions.add_argument('--observe', action='store_true')
    actions.add_argument('--launch-prepared', action='store_true')
    actions.add_argument('--retry-unclaimed', action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--shared-data', type=Path, default=PROJECT / 'shared-data')
    parser.add_argument('--official-manifest', type=Path, default=OFFICIAL)
    parser.add_argument('--method', type=Path, default=PROJECT / 'preparation/formal-inline-method.json')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--initial-probes-source-base')
    parser.add_argument('--reuse-published-bundle', action='store_true')
    args = parser.parse_args()
    if args.execute or args.observe or args.launch_prepared or args.retry_unclaimed:
        assert args.source_commit is None and args.bundle is None and args.initial_probes_source_base is None and not args.reuse_published_bundle, 'Use the persisted intent without replacement inputs'
        return perform(args)
    create_intent(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
