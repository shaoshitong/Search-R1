"""One bounded, published, read-only observation of the authoritative qprobe run.

Reuse the reviewed warmup scalar reader verbatim. This entry point cannot launch,
retry, signal, evaluate, load a model, poll, or change training/hourly anchors.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
AUTHORITY = ROOT/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json'
TEMPLATE = 'reproduction/operations/a800_qprobe_feature_operator.py'
TEMPLATE_COMMIT = '32da31fde5e8b8a9f095cb1235ef13584246a4ee'
TEMPLATE_SHA = '37bb99444eef2184a57ab73219e5924dfec474d05fc0facf57e1b46ff1c3238a'
REVIEWED_READER_SHA = 'fda6b547e6899f1ea4aff21e8073b616b740a160163df101a642be26d21d5c5e'
SUFFIXES = ('.py','.sh','.yaml','.yml','.json')
VERSIONS = ('verl/version/version','verl/single_controller/version/version')

REMOTE_COMMON = r'''
import hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
payload=json.load(sys.stdin);intent=payload['intent'];base=Path(intent['base']);repo=base/'repo'
def owned(path):
 path=Path(path);assert path.resolve()==path and path.stat().st_uid==os.getuid()
 return path
def read(path):
 path=owned(path);assert path.is_file() and path.stat().st_size<=1024**2
 return json.loads(path.read_text())
def pinned_run():
 assert intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800'
 assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id']) and re.fullmatch('[0-9a-f]{40}',intent['source_commit'])
 assert str(base)=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
 assert re.fullmatch('[0-9a-f]{32}',payload['expected_run']) and re.fullmatch('[0-9a-f]{32}',payload['expected_execution'])
 owned(base);owned(repo);run=read(base/'logs/run-manifest.json')
 assert read(base/'preparation/qprobe-intent.json')==intent
 assert run['run_id']==payload['expected_run'] and run['execution_id']==payload['expected_execution']
 assert run['source_commit']==intent['source_commit'] and run['source_sha256']==intent['source_sha256']
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==intent['source_commit']
 assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=10).strip()
 paths=[p for p in repo.rglob('*') if p.is_file() and (p.suffix in ('.py','.sh','.yaml','.yml','.json') or str(p.relative_to(repo)) in ('verl/version/version','verl/single_controller/version/version')) and not {'.git','__pycache__','evidence'}.intersection(p.relative_to(repo).parts)]
 # Hydra creates exactly these ignored launch-configuration YAMLs after startup.
 # Keep all declared source bytes and reject every other undeclared runtime file.
 generated=[]
 for path in paths:
  relative=str(path.relative_to(repo))
  if relative not in intent['source_sha256'] and re.fullmatch(r'outputs/[0-9]{4}-[0-9]{2}-[0-9]{2}/[0-9]{2}-[0-9]{2}-[0-9]{2}/\.hydra/(config|hydra|overrides)\.yaml',relative):
   owned(path);generated.append(path)
 paths=[path for path in paths if path not in generated]
 actual={str(path.relative_to(repo)):hashlib.sha256(owned(path).read_bytes()).hexdigest() for path in paths}
 assert actual==intent['source_sha256']
 assert run['completed_updates']==0 and run['training_world_size']==4
 assert run['execution_backend']=='ssh_local' and run['execution_host']==socket.gethostname()
 assert intent['gpu_placement_policy']=='launch_time_four_idle_excluding_auxiliary_v1' and intent['gpu_placement_bound'] is False
 sys.path.insert(0,str(repo))
 from reproduction.common8.native_allocation import local_placement
 training,auxiliary,_,_=local_placement(run)
 assert run['gpu_placement_binding_receipt']=='logs/local-gpu-placement-binding.json'
 binding_path=owned(base/run['gpu_placement_binding_receipt']);binding=read(binding_path)
 assert hashlib.sha256(binding_path.read_bytes()).hexdigest()==run['gpu_placement_binding_sha256']
 for key in ('run_id','execution_id','source_commit','training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','training_gpu_uuids','assigned_gpu_uuids','physical_gpu_inventory','gpu_placement_bound_unix'):
  assert binding[key]==run[key]
 assert binding['host']==socket.gethostname() and binding['cuda_initialized'] is False
 placement_keys=('training_physical_indices','training_gpu_uuids','auxiliary_physical_indices','assigned_gpu_uuids','excluded_physical_indices','gpu_placement_binding_sha256')
 for name in ('qprobe-launch-claim.json','deployment-launch-receipt.json'):
  path=base/'logs'/name
  if path.is_file():
   launch=read(path);assert launch['run_id']==run['run_id']
   for key in placement_keys:assert launch[key]==run[key]
 path=base/'logs/native-allocation-admission.json'
 if path.is_file():
  allocation=read(path);assert allocation['accepted'] is True
  assert allocation['run_id']==run['run_id'] and allocation['execution_id']==run['execution_id']
  assert allocation['actual_allocated_indices']==training+auxiliary
  assert allocation['training_physical_indices']==training and allocation['auxiliary_physical_indices']==auxiliary
  assert allocation['assigned_gpu_uuids']==run['assigned_gpu_uuids'] and allocation['excluded_physical_indices']==run['excluded_physical_indices']
  assert allocation['gpu_placement_binding_sha256']==run['gpu_placement_binding_sha256']
  assert allocation['actual_cuda_runtime_uuids']==run['training_gpu_uuids']
  assert allocation['assigned_cuda_runtime_uuids']==[run['assigned_gpu_uuids'][str(i)] for i in training+auxiliary]
 assert run['initial_model']==run['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct'
 return run
'''


def read(path):
    assert path.is_absolute() and path.resolve()==path and path.is_file()
    assert path.stat().st_uid==os.getuid() and path.stat().st_size<=1024**2
    return json.loads(path.read_text())


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],timeout=90)


def reviewed_reader():
    """Extract only the pinned literal read-only tail; never import its launcher."""
    commit=git(OPS,'rev-parse','HEAD').decode().strip()
    relative=str(Path(__file__).resolve().relative_to(OPS))
    assert git(OPS,'show',commit+':'+relative)==Path(__file__).read_bytes(), 'Publish this exact observer first'
    raw=git(OPS,'show',TEMPLATE_COMMIT+':'+TEMPLATE)
    assert hashlib.sha256(raw).hexdigest()==TEMPLATE_SHA
    assert git(OPS,'show',commit+':'+TEMPLATE)==raw==(OPS/TEMPLATE).read_bytes()
    values={target.id:node.value for node in ast.parse(raw).body if isinstance(node,ast.Assign)
            for target in node.targets if isinstance(target,ast.Name)}
    expression=values['REMOTE_INSPECT_WARMUP']
    assert isinstance(expression,ast.BinOp) and isinstance(expression.op,ast.Add)
    assert isinstance(expression.left,ast.Name) and expression.left.id=='REMOTE_COMMON'
    tail=ast.literal_eval(expression.right);old_common=ast.literal_eval(values['REMOTE_COMMON'])
    assert hashlib.sha256((old_common+tail).encode()).hexdigest()==REVIEWED_READER_SHA
    remote=REMOTE_COMMON+tail;ast.parse(remote)
    return commit,remote,hashlib.sha256(tail.encode()).hexdigest()


def authority(handle,intent,run_id,execution_id):
    handoff=read(AUTHORITY);current=handoff['current_authoritative']
    assert handoff['goal_active'] is True
    assert current['role']=='a800_qprobe_teacher_warmup_fresh200_current_authoritative' and current['host']=='baai-a800'
    assert (current['run_id'],current['execution_id'],current['source_commit'],current['base'])==(run_id,execution_id,intent['source_commit'],intent['base'])
    assert current['handle_path']==str(handle) and read(handle)==intent
    return hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()


def runtime_module(runtime,intent):
    assert runtime.is_absolute() and runtime.resolve()==runtime and runtime.is_dir()
    assert git(runtime,'rev-parse','HEAD').decode().strip()==intent['source_commit']
    assert not git(runtime,'status','--porcelain').strip(), 'The runtime checkout must be clean'
    relative='reproduction/a800_common/deploy.py';path=runtime/relative
    raw=path.read_bytes();assert raw==git(runtime,'show',intent['source_commit']+':'+relative)
    assert hashlib.sha256(raw).hexdigest()==intent['source_sha256'][relative]
    spec=importlib.util.spec_from_file_location('published_qprobe_readonly_route',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.published_source(runtime,intent['source_commit'])==intent['source_sha256']
    actual={str(path.relative_to(runtime)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in runtime.rglob('*') if path.is_file() and (path.suffix in SUFFIXES or str(path.relative_to(runtime)) in VERSIONS)
            and not {'.git','__pycache__','evidence'}.intersection(path.relative_to(runtime).parts)}
    assert actual==intent['source_sha256']
    assert all((runtime/name).resolve()==runtime/name for name in actual)
    return module


def once(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'w') as target:
        json.dump(value,target,sort_keys=True,allow_nan=False);target.write('\n');target.flush();os.fsync(target.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def observation(raw,expected):
    assert len(raw)<=128*1024, 'Bounded observation output exceeded'
    rows=[json.loads(line.split(':',1)[1]) for line in raw.decode(errors='replace').splitlines()
          if line.startswith('FEATURE_OPERATOR_JSON:')]
    assert len(rows)==1
    row=rows[0];assert (row['run_id'],row['execution_id'],row['source_commit'])==expected
    assert row['metadata_only'] is True
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle',type=Path,required=True);parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--expected-run',required=True);parser.add_argument('--expected-execution',required=True)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();handle=args.handle;handle.relative_to(ROOT/'preparation');intent=read(handle)
    assert intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800'
    assert re.fullmatch('[0-9a-f]{32}',args.expected_run) and re.fullmatch('[0-9a-f]{32}',args.expected_execution)
    assert re.fullmatch('[0-9a-f]{40}',intent['source_commit'])
    assert intent['base']=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
    commit,remote,tail_sha=reviewed_reader();authority(handle,intent,args.expected_run,args.expected_execution)
    module=runtime_module(args.runtime,intent);route=module.ssh_command(remote)
    assert route[-1].startswith('python3 -c ')
    route[-1]='/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    receipt=args.receipt or ROOT/'preparation'/('qprobe-flex-observation-'+str(time.time_ns())+'.json')
    assert receipt.is_absolute() and receipt.resolve()==receipt and receipt.suffix=='.json' and receipt.parent.is_dir()
    receipt.relative_to(ROOT/'preparation');assert not receipt.exists()
    authority_sha=authority(handle,intent,args.expected_run,args.expected_execution)
    result=dict(role='qprobe_authoritative_bounded_readonly_observation',run_id=args.expected_run,
        execution_id=args.expected_execution,source_commit=intent['source_commit'],base=intent['base'],handle_path=str(handle),
        operator_commit=commit,template_commit=TEMPLATE_COMMIT,template_sha256=TEMPLATE_SHA,
        reviewed_reader_sha256=REVIEWED_READER_SHA,reused_reader_tail_sha256=tail_sha,
        remote_reader_sha256=hashlib.sha256(remote.encode()).hexdigest(),authority_sha256=authority_sha,
        handle_sha256=hashlib.sha256(handle.read_bytes()).hexdigest(),no_automatic_retry=True,remote_requested=True)
    payload=json.dumps(dict(intent=intent,expected_run=args.expected_run,expected_execution=args.expected_execution),sort_keys=True,allow_nan=False).encode()
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            actual=subprocess.run(route,input=payload,stdout=out,stderr=err,timeout=90)
            result.update(exit_code=actual.returncode,status='READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            result.update(exit_code=None,status='TIMEOUT_OR_TRANSPORT_UNKNOWN',failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1)
        err.seek(0);stderr=err.read(64*1024)
        result['bounded_stderr_sha256']=hashlib.sha256(stderr).hexdigest()
        diagnostic=stderr.decode(errors='replace')
        result['remote_exception_frames']=[dict(file=Path(match.group(1)).name[:160],line=int(match.group(2)),function=match.group(3)[:96])
            for match in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',diagnostic,re.MULTILINE)][-8:]
        types=re.findall(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',diagnostic,re.MULTILINE)
        result['remote_exception_type']=types[-1] if types else None
    result['observation']=None
    if result['exit_code']==0:
        try:result['observation']=observation(raw,(args.expected_run,args.expected_execution,intent['source_commit']))
        except (AssertionError,ValueError,KeyError,TypeError) as failure:
            result.update(status='SANITIZED_OUTPUT_REJECTED',failure_type=type(failure).__name__)
    result['observed_unix']=time.time();once(receipt,result)
    print(json.dumps(dict(receipt=str(receipt),status=result['status'],exit_code=result['exit_code'],
        observation=result['observation'],no_automatic_retry=True),allow_nan=False))
    return 0 if result['exit_code']==0 and result['observation'] is not None else 1


if __name__=='__main__':
    raise SystemExit(main())
