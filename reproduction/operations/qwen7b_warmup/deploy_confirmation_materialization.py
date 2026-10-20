"""Published code-only confirmation preparation deployment; NEVER submit/sample.

Every SSH command is <16KB. A conservative recursive AST import closure,
package initializers, retriever subprocess module, and verl/version/version
resource are exported from an actually pushed commit. Partial matching code
uploads/extraction resume safely; conflicting versions require a new root.
CPU materialization is an owned background process with lock/PID/start-tick
identity and a durable terminal receipt. Observation failure never relaunches it.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


PARENT = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
DEFAULT_BASE = PARENT+'/confirmation-materialization-deploy-20261021'
DEFAULT_FROZEN = PARENT+'/algorithm-improvement-20261021/confirmation/manifest.json'
BRANCH = 'refs/heads/codex/qwen7b-probe-warmup-20261019'
PACKAGE = 'reproduction/operations/qwen7b_warmup/'
SEEDS = tuple(PACKAGE+name for name in (
    'deploy_confirmation_materialization.py', 'confirmation_materialize.py',
    'test_confirmation_materialize.py', 'validation_g10_materialize.py', 'test_validation_g10_materialize.py',
    'extra_g5_prepare.py', 'extra_g5_sampler.py', 'extra_g5_merge.py',
    'g10_grouping.py', 'shard.py', 'sampling.py',
    'retriever.py', 'native_capture.py', 'packed_features.py', 'download_model.py', 'runtime.sh'))
RESOURCE_FILES = ('verl/version/version',)
MANIFEST_NAME = '_confirmation_source_manifest.json'
CHUNK_BYTES = 6144


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def import_closure(paths, load, seeds=None, resource_files=None):
    """Conservative all-branches static closure; no source import/execution.

    Runtime-loaded third-party libraries remain in the already-qualified remote
    environment. The local retriever subprocess is an explicit entry seed.
    """
    paths = set(paths)
    seeds = SEEDS if seeds is None else tuple(seeds)
    resource_files = RESOURCE_FILES if resource_files is None else tuple(resource_files)
    pending, chosen, outside, unresolved = list(seeds)+list(resource_files), set(), set(), set()
    local_roots = {'reproduction', 'search_g1', 'search_r1', 'verl'}
    def add_module(parts, required=False):
        if not parts:
            return
        stem = '/'.join(parts)
        matches = [name for name in (stem+'.py', stem+'/__init__.py') if name in paths]
        if matches:
            name = matches[0]
            pending.append(name)
            for parent in Path(name).parents:
                if str(parent) == '.':
                    break
                initializer = str(parent/'__init__.py')
                if initializer in paths:
                    pending.append(initializer)
        elif any(name.startswith(stem+'/') for name in paths):
            # PEP420 namespace package without an initializer.
            return
        elif parts[0] in local_roots:
            if required:
                # The repo contains optional/version/backend imports whose
                # target is absent even in the full checkout. Record these;
                # the real collector's CPU critical-import smoke must pass.
                unresolved.add(stem.replace('/', '.'))
        else:
            outside.add(parts[0])
    while pending:
        name = pending.pop()
        if name in chosen:
            continue
        if name not in paths:
            raise ValueError('Required published entry/resource missing: '+name)
        chosen.add(name)
        if not name.endswith('.py'):
            continue
        raw = load(name)
        tree = ast.parse(raw.decode('utf8'), filename=name)
        package = list(Path(name).with_suffix('').parts)[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name.split('.'), required=True)
            elif isinstance(node, ast.ImportFrom):
                base = (package[:len(package)-node.level+1] if node.level else [])
                base += node.module.split('.') if node.module else []
                add_module(base, required=True)
                for alias in node.names:
                    if alias.name != '*':
                        # Imported names may be attributes, not submodules.
                        add_module(base+alias.name.split('.'), required=False)
    return sorted(chosen), sorted(outside), sorted(unresolved)


def make_bundle(paths, load, commit, seeds=None, resource_files=None, manifest_name=MANIFEST_NAME):
    own_seeds = SEEDS if seeds is None else tuple(seeds)
    own_resources = RESOURCE_FILES if resource_files is None else tuple(resource_files)
    selected, external, unresolved = import_closure(paths, load, own_seeds, own_resources)
    contents = {name:load(name) for name in selected}
    files = {name:dict(bytes=len(raw), sha256=digest(raw)) for name, raw in contents.items()}
    manifest = dict(commit=commit, files=files, entry_seeds=own_seeds,
        resource_files=own_resources, closure='recursive AST imports including conditional/function bodies and package initializers',
        explicit_subprocess_entry=('reproduction.operations.qwen7b_warmup.retriever' if seeds is None else None),
        external_environment_import_roots=external,
        unresolved_repository_optional_imports=unresolved,
        critical_runtime_import_smoke_required=True, code_only=True)
    manifest_raw = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    contents[manifest_name] = manifest_raw
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for name, raw in sorted(contents.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(raw), 0o644, 0
                archive.addfile(member, io.BytesIO(raw))
    return stream.getvalue(), dict(source_manifest_sha256=digest(manifest_raw),
        files=len(files), source_bytes=sum(len(raw) for raw in contents.values()),
        entry_seeds=list(own_seeds), resource_files=list(own_resources),
        external_environment_import_roots=external,
        unresolved_repository_optional_imports=unresolved)


def published_bundle(repo, prepared_commit=None):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', BRANCH], cwd=repo, text=True).split()[0]
    if head != remote:
        raise ValueError('Commit and push the latest source before deployment')
    relative = str(Path(__file__).relative_to(repo))
    if Path(__file__).read_bytes() != subprocess.check_output(['git', 'show', head+':'+relative], cwd=repo):
        raise ValueError('Deployment helper itself is not the published source')
    commit = subprocess.check_output(['git', 'rev-parse', prepared_commit or head], cwd=repo, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, head], cwd=repo, check=True)
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines()
    cache = {}
    def load(name):
        if name not in cache:
            cache[name] = subprocess.check_output(['git', 'show', commit+':'+name], cwd=repo)
        return cache[name]
    bundle, report = make_bundle(paths, load, commit)
    return commit, bundle, report


def self_test(repo):
    # Include unpublished local source only in this offline smoke, never deploy.
    paths = []
    for directory in ('reproduction', 'search_g1', 'search_r1', 'verl'):
        paths.extend(str(path.relative_to(repo)) for path in (repo/directory).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts)
    bundle, report = make_bundle(paths, lambda name:(repo/name).read_bytes(), 'a'*40)
    values = dict(base=DEFAULT_BASE, parent=PARENT, frozen=DEFAULT_FROZEN,
        commit='a'*40, bundle_sha256='b'*64, bundle_bytes=len(bundle), chunks=100,
        source_manifest_sha256=report['source_manifest_sha256'], fixed_seed=20261021)
    sizes = {}
    for name in ('INIT', 'CHUNK', 'EXTRACT', 'START', 'STATUS'):
        own = dict(values)
        if name == 'CHUNK':
            own.update(chunk_index=0, chunk_data=base64.b64encode(b'x'*CHUNK_BYTES).decode(), chunk_sha256='c'*64)
        source = 'inputs='+repr(own)+'\n'+COMMON+'\n'+globals()[name]
        compile(source, 'confirmation-remote-'+name, 'exec')
        command = 'python3 -c '+shlex.quote(source)
        if len(command.encode()) >= 16000:
            raise AssertionError('Remote command bound exceeded: '+name)
        sizes[name] = len(command.encode())
    from .confirmation_materialize import render_event_script
    script = render_event_script('/data/user/test/event-A', '/data/user/test/immutable-code')
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)
    exports = '\n'.join(line for line in script.splitlines() if line.startswith('export '))
    subprocess.run(['bash', '-euc', exports+'\ntest "$QWEN7B_CODE_ROOT" = /data/user/test/immutable-code\n'], check=True)
    result = dict(status='LOCAL_CONFIRMATION_DEPLOYMENT_SELFTEST_PASS', closure=report,
        bundle_bytes=len(bundle), chunks=(len(bundle)+CHUNK_BYTES-1)//CHUNK_BYTES,
        actual_remote_command_bytes=sizes, bash_syntax_and_code_export=True,
        remote_actions_executed=False, jobs_submitted=False)
    print(json.dumps(result), flush=True)
    return result


def remote_worker(request_path):
    """CPU-only background owner; never delegates to a sampler or scheduler."""
    import fcntl
    request_path = Path(request_path).resolve()
    if not str(request_path).startswith(PARENT+'/'):
        raise ValueError('Request must live inside the authorized isolated parent')
    request = json.loads(request_path.read_text())
    base, code = Path(request['base']), Path(request['code_root'])
    if request_path != base/'materialize-request.json' or code.parent != base:
        raise ValueError('Request/code root identity changed')
    lock = (base/'cpu-materialize.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    def atomic(path, value):
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
    def ticks(pid):
        return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[19]
    controller = dict(status='CPU_MATERIALIZATION_RUNNING', pid=os.getpid(), start_ticks=ticks(os.getpid()),
        code_root=str(code), commit=request['commit'], started=time.time(),
        cuda_visible_devices='', jobs_submitted=False, real_responses_generated=0)
    atomic(base/'cpu-controller.json', controller)
    runtime = code/'reproduction/operations/qwen7b_warmup/runtime.sh'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', QWEN7B_CODE_ROOT=str(code),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    command = ['bash', str(runtime), '-u', '-m',
        'reproduction.operations.qwen7b_warmup.confirmation_materialize',
        '--training-base', request['parent'], '--frozen-manifest', request['frozen'],
        '--base', request['destination'], '--code-root', str(code),
        '--source-commit', request['commit'], '--fixed-seed', str(request['fixed_seed'])]
    try:
        with (base/'materialization.log').open('ab') as log:
            smoke = '''import importlib,json,pathlib
root=pathlib.Path(__CODE_ROOT__).resolve()
names=['reproduction.operations.qwen7b_warmup.confirmation_materialize','reproduction.operations.qwen7b_warmup.shard','reproduction.operations.qwen7b_warmup.sampling','reproduction.operations.qwen7b_warmup.native_capture','reproduction.operations.qwen7b_warmup.packed_features','reproduction.operations.qwen7b_warmup.retriever','search_g1.async_rollout_runtime','search_g1.signals','search_r1.llm_agent.generation','search_r1.search.retrieval_server','verl.utils.reward_score.qa_em']
origins={}
for name in names:
 module=importlib.import_module(name);path=pathlib.Path(module.__file__).resolve()
 assert str(path).startswith(str(root)+'/'),(name,str(path));origins[name]=str(path)
print(json.dumps({'status':'ACTUAL_CPU_CRITICAL_COLLECTOR_IMPORTS_FROM_ISOLATED_CODE_PASS','origins':origins,'no_generation':True}),flush=True)
'''.replace('__CODE_ROOT__', repr(str(code)))
            check = subprocess.run(['bash', str(runtime), '-u', '-c', smoke], env=env, cwd=code,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if check.returncode:
                raise RuntimeError('Actual CPU critical collector import smoke failed; inspect preserved log')
            child = subprocess.Popen(command, env=env, cwd=code, stdout=log, stderr=subprocess.STDOUT)
            controller.update(child_pid=child.pid, child_start_ticks=ticks(child.pid))
            atomic(base/'cpu-controller.json', controller)
            returncode = child.wait()
        report = Path(request['destination'])/'materialization.json'
        if returncode:
            raise RuntimeError('CPU materializer exited%d; inspect preserved materialization.log' % returncode)
        value = json.loads(report.read_text())
        if value['actual_responses_generated'] != 0 or value['jobs_submitted'] is not False:
            raise ValueError('CPU preparation unexpectedly performed generation/submission')
        result = dict(status='CONFIRMATION_CPU_MATERIALIZATION_COMPLETE_NOT_GENERATED',
            returncode=returncode, materialization_report=dict(path=str(report), sha256=digest(report.read_bytes())),
            commit=request['commit'], code_root=str(code), time=time.time(),
            jobs_submitted=False, real_responses_generated=0)
    except BaseException as error:
        result = dict(status='CONFIRMATION_CPU_MATERIALIZATION_FAILED', error_type=type(error).__name__,
            error=str(error)[:1000], commit=request['commit'], time=time.time(),
            jobs_submitted=False, real_responses_generated=0)
    atomic(base/'cpu-result.json', result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control-root', type=Path)
    p.add_argument('--action', choices=('prepare', 'stage', 'status'), default='prepare')
    p.add_argument('--remote-base', default=DEFAULT_BASE)
    p.add_argument('--frozen-manifest', default=DEFAULT_FROZEN)
    p.add_argument('--prepared-commit')
    p.add_argument('--fixed-seed', type=int, default=20261021)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--remote-worker')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if a.remote_worker:
        remote_worker(a.remote_worker)
        return
    if a.self_test:
        self_test(repo)
        return
    if a.control_root is None:
        p.error('Remote prepare/status requires --control-root')
    base = Path(a.remote_base)
    if base.parent != Path(PARENT) or any(not (c.isalnum() or c in '-_.') for c in base.name):
        raise ValueError('Use a new isolated child of the authorized experiment root')
    if not str(Path(a.frozen_manifest).resolve()).startswith(PARENT+'/'):
        raise ValueError('Frozen manifest must remain in the authorized experiment')
    if a.action in ('prepare', 'stage'):
        commit, bundle, report = published_bundle(repo, a.prepared_commit)
    else:
        commit = subprocess.check_output(['git', 'rev-parse', a.prepared_commit or 'HEAD'], cwd=repo, text=True).strip()
        bundle, report = b'', {}
    chunks = [bundle[i:i+CHUNK_BYTES] for i in range(0, len(bundle), CHUNK_BYTES)]
    values = dict(base=str(base), parent=PARENT, frozen=a.frozen_manifest,
        commit=commit, bundle_sha256=digest(bundle), bundle_bytes=len(bundle), chunks=len(chunks),
        source_manifest_sha256=report.get('source_manifest_sha256'), fixed_seed=a.fixed_seed)
    adapter = a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py'
    spec = importlib.util.spec_from_file_location('confirmation_private_control', adapter)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
    def call(block, extra=None):
        own = dict(values)
        own.update(extra or {})
        source = 'inputs='+repr(own)+'\n'+COMMON+'\n'+block
        command = 'python3 -c '+shlex.quote(source)
        if len(command.encode()) >= 16000:
            raise ValueError('SSH control command exceeds the deliberate16KB bound')
        outcome = control.run(command, account='mou176', source_address=address, timeout=45)
        records = [json.loads(line) for line in outcome.stdout.decode(errors='replace').splitlines()
                   if line.strip().startswith('{')]
        if outcome.returncode:
            print(outcome.stderr.decode(errors='replace'), flush=True)
            raise SystemExit(outcome.returncode)
        if not records:
            raise ValueError('No fresh structured control result; do not reuse stale local state')
        return records[-1]
    if a.action == 'status':
        print(json.dumps(call(STATUS)), flush=True)
        return
    print(json.dumps(call(INIT)), flush=True)
    for index, raw in enumerate(chunks):
        result = call(CHUNK, dict(chunk_index=index, chunk_data=base64.b64encode(raw).decode(), chunk_sha256=digest(raw)))
        if index % 16 == 0 or index+1 == len(chunks):
            print(json.dumps(dict(stage='CODE_ONLY_UPLOAD_PROGRESS', verified_chunks=index+1,
                                  total_chunks=len(chunks), last=result)), flush=True)
    print(json.dumps(call(EXTRACT)), flush=True)
    if a.action == 'stage':
        print(json.dumps(call(STATUS)), flush=True)
        return
    print(json.dumps(call(START)), flush=True)


COMMON = r'''
import pathlib,json,hashlib,base64,os,fcntl,time,subprocess,tarfile,io,sys
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);parent=pathlib.Path(inputs['parent']);commit=inputs['commit']
assert b.is_absolute() and b.resolve()==b and b.parent==parent
code=b/('code-'+commit[:12]);uploads=b/('upload-'+inputs['bundle_sha256'][:16])
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 p=pathlib.Path(p);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
def lock():
 f=(b/'deployment.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def identity(pid,ticks):
 p=pathlib.Path('/proc/%d'%pid)
 try:return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==str(ticks)
 except OSError:return False
def code_manifest():
 p=read(b/'source-publication.json');assert p['commit']==commit and p['code_root']==str(code)
 m=read(code/'_confirmation_source_manifest.json');assert m['commit']==commit
 for name,item in m['files'].items():assert sha(code/name)==item['sha256'] and (code/name).stat().st_size==item['bytes'],name
 return p,m
'''

INIT = r'''
b.mkdir(parents=True,exist_ok=True);guard=lock()
expected=dict(commit=commit,bundle_sha256=inputs['bundle_sha256'],bundle_bytes=inputs['bundle_bytes'],chunks=inputs['chunks'],source_manifest_sha256=inputs['source_manifest_sha256'],frozen=inputs['frozen'],fixed_seed=inputs['fixed_seed'])
p=b/'upload-controller.json'
if p.exists():assert read(p)==expected,'Conflicting partial/version upload; use another isolated root'
else:atomic(p,expected)
uploads.mkdir(exist_ok=True)
print(json.dumps(dict(status='CONFIRMATION_CODE_UPLOAD_BOUND',base=str(b),commit=commit,chunks=inputs['chunks'])))
'''

CHUNK = r'''
guard=lock();c=read(b/'upload-controller.json');assert c['bundle_sha256']==inputs['bundle_sha256'] and c['commit']==commit
i=inputs['chunk_index'];assert 0<=i<inputs['chunks'];raw=base64.b64decode(inputs['chunk_data'])
assert len(raw)<=6144 and hashlib.sha256(raw).hexdigest()==inputs['chunk_sha256']
p=uploads/('part-%05d'%i)
if p.exists():assert p.read_bytes()==raw,'Previously staged chunk changed'
else:
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='CODE_CHUNK_SHA_VERIFIED',index=i,bytes=len(raw))))
'''

EXTRACT = r'''
guard=lock();parts=sorted(uploads.glob('part-*'));assert len(parts)==inputs['chunks']
raw=b''.join(p.read_bytes() for p in parts);assert len(raw)==inputs['bundle_bytes'] and hashlib.sha256(raw).hexdigest()==inputs['bundle_sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 members=archive.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
 item=archive.getmember('_confirmation_source_manifest.json');manifest_raw=archive.extractfile(item).read();assert hashlib.sha256(manifest_raw).hexdigest()==inputs['source_manifest_sha256']
 manifest=json.loads(manifest_raw);assert manifest['commit']==commit
 expected=set(manifest['files'])|{'_confirmation_source_manifest.json'};assert set(names)==expected
 target=code if code.exists() else code.with_name(code.name+'.staging');target.mkdir(exist_ok=True)
 for member in members:
  rel=pathlib.Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  data=archive.extractfile(member).read()
  if member.name!='_confirmation_source_manifest.json':assert hashlib.sha256(data).hexdigest()==manifest['files'][member.name]['sha256']
  destination=target/rel;destination.parent.mkdir(parents=True,exist_ok=True)
  if destination.exists():assert destination.read_bytes()==data,'Conflicting interrupted source staging'
  else:
   with destination.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}==expected,'Unpublished file in isolated code root'
 if target!=code:target.rename(code)
publication=dict(commit=commit,remote_sha=commit,code_root=str(code),bundle_sha256=inputs['bundle_sha256'],source_manifest_sha256=inputs['source_manifest_sha256'],files=len(manifest['files']),collector_import_closure=True,active_source_unchanged=True)
p=b/'source-publication.json'
if p.exists():assert read(p)==publication
else:atomic(p,publication)
print(json.dumps(dict(status='ISOLATED_COLLECTOR_IMPORT_CLOSURE_SHA_READY',code_root=str(code),files=len(manifest['files']),commit=commit)))
'''

START = r'''
guard=lock();publication,manifest=code_manifest()
if (b/'cpu-result.json').exists():print(json.dumps(read(b/'cpu-result.json')));raise SystemExit(0)
if (b/'launch-intent.json').exists():
 print(json.dumps(dict(status='EXISTING_CPU_MATERIALIZATION_INTENT_INSPECT_STATUS_DO_NOT_DUPLICATE',intent=read(b/'launch-intent.json'))));raise SystemExit(0)
assert not (b/'materialized').exists(),'Existing unowned materialization outputs must be reconciled'
assert pathlib.Path(inputs['frozen']).is_file()
request=dict(base=str(b),parent=str(parent),frozen=inputs['frozen'],destination=str(b/'materialized'),code_root=str(code),commit=commit,fixed_seed=inputs['fixed_seed'])
atomic(b/'materialize-request.json',request);atomic(b/'launch-intent.json',dict(time=time.time(),request_sha256=sha(b/'materialize-request.json'),commit=commit))
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (b/'cpu-runner.log').open('ab') as log:
 process=subprocess.Popen(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.deploy_confirmation_materialization','--remote-worker',str(b/'materialize-request.json')],cwd=str(code),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=pathlib.Path('/proc/%d/stat'%process.pid).read_text().rsplit(')',1)[1].split()[19]
launch=dict(pid=process.pid,start_ticks=ticks,commit=commit,time=time.time(),jobs_submitted=False,real_responses_generated=0)
atomic(b/'cpu-launch.json',launch)
print(json.dumps(dict(status='OWNED_CPU_MATERIALIZATION_LAUNCHED_VERIFY_STATUS',launch=launch,source=publication)))
'''

STATUS = r'''
if not b.exists():print(json.dumps(dict(status='NO_SUCH_ISOLATED_DEPLOYMENT',base=str(b))));raise SystemExit(0)
result=dict(base=str(b),observed_time=time.time(),jobs_submitted=False,real_responses_generated=0)
for name in ('source-publication.json','cpu-launch.json','cpu-controller.json','cpu-result.json'):
 if (b/name).exists():result[name[:-5].replace('-','_')]=read(b/name)
launch=result.get('cpu_launch',{});controller=result.get('cpu_controller',{})
result['runner_identity_live']=identity(launch['pid'],launch['start_ticks']) if launch else False
result['worker_identity_live']=identity(controller['pid'],controller['start_ticks']) if controller else False
result['child_identity_live']=identity(controller['child_pid'],controller['child_start_ticks']) if controller.get('child_pid') else False
default_status='CPU_MATERIALIZATION_LIVE' if result['runner_identity_live'] or result['worker_identity_live'] or result['child_identity_live'] else ('CODE_STAGED_NO_CPU_MATERIALIZATION' if result.get('source_publication') and not (b/'launch-intent.json').exists() else 'OUTCOME_UNPROVEN_INSPECT_DURABLE_INTENT_DO_NOT_RESTART')
result['status']=result.get('cpu_result',{}).get('status',default_status)
if (b/'materialization.log').exists():result['cpu_log_bytes']=(b/'materialization.log').stat().st_size
print(json.dumps(result))
'''


if __name__ == '__main__':
    main()
