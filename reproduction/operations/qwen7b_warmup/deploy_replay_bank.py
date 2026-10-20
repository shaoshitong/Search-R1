"""Independent published-code deployment / CPU fixture preparation / shard control.

No remote action on import or --self-test. prepare starts CPU metadata work only;
submit is separate, per-shard, explicitly guarded, no dependencies. Existing
Q1 official view and exact18c46aff replay primitives are reused, never modified.
All control commands (including shell quoting) are strictly below16KB.
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

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
DEFAULT_BASE=PARENT+'/replay-matched9088-bank-20261021'
DEFAULT_CONFIG=PARENT+'/algorithm-improvement-20261021/matched-loss128-v2/event/config.json'
LEGACY='18c46affb9c6c69330aa19eb1e00124e1f39d019'
BRANCH='refs/heads/codex/qwen7b-probe-warmup-20261019'
PACKAGE='reproduction/operations/qwen7b_warmup/'
MANIFEST='_replay_bank_source_manifest.json'
SEEDS=tuple(PACKAGE+n for n in ('deploy_replay_bank.py','replay_bank_prepare.py',
    'replay_bank_capture.py','replay_probe_layers.py','replay_feature_reader.py',
    'test_replay_bank.py','runtime.sh'))
CHUNK_BYTES=4096


def digest(raw):return hashlib.sha256(raw).hexdigest()


def bundle_from_sources(paths,load,commit):
    paths=set(paths);pending=list(SEEDS);selected=set();external=set()
    def module(parts):
        stem='/'.join(parts)
        for p in (stem+'.py',stem+'/__init__.py'):
            if p in paths:
                pending.append(p)
                for parent in Path(p).parents:
                    if str(parent)=='.':break
                    init=str(parent/'__init__.py')
                    if init in paths:pending.append(init)
                return
        if parts and parts[0] not in ('reproduction','search_g1','search_r1','verl'):
            external.add(parts[0])
    while pending:
        p=pending.pop()
        if p in selected:continue
        if p not in paths:raise ValueError('Missing actual source/import entry '+p)
        selected.add(p)
        if not p.endswith('.py'):continue
        package=list(Path(p).with_suffix('').parts)[:-1]
        for node in ast.walk(ast.parse(load(p).decode(),filename=p)):
            if isinstance(node,ast.Import):
                for a in node.names:module(a.name.split('.'))
            elif isinstance(node,ast.ImportFrom):
                parts=(package[:len(package)-node.level+1] if node.level else [])
                parts+=node.module.split('.') if node.module else []
                module(parts)
                for a in node.names:
                    if a.name!='*':module(parts+a.name.split('.'))
    contents={p:load(p) for p in selected}
    files={p:dict(bytes=len(raw),sha256=digest(raw)) for p,raw in contents.items()}
    manifest=dict(commit=commit,files=files,entry_seeds=list(SEEDS),external_environment_imports=sorted(external),
        closure='all-branches AST local imports and package initializers',code_only=True,
        legacy_replay_commit=LEGACY,legacy_replay_path=PACKAGE+'replay_probe_layers.py')
    raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode();contents[MANIFEST]=raw
    output=io.BytesIO()
    with gzip.GzipFile(fileobj=output,mode='wb',mtime=0,filename='') as stream:
        with tarfile.open(fileobj=stream,mode='w') as tar:
            for name,data in sorted(contents.items()):
                item=tarfile.TarInfo(name);item.size=len(data);item.mtime=0;item.mode=0o644
                tar.addfile(item,io.BytesIO(data))
    return output.getvalue(),dict(source_manifest_sha256=digest(raw),files=len(files),
        source_bytes=sum(len(v) for v in contents.values()),external_environment_imports=sorted(external))


def published_bundle(repo,prepared_commit=None):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin',BRANCH],cwd=repo,text=True).split()[0]
    if head!=remote:raise ValueError('Publish latest source before deployment')
    rel=str(Path(__file__).relative_to(repo))
    if Path(__file__).read_bytes()!=subprocess.check_output(['git','show',head+':'+rel],cwd=repo):
        raise ValueError('Deployment helper is not published')
    commit=subprocess.check_output(['git','rev-parse',prepared_commit or head],cwd=repo,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,head],cwd=repo,check=True)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines()
    cache={}
    def load(name):
        if name not in cache:cache[name]=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
        return cache[name]
    legacy_path=PACKAGE+'replay_probe_layers.py'
    if load(legacy_path)!=subprocess.check_output(['git','show',LEGACY+':'+legacy_path],cwd=repo):
        raise ValueError('Qualified replay primitive changed; refuse silent backend change')
    raw,report=bundle_from_sources(paths,load,commit)
    return commit,raw,report


def self_test(repo):
    paths=[str(p.relative_to(repo)) for p in (repo/'reproduction').rglob('*')
           if p.is_file() and '__pycache__' not in p.parts]
    raw,report=bundle_from_sources(paths,lambda name:(repo/name).read_bytes(),'a'*40)
    values=dict(base=DEFAULT_BASE,parent=PARENT,commit='a'*40,bundle_sha256=digest(raw),
        bundle_bytes=len(raw),chunks=10,source_manifest_sha256=report['source_manifest_sha256'],
        matched_config=DEFAULT_CONFIG,matched_sha='b'*64,hours=1,shard=0)
    sizes={}
    for name in ('INIT','CHUNK','EXTRACT','START','STATUS','SUBMIT'):
        own=dict(values)
        if name=='CHUNK':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*CHUNK_BYTES).decode(),chunk_sha256='c'*64)
        source='inputs='+repr(own)+'\n'+COMMON+'\n'+globals()[name]
        compile(source,'replay-bank-control-'+name,'exec')
        command='python3 -c '+shlex.quote(source)
        if len(command.encode())>=16000:raise ValueError('Quoted control command exceeds16KB: '+name)
        sizes[name]=len(command.encode())
    result=dict(status='LOCAL_REPLAY_BANK_CLOSURE_TRANSPORT_SELFTEST_PASS',closure=report,
        bundle_bytes=len(raw),chunks=(len(raw)+CHUNK_BYTES-1)//CHUNK_BYTES,
        quoted_command_bytes=sizes,remote_actions_executed=False,jobs_submitted=False)
    print(json.dumps(result),flush=True);return result


def worker(request_path):
    request_path=Path(request_path);request=json.loads(request_path.read_text());base=Path(request['base']);code=Path(request['code_root'])
    if (not str(base).startswith(PARENT+'/') or request_path!=base/'cpu-request.json' or code.parent!=base):
        raise ValueError('Explicit isolated CPU worker identity required')
    with (base/'cpu-worker.lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def atomic(path,value):
            p=Path(path);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
        def ticks(pid):return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
        controller=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),code_root=str(code),
            source_commit=request['commit'],status='CPU_PREFIX_METADATA_PREPARATION_RUNNING',time=time.time(),GPU_used=False)
        atomic(base/'cpu-controller.json',controller)
        runtime=code/(PACKAGE+'runtime.sh')
        env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        command=['bash',str(runtime),'-u','-m','reproduction.operations.qwen7b_warmup.replay_bank_prepare',
            '--base',str(base/'prepared'),'--training-base',PARENT,'--matched-config',request['matched_config'],
            '--matched-config-sha256',request['matched_sha'],'--code-root',str(code),
            '--source-commit',request['commit'],'--hours',str(request['hours'])]
        try:
            with (base/'cpu-prepare.log').open('ab') as log:
                smoke=("import importlib,pathlib\nroot=pathlib.Path(%r)\n"%str(code)+
                    "for name in ['replay_bank_prepare','replay_bank_capture','replay_probe_layers','replay_feature_reader','packed_features','native_capture']:\n"
                    " m=importlib.import_module('reproduction.operations.qwen7b_warmup.'+name)\n"
                    " assert str(pathlib.Path(m.__file__).resolve()).startswith(str(root)+'/'),name\n")
                check=subprocess.run(['bash',str(runtime),'-u','-c',smoke],env=env,cwd=code,stdout=log,stderr=subprocess.STDOUT,timeout=90)
                if check.returncode:raise ValueError('CPU actual isolated replay import smoke failed')
                process=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
                controller.update(child_pid=process.pid,child_start_ticks=ticks(process.pid));atomic(base/'cpu-controller.json',controller)
                rc=process.wait()
            if rc:raise ValueError('CPU fixture preparation failed; preserve actual log/partial metadata')
            prep=base/'prepared/preparation.json';result=json.loads(prep.read_text())
            if result['identity']['questions']!=9088 or result['identity']['trajectories']!=45440 or result['jobs_submitted']:
                raise ValueError('Actual prepared fixture coverage/side effects differ')
            done=dict(status='ACTUAL_REAL_9088Q_FOUR_SHARD_CPU_FIXTURES_PREPARED_NOT_CAPTURED',
                preparation=dict(path=str(prep),sha256=digest(prep.read_bytes()),bytes=prep.stat().st_size),
                identity=result['identity'],selections=result['selections'],scripts=result['scripts'],
                per_shard_questions=result['per_shard_questions'],per_shard_trajectories=result['per_shard_trajectories'],
                jobs_submitted=False,replay_features_generated=0,time=time.time())
        except BaseException as error:
            done=dict(status='CPU_REPLAY_BANK_PREPARATION_FAILED',type=type(error).__name__,error=str(error)[:1000],time=time.time(),jobs_submitted=False)
        atomic(base/'cpu-result.json',done)


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=('stage','prepare','status','submit'),default='prepare')
    p.add_argument('--remote-base',default=DEFAULT_BASE);p.add_argument('--prepared-commit');p.add_argument('--matched-config',default=DEFAULT_CONFIG)
    p.add_argument('--matched-config-sha256');p.add_argument('--hours',type=int,choices=(1,2),default=1)
    p.add_argument('--shard',type=int,choices=range(4));p.add_argument('--submit-authorized',action='store_true')
    p.add_argument('--self-test',action='store_true');p.add_argument('--remote-worker');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    if a.remote_worker:worker(a.remote_worker);return
    if a.self_test:self_test(repo);return
    if a.control_root is None:p.error('--control-root required for remote actions')
    if a.action=='submit' and (not a.submit_authorized or a.shard is None):p.error('Separate --submit-authorized --shard N required')
    base=Path(a.remote_base)
    if base.parent!=Path(PARENT) or any(not(c.isalnum() or c in '-_.') for c in base.name):raise ValueError('Use a new isolated experiment child')
    if a.action in ('stage','prepare'):
        if not a.matched_config_sha256 or len(a.matched_config_sha256)!=64:raise ValueError('Actual matched config SHA required')
        commit,raw,report=published_bundle(repo,a.prepared_commit)
    else:
        commit=subprocess.check_output(['git','rev-parse',a.prepared_commit or 'HEAD'],cwd=repo,text=True).strip();raw=b'';report={}
    chunks=[raw[i:i+CHUNK_BYTES] for i in range(0,len(raw),CHUNK_BYTES)]
    values=dict(base=str(base),parent=PARENT,commit=commit,bundle_sha256=digest(raw),bundle_bytes=len(raw),chunks=len(chunks),
        source_manifest_sha256=report.get('source_manifest_sha256'),matched_config=a.matched_config,
        matched_sha=a.matched_config_sha256,hours=a.hours,shard=a.shard)
    spec=importlib.util.spec_from_file_location('replay_bank_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(block,extra=None):
        inputs=dict(values);inputs.update(extra or {})
        source='inputs='+repr(inputs)+'\n'+COMMON+'\n'+block;command='python3 -c '+shlex.quote(source)
        if len(command.encode())>=16000:raise ValueError('Quoted SSH command exceeds16KB')
        out=control.run(command,account='mou176',source_address=address,timeout=45)
        rows=[json.loads(line) for line in out.stdout.decode(errors='replace').splitlines() if line.strip().startswith('{')]
        if out.returncode:raise RuntimeError('Control action unconfirmed; inspect status/intent before retry: '+out.stderr.decode(errors='replace')[-1000:])
        if not rows:raise ValueError('No fresh structured remote result')
        return rows[-1]
    if a.action=='status':print(json.dumps(call(STATUS)),flush=True);return
    if a.action=='submit':print(json.dumps(call(SUBMIT)),flush=True);return
    print(json.dumps(call(INIT)),flush=True)
    for index,chunk in enumerate(chunks):
        row=call(CHUNK,dict(chunk_index=index,chunk_data=base64.b64encode(chunk).decode(),chunk_sha256=digest(chunk)))
        if index%16==0 or index+1==len(chunks):print(json.dumps(dict(code_chunks_verified=index+1,total=len(chunks))),flush=True)
    print(json.dumps(call(EXTRACT)),flush=True)
    print(json.dumps(call(START if a.action=='prepare' else STATUS)),flush=True)


COMMON=r'''
import pathlib,json,hashlib,os,fcntl,time,subprocess,base64,tarfile,io,sys
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);parent=pathlib.Path(inputs['parent']);commit=inputs['commit'];code=b/('code-'+commit[:12]);uploads=b/('upload-'+inputs['bundle_sha256'][:16])
assert b.is_absolute() and b.resolve()==b and b.parent==parent
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(v):
 a=ref(v['path']);assert a['sha256']==v['sha256'] and ('bytes' not in v or a['bytes']==v['bytes']);return pathlib.Path(v['path'])
def atomic(p,v):
 p=pathlib.Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True));t.replace(p)
def guard():
 f=(b/'deployment.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def published():
 pub=read(b/'source-publication.json');assert pub['commit']==commit and pub['code_root']==str(code)
 m=read(code/'_replay_bank_source_manifest.json');assert m['commit']==commit
 for name,v in m['files'].items():assert (code/name).stat().st_size==v['bytes'] and sha(code/name)==v['sha256'],name
 return pub
def live(pid,ticks):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]==str(ticks)
 except OSError:return False
'''
INIT=r'''
b.mkdir(exist_ok=True,parents=True);lock=guard()
expected={k:inputs[k] for k in ('commit','bundle_sha256','bundle_bytes','chunks','source_manifest_sha256','matched_config','matched_sha','hours')}
p=b/'upload-controller.json'
if p.exists():assert read(p)==expected,'Conflicting partial/version upload: use a new isolated root'
else:atomic(p,expected)
uploads.mkdir(exist_ok=True)
print(json.dumps(dict(status='REPLAY_BANK_CODE_UPLOAD_BOUND',commit=commit,chunks=inputs['chunks'])))
'''
CHUNK=r'''
lock=guard();assert read(b/'upload-controller.json')['bundle_sha256']==inputs['bundle_sha256']
i=inputs['chunk_index'];assert 0<=i<inputs['chunks'];raw=base64.b64decode(inputs['chunk_data']);assert len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==inputs['chunk_sha256']
p=uploads/('part-%05d'%i)
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='REPLAY_CODE_CHUNK_SHA_VERIFIED',index=i)))
'''
EXTRACT=r'''
lock=guard();parts=sorted(uploads.glob('part-*'));assert len(parts)==inputs['chunks'];raw=b''.join(p.read_bytes() for p in parts)
assert len(raw)==inputs['bundle_bytes'] and hashlib.sha256(raw).hexdigest()==inputs['bundle_sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 members=tar.getmembers();names=[m.name for m in members];assert len(set(names))==len(names)
 manifest_raw=tar.extractfile('_replay_bank_source_manifest.json').read();assert hashlib.sha256(manifest_raw).hexdigest()==inputs['source_manifest_sha256'];manifest=json.loads(manifest_raw);assert manifest['commit']==commit
 assert set(names)==set(manifest['files'])|{'_replay_bank_source_manifest.json'}
 target=code if code.exists() else code.with_name(code.name+'.staging');target.mkdir(exist_ok=True)
 for m in members:
  p=pathlib.Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts;raw=tar.extractfile(m).read()
  if m.name!='_replay_bank_source_manifest.json':assert len(raw)==manifest['files'][m.name]['bytes'] and hashlib.sha256(raw).hexdigest()==manifest['files'][m.name]['sha256']
  dest=target/p;dest.parent.mkdir(exist_ok=True,parents=True)
  if dest.exists():assert dest.read_bytes()==raw,'Conflicting partial immutable source'
  else:
   with dest.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}==set(names)
 if target!=code:target.rename(code)
pub=dict(commit=commit,code_root=str(code),files=manifest['files'],source_manifest_sha256=inputs['source_manifest_sha256'],legacy_replay_commit=manifest['legacy_replay_commit'],active_source_unchanged=True)
p=b/'source-publication.json'
if p.exists():assert read(p)==pub
else:atomic(p,pub)
print(json.dumps(dict(status='INDEPENDENT_REPLAY_BANK_IMPORT_CLOSURE_SHA_READY',commit=commit,code_root=str(code),files=len(manifest['files']))))
'''
START=r'''
lock=guard();pub=published()
if (b/'cpu-result.json').exists():print(json.dumps(read(b/'cpu-result.json')));raise SystemExit(0)
if (b/'cpu-launch-intent.json').exists():print(json.dumps(dict(status='EXISTING_CPU_INTENT_INSPECT_STATUS_DO_NOT_DUPLICATE')));raise SystemExit(0)
assert not (b/'prepared').exists(),'Existing unowned preparation outputs'
request=dict(base=str(b),code_root=str(code),commit=commit,matched_config=inputs['matched_config'],matched_sha=inputs['matched_sha'],hours=inputs['hours'])
atomic(b/'cpu-request.json',request);atomic(b/'cpu-launch-intent.json',dict(request=ref(b/'cpu-request.json'),time=time.time()))
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (b/'cpu-worker.log').open('ab') as log:
 p=subprocess.Popen(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.deploy_replay_bank','--remote-worker',str(b/'cpu-request.json')],cwd=str(code),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=pathlib.Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]
atomic(b/'cpu-launch.json',dict(pid=p.pid,start_ticks=ticks,time=time.time(),jobs_submitted=False))
print(json.dumps(dict(status='OWNED_CPU_PREFIX_ONLY_PREPARATION_LAUNCHED_VERIFY_REAL_STATUS',pid=p.pid,jobs_submitted=False)))
'''
STATUS=r'''
if not b.exists():print(json.dumps(dict(status='NO_SUCH_ISOLATED_REPLAY_BANK')));raise SystemExit(0)
out=dict(base=str(b),time=time.time())
for name in ('cpu-launch.json','cpu-controller.json','cpu-result.json'):
 if (b/name).exists():out[name[:-5]]=read(b/name)
l=out.get('cpu-launch',{});c=out.get('cpu-controller',{});out['cpu_identity_live']=bool(l and live(l['pid'],l['start_ticks'])) or bool(c and live(c['pid'],c['start_ticks']))
out['jobs']=[]
for i in range(4):
 root=b/'prepared'/('shard-%02d'%i);own=dict(shard=i)
 for name in ('job.id','submission-intent.json','capture/progress.json','capture/failure.json','capture/complete.json'):
  p=root/name
  if p.exists():
   if name=='job.id':
    job=p.read_text().strip();r=subprocess.run(['sacct','-n','-P','-j',job,'-o','JobID,JobName,State,ExitCode'],capture_output=True,text=True,timeout=20);own['job']=job;own['accounting']=r.stdout[-4000:]
   elif name=='capture/complete.json':v=read(p);own['complete']={k:v.get(k) for k in ('status','questions','trajectories','capture_layers','native_bitexact_equivalence_claim')}
   else:own[name]=read(p)
 out['jobs'].append(own)
print(json.dumps(out))
'''
SUBMIT=r'''
lock=guard();published();result=read(b/'cpu-result.json');assert result['status']=='ACTUAL_REAL_9088Q_FOUR_SHARD_CPU_FIXTURES_PREPARED_NOT_CAPTURED'
prep=read(verify(result['preparation']));i=inputs['shard'];assert type(i)is int and 0<=i<4;root=b/'prepared'/('shard-%02d'%i)
for v in (prep['selections'][i],prep['scripts'][i]):verify(v)
assert not (root/'capture/complete.json').exists(),'Already complete; do not resubmit'
if (root/'job.id').exists():print(json.dumps(dict(status='EXISTING_REPLAY_SHARD_JOB_NOT_RESUBMITTED',shard=i,job=(root/'job.id').read_text().strip())));raise SystemExit(0)
assert not (root/'submission-intent.json').exists(),'Unknown prior submission; inspect queue/accounting before any retry'
name='q7b-replay-bank-'+str(i);q=subprocess.run(['squeue','-h','-u','mou176','-n',name,'-o','%A|%T|%j'],capture_output=True,text=True,timeout=20);assert q.returncode==0 and not q.stdout.strip(),'Existing same-name job; reconcile identity before submission'
assert not (root/'capture').exists(),'Unowned/partial writer output; preserve and inspect'
atomic(root/'submission-intent.json',dict(name=name,script=prep['scripts'][i],selection=prep['selections'][i],time=time.time()))
r=subprocess.run(['sbatch','--parsable',prep['scripts'][i]['path']],capture_output=True,text=True,timeout=25)
assert r.returncode==0,(r.stdout+r.stderr)[-2000:];job=r.stdout.strip().split(';')[0];assert job.isdigit();(root/'job.id').write_text(job+'\n')
print(json.dumps(dict(status='REPLAY_SHARD_SUBMITTED_NOT_CAPTURE_ACCEPTED',shard=i,job=job,partition='acd_u',qos='normal',dependency=None,gpus=1,time=time.time())))
'''


if __name__=='__main__':main()
