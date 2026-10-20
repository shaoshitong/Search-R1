"""Published, isolated E1/E3 native-vs-replay comparison deployment.

Only ``prepare`` stages code and small SHA-bound metadata. It first requires
actual completed extended qualification (120 trajectories / 24 real G5 groups).
It never recaptures tokens, reads labels, loads model tensors on the login node,
or modifies the earlier qualification/runtime. ``submit --submit-authorized``
requests one normal acd_u GPU for 30 minutes, with no dependency. Any prior
submit intent, including an unknown result, blocks another submission.
"""
import argparse
import base64
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shlex
import subprocess

from .deploy_confirmation_materialization import make_bundle

PARENT = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
DEFAULT_BASE = PARENT+'/replay-downstream-comparison-20261021'
DEFAULT_QUALIFICATION = PARENT+'/replay-extended-qualification-20261021/qualification-output'
DEFAULT_E1 = PARENT+'/mixed-only-ft-cp2-prefix20-20261020/probe/epoch-0001.json'
DEFAULT_E3 = PARENT+'/balanced-rank1-from-epoch1-prefix20-20261020/probe/epoch-0003.json'
BRANCH = 'refs/heads/codex/qwen7b-probe-warmup-20261019'
PACKAGE = 'reproduction/operations/qwen7b_warmup/'
SEEDS = tuple(PACKAGE+name for name in (
    'compare_replay_probe_outputs.py', 'replay_feature_reader.py',
    'model.py', 'train.py', 'packed_features.py', 'replay_probe_layers.py', 'runtime.sh'))
MANIFEST = '_replay_comparison_source_manifest.json'
CHUNK_BYTES = 4096
JOB_NAME = 'q7b-replay-compare'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def render_sbatch(base, code_root, qualification, qualification_sha256,
                  checkpoints, checkpoints_sha256, job_name=JOB_NAME):
    paths = (str(base), str(code_root), str(qualification), str(checkpoints))
    if any(not Path(p).is_absolute() or '\n' in p or '\r' in p for p in paths):
        raise ValueError('Absolute newline-free remote paths required')
    if any(c.isspace() for c in str(base)):
        raise ValueError('Slurm output base must not contain whitespace')
    if not job_name or any(not (c.isalnum() or c in '-_') for c in job_name):
        raise ValueError('Safe isolated job name required')
    for value in (qualification_sha256, checkpoints_sha256):
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Real SHA256 bindings required')
    script = '''#!/usr/bin/env bash
#SBATCH --job-name=@@CMP_JOB@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=@@CMP_BASE_RAW@@/logs/compare-%j.log
set -euo pipefail
CMP_BASE=@@CMP_BASE_Q@@
CMP_CODE_ROOT=@@CMP_CODE_Q@@
CMP_QUALIFICATION=@@CMP_QUALIFICATION_Q@@
CMP_CHECKPOINTS=@@CMP_CHECKPOINTS_Q@@
exec 9>"$CMP_BASE/comparison-run.lock"
flock -n 9
export QWEN7B_CODE_ROOT="$CMP_CODE_ROOT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.compare_replay_probe_outputs --qualification-root "$CMP_QUALIFICATION" --qualification-sha256 @@CMP_QUALIFICATION_SHA@@ --checkpoints "$CMP_CHECKPOINTS" --checkpoints-sha256 @@CMP_CHECKPOINTS_SHA@@ --output "$CMP_BASE/comparison-output" --device cuda:0 --micro-trajectories 4
'''
    substitutions = {
        '@@CMP_JOB@@':job_name, '@@CMP_BASE_RAW@@':str(base),
        '@@CMP_BASE_Q@@':shlex.quote(str(base)), '@@CMP_CODE_Q@@':shlex.quote(str(code_root)),
        '@@CMP_QUALIFICATION_Q@@':shlex.quote(str(qualification)),
        '@@CMP_CHECKPOINTS_Q@@':shlex.quote(str(checkpoints)),
        '@@CMP_QUALIFICATION_SHA@@':qualification_sha256,
        '@@CMP_CHECKPOINTS_SHA@@':checkpoints_sha256,
    }
    for key, value in substitutions.items():
        script = script.replace(key, value)
    if '@@CMP_' in script:
        raise ValueError('Unrendered comparison placeholder')
    return script


def bundle_from(repo, commit, published=True):
    """Export the true AST import closure from one published Git revision."""
    if published:
        paths = subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines()
        cache = {}
        def load(name):
            if name not in cache:
                cache[name] = subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
            return cache[name]
    else:
        paths = [str(p.relative_to(repo)) for directory in ('reproduction','search_g1','search_r1','verl')
                 for p in (repo/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        load = lambda name:(repo/name).read_bytes()
    return make_bundle(paths,load,commit,seeds=SEEDS,resource_files=(),manifest_name=MANIFEST)


def published_bundle(repo, prepared_commit=None):
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin',BRANCH],cwd=repo,text=True).split()[0]
    if head != remote:
        raise ValueError('Publish latest source before deployment')
    for module in (__file__, Path(__file__).with_name('deploy_confirmation_materialization.py')):
        path = Path(module)
        rel = str(path.relative_to(repo))
        if path.read_bytes() != subprocess.check_output(['git','show',head+':'+rel],cwd=repo):
            raise ValueError('Unpublished deployment helper: '+rel)
    commit = subprocess.check_output(['git','rev-parse',prepared_commit or head],cwd=repo,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,head],cwd=repo,check=True)
    bundle, report = bundle_from(repo,commit)
    return commit,bundle,report


def control_source(values, block, extra=None):
    own = dict(values)
    own.update(extra or {})
    source = 'inputs='+repr(own)+'\n'+COMMON+'\n'
    if block == SUBMIT:
        source += inspect.getsource(render_sbatch)+'\n'
    return source+block


def self_test(repo):
    bundle, report = bundle_from(repo,'a'*40,published=False)
    values = dict(base=DEFAULT_BASE,parent=PARENT,qualification=DEFAULT_QUALIFICATION,
        e1_receipt=DEFAULT_E1,e3_receipt=DEFAULT_E3,commit='a'*40,
        job_name=JOB_NAME,bundle_sha256=digest(bundle),bundle_bytes=len(bundle),
        chunks=(len(bundle)+CHUNK_BYTES-1)//CHUNK_BYTES,
        source_manifest_sha256=report['source_manifest_sha256'],
        renderer_sha256=digest(inspect.getsource(render_sbatch).encode()))
    sizes = {}
    for name in ('CHECK','INIT','CHUNK','EXTRACT','PREPARE','SUBMIT','STATUS'):
        extra = dict(chunk_index=0,chunk_data=base64.b64encode(b'x'*CHUNK_BYTES).decode(),chunk_sha256='b'*64) if name=='CHUNK' else None
        source = control_source(values,globals()[name],extra)
        compile(source,'comparison-remote-'+name,'exec')
        sizes[name] = len(('python3 -c '+shlex.quote(source)).encode())
        if sizes[name] >= 16000:
            raise AssertionError('Remote command bound exceeded: '+name)
    base = '/data/user/test/replay-compare'
    code = base+'/code-0123456789ab'
    script = render_sbatch(base,code,base+'/Q2','a'*64,base+'/inputs/checkpoints.json','b'*64)
    subprocess.run(['bash','-n'],input=script,text=True,check=True)
    setup = '\n'.join(line for line in script.splitlines() if line.startswith(('CMP_','export ')))
    subprocess.run(['bash','-euc',setup+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(code)+'\ntest "$CMP_QUALIFICATION" = '+shlex.quote(base+'/Q2')+'\n'],check=True)
    if '--dependency' in script or 'afterok' in script:
        raise AssertionError('Unexpected previous-job dependency')
    result = dict(status='LOCAL_REPLAY_COMPARISON_DEPLOYMENT_SELFTEST_PASS',
        closure=report,bundle_bytes=len(bundle),remote_command_bytes=sizes,
        actual_bash_syntax_and_export_execution=True,remote_actions_executed=False,jobs_submitted=False)
    print(json.dumps(result),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-root',type=Path)
    parser.add_argument('--action',choices=('prepare','submit','status'),default='prepare')
    parser.add_argument('--remote-base',default=DEFAULT_BASE)
    parser.add_argument('--qualification-root',default=DEFAULT_QUALIFICATION)
    parser.add_argument('--e1-receipt',default=DEFAULT_E1)
    parser.add_argument('--e3-receipt',default=DEFAULT_E3)
    parser.add_argument('--prepared-commit')
    parser.add_argument('--job-name',default=JOB_NAME)
    parser.add_argument('--submit-authorized',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if args.self_test:
        self_test(repo)
        return
    if args.control_root is None:
        parser.error('Remote actions require --control-root')
    base = Path(args.remote_base)
    if base.parent != Path(PARENT) or any(not (c.isalnum() or c in '-_.') for c in base.name):
        raise ValueError('Use an isolated direct child of the authorized experiment root')
    for value in (args.qualification_root,args.e1_receipt,args.e3_receipt):
        path = Path(value)
        if not path.is_absolute() or not str(path).startswith(PARENT+'/') or '..' in path.parts:
            raise ValueError('All source metadata must remain in the remote experiment')
    if args.action == 'submit' and not args.submit_authorized:
        parser.error('Submit requires explicit --submit-authorized')
    if args.action == 'prepare':
        commit,bundle,report = published_bundle(repo,args.prepared_commit)
    else:
        commit = subprocess.check_output(['git','rev-parse',args.prepared_commit or 'HEAD'],cwd=repo,text=True).strip()
        bundle,report = b'',{}
    chunks = [bundle[i:i+CHUNK_BYTES] for i in range(0,len(bundle),CHUNK_BYTES)]
    values = dict(base=str(base),parent=PARENT,qualification=args.qualification_root,
        e1_receipt=args.e1_receipt,e3_receipt=args.e3_receipt,commit=commit,job_name=args.job_name,
        bundle_sha256=digest(bundle),bundle_bytes=len(bundle),chunks=len(chunks),
        source_manifest_sha256=report.get('source_manifest_sha256'),
        renderer_sha256=digest(inspect.getsource(render_sbatch).encode()))
    adapter = args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py'
    spec = importlib.util.spec_from_file_location('comparison_private_control',adapter)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    address = subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(block,extra=None):
        source = control_source(values,block,extra)
        command = 'python3 -c '+shlex.quote(source)
        if len(command.encode()) >= 16000:
            raise ValueError('SSH control command exceeds 16KB')
        outcome = control.run(command,account='mou176',source_address=address,timeout=55)
        records = [json.loads(line) for line in outcome.stdout.decode(errors='replace').splitlines()
                   if line.strip().startswith('{')]
        if outcome.returncode:
            print(outcome.stderr.decode(errors='replace')[-3000:],flush=True)
            raise SystemExit(outcome.returncode)
        if not records:
            raise RuntimeError('Missing fresh structured result; inspect before retrying')
        return records[-1]
    if args.action == 'status':
        print(json.dumps(call(STATUS)),flush=True)
        return
    if args.action == 'submit':
        print(json.dumps(call(SUBMIT)),flush=True)
        return
    print(json.dumps(call(CHECK)),flush=True)
    print(json.dumps(call(INIT)),flush=True)
    for index,raw in enumerate(chunks):
        result = call(CHUNK,dict(chunk_index=index,chunk_data=base64.b64encode(raw).decode(),chunk_sha256=digest(raw)))
        if index%16 == 0 or index+1 == len(chunks):
            print(json.dumps(dict(status='COMPARISON_CODE_UPLOAD_PROGRESS',verified_chunks=index+1,total_chunks=len(chunks),last=result)),flush=True)
    print(json.dumps(call(EXTRACT)),flush=True)
    print(json.dumps(call(PREPARE)),flush=True)


COMMON = r'''
import pathlib,json,hashlib,base64,os,fcntl,time,subprocess,tarfile,io,sys,shlex
from pathlib import Path
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);parent=pathlib.Path(inputs['parent']);commit=inputs['commit']
JOB_NAME=inputs['job_name']
assert b.is_absolute() and b.resolve()==b and b.parent==parent
code=b/('code-'+commit[:12]);uploads=b/('upload-'+inputs['bundle_sha256'][:16])
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def command(args,timeout=25):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 assert r.returncode==0,(args,r.stderr[-2000:]);return r.stdout.strip()
def lock():
 h=(b/'controller.lock').open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def unsubmitted():
 assert not (b/'job.id').exists(),'Existing job identity; inspect rather than duplicate'
 assert not (b/'submission-intent.json').exists(),'Prior submit outcome must be reconciled; do not repeat'
def verify_code():
 p=read(b/'source-publication.json');assert p['commit']==commit and p['code_root']==str(code)
 manifest_path=code/'_replay_comparison_source_manifest.json';assert sha(manifest_path)==p['source_manifest_sha256']
 m=read(manifest_path);assert m['commit']==commit
 for name,item in m['files'].items():assert fp(code/name)==dict(path=str(code/name),**item),name
 return p
def checked_qualification():
 q=pathlib.Path(inputs['qualification']);assert str(q).startswith(str(parent)+'/') and q.resolve()==q
 p=q/'complete.json';v=read(p)
 assert v['status']=='BOUNDED_HF_REPLAY_CAPTURE_AND_NATIVE_COMPARISON_COMPLETE'
 assert v['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
 assert v['trajectories']==120 and v['bit_exact_native_vllm_claimed'] is False
 assert v['accepted_for_training'] is False
 source=v['source_selection'];assert fp(source['path'])==source
 selection=read(source['path']);rows=selection['trajectories'];groups={}
 for row in rows:groups.setdefault(str(row['qid']),[]).append(row['trajectory_id'])
 assert len(rows)==len(set(row['trajectory_id'] for row in rows))==120 and len(groups)==24
 assert all(len(tids)==5 and len(set(tids))==5 for tids in groups.values())
 replay=v['feature_manifests']['26'];assert fp(replay['path'])==replay
 return fp(p),v
'''

CHECK = r'''
f,v=checked_qualification();print(json.dumps(dict(status='ACTUAL_Q2_COMPLETE_120_TRAJECTORIES_G5',qualification=f,trajectories=v['trajectories'],accepted_for_training=False,submitted=False)))
'''

INIT = r'''
b.mkdir(parents=True,exist_ok=True);guard=lock();unsubmitted()
for name in ('logs','inputs'):(b/name).mkdir(exist_ok=True)
expected={k:inputs[k] for k in ('commit','bundle_sha256','bundle_bytes','chunks','source_manifest_sha256','qualification','e1_receipt','e3_receipt','job_name','renderer_sha256')}
p=b/'upload-controller.json'
if p.exists():assert read(p)==expected,'Conflicting deployment/version; use a new isolated root'
else:atomic(p,expected)
uploads.mkdir(exist_ok=True)
print(json.dumps(dict(status='COMPARISON_CODE_UPLOAD_BOUND',base=str(b),commit=commit,chunks=inputs['chunks'])))
'''

CHUNK = r'''
guard=lock();unsubmitted();controller=read(b/'upload-controller.json');assert controller['commit']==commit and controller['bundle_sha256']==inputs['bundle_sha256']
i=inputs['chunk_index'];assert 0<=i<inputs['chunks'];raw=base64.b64decode(inputs['chunk_data']);assert len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==inputs['chunk_sha256']
p=uploads/('part-%05d'%i)
if p.exists():assert p.read_bytes()==raw,'Conflicting partial code upload'
else:
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='CODE_CHUNK_SHA_VERIFIED',chunk=i,bytes=len(raw))))
'''

EXTRACT = r'''
guard=lock();unsubmitted();parts=sorted(uploads.glob('part-*'));assert len(parts)==inputs['chunks'];raw=b''.join(p.read_bytes() for p in parts)
assert len(raw)==inputs['bundle_bytes'] and hashlib.sha256(raw).hexdigest()==inputs['bundle_sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 members=archive.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
 manifest_raw=archive.extractfile('_replay_comparison_source_manifest.json').read();assert hashlib.sha256(manifest_raw).hexdigest()==inputs['source_manifest_sha256']
 manifest=json.loads(manifest_raw);assert manifest['commit']==commit
 expected=set(manifest['files'])|{'_replay_comparison_source_manifest.json'};assert set(names)==expected
 target=code if code.exists() else code.with_name(code.name+'.staging');target.mkdir(exist_ok=True)
 for member in members:
  rel=pathlib.Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  data=archive.extractfile(member).read()
  if member.name!='_replay_comparison_source_manifest.json':assert len(data)==manifest['files'][member.name]['bytes'] and hashlib.sha256(data).hexdigest()==manifest['files'][member.name]['sha256']
  dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==data,'Conflicting isolated code'
  else:
   with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}==expected
 if target!=code:target.rename(code)
publication=dict(commit=commit,remote_sha=commit,code_root=str(code),bundle_sha256=inputs['bundle_sha256'],source_manifest_sha256=inputs['source_manifest_sha256'],files=len(manifest['files']),active_source_unchanged=True)
p=b/'source-publication.json'
if p.exists():assert read(p)==publication
else:atomic(p,publication)
print(json.dumps(dict(status='ISOLATED_COMPARISON_IMPORT_CLOSURE_SHA_READY',**publication)))
'''

PREPARE = r'''
guard=lock();unsubmitted();verify_code();qualification,q=checked_qualification();descriptors=[];receipts={}
for name,epoch,key in [('E1',1,'e1_receipt'),('E3',3,'e3_receipt')]:
 p=pathlib.Path(inputs[key]);assert str(p).startswith(str(parent)+'/') and p.resolve()==p
 receipt=read(p)
 assert receipt['epoch']==epoch and receipt['cursor']==0 and receipt['completed_epoch'] is True
 assert receipt['actor_updates']==0 and receipt['independent_cpu_reload'] is True
 assert isinstance(receipt['steps'],int) and receipt['steps']>0
 assert len(receipt['sha256'])==64 and all(c in '0123456789abcdef' for c in receipt['sha256'])
 checkpoint=pathlib.Path(receipt['path']);assert str(checkpoint).startswith(str(parent)+'/') and checkpoint.resolve()==checkpoint
 assert checkpoint.is_file() and checkpoint.stat().st_size==receipt['bytes']>0
 descriptors.append(dict(name=name,path=str(checkpoint),bytes=receipt['bytes'],sha256=receipt['sha256'],epoch=epoch,steps=receipt['steps']))
 receipts[name]=fp(p)
 sealed=b/'inputs'/('source-'+name+'-sealed-receipt.json')
 if sealed.exists():assert sealed.read_bytes()==p.read_bytes(),'Sealed epoch receipt changed'
 else:
  with sealed.open('xb') as h:h.write(p.read_bytes());h.flush();os.fsync(h.fileno())
assert descriptors[1]['steps']>descriptors[0]['steps']
spec=dict(checkpoints=descriptors,source_receipts=receipts,fixed_epochs=[1,3],labels_read=False,checkpoint_selection_performed=False)
spec_path=b/'inputs/checkpoints.json'
if spec_path.exists():assert read(spec_path)==spec,'Immutable E1/E3 source binding changed'
else:atomic(spec_path,spec)
prepared=dict(status='FIXED_E1_E3_REPLAY_DOWNSTREAM_COMPARISON_PREPARED',commit=commit,code_root=str(code),qualification=qualification,checkpoints=fp(spec_path),source_receipts=receipts,trajectories=120,questions=24,labels_read=False,accepted_for_training=False,full_checkpoint_SHA_recheck_deferred_to_GPU=True,login_model_or_tensor_loads=False,job_name=inputs['job_name'],renderer_sha256=inputs['renderer_sha256'])
p=b/'preparation.json'
if p.exists():assert read(p)==prepared
else:atomic(p,prepared)
print(json.dumps(prepared))
'''

SUBMIT = r'''
guard=lock();unsubmitted();verify_code();prepared=read(b/'preparation.json')
assert prepared['status']=='FIXED_E1_E3_REPLAY_DOWNSTREAM_COMPARISON_PREPARED' and prepared['commit']==commit
assert prepared['job_name']==inputs['job_name'] and prepared['renderer_sha256']==inputs['renderer_sha256'],'Prepared job name/renderer changed'
assert prepared['trajectories']==120 and prepared['questions']==24
assert fp(prepared['qualification']['path'])==prepared['qualification'] and fp(prepared['checkpoints']['path'])==prepared['checkpoints']
assert not command(['squeue','-h','-u','mou176','-n',inputs['job_name'],'-o','%i']),'Same-name queued/running job exists; do not duplicate'
assert not (b/'comparison-output').exists(),'Prior GPU output exists; preserve and inspect'
with (b/'comparison-run.lock').open('a') as run_lock:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
script=render_sbatch(str(b),str(code),str(pathlib.Path(prepared['qualification']['path']).parent),prepared['qualification']['sha256'],prepared['checkpoints']['path'],prepared['checkpoints']['sha256'],inputs['job_name'])
p=b/'comparison.sbatch'
if p.exists():assert p.read_text()==script
else:p.write_text(script)
command(['bash','-n',str(p)])
intent=dict(status='REPLAY_COMPARISON_SUBMIT_INTENT',commit=commit,script=fp(p),job_name=inputs['job_name'],requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='00:30:00'),dependency=None,epoch_time=time.time())
atomic(b/'submission-intent.json',intent)
try:
 result=subprocess.run(['sbatch','--hold','--parsable',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
except subprocess.TimeoutExpired:
 intent.update(status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT');atomic(b/'submission-intent.json',intent);raise
if result.returncode:
 intent.update(status='SBATCH_REJECTED',stderr=result.stderr[-2000:]);atomic(b/'submission-intent.json',intent);raise RuntimeError(result.stderr[-2000:])
job=result.stdout.strip().split(';')[0]
if not job.isdigit():
 intent.update(status='SUBMIT_RESPONSE_UNKNOWN_DO_NOT_REPEAT',response=result.stdout[-1000:]);atomic(b/'submission-intent.json',intent);raise RuntimeError('Unknown sbatch response; inspect same-name job')
with (b/'job.id').open('x') as h:h.write(job+'\n');h.flush();os.fsync(h.fileno())
submission=dict(intent,status='REPLAY_COMPARISON_JOB_HELD',job_id=job,code_root=str(code),qualification=prepared['qualification'],checkpoints=prepared['checkpoints'])
atomic(b/'submission.json',submission)
state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
command(['scontrol','release',job]);submission.update(status='REPLAY_COMPARISON_JOB_RELEASED',queue_observation=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R']),epoch_time=time.time());atomic(b/'submission.json',submission)
print(json.dumps(submission))
'''

STATUS = r'''
result=dict(base=str(b),observed_epoch=time.time(),exists=b.exists())
for name in ('source-publication.json','preparation.json','submission-intent.json','submission.json'):
 p=b/name
 if p.exists():
  item=read(p);result[name]={k:item[k] for k in ('status','commit','job_id','code_root','trajectories','questions','requested_resources') if k in item}
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();assert job.isdigit();result['job_id']=job
 result['squeue']=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode,Elapsed'])
 result['start_estimate']=command(['squeue','--start','-h','-j',job,'-o','%i|%S|%R'])
for name in ('intent.json','progress.json','complete.json'):
 p=b/'comparison-output'/name
 if p.exists():result[name]=read(p)
print(json.dumps(result))
'''


if __name__ == '__main__':
    main()
