"""Stage a SHA-bound2-question/G5 replay fixture; explicit guarded submit only.

No action runs by importing this module.  ``prepare`` uploads only published
code to an isolated root and performs small CPU metadata/prefix checks.
``submit --submit-authorized`` is a separate action and requests one normal
acd_u GPU (8CPU/64GB/2h), without dependency on any previous experiment.

The actual migrated model directory has a verified migration complete.json but
does not necessarily contain official-inventory.json.  Supply an authentic
saved pinned inventory with --inventory-source when needed.  This helper never
fabricates an official API inventory from migration receipt fields.  Its model
view consists of remote hardlinks plus that small inventory copy; original
assets and active source/configs are unchanged.  Full model shard SHA checks
remain inside the GPU qualification job, not on the login node.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


BRANCH = 'refs/heads/codex/qwen7b-probe-warmup-20261019'
REMOTE_PARENT = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
DEFAULT_BASE = REMOTE_PARENT+'/replay-layer-qualification-20261021'
JOB_NAME = 'q7b-replay-qual'
CODE_FILES = (
    'reproduction/operations/qwen7b_warmup/replay_probe_layers.py',
    'reproduction/operations/qwen7b_warmup/test_replay_probe_layers.py',
    'reproduction/operations/qwen7b_warmup/packed_features.py',
    'reproduction/operations/qwen7b_warmup/native_capture.py',
    'reproduction/operations/qwen7b_warmup/download_model.py',
    'reproduction/operations/qwen7b_warmup/runtime.sh',
    'reproduction/operations/qwen7b_warmup/deploy_replay_qualification.py',
)


def render_sbatch(base, code_root, selection, selection_sha256, model_view,
                  job_name=JOB_NAME):
    """Render unique full placeholders; never replace a generic ROOT token."""
    paths = [str(base), str(code_root), str(selection), str(model_view)]
    if any(not Path(path).is_absolute() or '\n' in path or '\r' in path for path in paths):
        raise ValueError('Absolute newline-free artifact paths are required')
    if any(char.isspace() for char in str(base)):
        raise ValueError('The Slurm output directory must not contain whitespace')
    if not job_name or any(not (char.isalnum() or char in '-_') for char in job_name):
        raise ValueError('Safe isolated Slurm job name required')
    if len(selection_sha256) != 64 or any(char not in '0123456789abcdef' for char in selection_sha256):
        raise ValueError('Real selection SHA256 required')
    template = '''#!/usr/bin/env bash
#SBATCH --job-name=@@REPLAY_JOB_NAME@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=@@REPLAY_BASE_RAW@@/logs/replay-%j.log
set -euo pipefail
REPLAY_BASE=@@REPLAY_BASE_QUOTED@@
REPLAY_CODE_ROOT=@@REPLAY_CODE_QUOTED@@
REPLAY_SELECTION=@@REPLAY_SELECTION_QUOTED@@
REPLAY_MODEL_VIEW=@@REPLAY_MODEL_QUOTED@@
exec 9>"$REPLAY_BASE/replay-run.lock"
flock -n 9
export QWEN7B_CODE_ROOT="$REPLAY_CODE_ROOT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.replay_probe_layers --selection "$REPLAY_SELECTION" --selection-sha256 @@REPLAY_SELECTION_SHA256@@ --model "$REPLAY_MODEL_VIEW" --output "$REPLAY_BASE/qualification-output" --max-trajectories 10 --device cuda:0
'''
    replacements = {
        '@@REPLAY_JOB_NAME@@':job_name, '@@REPLAY_BASE_RAW@@':str(base),
        '@@REPLAY_BASE_QUOTED@@':shlex.quote(str(base)),
        '@@REPLAY_CODE_QUOTED@@':shlex.quote(str(code_root)),
        '@@REPLAY_SELECTION_QUOTED@@':shlex.quote(str(selection)),
        '@@REPLAY_MODEL_QUOTED@@':shlex.quote(str(model_view)),
        '@@REPLAY_SELECTION_SHA256@@':selection_sha256,
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    if '@@REPLAY_' in template:
        raise ValueError('Unrendered Slurm template token')
    return template


def self_test():
    """Local syntax/transport/export smoke only; no SSH, sbatch or GPU calls."""
    base = '/tmp/replay-qualification-selftest'
    code = base+'/code-0123456789ab'
    script = render_sbatch(base, code, base+'/inputs/fixture.json', 'a'*64,
                           base+'/official-snapshot-view')
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)
    exports = '\n'.join(line for line in script.splitlines()
                        if line.startswith(('REPLAY_', 'export ')))
    check = exports+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(code)+'\n'
    subprocess.run(['bash', '-euc', check], check=True)
    values = dict(base=DEFAULT_BASE, parent=REMOTE_PARENT, commit='a'*40,
                  bundle_sha256='b'*64, bundle_bytes=32000, chunks=8,
                  files={name:dict(bytes=10000, sha256='c'*64) for name in CODE_FILES},
                  inventory_source=None, job_name=JOB_NAME)
    sizes = {}
    for name in ('INIT', 'CHUNK', 'EXTRACT', 'PREPARE', 'SUBMIT', 'STATUS'):
        own = dict(values)
        if name == 'CHUNK':
            own.update(chunk_index=0, chunk_data=base64.b64encode(b'x'*4096).decode(), chunk_sha256='d'*64)
        source = 'inputs='+repr(own)+'\n'+COMMON+'\n'+globals()[name]
        compile(source, 'remote-'+name, 'exec')
        if len(source.encode()) >= 16000:
            raise AssertionError('Transport bound exceeded: '+name)
        sizes[name] = len(source.encode())
    result = dict(status='LOCAL_REPLAY_DEPLOYMENT_SMOKE_PASSED', bash_syntax=True,
                  rendered_QWEN7B_CODE_ROOT_export_executed_and_verified=True,
                  generic_ROOT_replacement_used=False, remote_blocks_compiled=6,
                  remote_command_bytes=sizes, remote_actions_executed=False)
    print(json.dumps(result), flush=True)
    return result


def published_bundle(repo, selected_commit=None):
    """Reproducible code-only tar from an actually published commit."""
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', BRANCH], cwd=repo, text=True).split()[0]
    if head != remote:
        raise RuntimeError('Current source is not the published branch head')
    helper_name=str(Path(__file__).relative_to(repo))
    if Path(__file__).read_bytes()!=subprocess.check_output(['git','show',head+':'+helper_name],cwd=repo):
        raise RuntimeError('Commit and push deployment helper before execution')
    commit = selected_commit or head
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, head], cwd=repo, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', commit], cwd=repo, text=True).strip()
    files, stream = {}, io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for name in CODE_FILES:
                data = subprocess.check_output(['git', 'show', commit+':'+name], cwd=repo)
                files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                item = tarfile.TarInfo(name)
                item.size, item.mode, item.mtime = len(data), 0o644, 0
                archive.addfile(item, io.BytesIO(data))
    return commit, stream.getvalue(), files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--action', choices=('prepare', 'submit', 'status'), default='prepare')
    parser.add_argument('--remote-base', default=DEFAULT_BASE)
    parser.add_argument('--inventory-source', help='Remote authentic saved official-inventory.json, if absent in model directory')
    parser.add_argument('--prepared-commit', help='Previously published prepared commit; defaults to current published HEAD')
    parser.add_argument('--submit-authorized', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.control_root is None:
        parser.error('--control-root is required for remote actions')
    if args.action == 'submit' and not args.submit_authorized:
        raise SystemExit('Submission is separate: explicit --submit-authorized is required')
    if (Path(args.remote_base).parent != Path(REMOTE_PARENT) or
            any(not (char.isalnum() or char in '-_.') for char in Path(args.remote_base).name)):
        raise SystemExit('Use a new isolated child directory of the authorized experiment')
    repo = Path(__file__).resolve().parents[3]
    if args.action == 'status':
        commit = subprocess.check_output(['git', 'rev-parse', args.prepared_commit or 'HEAD'], cwd=repo, text=True).strip()
        bundle, files = b'', {}
    else:
        commit, bundle, files = published_bundle(repo, args.prepared_commit)
    digest = hashlib.sha256(bundle).hexdigest()
    chunks = [bundle[start:start+4096] for start in range(0, len(bundle), 4096)]
    inputs = dict(base=args.remote_base, parent=REMOTE_PARENT, commit=commit,
                  bundle_sha256=digest, bundle_bytes=len(bundle), chunks=len(chunks),
                  files=files, inventory_source=args.inventory_source, job_name=JOB_NAME)
    spec = importlib.util.spec_from_file_location('replay_control', args.control_root/
        'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    address = subprocess.check_output(['/usr/sbin/ipconfig', 'getifaddr', 'en0'], text=True).strip()
    def call(code, extra=None, timeout=45):
        values = dict(inputs)
        values.update(extra or {})
        source = 'inputs='+repr(values)+'\n'+COMMON+'\n'+code
        if len(source.encode()) >= 16000:
            raise RuntimeError('Remote command exceeds the deliberate16KB transport bound')
        result = control.run('python3 -c '+shlex.quote(source), account='mou176',
                             source_address=address, timeout=timeout)
        records = []
        for line in result.stdout.decode(errors='replace').splitlines():
            if line.strip().startswith('{'):
                records.append(json.loads(line))
        for row in records:
            print(json.dumps(row), flush=True)
        if result.returncode:
            print(result.stderr.decode(errors='replace'), flush=True)
            raise SystemExit(result.returncode)
        if not records:
            raise RuntimeError('No structured fresh deployment result; do not reuse an old result')
        return records[-1]
    if args.action == 'status':
        call(STATUS)
        return
    if args.action == 'submit':
        call(SUBMIT, timeout=55)
        return
    call(INIT)
    for index, chunk in enumerate(chunks):
        call(CHUNK, dict(chunk_index=index, chunk_data=base64.b64encode(chunk).decode(),
                         chunk_sha256=hashlib.sha256(chunk).hexdigest()))
    call(EXTRACT)
    call(PREPARE, timeout=55)


COMMON = r'''
import pathlib,json,hashlib,base64,os,fcntl,time,subprocess,tarfile,io,shutil,gzip,sys
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);parent=pathlib.Path(inputs['parent']);commit=inputs['commit']
assert b.is_absolute() and b.resolve()==b and b.parent==parent
code=b/('code-'+commit[:12]);uploads=b/('upload-'+inputs['bundle_sha256'][:16])
def read(p):return json.loads(pathlib.Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def atomic(p,v):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('wb') as h:h.write(canonical(v)+b'\n');h.flush();os.fsync(h.fileno())
 tmp.replace(p)
def sha(p):
 d=hashlib.sha256()
 with pathlib.Path(p).open('rb') as h:
  for raw in iter(lambda:h.read(8*1024**2),b''):d.update(raw)
 return d.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def command(args,timeout=25):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 assert r.returncode==0,(args,r.stderr[-2000:]);return r.stdout.strip()
def lock():
 h=(b/'controller.lock').open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def unsubmitted():
 assert not (b/'job.id').exists(),'Existing job identity: inspect it, do not duplicate'
 assert not (b/'submission-intent.json').exists(),'Prior submit outcome must be reconciled before another submission'
def verify_code():
 p=read(b/'source-publication.json');assert p['commit']==commit and p['bundle_sha256']==inputs['bundle_sha256'] and p['code_root']==str(code)
 for name,item in inputs['files'].items():assert sha(code/name)==item['sha256'] and (code/name).stat().st_size==item['bytes'],name
 return p
'''


INIT = r'''
b.mkdir(parents=True,exist_ok=True)
guard=lock();unsubmitted()
for name in ('logs','inputs'): (b/name).mkdir(exist_ok=True)
controller=dict(status='PUBLISHED_CODE_UPLOAD_INTENT',commit=commit,bundle_sha256=inputs['bundle_sha256'],bundle_bytes=inputs['bundle_bytes'],chunks=inputs['chunks'],files=inputs['files'])
path=b/'upload-controller.json'
if path.exists():assert read(path)==controller,'Another immutable upload/version exists; use an isolated new base'
else:atomic(path,controller)
uploads.mkdir(exist_ok=True)
print(json.dumps(dict(status='REPLAY_QUALIFICATION_UPLOAD_BOUND',base=str(b),commit=commit,chunks=inputs['chunks'])))
'''


CHUNK = r'''
guard=lock();unsubmitted();controller=read(b/'upload-controller.json')
assert controller['commit']==commit and controller['bundle_sha256']==inputs['bundle_sha256']
index=inputs['chunk_index'];assert 0<=index<inputs['chunks']
raw=base64.b64decode(inputs['chunk_data']);assert len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==inputs['chunk_sha256']
part=uploads/('part-%05d'%index)
if part.exists():assert part.read_bytes()==raw,'Partial upload changed'
else:
 with part.open('xb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
print(json.dumps(dict(status='CODE_CHUNK_SHA_VERIFIED',chunk=index,bytes=len(raw))))
'''


EXTRACT = r'''
guard=lock();unsubmitted()
parts=sorted(uploads.glob('part-*'));assert len(parts)==inputs['chunks']
raw=b''.join(p.read_bytes() for p in parts);assert len(raw)==inputs['bundle_bytes'] and hashlib.sha256(raw).hexdigest()==inputs['bundle_sha256']
if not code.exists():
 temporary=code.with_name(code.name+'.staging');temporary.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  members=archive.getmembers();assert {m.name for m in members}==set(inputs['files'])
  for member in members:
   rel=pathlib.Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
   payload=archive.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==inputs['files'][member.name]['sha256']
   target=temporary/rel;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():assert sha(target)==inputs['files'][member.name]['sha256']
   else:
    with target.open('xb') as h:h.write(payload);h.flush();os.fsync(h.fileno())
 assert {str(p.relative_to(temporary)) for p in temporary.rglob('*') if p.is_file()}==set(inputs['files']),'Unpublished file in interrupted code staging root'
 temporary.rename(code)
for name,item in inputs['files'].items():assert sha(code/name)==item['sha256']
publication=dict(commit=commit,remote_sha=commit,code_root=str(code),bundle_sha256=inputs['bundle_sha256'],files=inputs['files'],active_code_unchanged=True)
path=b/'source-publication.json'
if path.exists():assert read(path)==publication
else:atomic(path,publication)
print(json.dumps(dict(status='ISOLATED_PUBLISHED_CODE_READY',commit=commit,code_root=str(code))))
'''


PREPARE = r'''
guard=lock();unsubmitted();publication=verify_code()
sys.path.insert(0,str(code))
from reproduction.operations.qwen7b_warmup.replay_probe_layers import normalize_prefix,REVISION,MODEL
c=read(parent/'config.json');model=pathlib.Path(c['model_dir']);done=read(model/'complete.json')
assert c['model_revision']==done['revision']==REVISION and done['model']==MODEL
assert done['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' and done['fresh_all_files_SHA256'] is True
geometry=read(model/'config.json');assert (geometry['num_hidden_layers'],geometry['hidden_size'],geometry['num_attention_heads'],geometry['num_key_value_heads'],geometry['intermediate_size'])==(28,3584,28,4,18944)
index=read(model/'model.safetensors.index.json');needed={'config.json','model.safetensors.index.json'}|set(index['weight_map'].values())
assert all(pathlib.Path(name).name==name for name in needed) and needed.issubset(done['files'])
for name in needed:
 item=done['files'][name];assert (model/name).stat().st_size==item['bytes'] and len(item['sha256'])==64
 if not name.endswith('.safetensors'):assert sha(model/name)==item['sha256']
inventory_candidates=[pathlib.Path(inputs['inventory_source'])] if inputs.get('inventory_source') else [model/'official-inventory.json',b/'inputs/official-inventory.json',parent/'assets/official-inventory.json',parent/'preparation/official-model-inventory.json',parent/'preparation/official-inventory.json']
found=[p for p in inventory_candidates if p.is_file()]
assert found,'Authentic pinned official-inventory.json is absent: stage that small saved metadata remotely and pass --inventory-source; no fabricated API inventory is permitted'
inventory_path=found[0];assert inventory_path.is_absolute() and inventory_path.resolve()==inventory_path
inventory=read(inventory_path);assert inventory['sha']==REVISION and inventory.get('id',MODEL)==MODEL
official={item['rfilename']:item for item in inventory['siblings']}
for name in needed:
 item=official[name];source=done['files'][name]
 if item.get('lfs'):assert item['lfs']['sha256']==source['sha256'] and item['size']==source['bytes']
 else:
  payload=(model/name).read_bytes();assert hashlib.sha1(('blob %d\0'%len(payload)).encode()+payload).hexdigest()==item['blobId']==source['git_blob_sha1']
view=b/'official-snapshot-view';view.mkdir(exist_ok=True)
for name in needed:
 destination=view/name
 if destination.exists():assert os.path.samefile(destination,model/name)
 else:os.link(str(model/name),str(destination))
inventory_bytes=inventory_path.read_bytes();destination=view/'official-inventory.json'
if destination.exists():assert destination.read_bytes()==inventory_bytes
else:
 with destination.open('xb') as h:h.write(inventory_bytes);h.flush();os.fsync(h.fileno())
model_files={name:dict(bytes=done['files'][name]['bytes'],sha256=done['files'][name]['sha256']) for name in needed}
model_files['official-inventory.json']=dict(bytes=len(inventory_bytes),sha256=hashlib.sha256(inventory_bytes).hexdigest())
shard=read(parent/'sampling/shard-00/complete.json');assert shard['model_revision']==REVISION and shard['run_id']==c['run_id']
batch_path=pathlib.Path(shard['batches'][0]);batch=read(batch_path)
assert batch['model_revision']==REVISION and batch['run_id']==c['run_id'] and batch['actor_optimizer_updates']==0
input_path=parent/'inputs/shard-00'/(batch['batch']+'.json');assert sha(input_path)==batch['input_sha256']
original={str(row['qid']):row for row in read(input_path)['rows']}
prefix=batch['prefix']['prefix'];native=batch['prefix']['actor_hidden']
for item in (prefix,native):assert fp(item['path'])['sha256']==item['sha256'] and pathlib.Path(item['path']).stat().st_size==item['bytes']
chosen=[];groups={};seen=set()
with gzip.open(prefix['path'],'rt') as h:
 for line in h:
  row=json.loads(line);qid=str(row['qid']);tid=row['trajectory_id']
  assert tid not in seen;seen.add(tid)
  if qid not in groups and len(chosen)<2:chosen.append(qid);groups[qid]=[]
  if qid not in groups:continue
  normalized=normalize_prefix(row,tid,qid,True)
  assert qid in original and row['initial_prompt_token_ids']==original[qid]['initial_prompt_token_ids'] and row['question_hash']==original[qid]['question_hash']
  assert normalized['source_native_manifest']==native['path'] and row['feature_layer']==26
  groups[qid].append(dict(trajectory_id=tid,qid=qid,response_index=row['response_index']))
assert len(groups)==2
selected=[]
for qid in chosen:
 own=sorted(groups[qid],key=lambda row:row['response_index']);assert [row['response_index'] for row in own]==list(range(5));selected.extend(own)
assert len(selected)==10 and len({row['trajectory_id'] for row in selected})==10
native_meta=read(native['path']);assert native_meta['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE' and native_meta['feature_layer']==26 and native_meta['dimension']==3584 and native_meta['question_max_tokens']==273 and native_meta['prefix_tokens']==20
selection=dict(schema_version=1,model_revision=REVISION,prefix_files=[{k:prefix[k] for k in ('path','bytes','sha256')}],trajectories=[{k:row[k] for k in ('trajectory_id','qid')} for row in selected],native_layer26_manifests=[{k:native[k] for k in ('path','bytes','sha256')}],model_files=model_files,label_blind=True,selection_rule='first2encountered_original_question_ids_each_completeG5_noEMlabels_read',model_source_receipt=fp(model/'complete.json'))
selection_path=b/'inputs/fixture-selection.json'
if selection_path.exists():assert read(selection_path)==selection,'Existing immutable fixture changed'
else:atomic(selection_path,selection)
acceptance=dict(status='BOUNDED_REPLAY_FIXTURE_PREPARED',commit=commit,code_root=str(code),model_view=str(view),selection=fp(selection_path),trajectories=10,questions=2,label_blind=True,model_source_receipt=fp(model/'complete.json'),model_source_config=fp(parent/'config.json'),inventory_source=fp(inventory_path),source_batch=fp(batch_path),source_input=fp(input_path),native_manifest=fp(native['path']),full_model_shards_freshly_hashed_on_login=False,gpu_capture_or_qualification_claimed=False)
acceptance_path=b/'preparation.json'
if acceptance_path.exists():assert read(acceptance_path)==acceptance
else:atomic(acceptance_path,acceptance)
print(json.dumps(dict(status=acceptance['status'],base=str(b),commit=commit,selection=acceptance['selection'],model_view=str(view),trajectories=10,questions=2,submitted=False)))
'''


SUBMIT = r'''
guard=lock();unsubmitted();verify_code();prepared=read(b/'preparation.json')
assert prepared['status']=='BOUNDED_REPLAY_FIXTURE_PREPARED' and prepared['commit']==commit and prepared['trajectories']==10 and prepared['questions']==2
assert fp(prepared['selection']['path'])==prepared['selection']
assert not command(['squeue','-h','-u','mou176','-n',inputs['job_name'],'-o','%i']),'A same-name job is already queued/running; inspect instead of duplicating'
assert not (b/'qualification-output').exists(),'A prior GPU attempt/output exists; preserve and inspect it'
sys.path.insert(0,str(code))
from reproduction.operations.qwen7b_warmup.deploy_replay_qualification import render_sbatch
script=render_sbatch(str(b),str(code),prepared['selection']['path'],prepared['selection']['sha256'],prepared['model_view'],inputs['job_name'])
script_path=b/'replay-qualification.sbatch'
if script_path.exists():assert script_path.read_text()==script
else:script_path.write_text(script)
intent=dict(status='REPLAY_QUALIFICATION_SUBMIT_INTENT',commit=commit,script=fp(script_path),job_name=inputs['job_name'],requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='02:00:00'),dependency=None,epoch_time=time.time())
atomic(b/'submission-intent.json',intent)
try:
 result=subprocess.run(['sbatch','--hold','--parsable',str(script_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
except subprocess.TimeoutExpired:
 intent.update(status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT');atomic(b/'submission-intent.json',intent);raise
if result.returncode:
 intent.update(status='SBATCH_REJECTED',stderr=result.stderr[-2000:]);atomic(b/'submission-intent.json',intent);raise RuntimeError(result.stderr[-2000:])
job=result.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch response; inspect same-name job before retry'
with (b/'job.id').open('x') as h:h.write(job+'\n');h.flush();os.fsync(h.fileno())
submission=dict(intent,job_id=job,status='REPLAY_QUALIFICATION_JOB_HELD',code_root=str(code),selection=prepared['selection'],model_view=prepared['model_view'],active_other_tasks_unchanged=True)
atomic(b/'submission.json',submission)
state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
command(['scontrol','release',job]);submission.update(status='REPLAY_QUALIFICATION_JOB_RELEASED',queue_observation=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R']),epoch_time=time.time());atomic(b/'submission.json',submission)
print(json.dumps(submission))
'''


STATUS = r'''
result=dict(base=str(b),observed_epoch=time.time(),exists=b.exists())
for name in ('source-publication.json','preparation.json','submission-intent.json','submission.json'):
 p=b/name
 if p.exists():
  item=read(p);result[name]={k:item[k] for k in ('status','commit','job_id','code_root','trajectories','questions') if k in item}
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();assert job.isdigit();result['job_id']=job
 result['squeue']=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode,Elapsed'])
for name in ('progress.json','complete.json','native-layer26-comparison.json'):
 p=b/'qualification-output'/name
 if p.exists():result[name]=read(p)
print(json.dumps(result))
'''


if __name__ == '__main__':
    main()
