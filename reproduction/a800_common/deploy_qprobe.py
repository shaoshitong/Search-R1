"""Prospective A800 qprobe preparation, with an explicit final launch action.

Only published Git objects cross the controller. Canonical data, model metadata
and the unchanged formal recipe are copied and verified on A800 itself. Stage,
prepare and bind never launch the pipeline; failures retain their claims.
"""
import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

from reproduction.a800_common import deploy as original_deploy
from reproduction.a800_common.deploy import (
    PLACEMENT_POLICY, REMOTE_LAUNCH_PREPARED, REMOTE_OBSERVE, digest, published_source, ssh_command, write_once,
)

ROLE = 'a800_qprobe_prospective_intent'
BASELINE = '/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab'
BASELINE_COMMIT = '34e63b7b12f4056503803905c1b3dd1be95f0adc'
BASELINE_RUN_ID = '1db662001a0f4d998928bcf6f5f33c9d'
INITIAL_PROBES = '/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef'
TEACHER_CACHE = '/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v3-async-top512'
FEATURE_SOURCE_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-27bab88929e048b4869f3889b99d8e18'
FEATURE_SOURCE_COMMIT = 'bf4b88d04c166951691029b18d05d91ece7c7054'
FEATURE_SOURCE_RUN = '0f49c2a664b448f59b5f21dd49f9b10b'
FEATURE_SOURCE_EXECUTION = '7d85bd0daa93462981aefc2a9d87fdb7'
FEATURE_MANIFEST_SHA = '7228f5005cf8ac9f1f67f48a4529a24577dc8903f89c2bfa5e49ee0faf74a1c9'
FEATURE_FILE_SHA = 'e2186ca7ea6735470ad5cd1302b03054529c675ba45b9f3297c1b07bab7e97c3'
FEATURE_SPEC_SHA = 'aaea09aac50ebe4cc3c424af0c2273cc9e0e13a8ab9049acd8f1804d3394a9d4'
FEATURE_UUID = 'GPU-92249a14-a344-f77f-e18c-d1cea5c638a7'
FEATURE_PRODUCERS = ('search_g1/qprobe_features.py','search_g1/qprobe_prepare.py',
    'reproduction/a800_common/precompute_qprobe.py','reproduction/common8/prepare_run.py',
    'reproduction/common8/verify_data.py')
MAX_BUNDLE = 32 * 1024**2
REVIEW_CHECKS = (
    'official_model_and_canonical_data', 'frozen_teacher_5120_x_5',
    'bootstrap_all_5120_groups', 'exploration_0_top512', 'formal_200_not_201',
    'independent_trajectory_phase_v1',
    'four_rank_bf16_local4_global16_accum8_ref8', 'recoverable_guards_lr_warmup57',
)


def validate_intent(intent):
    assert intent['role'] == ROLE and intent['host'] == 'baai-a800'
    assert re.fullmatch('[0-9a-f]{32}', intent['deployment_id'])
    assert re.fullmatch('[0-9a-f]{40}', intent['source_commit'])
    assert re.fullmatch('[0-9a-f]{64}', intent['baseline_intent_sha256'])
    assert re.fullmatch(r'refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*', intent['published_ref'])
    assert intent['base'] == '/data/shaoshitong/Search-G1-a800-fresh200-' + intent['deployment_id']
    assert intent['source_base'] == '/data/shaoshitong/Search-G1-qprobe-source-' + intent['deployment_id']
    assert intent['baseline_base'] == BASELINE and intent['baseline_commit'] == BASELINE_COMMIT
    assert re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}', intent['initial_probes_source_base'])
    assert intent['initial_probes_source_base'] != intent['base']
    validate_teacher_cache(intent['teacher_warmup_cache'])
    assert intent['launch_services'] is False and intent['target_updates'] == 200
    assert intent['gpu_placement_policy']==PLACEMENT_POLICY and intent['gpu_placement_bound'] is False
    return intent


def validate_teacher_cache(value):
    assert isinstance(value, str) and re.fullmatch(r'/(data|mnt|backup)/[A-Za-z0-9._/-]+', value)
    path = Path(value)
    assert path.is_absolute() and str(path) == value and '..' not in path.parts
    return path


def verified_launcher(repo, intent):
    """Pin current disk bytes, committed blobs and already imported launcher text."""
    repo = Path(repo).resolve()
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert commit == intent['source_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
    assert published_source(repo, commit) == intent['source_sha256']
    from reproduction.common8.source_files import runtime_files
    assert {str(path.relative_to(repo)):digest(path) for path in runtime_files(repo)} == intent['source_sha256']
    blobs = {}
    for relative in ('reproduction/a800_common/deploy_qprobe.py','reproduction/a800_common/deploy.py'):
        raw = subprocess.check_output(['git','-C',str(repo),'show',commit+':'+relative])
        assert hashlib.sha256(raw).hexdigest() == digest(repo/relative) == intent['source_sha256'][relative]
        blobs[relative] = raw
    assert Path(original_deploy.__file__).resolve() == repo/'reproduction/a800_common/deploy.py'
    snippets = [ast.literal_eval(node.value) for node in ast.parse(blobs['reproduction/a800_common/deploy.py']).body
                if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='REMOTE_LAUNCH_PREPARED'
                                                      for target in node.targets)]
    assert len(snippets) == 1 and snippets[0] == original_deploy.REMOTE_LAUNCH_PREPARED == REMOTE_LAUNCH_PREPARED
    return commit, blobs['reproduction/a800_common/deploy_qprobe.py'], snippets[0]


def validate_review(review, intent):
    """A caller supplies an actual independent review; the adapter never creates one."""
    assert review['role'] == 'a800_qprobe_independent_code_review'
    assert review['approved'] is True and review['independent'] is True
    assert isinstance(review['reviewer'], str) and review['reviewer'].strip()
    assert isinstance(review['implementation_actor'], str) and review['implementation_actor'].strip()
    assert review['reviewer'] != review['implementation_actor']
    assert review['source_commit'] == intent['source_commit']
    assert review['source_sha256'] == intent['source_sha256']
    assert set(review['checks']) == set(REVIEW_CHECKS)
    assert all(review['checks'][key] is True for key in REVIEW_CHECKS)
    assert type(review['reviewed_unix']) in (int, float) and 0 < review['reviewed_unix'] <= time.time()
    return review


def validate_feature_origin(old, new, ready, exit_receipt, manifest):
    """Authenticate the completed encoding separately from a new run's identity."""
    assert (old['run_id'],old['execution_id'],old['source_commit']) == (FEATURE_SOURCE_RUN,FEATURE_SOURCE_EXECUTION,FEATURE_SOURCE_COMMIT)
    assert new['run_id'] != old['run_id'] and new['execution_id'] != old['execution_id'] and new['source_commit'] != old['source_commit']
    for candidate in (old,new):
        assert candidate['profile']=='common8' and candidate['execution_backend']=='ssh_local'
        assert candidate['training_world_size']==4 and candidate['completed_updates']==0 and not candidate.get('full_state_restore')
        assert candidate['initial_model']==candidate['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct'
        assert candidate['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert len(old['initial_model_sha256'])==9 and old['initial_model_sha256']==new['initial_model_sha256']
    assert old['input_sha256']==new['input_sha256'] and old['calibration_question_ids']==new['calibration_question_ids']
    for name in FEATURE_PRODUCERS:
        assert old['source_sha256'].get(name) is not None and old['source_sha256'][name]==new['source_sha256'].get(name)
    assert ready['run_id']==old['run_id'] and ready['source_commit']==old['source_commit']
    assert ready['cache_dir']==FEATURE_SOURCE_BASE+'/question-features'
    assert (ready['pid'],ready['start_ticks'],ready['sid'])==(2861772,604134066,2861771)
    assert ready['count']==169028 and ready['feature_dim']==2048 and ready['completed_training_outers']==0
    assert ready['stage']=='question_feature_cache_complete' and ready['target_training_outers']==200
    assert ready['exit_code']==0 and ready['actual_visible_cuda_count']==1 and ready['physical_index']==6
    assert ready['physical_gpu_uuid']==FEATURE_UUID and ready['actual_cuda_uuids']==[FEATURE_UUID]
    assert ready['manifest_sha256']==FEATURE_MANIFEST_SHA and ready['feature_file_sha256']==FEATURE_FILE_SHA and ready['spec_sha']==FEATURE_SPEC_SHA
    assert (exit_receipt['run_id'],exit_receipt['execution_id'],exit_receipt['source_commit'])==(old['run_id'],old['execution_id'],old['source_commit'])
    assert exit_receipt['exit_code']==0 and exit_receipt['status']=='FEATURE_CHILD_EXITED'
    child=exit_receipt['child'];assert (child['pid'],child['start_ticks'],child['sid'])==(2861772,604134066,2861771)
    assert manifest['count']==169028 and manifest['feature_dim']==2048 and manifest['spec_sha']==FEATURE_SPEC_SHA
    assert manifest['files']['features.npy']['sha256']==FEATURE_FILE_SHA
    assert manifest['spec']['train_parquet_sha256']==new['input_sha256']['train.parquet']
    assert manifest['spec']['encoder_compute_dtype']=='bfloat16' and manifest['spec']['feature_dtype']=='float16'
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode()
    model_sha=hashlib.sha256(canonical(new['initial_model_sha256'])).hexdigest()
    assert manifest['spec']['encoder_files_sha256']==model_sha
    assert manifest['spec']['encoder_revision']==new['model_revision']
    assert manifest['spec']['feature_layer']==27 and manifest['spec']['max_prompt_tokens']==4096
    assert set(manifest['files'])=={'index.jsonl','features.npy'}
    assert all(type(row['bytes']) is int and row['bytes']>0 and re.fullmatch('[0-9a-f]{64}',row['sha256']) for row in manifest['files'].values())
    return True


def validate_prepared_features(features, run):
    assert features['run_id']==run['run_id'] and features['source_commit']==run['source_commit']
    assert features['count']==169028 and features['feature_dim']==2048 and features['exit_code']==0
    assert features['completed_training_outers']==0 and features['target_training_outers']==200
    assert features['cache_dir'].startswith('/data/shaoshitong/Search-G1-a800-fresh200-') and features['cache_dir'].endswith('/question-features')
    if features.get('kind')=='copied_verified_existing_encoding':
        assert features['fresh_gpu_encoding'] is False and features['actual_visible_cuda_count'] is None
        assert features['origin']['base']==FEATURE_SOURCE_BASE and features['origin']['run_id']==FEATURE_SOURCE_RUN
        assert features['origin']['source_commit']==FEATURE_SOURCE_COMMIT and features['origin']['execution_id']==FEATURE_SOURCE_EXECUTION
        assert features['execution_id']==run['execution_id']
        process=features['copy_process']
        assert all(type(process[key]) is int and process[key]>0 for key in ('pid','start_ticks','sid'))
        assert type(process['uid']) is int and process['uid']>=0
        assert (process['pid'],process['start_ticks'],process['sid'])!=(2861772,604134066,2861771)
        assert (features['origin']['encoding_pid'],features['origin']['encoding_start_ticks'],features['origin']['encoding_exit_code'])==(2861772,604134066,0)
        assert features['origin']['encoding_cuda_uuid']==FEATURE_UUID and features['model_loaded'] is False
        assert features['origin']['immutable_encoder_producer_sha256']=={name:run['source_sha256'][name] for name in FEATURE_PRODUCERS}
        assert features['manifest_sha256']==FEATURE_MANIFEST_SHA and features['feature_file_sha256']==FEATURE_FILE_SHA and features['spec_sha']==FEATURE_SPEC_SHA
    else:
        assert features.get('kind') is None and features['actual_visible_cuda_count']==1
        assert features['stage']=='question_feature_cache_complete' and features['physical_index'] in (5,6)
        assert features['actual_cuda_uuids']==[features['physical_gpu_uuid']]
    return features


REMOTE_COMMON = r'''
import hashlib,io,json,os,re,shutil,socket,subprocess,sys,tarfile,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''  # CPU-only until the explicit launch gate.
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def owned(path):
 path=Path(path);assert path==path.resolve() and path.stat().st_uid==os.getuid()
 return path
def read(path):
 path=owned(path);assert path.is_file() and path.stat().st_size<=8*1024**2
 return json.loads(path.read_bytes())
def once(path,value):
 data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def git(*args):
 result=subprocess.run(['git',*args],capture_output=True,text=True,timeout=90,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
 assert result.returncode==0, 'Offline Git preparation failed; inspect this namespace'
 return result.stdout.strip()
def output(value):print('QPROBE_DEPLOYMENT_JSON:'+json.dumps(value),flush=True)
def check(intent):
 assert intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800'
 assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id']) and re.fullmatch('[0-9a-f]{40}',intent['source_commit'])
 assert re.fullmatch('[0-9a-f]{64}',intent['baseline_intent_sha256'])
 assert re.fullmatch(r'refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*',intent['published_ref'])
 assert intent['base']=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id']
 assert intent['source_base']=='/data/shaoshitong/Search-G1-qprobe-source-'+intent['deployment_id']
 assert intent['baseline_base']=='/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab'
 assert intent['baseline_commit']=='34e63b7b12f4056503803905c1b3dd1be95f0adc'
 assert re.fullmatch('/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',intent['initial_probes_source_base'])
 assert intent['initial_probes_source_base']!=intent['base']
 cache=Path(intent['teacher_warmup_cache'])
 assert re.fullmatch(r'/(data|mnt|backup)/[A-Za-z0-9._/-]+',str(cache)) and '..' not in cache.parts
 assert cache.is_absolute() and str(cache)==intent['teacher_warmup_cache']
 assert intent['launch_services'] is False and intent['target_updates']==200
 assert intent['gpu_placement_policy']=='launch_time_four_idle_excluding_auxiliary_v1' and intent['gpu_placement_bound'] is False
def teacher_storage(intent,create_parent=False):
 cache=Path(intent['teacher_warmup_cache']);assert cache.resolve()==cache and not cache.is_symlink()
 existing=cache if cache.exists() else cache.parent
 while not existing.exists():existing=existing.parent
 assert owned(existing).is_dir()
 if create_parent:cache.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 if cache.parent.exists():assert owned(cache.parent).is_dir()
 if cache.exists():assert owned(cache).is_dir()
 assert cache.resolve()==cache
 return cache
def source_map(repo):
 sys.path.insert(0,str(repo))
 from reproduction.common8.source_files import runtime_files
 return {str(p.relative_to(repo)):sha(owned(p)) for p in runtime_files(repo)}
def staged(intent):
 check(intent);source=owned(intent['source_base'])
 assert read(source/'preparation/qprobe-intent.json')==intent
 receipt=read(source/'logs/source-staging-complete.json')
 assert receipt['source_commit']==intent['source_commit'] and receipt['source_sha256']==intent['source_sha256']
 assert receipt['exact_published_source_verified'] is True and receipt['trainer_started'] is False
 repo=owned(source/'repo');assert git('-C',str(repo),'rev-parse','HEAD')==intent['source_commit']
 assert not git('-C',str(repo),'status','--porcelain') and source_map(repo)==intent['source_sha256']
 return source,repo
def unlaunched(base):
 for name in ('deployment-launch-receipt.json','prepared-launch-claim.json','pipeline-launch-claim.json',
              'training-launch-claim.json','process-ledger.json','actual-attempts.jsonl','actual-updates.jsonl',
              'resolved-training-config.json','policy-safety-calibration.json'):
  assert not (base/'logs'/name).exists(), 'Existing execution evidence requires observation'
def environment(base,repo):
 env=dict(os.environ)
 for key in tuple(env):
  if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
 env.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',
  CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(repo),PATH='/data/search-g1/env/bin:'+env.get('PATH','/usr/bin:/bin'),
  HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
 for key,name in dict(TMPDIR='tmp',HF_HOME='cache/hf',XDG_CACHE_HOME='cache',TRITON_CACHE_DIR='cache/triton',TORCHINDUCTOR_CACHE_DIR='cache/inductor').items():env[key]=str(base/name)
 return env
'''

REMOTE_STAGE = REMOTE_COMMON + r'''
archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|')
first=archive.next();assert first and first.isfile() and first.name=='qprobe-intent.json' and first.size<=1024**2
intent=json.load(archive.extractfile(first));check(intent)
base=Path(intent['source_base']);assert base.parent==base.parent.resolve() and base.parent.is_dir()
baseline=owned(intent['baseline_base']);original=baseline/'preparation/deployment-intent.json'
assert sha(owned(original))==intent['baseline_intent_sha256']
old_run=read(baseline/'logs/run-manifest.json')
assert old_run['run_id']=='1db662001a0f4d998928bcf6f5f33c9d' and old_run['source_commit']==intent['baseline_commit']
assert old_run['execution_backend']=='ssh_local' and old_run['execution_host']==socket.gethostname()
old=owned(baseline/'repo');assert git('-C',str(old),'rev-parse','HEAD')==intent['baseline_commit']
base.mkdir(mode=0o700,exist_ok=False)
for name in ('preparation','logs'):(base/name).mkdir()
once(base/'preparation/qprobe-intent.json',intent)
once(base/'logs/source-stage-claim.json',dict(deployment_id=intent['deployment_id'],claimed_unix=time.time(),no_automatic_retry=True))
item=archive.next();assert item and item.isfile() and item.name=='published-source.bundle'
assert item.size==intent['bundle_bytes'] and 0<item.size<=32*1024**2
bundle=base/'preparation/published-source.bundle';h=hashlib.sha256();count=0
with bundle.open('xb') as f:
 src=archive.extractfile(item)
 for part in iter(lambda:src.read(1024**2),b''):count+=len(part);h.update(part);f.write(part)
 f.flush();os.fsync(f.fileno())
assert count==item.size and h.hexdigest()==intent['bundle_sha256'] and archive.next() is None
repo=base/'repo';git('clone','--local','--no-hardlinks','--no-checkout',str(old),str(repo))
git('-C',str(repo),'bundle','verify',str(bundle))
assert git('bundle','list-heads',str(bundle)).splitlines()==[intent['source_commit']+' '+intent['published_ref']]
git('-C',str(repo),'fetch',str(bundle),intent['published_ref'])
assert git('-C',str(repo),'rev-parse','FETCH_HEAD')==intent['source_commit']
git('-C',str(repo),'checkout','--detach',intent['source_commit'])
assert git('-C',str(repo),'rev-parse','HEAD')==intent['source_commit'] and not git('-C',str(repo),'status','--porcelain')
actual=source_map(repo);assert actual==intent['source_sha256']
assert git('-C',str(old),'rev-parse','HEAD')==intent['baseline_commit'] and sha(original)==intent['baseline_intent_sha256']
receipt=dict(role='a800_published_source_staging_only',deployment_id=intent['deployment_id'],base=str(base),
 source_commit=intent['source_commit'],source_sha256=actual,bundle_sha256=sha(bundle),
 exact_published_source_verified=True,git_tree_clean=True,baseline_source_unchanged=True,
 services_started=False,trainer_started=False,production_admission=False,checked_unix=time.time())
once(base/'logs/source-staging-complete.json',receipt);output(receipt)
'''

REMOTE_PREPARE = REMOTE_COMMON + r'''
payload=json.load(sys.stdin);intent=payload['intent'];source,staged_repo=staged(intent)
base=Path(intent['base']);assert base.parent==base.parent.resolve() and base.parent.is_dir()
baseline=owned(intent['baseline_base']);original=baseline/'preparation/deployment-intent.json'
assert sha(owned(original))==intent['baseline_intent_sha256'];old=read(original)
assert old['deployment_id']=='5e1f65d8e7d44363b02938c8d19852ab' and old['base']==str(baseline)
assert old['source_commit']==intent['baseline_commit'] and old['host']=='baai-a800'
allowed={'data/'+name for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json')}
allowed.update(('preparation/official-model-manifest-original.json','preparation/formal-inline-method-template.json'))
rows=[{k:row[k] for k in ('target','bytes','sha256')} for row in old['inputs'] if row['target'] in allowed]
assert len(rows)==len(allowed) and {row['target'] for row in rows}==allowed
assert next(row['sha256'] for row in rows if row['target']=='data/manifest.json')=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert next(row['sha256'] for row in rows if row['target']=='preparation/official-model-manifest-original.json')=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
base.mkdir(mode=0o700,exist_ok=False)
for name in ('logs','preparation','data','tmp','ray','cache/triton','cache/inductor','cache/hf','artifacts','checkpoints'):(base/name).mkdir(parents=True,exist_ok=True)
once(base/'preparation/qprobe-intent.json',intent)
once(base/'logs/qprobe-prepare-claim.json',dict(deployment_id=intent['deployment_id'],claimed_unix=time.time(),no_automatic_retry=True))
teacher_storage(intent,create_parent=True)
for row in rows:
 origin=owned(baseline/row['target']);assert origin.is_file() and origin.stat().st_size==row['bytes'] and sha(origin)==row['sha256']
 target=base/row['target'];assert target.parent==target.parent.resolve() and not target.exists()
 with origin.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
 assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'] and sha(origin)==row['sha256']
deployment=dict(schema_version=1,role='fresh_a800_common_deployment_intent',deployment_id=intent['deployment_id'],
 created_unix=intent['created_unix'],host='baai-a800',base=str(base),source_commit=intent['source_commit'],
 source_sha256=intent['source_sha256'],inputs=rows,training_world_size=4,training_physical_indices=None,
 gpu_placement_policy=intent['gpu_placement_policy'],gpu_placement_bound=False,
 auxiliary_physical_indices=[5,6],target_updates=200,same_process_after_five=True,
 initial_probes_source_base=intent['initial_probes_source_base'],
 remote_canonical_source_base=str(baseline),remote_original_intent_sha256=intent['baseline_intent_sha256'])
once(base/'preparation/deployment-intent.json',deployment)
once(base/'logs/deployment-input-receipt.json',dict(deployment_id=intent['deployment_id'],inputs=rows,
 exact_original_remote_bytes=True,controller_data_or_model_bytes_transferred=False,training_data_partitioned=False,
 original_intent_sha256=intent['baseline_intent_sha256']))
repo=base/'repo';git('clone','--local','--no-hardlinks','--no-checkout',str(staged_repo),str(repo));git('-C',str(repo),'checkout','--detach',intent['source_commit'])
assert source_map(repo)==intent['source_sha256'] and not git('-C',str(repo),'status','--porcelain')
once(base/'logs/deployment-source-receipt.json',dict(deployment_id=intent['deployment_id'],
 source_commit=intent['source_commit'],source_sha256=intent['source_sha256'],source_transport='published_git_bundle_remote_staging',
 source_is_exact_published_commit=True,uncommitted_source_copied=False,source_staging_base=str(source)))
from reproduction.common8.policy_calibration_plan import document_sha256
template=read(base/'preparation/formal-inline-method-template.json');method=dict(template)
method.update(source_commit=intent['source_commit'],source_map_sha256=document_sha256(intent['source_sha256']),created_unix=time.time())
once(base/'preparation/formal-inline-method.json',method)
args=['/data/search-g1/env/bin/python','-m','reproduction.a800_common.prepare_run','--base',str(base),
 '--model','/data/search-g1/assets/qwen2.5-3b-instruct','--official-manifest',str(base/'preparation/official-model-manifest-original.json'),
 '--source-commit',intent['source_commit'],'--initial-probes-source-base',intent['initial_probes_source_base']]
with (base/'preparation/prepare-run.log').open('xb') as log:
 result=subprocess.run(args,cwd=repo,env=environment(base,repo),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 log.flush();os.fsync(log.fileno())
once(base/'logs/qprobe-prepare-exit.json',dict(exit_code=result.returncode,checked_unix=time.time(),log_bytes=(base/'preparation/prepare-run.log').stat().st_size))
assert result.returncode==0, 'CPU run preparation failed; inspect this claimed base'
run=read(base/'logs/run-manifest.json')
from reproduction.common8.formal_inline_safety import verify_method
verify_method(method,run);unlaunched(base)
assert run['source_sha256']==intent['source_sha256'] and run.get('question_probe') is None
assert run['initial_model']==run['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct'
assert run['training_world_size']==4 and run['target_updates']==200 and run['completed_updates']==0
assert sha(original)==intent['baseline_intent_sha256']
receipt=dict(stage='CPU_PREPARED_NO_GPU_OR_PIPELINE_LAUNCH',deployment_id=intent['deployment_id'],base=str(base),
 run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 qprobe_binding_pending=True,teacher_warmup_pending=True,trainer_started=False,checked_unix=time.time())
once(base/'logs/qprobe-prepared.json',receipt);output(receipt)
'''

REMOTE_BIND = REMOTE_COMMON + r'''
payload=json.load(sys.stdin);intent=payload['intent'];staged(intent);base=owned(intent['base']);unlaunched(base)
assert read(base/'preparation/qprobe-intent.json')==intent
before=read(base/'logs/run-manifest.json');features=read(base/'preparation/features-ready.json')
sys.path.insert(0,str(base/'repo'))
from reproduction.a800_common.deploy_qprobe import validate_prepared_features
validate_prepared_features(features,before)
assert features['cache_dir']==str(base/'question-features')
assert sha(owned(base/'question-features/manifest.json'))==features['manifest_sha256']
if features.get('kind')=='copied_verified_existing_encoding':assert read(base/'logs/qprobe-feature-adoption.json')==features
once(base/'logs/qprobe-bind-claim.json',dict(deployment_id=intent['deployment_id'],claimed_unix=time.time(),no_automatic_retry=True))
sys.path.insert(0,str(base/'repo'))
from reproduction.common8.qprobe_bind import bind
from reproduction.common8.stage_retrieval import ASSETS
from search_g1.qprobe_features import sha_json
from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
run=bind(base,base/'question-features',exploration_questions=0,retriever_version=sha_json(ASSETS),
 seed=20261013,teacher_warmup_cache=teacher_storage(intent),infra_rollout=DEFAULT_ROLLOUT_PHASE)
receipt=dict(stage='QPROBE_BOUND_PRELAUNCH',deployment_id=intent['deployment_id'],run_id=run['run_id'],
 source_commit=run['source_commit'],question_probe=run['question_probe'],target_updates=200,
 extra_frozen_teacher_warmup_outers=1,completed_training_outers=0,trainer_started=False,checked_unix=time.time())
once(base/'logs/qprobe-bound.json',receipt);output(receipt)
'''

REMOTE_FEATURE_ADOPT = REMOTE_COMMON + r'''
import stat
payload=json.load(sys.stdin);intent=payload['intent'];staged(intent);base=owned(intent['base']);unlaunched(base)
assert read(base/'preparation/qprobe-intent.json')==intent
sys.path.insert(0,str(base/'repo'))
from reproduction.a800_common.deploy_qprobe import (FEATURE_SOURCE_BASE,FEATURE_SOURCE_COMMIT,FEATURE_MANIFEST_SHA,
 FEATURE_FILE_SHA,FEATURE_SPEC_SHA,FEATURE_PRODUCERS,validate_feature_origin)
origin=owned(payload['feature_source_base']);assert str(origin)==FEATURE_SOURCE_BASE and origin!=base
new=read(base/'logs/run-manifest.json');old=read(origin/'logs/run-manifest.json')
assert new.get('question_probe') is None and new['source_commit']==intent['source_commit']
for name in ('qprobe-feature-claim.json','qprobe-bind-claim.json','qprobe-feature-adopt-claim.json'):
 assert not (base/'logs'/name).exists(), 'Feature preparation or binding already claimed'
target=base/'question-features';assert not target.exists() and not target.is_symlink()
assert not any(base.glob('question-features.partial*')) and not (base/'preparation/features-ready.json').exists()
ready_path=origin/'preparation/features-ready.json';exit_path=origin/'logs/qprobe-feature-background-exit.json'
ready=read(ready_path);completed=read(exit_path);cache=owned(origin/'question-features');manifest_path=cache/'manifest.json'
assert sha(owned(manifest_path))==FEATURE_MANIFEST_SHA;manifest=read(manifest_path)
validate_feature_origin(old,new,ready,completed,manifest)
assert old['execution_host']==new['execution_host']==socket.gethostname() and ready['hostname']==socket.gethostname()
assert ready['uid']==completed['child']['uid']==os.getuid()
old_repo=owned(origin/'repo');new_repo=owned(base/'repo')
assert git('-C',str(old_repo),'rev-parse','HEAD')==FEATURE_SOURCE_COMMIT and not git('-C',str(old_repo),'status','--porcelain')
assert git('-C',str(new_repo),'rev-parse','HEAD')==intent['source_commit'] and source_map(new_repo)==intent['source_sha256']
for name in FEATURE_PRODUCERS:
 assert sha(owned(old_repo/name))==old['source_sha256'][name]==sha(owned(new_repo/name))==new['source_sha256'][name]
for name,expected in new['input_sha256'].items():
 assert Path(name).name==name
 previous=owned(origin/'data'/name);current=owned(base/'data'/name)
 assert stat.S_ISREG(previous.stat().st_mode) and stat.S_ISREG(current.stat().st_mode)
 assert previous.stat().st_size==current.stat().st_size and sha(previous)==sha(current)==expected
official=base/'preparation/official-model-manifest-original.json'
assert sha(owned(official))==sha(owned(origin/'preparation/official-model-manifest-original.json'))=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
from reproduction.common8.prepare_run import verify_model
model=owned(new['initial_model']);assert model.is_dir()
for name in new['initial_model_sha256']:
 assert Path(name).name==name and stat.S_ISREG(owned(model/name).stat().st_mode)
assert verify_model(Path(new['initial_model']),official)==new['initial_model_sha256']
rows=[dict(name='manifest.json',bytes=manifest_path.stat().st_size,sha256=FEATURE_MANIFEST_SHA)]
rows += [dict(name=name,**manifest['files'][name]) for name in ('index.jsonl','features.npy')]
assert set(manifest['files'])=={'index.jsonl','features.npy'}
assert set(path.name for path in cache.iterdir())=={'manifest.json','index.jsonl','features.npy'}
for row in rows:
 source=owned(cache/row['name']);assert stat.S_ISREG(source.stat().st_mode) and source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
receipt_shas={str(path.relative_to(origin)):sha(path) for path in (ready_path,exit_path,origin/'logs/run-manifest.json')}
text=(Path('/proc')/str(os.getpid())/'stat').read_text();fields=text[text.rfind(')')+2:].split()
copy_process=dict(pid=os.getpid(),start_ticks=int(fields[19]),sid=int(fields[3]),uid=os.getuid())
once(base/'logs/qprobe-feature-claim.json',dict(copy_process,kind='copied_verified_existing_encoding',run_id=new['run_id'],source_commit=new['source_commit'],
 source_base=str(origin),claimed_unix=time.time(),gpu_job_started=False,no_automatic_retry=True))
once(base/'logs/qprobe-feature-adopt-claim.json',dict(run_id=new['run_id'],source_commit=new['source_commit'],
 source_base=str(origin),origin_receipt_sha256=receipt_shas,claimed_unix=time.time(),no_automatic_retry=True))
target.mkdir(mode=0o700,exist_ok=False)
for row in rows:
 source=owned(cache/row['name']);destination=target/row['name']
 with source.open('rb') as reader,destination.open('xb') as writer:
  shutil.copyfileobj(reader,writer,1024**2);writer.flush();os.fsync(writer.fileno())
 assert destination.stat().st_size==row['bytes'] and sha(destination)==sha(source)==row['sha256']
from search_g1.qprobe_features import FeatureCache
verification=FeatureCache(target,expected_spec_sha=FEATURE_SPEC_SHA).verify_all()
assert verification['count']==169028 and verification['feature_dim']==2048 and verification['feature_file_sha256']==FEATURE_FILE_SHA
for path in (ready_path,exit_path,origin/'logs/run-manifest.json'):
 assert sha(path)==receipt_shas[str(path.relative_to(origin))], 'Origin metadata changed during adoption'
receipt=dict(stage='CPU_EXISTING_FEATURE_CACHE_ADOPTED',kind='copied_verified_existing_encoding',
 run_id=new['run_id'],source_commit=new['source_commit'],execution_id=new['execution_id'],
 cache_dir=str(target),manifest_sha256=sha(target/'manifest.json'),**verification,
 fresh_gpu_encoding=False,actual_visible_cuda_count=None,completed_training_outers=0,target_training_outers=200,
 exit_code=0,model_loaded=False,qa_partitioned=False,copy_process=copy_process,files=rows,checked_unix=time.time(),
 origin=dict(base=str(origin),run_id=old['run_id'],execution_id=old['execution_id'],source_commit=old['source_commit'],
  encoding_pid=ready['pid'],encoding_start_ticks=ready['start_ticks'],encoding_exit_code=completed['exit_code'],
  encoding_cuda_uuid=ready['physical_gpu_uuid'],receipt_sha256=receipt_shas,
  immutable_encoder_producer_sha256={name:new['source_sha256'][name] for name in FEATURE_PRODUCERS}))
once(base/'logs/qprobe-feature-adoption.json',receipt);once(base/'preparation/features-ready.json',receipt)
output(dict(stage=receipt['stage'],kind=receipt['kind'],run_id=new['run_id'],source_commit=new['source_commit'],
 count=verification['count'],feature_dim=verification['feature_dim'],fresh_gpu_encoding=False,exit_code=0))
'''

REMOTE_INSPECT = REMOTE_COMMON + r'''
payload=json.load(sys.stdin);intent=payload['intent'];check(intent)
base=Path(intent['base']);source=Path(intent['source_base'])
result=dict(deployment_id=intent['deployment_id'],source_base_exists=source.is_dir(),base_exists=base.is_dir(),
 teacher_warmup_cache=intent['teacher_warmup_cache'],observed_unix=time.time())
if source.is_dir():
 assert read(source/'preparation/qprobe-intent.json')==intent
 result['source_staging_complete']=(source/'logs/source-staging-complete.json').is_file()
if base.is_dir():
 assert read(base/'preparation/qprobe-intent.json')==intent
 for name,key in [('qprobe-prepared.json','prepared'),('qprobe-bound.json','bound'),('qprobe-feature-claim.json','feature_claimed')]:result[key]=(base/'logs'/name).is_file()
 run_path=base/'logs/run-manifest.json'
 if run_path.is_file():
  run=read(run_path);result['run_id']=run['run_id'];result['question_probe']=run.get('question_probe')
 for name,key in [('manifest.json','feature_manifest'),('complete.json','teacher_warmup_complete')]:
  root=base/'question-features' if key=='feature_manifest' else teacher_storage(intent);path=root/name
  if path.is_file():result[key]=read(path)
 for name,key in [('question-features-progress.json','feature_progress'),('features-ready.json','features_ready')]:
  path=base/'preparation'/name
  if path.is_file():result[key]=read(path)
 for name,key in [('qprobe-feature-claim.json','feature_process'),('qprobe-prepare-exit.json','prepare_exit')]:
  path=base/'logs'/name
  if path.is_file():
   row=read(path);result[key]=row
   if key=='feature_process':
    row.update(birth_matches=False,alive=False,state=None)
    try:
     proc=Path('/proc')/str(row['pid']);text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
     matches=int(fields[19])==row['start_ticks'] and int(fields[3])==row['sid'] and proc.stat().st_uid==row['uid']==os.getuid()
     row.update(birth_matches=matches,alive=matches and fields[0] not in ('Z','X'),state=fields[0] if matches else None)
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 for name,key in [('prepare-run.log','prepare_log'),('question-feature-precompute.log','feature_log')]:
  path=base/'preparation'/name
  if path.is_file():
   info=owned(path).stat();result[key]=dict(bytes=info.st_size,modified_unix=info.st_mtime)
 if (base/'preparation/deployment-intent.json').is_file():
  deployment=read(base/'preparation/deployment-intent.json')
  sys.stdin=io.StringIO(json.dumps(deployment));exec(payload['observe_code'])
output(result)
'''

REMOTE_LAUNCH_GATE = REMOTE_COMMON + r'''
payload=json.load(sys.stdin);intent=payload['intent'];staged(intent);base=owned(intent['base']);unlaunched(base)
assert read(base/'preparation/qprobe-intent.json')==intent
sys.path.insert(0,str(base/'repo'))
from reproduction.a800_common.deploy_qprobe import validate_review,validate_prepared_features
from reproduction.a800_common import deploy as published_deploy
repo=owned(base/'repo');assert git('-C',str(repo),'rev-parse','HEAD')==intent['source_commit']
assert not git('-C',str(repo),'status','--porcelain') and source_map(repo)==intent['source_sha256']
assert sha(owned(published_deploy.__file__))==intent['source_sha256']['reproduction/a800_common/deploy.py']
assert payload['launcher_source_commit']==intent['source_commit']
assert payload['launcher_source_sha256']==intent['source_sha256']['reproduction/a800_common/deploy_qprobe.py']
assert payload['launch_code']==published_deploy.REMOTE_LAUNCH_PREPARED
assert payload['embedded_launcher_sha256']==hashlib.sha256(published_deploy.REMOTE_LAUNCH_PREPARED.encode()).hexdigest()
review=validate_review(payload['review'],intent)
run=read(base/'logs/run-manifest.json');q=run['question_probe']
ready=validate_prepared_features(read(base/'preparation/features-ready.json'),run)
assert ready['cache_dir']==str(base/'question-features')
assert sha(owned(base/'question-features/manifest.json'))==ready['manifest_sha256']
if ready.get('kind')=='copied_verified_existing_encoding':assert read(base/'logs/qprobe-feature-adoption.json')==ready
assert read(base/'logs/qprobe-bound.json')['run_id']==run['run_id']
assert q['candidate_questions']==5120 and q['selected_questions']==512 and q['responses_per_question']==5
assert q['exploration_questions']==0 and q['bootstrap_max_groups_per_outer']==5120
from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
assert run['infra_rollout']==DEFAULT_ROLLOUT_PHASE and run['infra_rollout']['mode']=='independent_trajectory_phase_v1'
assert q['teacher_warmup_cache']==str(teacher_storage(intent)) and q['cache_dir']==str(base/'question-features')
assert not q.get('initial_head_checkpoint') and not run.get('full_state_restore') and run['mixed_first_selection'] is False
assert run['target_updates']==200 and run['completed_updates']==0 and run['training_world_size']==4
from search_g1.qprobe_features import FeatureCache,sha_json
from reproduction.common8.stage_retrieval import ASSETS
features=FeatureCache(q['cache_dir']);features.verify_all()
assert features.manifest['count']==169028 and features.spec['train_parquet_sha256']==run['input_sha256']['train.parquet']
assert run['retriever_version']==sha_json(ASSETS)
from reproduction.common8.precision_contract import BF16_PROFILE
assert run['precision_profile']==BF16_PROFILE
assert run['actor_micro_batch_per_rank']==4
budget=run['budget'];assert budget['per_rank_actor_micro_batch']==4 and budget['ppo_micro_batch_size']==16
assert budget['ppo_mini_batch_size']==128 and budget['per_rank_log_prob_micro_batch']==8 and budget['lr_warmup_steps']==57
assert (128//4)//4==8
for path in (base/'checkpoints',base/'artifacts/question-probe',base/'cache'):
 assert str(path).startswith('/data/') and path.resolve()==path
from reproduction.common8.native_allocation import bind_idle_local_placement
run=bind_idle_local_placement(base);placement=published_deploy.launch_placement(run)
once(base/'preparation/qprobe-independent-review.json',review)
once(base/'logs/qprobe-launch-claim.json',dict(deployment_id=intent['deployment_id'],run_id=run['run_id'],
 review_sha256=sha(base/'preparation/qprobe-independent-review.json'),claimed_unix=time.time(),no_automatic_retry=True,**placement))
payload['intent']=read(base/'preparation/deployment-intent.json')
sys.stdin=io.StringIO(json.dumps(payload));exec(payload['launch_code'])
'''


def create_bundle(repo, commit, ref, bundle):
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', ref], text=True).strip() == commit
    assert not bundle.exists() and not bundle.is_symlink()
    subprocess.run(['git', '-C', str(repo), 'bundle', 'create', str(bundle), ref, '^' + BASELINE_COMMIT],
                   check=True, capture_output=True, timeout=90)
    assert subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], text=True).splitlines() == [commit + ' ' + ref]
    assert 0 < bundle.stat().st_size <= MAX_BUNDLE
    return bundle


def stage(args):
    repo = Path(__file__).resolve().parents[2]
    assert re.fullmatch('[0-9a-f]{40}', args.source_commit)
    assert re.fullmatch(r'refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*', args.published_ref)
    assert re.fullmatch('[0-9a-f]{64}', args.baseline_intent_sha256)
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == args.source_commit
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip()
    remote = subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'origin', args.published_ref], text=True, timeout=60).splitlines()
    assert remote == [args.source_commit + '\t' + args.published_ref], 'Stage only the actually published branch tip'
    handle = args.handle.resolve();handle.parent.mkdir(parents=True, exist_ok=True)
    assert not handle.exists()
    assert shutil.disk_usage(handle.parent).free >= 2 * MAX_BUNDLE + 8 * 1024**2, 'Insufficient controller space for bounded source relay'
    identifier = uuid.uuid4().hex
    bundle = create_bundle(repo, args.source_commit, args.published_ref, handle.with_name(handle.stem + '.source.bundle'))
    intent = dict(role=ROLE, host='baai-a800', deployment_id=identifier,
        base='/data/shaoshitong/Search-G1-a800-fresh200-' + identifier,
        source_base='/data/shaoshitong/Search-G1-qprobe-source-' + identifier,
        source_commit=args.source_commit, published_ref=args.published_ref,
        source_sha256=published_source(repo, args.source_commit), baseline_base=BASELINE,
        baseline_commit=BASELINE_COMMIT, baseline_intent_sha256=args.baseline_intent_sha256,
        initial_probes_source_base=args.initial_probes_source_base or INITIAL_PROBES,
        teacher_warmup_cache=str(validate_teacher_cache(args.teacher_warmup_cache or TEACHER_CACHE)),
        bundle_bytes=bundle.stat().st_size, bundle_sha256=digest(bundle), target_updates=200,
        launch_services=False, gpu_placement_policy=PLACEMENT_POLICY,gpu_placement_bound=False,created_unix=time.time())
    write_once(handle, validate_intent(intent))
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as archive:
        for name, raw in [('qprobe-intent.json', json.dumps(intent, allow_nan=False).encode()), ('published-source.bundle', bundle.read_bytes())]:
            member = tarfile.TarInfo(name);member.size = len(raw);archive.addfile(member, io.BytesIO(raw))
    result = invoke(handle, intent, 'stage', REMOTE_STAGE, data.getvalue(), 180)
    if result['exit_code'] == 0 and any(row.get('exact_published_source_verified') is True and
            row.get('bundle_sha256') == intent['bundle_sha256'] and row.get('source_sha256') == intent['source_sha256'] and
            row.get('source_commit') == intent['source_commit'] and row.get('deployment_id') == identifier
            for row in result['observations']):
        bundle.unlink()  # Actual A800 SHA receipt verified; no local data/model/environment relay.
    return result


def invoke(handle, intent, action, code, payload, timeout):
    if action != 'inspect':
        write_once(handle.with_name(handle.stem + '.' + action + '-once.json'),
                   dict(deployment_id=intent['deployment_id'], claimed_unix=time.time(), no_automatic_retry=True))
    stderr = handle.with_name(handle.stem + '.' + action + '-' + str(time.time_ns()) + '.stderr')
    output = b''; exit_code = None; failure_type = None
    with stderr.open('xb') as errors:
        try:
            route = ssh_command(code)
            assert route[-1].startswith('python3 -c ')
            route[-1] = '/data/search-g1/env/bin/python -c ' + route[-1][len('python3 -c '):]
            result = subprocess.run(route, input=payload, stdout=subprocess.PIPE, stderr=errors, timeout=timeout)
            output, exit_code = result.stdout, result.returncode
        except (subprocess.TimeoutExpired, OSError) as failure:
            failure_type = type(failure).__name__; output = getattr(failure, 'stdout', None) or b''
        errors.flush(); os.fsync(errors.fileno())
    observations = [json.loads(line.split(':', 1)[1]) for line in output.decode(errors='replace').splitlines()
                    if line.startswith(('QPROBE_DEPLOYMENT_JSON:', 'DEPLOYMENT_JSON:'))]
    result = dict(action=action, deployment_id=intent['deployment_id'], exit_code=exit_code,
                  failure_type=failure_type, observations=observations, transport_stderr=str(stderr), observed_unix=time.time())
    receipt = handle.with_name(handle.stem + '.' + action + '-' + str(time.time_ns()) + '.json')
    write_once(receipt, result)
    return dict(result, handle=str(handle), receipt=str(receipt), next_action='inspect the same handle; no automatic retry')


def feature_plan(intent, physical_index=6, batch_size=32):
    validate_intent(intent)
    assert physical_index in (5, 6) and type(batch_size) is int and batch_size > 0
    base = Path(intent['base'])
    return dict(base=str(base), repo=str(base/'repo'), source_commit=intent['source_commit'],
        argv=['/data/search-g1/env/bin/python', '-m', 'reproduction.a800_common.precompute_qprobe',
              '--base', str(base), '--physical-index', str(physical_index), '--batch-size', str(batch_size)],
        required_environment={'PYTHONPATH':str(base/'repo'), 'CUDA_VISIBLE_DEVICES':str(physical_index)},
        progress_path=str(base/'preparation/question-features-progress.json'),
        log_path=str(base/'preparation/question-feature-precompute.log'),
        completion_path=str(base/'preparation/features-ready.json'), gpu_job_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'prepare', 'feature-plan', 'feature-adopt', 'bind', 'inspect', 'launch'))
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--source-commit');parser.add_argument('--published-ref')
    parser.add_argument('--baseline-intent-sha256');parser.add_argument('--review-receipt', type=Path)
    parser.add_argument('--initial-probes-source-base')
    parser.add_argument('--teacher-warmup-cache')
    parser.add_argument('--feature-source-base')
    parser.add_argument('--physical-index', type=int, choices=(5,6), default=6)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    assert args.action=='feature-adopt' or args.feature_source_base is None
    if args.action == 'stage':
        assert args.source_commit and args.published_ref and args.baseline_intent_sha256
        result = stage(args)
    else:
        assert args.source_commit is None and args.published_ref is None and args.baseline_intent_sha256 is None and args.initial_probes_source_base is None and args.teacher_warmup_cache is None
        handle = args.handle.resolve();intent = validate_intent(json.loads(handle.read_text()))
        if args.action == 'feature-plan':
            result = feature_plan(intent, args.physical_index, args.batch_size)
        else:
            payload = dict(intent=intent)
            code, timeout = {'prepare':(REMOTE_PREPARE,1000),'bind':(REMOTE_BIND,600),'feature-adopt':(REMOTE_FEATURE_ADOPT,1000),
                             'inspect':(REMOTE_INSPECT,60),'launch':(REMOTE_LAUNCH_GATE,600)}[args.action]
            if args.action=='feature-adopt':
                assert args.feature_source_base==FEATURE_SOURCE_BASE, 'Select the exact completed BF4 encoding source explicitly'
                payload['feature_source_base']=args.feature_source_base
            if args.action == 'inspect':payload['observe_code'] = REMOTE_OBSERVE
            if args.action == 'launch':
                assert args.review_receipt is not None, 'An independent review of the exact published source is required'
                assert args.review_receipt.stat().st_size <= 1024**2
                payload['review'] = validate_review(json.loads(args.review_receipt.read_text()), intent)
                repo = Path(__file__).resolve().parents[2]
                commit, launcher, launch_code = verified_launcher(repo, intent)
                payload.update(launcher_source_commit=commit, launcher_source_sha256=hashlib.sha256(launcher).hexdigest(),
                    embedded_launcher_sha256=hashlib.sha256(launch_code.encode()).hexdigest(), launch_code=launch_code)
            result = invoke(handle,intent,args.action,code,json.dumps(payload,sort_keys=True,allow_nan=False).encode(),timeout)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result.get('exit_code', 0) == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
