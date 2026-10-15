"""One explicit BF4/A800 feature job plus bounded, read-only observations.

The published precompute wrapper owns the sole qprobe-feature-claim and actual
GPU6 UUID admission. This operator only detaches its CPU parent, persists the
child's real wait result, and never retries, signals, cleans or loads a model.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
RUNTIME = ROOT/'qprobe-teacher-warmup-a800'
HANDLE = ROOT/'preparation/qprobe-teacher-warmup-a800-deployment.json'
SOURCE = 'bf4b88d04c166951691029b18d05d91ece7c7054'
BASE = '/data/shaoshitong/Search-G1-a800-fresh200-27bab88929e048b4869f3889b99d8e18'
OLD_SOURCE = 'c4d156de124de7a94652d75858bcbabc8c56083c'
OLD_BASE = '/data/shaoshitong/Search-G1-a800-fresh200-6047f7f9674b49fdad718a256ab3db95'
OLD_RUN = '6a26247f538b4145a24b156e79fb04d1'
OLD_EXECUTION = '140e43d2243d49a99d88f77399f263b8'

REMOTE_COMMON = r'''
import hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
payload=json.load(sys.stdin)
base=Path('/data/shaoshitong/Search-G1-a800-fresh200-27bab88929e048b4869f3889b99d8e18');repo=base/'repo'
old_base=Path('/data/shaoshitong/Search-G1-a800-fresh200-6047f7f9674b49fdad718a256ab3db95')
def owned(path):
 path=Path(path);assert path.resolve()==path and path.stat().st_uid==os.getuid()
 return path
def read(path):
 path=owned(path);assert path.is_file() and path.stat().st_size<=1024**2
 return json.loads(path.read_text())
def once(path,value):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def birth(pid):
 proc=Path('/proc')/str(pid);text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=proc.stat().st_uid,state=fields[0])
def process(saved):
 result=dict(saved,birth_matches=False,alive=False,state=None)
 try:
  actual=birth(saved['pid']);matches=all(actual[key]==saved[key] for key in ('pid','start_ticks','sid','uid'))
  result.update(birth_matches=matches,alive=matches and actual['state'] not in ('Z','X'),state=actual['state'] if matches else None)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return result
def pinned_run():
 owned(base);owned(repo);run=read(base/'logs/run-manifest.json');intent=read(base/'preparation/qprobe-intent.json')
 assert intent==payload['intent'] and intent['source_commit']=='bf4b88d04c166951691029b18d05d91ece7c7054'
 assert run['source_commit']==intent['source_commit'] and run['source_sha256']==intent['source_sha256']
 assert len(run['source_sha256'])==469 and run['completed_updates']==0
 assert run['execution_backend']=='ssh_local' and run['execution_host']==socket.gethostname()
 assert run['training_physical_indices']==[0,1,2,3] and run['excluded_physical_indices']==[4,7]
 assert run['initial_model']==run['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct'
 return run
def old_cleanup():
 old_repo=owned(old_base/'repo');old=read(old_base/'logs/run-manifest.json')
 assert old['run_id']=='6a26247f538b4145a24b156e79fb04d1' and old['execution_id']=='140e43d2243d49a99d88f77399f263b8'
 assert old['source_commit']=='c4d156de124de7a94652d75858bcbabc8c56083c'
 assert subprocess.check_output(['git','-C',str(old_repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==old['source_commit']
 assert not subprocess.check_output(['git','-C',str(old_repo),'status','--porcelain'],timeout=10).strip()
 relative='reproduction/common8/actor_probe_owned_cleanup.py'
 assert hashlib.sha256(owned(old_repo/relative).read_bytes()).hexdigest()==old['source_sha256'][relative]
 sys.path.insert(0,str(old_repo))
 from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup
 result=verify_cleanup(old_base,old)
 assert result['accepted'] is True and result['owned_cuda_contexts_remaining']==0
 assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
 return result
'''

SUPERVISOR = r'''
import json,os,subprocess,sys,time
from pathlib import Path
base=Path(os.environ['SEARCH_G1_BASE']);repo=base/'repo'
sys.path.insert(0,str(repo))
from reproduction.a800_common.deploy import write_once
def birth(pid):
 proc=Path('/proc')/str(pid);text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=proc.stat().st_uid)
run=json.loads((base/'logs/run-manifest.json').read_text());supervisor=birth(os.getpid());child=None
record=dict(run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 supervisor=supervisor,physical_index=6,protected_physical_indices=[4,7],started_unix=time.time(),no_automatic_retry=True)
try:
 argv=['/data/search-g1/env/bin/python','-m','reproduction.a800_common.precompute_qprobe',
       '--base',str(base),'--physical-index','6','--batch-size','32']
 child=subprocess.Popen(argv,cwd=repo,env=dict(os.environ),stdin=subprocess.DEVNULL)
 actual=birth(child.pid);assert actual['uid']==os.getuid() and actual['sid']==supervisor['sid']
 record.update(child=actual,argv=argv)
 write_once(base/'logs/qprobe-feature-background-child.json',record)
 code=child.wait()
 write_once(base/'logs/qprobe-feature-background-exit.json',dict(record,exit_code=code,
  status='FEATURE_CHILD_EXITED',finished_unix=time.time()))
 sys.exit(0 if code==0 else 1)
except Exception as failure:
 write_once(base/'logs/qprobe-feature-background-exit.json',dict(record,exit_code=None,
  status='SUPERVISOR_FAILED',failure_type=type(failure).__name__,finished_unix=time.time()))
 raise
'''

REMOTE_LAUNCH = REMOTE_COMMON + r'''
run=pinned_run();assert run.get('question_probe') is None
for name in ('qprobe-feature-claim.json','qprobe-bind-claim.json','pipeline-launch-claim.json',
 'training-launch-claim.json','deployment-launch-receipt.json','process-ledger.json'):
 assert not (base/'logs'/name).exists(), 'Existing feature or pipeline evidence requires inspection'
assert not (base/'question-features').exists() and not (base/'question-features.partial').exists()
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==run['source_commit']
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=10).strip()
cleanup=old_cleanup()
once(base/'logs/qprobe-feature-background-start-once.json',dict(run_id=run['run_id'],execution_id=run['execution_id'],
 source_commit=run['source_commit'],old_cleanup_accepted=True,checked_unix=cleanup['checked_unix'],
 control_code_sha256=hashlib.sha256(payload['supervisor_code'].encode()).hexdigest(),claimed_unix=time.time(),no_automatic_retry=True))
env=dict(os.environ)
for key in tuple(env):
 if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
env.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='4',
 PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES='6',PATH='/data/search-g1/env/bin:'+env.get('PATH','/usr/bin:/bin'),
 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',
 HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
out_path=base/'logs/qprobe-feature-background.stdout.log';err_path=base/'logs/qprobe-feature-background.stderr.log'
with out_path.open('xb') as out,err_path.open('xb') as err:
 parent=subprocess.Popen(['/data/search-g1/env/bin/python','-c',payload['supervisor_code']],cwd=repo,
  env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 actual=birth(parent.pid);assert actual['sid']==parent.pid and actual['uid']==os.getuid()
 receipt=dict(run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
  supervisor=actual,stdout_log=str(out_path),stderr_log=str(err_path),physical_index=6,
  protected_physical_indices=[4,7],launched_unix=time.time(),no_automatic_retry=True)
 once(base/'logs/qprobe-feature-background-receipt.json',receipt)
print('FEATURE_OPERATOR_JSON:'+json.dumps(receipt),flush=True)
'''

REMOTE_INSPECT = REMOTE_COMMON + r'''
run=pinned_run();result=dict(run_id=run['run_id'],source_commit=run['source_commit'],observed_unix=time.time())
for relative,key in [('logs/qprobe-feature-background-receipt.json','background'),
 ('logs/qprobe-feature-background-child.json','child'),('logs/qprobe-feature-background-exit.json','exit'),
 ('logs/qprobe-feature-claim.json','feature_claim'),('preparation/question-features-progress.json','progress'),
 ('preparation/features-ready.json','features_ready')]:
 path=base/relative
 if path.is_file():
  value=read(path);assert value['run_id']==run['run_id'] and value['source_commit']==run['source_commit']
  result[key]=value
  if key=='background':result['supervisor_process']=process(value['supervisor'])
  if key=='child':result['feature_child_process']=process(value['child'])
  if key=='progress':result['completed_questions']=value.get('completed',0)
for relative,key in [('logs/qprobe-feature-background.stdout.log','stdout_log'),
 ('logs/qprobe-feature-background.stderr.log','stderr_log'),('preparation/question-feature-precompute.log','feature_log')]:
 path=base/relative
 if path.is_file():
  info=owned(path).stat();result[key]=dict(bytes=info.st_size,modified_unix=info.st_mtime)
stderr=base/'logs/qprobe-feature-background.stderr.log'
if stderr.is_file():
 with stderr.open('rb') as f:f.seek(max(0,stderr.stat().st_size-8192));lines=f.read(8192).decode(errors='replace').splitlines()
 errors=[]
 for line in lines:
  text=line.strip()
  if re.match(r'^(AssertionError|ValueError|RuntimeError|OSError|FileNotFoundError|PermissionError|ModuleNotFoundError|ImportError|KeyError|TypeError)(:|$)',text):
   if any(word in text.lower() for word in ('password','authorization','bearer','ghp_','prompt','question','answer','response')):text=text.split(':',1)[0]
   errors.append(re.split(r'[\{\[]',text,1)[0][:240])
 result['exception_diagnostic']=errors[-6:]
print('FEATURE_OPERATOR_JSON:'+json.dumps(result),flush=True)
'''

REMOTE_OLD_CLEANUP = REMOTE_COMMON + r'''
print('FEATURE_OPERATOR_JSON:'+json.dumps(old_cleanup()),flush=True)
'''

REMOTE_INSPECT_WARMUP = REMOTE_COMMON + r'''
import contextlib,importlib.util,io,math,stat
sys.dont_write_bytecode=True
bound_run=pinned_run();bound_base=base;bound_repo=repo
prepared=read(bound_base/'preparation/deployment-intent.json')
assert prepared['base']==str(bound_base) and prepared['source_commit']==bound_run['source_commit']
assert prepared['source_sha256']==bound_run['source_sha256'] and prepared['deployment_id']==payload['intent']['deployment_id']
module_path=owned(bound_repo/'reproduction/a800_common/deploy.py')
assert hashlib.sha256(module_path.read_bytes()).hexdigest()==bound_run['source_sha256']['reproduction/a800_common/deploy.py']
spec=importlib.util.spec_from_file_location('verified_bf4_metadata_observer',module_path)
published=importlib.util.module_from_spec(spec);spec.loader.exec_module(published)
captured=io.StringIO();original_stdin=sys.stdin
try:
 sys.stdin=io.StringIO(json.dumps(prepared))
 with contextlib.redirect_stdout(captured):exec(published.REMOTE_OBSERVE,{'__name__':'verified_bf4_metadata_observation'})
finally:sys.stdin=original_stdin
rows=[json.loads(line.split(':',1)[1]) for line in captured.getvalue().splitlines() if line.startswith('DEPLOYMENT_JSON:')]
assert len(rows)==1;standard=rows[0]
assert standard.get('run_id',bound_run['run_id'])==bound_run['run_id']
assert standard.get('execution_id',bound_run['execution_id'])==bound_run['execution_id']
def select(value,keys):
 result={}
 for key in keys:
  item=value.get(key)
  if item is None or type(item) in (bool,int) or type(item) is float and math.isfinite(item):result[key]=item
  elif isinstance(item,str):result[key]=item[:256]
 return result
def birth_summary(value):return select(value,('pid','start_ticks','sid','uid','birth_matches','alive','state'))
result=dict(run_id=bound_run['run_id'],execution_id=bound_run['execution_id'],source_commit=bound_run['source_commit'],
 observed_unix=time.time(),training_world_size=bound_run['training_world_size'],
 driver_rng_status='source_restore_only_not_independent_persisted_readback',
 head_output_classes_contract=6,head_k_class_histogram=None,metadata_only=True)
result.update(select(standard,('pipeline_birth_matches','pipeline_process_state','pipeline_alive','pid','start_ticks',
 'pipeline_claimed','trainer_claimed','pipeline_stage','pipeline_status_unix','recorded_completed_outers','last_recorded_outer',
 'stdout_log_bytes','stdout_log_modified_unix','failure_type','last_command')))
for key in ('failure_diagnostic','startup_diagnostic'):
 if key in standard:result[key]=standard[key][:6]
update=standard.get('latest_actual_update')
if update:
 result['latest_actual_update']=select(update,('step','opportunities_processed','actual_optimizer_writes','optimizer_writes_before','optimizer_writes_after'))
 result['latest_actual_update']['metrics']=select(update.get('metrics',{}),('critic/rewards/mean','actor/pg_loss','actor/ppo_kl',
  'actor/grad_norm','actor/optimizer_steps_this_update','actor/update_opportunities_this_attempt','actor/skipped_update_opportunities'))
result['rank_optimizer_counts']=[]
for row in standard.get('opportunity_ledgers',[])[:4]:
 item=select(row,('rank','exists','file_bytes','scan_from_start','malformed_complete_lines','oversized_lines','partial_final_line',
  'identity_rejected_rows','processed_opportunities_per_rank','successful_writes_per_rank','cumulative_optimizer_writes'))
 processed=row.get('processed_opportunities_per_rank');written=row.get('successful_writes_per_rank')
 item['processed_without_successful_write']=processed-written if type(processed) is int and type(written) is int else None
 result['rank_optimizer_counts'].append(item)
result['owned_job_status']=[]
for job in standard.get('owned_jobs',[])[:32]:
 item=select(job,('name','ledger_state','live_registered_count','exit_code','log_bytes','log_modified_unix'))
 item['leader']=birth_summary(job.get('leader',{}));item['exception_diagnostic']=job.get('exception_diagnostic',[])[:6]
 result['owned_job_status'].append(item)
gpu=standard.get('gpu_samples',{})
result['gpu_sampler']=select(gpu,('matching_sample_count','cumulative_sample_count','latest_sample_unix','latest_sample_age_seconds',
 'maximum_scanned_sampling_gap_seconds','identity_or_scalar_rejected_rows','training_claimed','pretraining_history_only'))
if gpu.get('sampler'):result['gpu_sampler']['process']=birth_summary(gpu['sampler'])
result['gpu_sampler']['latest_devices']=[select(device,('physical_index','used_mib','total_mib','util_percent','sampled_run_peak_mib'))
 for device in gpu.get('latest_devices',[])[:6]]
result['training_log']=select(standard.get('training_log',{}),('exists','bytes','modified_unix'))
result['training_log']['exception_diagnostic']=standard.get('training_log',{}).get('exception_diagnostic',[])[:6]
configuration=bound_base/'logs/resolved-training-config.json'
result['resolved_selection_mode']=None
if configuration.is_file():
 document=read(configuration);assert document['run_id']==bound_run['run_id'] and document['execution_id']==bound_run['execution_id']
 result['resolved_selection_mode']=document['resolved_config']['data'].get('selection_mode')
 result['resolved_config_sha256']=document['resolved_config_sha256']
question_probe=bound_run.get('question_probe') or {}
result['question_probe']=select(question_probe,('candidate_questions','selected_questions','responses_per_question',
 'exploration_questions','bootstrap_max_groups_per_outer','head_batch_size','head_epochs_per_outer'))
train_log=owned(bound_run['train_log']) if Path(bound_run['train_log']).is_file() else None
result['teacher_partial_progress']=None
if train_log:
 train_log.relative_to(bound_base/'logs');info=train_log.stat();limit=256*1024;offset=max(0,info.st_size-limit)
 with train_log.open('rb') as file:file.seek(offset);tail=file.read(limit)
 lines=tail.splitlines()
 if offset and lines:lines=lines[1:]
 if tail and not tail.endswith(b'\n') and lines:lines=lines[:-1]
 for line in lines:
  text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',line.decode(errors='replace')).strip()
  text=re.sub(r'^\((?:TaskRunner|main_task) pid=[1-9][0-9]*(?:, ip=[0-9a-fA-F:.]+)?\)\s*','',text,count=1)
  try:value=json.loads(text)
  except ValueError:continue
  if not isinstance(value,dict) or value.get('stage')!='frozen_teacher_warmup_chunk_complete':continue
  count=value.get('completed_questions');trajectories=value.get('completed_trajectories')
  if type(count) is int and 0<count<=5120 and count%512==0 and trajectories==5*count and value.get('total_questions')==5120:
   result['teacher_partial_progress']=dict(completed_questions=count,completed_trajectories=trajectories,
    total_questions=5120,log_modified_unix=info.st_mtime,bound_to_current_run_log=True)
 result['teacher_tail_scan_bytes']=len(tail)
ready_path=bound_base/'artifacts/question-probe/teacher-warmup-ready.json'
result['teacher_warmup_ready_exists']=ready_path.is_file()
if ready_path.is_file():
 ready=read(ready_path);head=ready['head_checkpoint'];expected_snapshot=bound_run['run_id']+':0'
 assert ready['status']=='FROZEN_TEACHER_WARMUP_HEAD_READY' and head['actor_snapshot_id']==expected_snapshot
 before=sorted(ready['before_workers'],key=lambda row:row['rank']);after=sorted(ready['after_workers'],key=lambda row:row['rank'])
 restored=sorted(ready['actor_rng_restored'],key=lambda row:row['rank'])
 assert [row['rank'] for row in before]==[row['rank'] for row in after]==[row['rank'] for row in restored]==[0,1,2,3]
 worker_rows=[]
 for first,last,rng in zip(before,after,restored):
  boundary=bound_base/'artifacts/question-probe/teacher-warmup-boundary'
  for worker,phase in ((first,'before'),(last,'after')):
   proof=worker['rng_receipt'];assert proof['run_id']==bound_run['run_id'] and proof['rank']==worker['rank']
   expected=boundary/phase/('rank-%d.pt'%worker['rank']);assert Path(proof['path'])==expected
   info=owned(expected).stat()
   assert stat.S_ISREG(info.st_mode) and type(proof['bytes']) is int and 0<proof['bytes']<=64*1024**2 and info.st_size==proof['bytes']
  expected_before=boundary/'before'/('rank-%d.pt'%rng['rank']);assert Path(rng['source'])==expected_before
  info=owned(expected_before).stat();assert stat.S_ISREG(info.st_mode) and 0<info.st_size<=64*1024**2 and info.st_size==first['rng_receipt']['bytes']
  worker_rows.append(dict(rank=first['rank'],parameter_sha256_before=first['parameter_sha256'],parameter_sha256_after=last['parameter_sha256'],
   optimizer_sha256_before=first['optimizer_sha256'],optimizer_sha256_after=last['optimizer_sha256'],
   scheduler_sha256_before=first['scheduler_sha256'],scheduler_sha256_after=last['scheduler_sha256'],
   state_hashes_match=all(first[key]==last[key] for key in ('parameter_sha256','optimizer_sha256','scheduler_sha256')),
   actor_optimizer_writes_before=first['actual_optimizer_writes'],actor_optimizer_writes_after=last['actual_optimizer_writes'],
   actor_completed_outer_before=first['completed_outer'],actor_completed_outer_after=last['completed_outer'],
   rng_restored=rng['rng_equal'],rng_restore_actor_optimizer_writes=rng['actual_optimizer_writes'],
   rng_restore_completed_training_outers=rng['completed_training_outers']))
 cache=Path(payload['intent']['teacher_warmup_cache']);assert cache.resolve()==cache
 complete=read(cache/'complete.json');proof=ready['cache_receipt'];assert proof==complete
 chunk_map={row['relative_path']:row['sha256'] for row in proof['chunks']}
 result['teacher_warmup_ready']=dict(**select(ready,('status','reused_questions','fitted_questions','extra_warmup_outers',
  'completed_training_outers','target_training_outers','checked_unix')),
  ready_metadata_sha256=hashlib.sha256(ready_path.read_bytes()).hexdigest(),actor_snapshot_id=head['actor_snapshot_id'],
  head=select(head,('checkpoint_sha','version','completed_outer','feature_spec_sha','update_count')),
  head_metrics=select(ready['head_metrics'],('bootstrap_groups','replay_groups','head_optimizer_updates','replay_weighted_loss',
   'count_ce','mean_mse','std_mse','completed_outer')),
  frozen_workers=worker_rows,actor_rng_restored_rank_count=sum(row['rng_equal'] is True for row in restored),
  cache_complete_metadata_matches_ready=True,cache_questions=proof['questions'],cache_trajectories=proof['trajectories'],
  cache_identity_sha256=proof['identity_sha256'],cache_chunk_count=len(chunk_map),
  cache_chunk_hash_map_sha256=hashlib.sha256(json.dumps(chunk_map,sort_keys=True,separators=(',',':')).encode()).hexdigest())
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
print('FEATURE_OPERATOR_JSON:'+json.dumps(result,allow_nan=False),flush=True)
'''


def write_once(path, value):
    with path.open('x') as target:
        json.dump(value,target,sort_keys=True,allow_nan=False);target.write('\n');target.flush();os.fsync(target.fileno())
    descriptor = os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--launch-features',action='store_true')
    actions.add_argument('--inspect-features',action='store_true')
    actions.add_argument('--inspect-old-cleanup',action='store_true')
    actions.add_argument('--inspect-warmup',action='store_true')
    args = parser.parse_args()
    operator_repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git','-C',str(operator_repo),'rev-parse','HEAD'],text=True).strip()
    committed = subprocess.check_output(['git','-C',str(operator_repo),'show',commit+':reproduction/operations/a800_qprobe_feature_operator.py'])
    assert committed == Path(__file__).read_bytes(), 'Publish this exact operator before execution'
    intent = json.loads(HANDLE.read_text())
    assert intent['source_commit'] == SOURCE and intent['base'] == BASE and len(intent['source_sha256']) == 469
    assert subprocess.check_output(['git','-C',str(RUNTIME),'rev-parse','HEAD'],text=True).strip() == SOURCE
    assert not subprocess.check_output(['git','-C',str(RUNTIME),'status','--porcelain']).strip()
    module_path = RUNTIME/'reproduction/a800_common/deploy.py'
    assert hashlib.sha256(module_path.read_bytes()).hexdigest() == intent['source_sha256']['reproduction/a800_common/deploy.py']
    spec = importlib.util.spec_from_file_location('published_a800_deploy',module_path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    action = 'launch' if args.launch_features else 'inspect' if args.inspect_features else 'old-cleanup'
    if args.launch_features:
        write_once(ROOT/'preparation/qprobe-feature-background-local-start-once.json',dict(
            source_commit=SOURCE,base=BASE,operator_commit=commit,claimed_unix=time.time(),no_automatic_retry=True))
    remote = {'launch':REMOTE_LAUNCH,'inspect':REMOTE_INSPECT,'old-cleanup':REMOTE_OLD_CLEANUP}[action]
    if args.inspect_warmup:
        action = 'warmup';remote = REMOTE_INSPECT_WARMUP
    route = module.ssh_command(remote)
    assert route[-1].startswith('python3 -c ')
    route[-1] = '/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    payload = json.dumps(dict(intent=intent,supervisor_code=SUPERVISOR),sort_keys=True,allow_nan=False).encode()
    stamp = str(time.time_ns());stderr = ROOT/'preparation'/('qprobe-feature-operator-'+action+'-'+stamp+'.stderr')
    output=b'';exit_code=None;failure_type=None
    with stderr.open('xb') as errors:
        try:
            result=subprocess.run(route,input=payload,stdout=subprocess.PIPE,stderr=errors,timeout=90)
            output,exit_code=result.stdout,result.returncode
        except (OSError,subprocess.TimeoutExpired) as failure:
            failure_type=type(failure).__name__;output=getattr(failure,'stdout',None) or b''
        errors.flush();os.fsync(errors.fileno())
    observations=[json.loads(line.split(':',1)[1]) for line in output.decode(errors='replace').splitlines()
                  if line.startswith('FEATURE_OPERATOR_JSON:')]
    receipt=ROOT/'preparation'/('qprobe-feature-operator-'+action+'-'+stamp+'.json')
    write_once(receipt,dict(action=action,source_commit=SOURCE,base=BASE,operator_commit=commit,
        exit_code=exit_code,failure_type=failure_type,observations=observations,stderr=str(stderr),observed_unix=time.time()))
    print(json.dumps(dict(receipt=str(receipt),exit_code=exit_code,failure_type=failure_type,
                         observation=observations[-1] if observations else None,no_automatic_retry=True)))
    return 0 if exit_code==0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
