"""Run one published remote CPU saved-CP5 qualification in a new namespace.

Authenticate failed-run evidence and source-only staging before the expensive
CPU validator. Never train, evaluate, signal, retry, modify old source or claim
CUDA/FSDP acceptance. All cache, logs, temporary files and state reads stay remote.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
METADATA = OPS/'reproduction/operations/a800_qprobe_restore5_qualification.py'
METADATA_SHA = '14358272539c36c578040018a284924b7926abd37ab6c92e90934a3fbef00469'
METADATA_COMMIT = '93fbb28f7dbce8f993212b52617e3f9b5ed04283'

REMOTE_CPU = r'''
raw_payload=sys.stdin.read(1024*1024+1);need(len(raw_payload.encode())<=1024*1024,'payload_limit')
request=json.loads(raw_payload);sys.stdin=io.StringIO(raw_payload)
result=dict(role='remote_saved_cp5_cpu_qualification',run_id=request['expected_run'],execution_id=request['expected_execution'],
 source_commit=request['intent']['source_commit'],base=request['intent']['base'],status='PRECHECK_UNKNOWN',
 actual_cuda_restore_accepted=False,training_started=False,evaluation_started=False,no_automatic_retry=True)
child=None;work=None;log=None;identity=None;launch_handle=None
def immutable_once(path,value):
 data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def current_identity(pid):
 proc=Path('/proc')/str(pid);text=(proc/'stat').read_text();fields=text[text.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=proc.stat().st_uid)
def safe_trace(raw):
 text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',raw.decode(errors='replace'))
 frames=[dict(file=Path(m.group(1)).name[:160],line=int(m.group(2)),function=m.group(3)[:96]) for m in
  re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',text,re.MULTILINE)][-8:]
 classes=re.findall(r'^[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',text,re.MULTILINE)
 codes=re.findall(r'\b(QPROBE_SAVED_CHECKPOINT_[A-Z0-9_]{1,96}|QPROBE_PAIR_[A-Z0-9_]{1,96})\b',text)
 return dict(bounded_tail_sha256=hashlib.sha256(raw).hexdigest(),exception_frames=frames,
  exception_type=classes[-1][:96] if classes else None,fixed_message_code=codes[-1] if codes else None)
try:
 exec(common_code,globals());run=pinned_run()
 source_map_sha=digest(canonical(run['source_sha256']));run_manifest_sha=digest(budget_bytes(base/'logs/run-manifest.json',1024*1024))
 terminal=request['terminal'];need((terminal['run_id'],terminal['execution_id'],terminal['source_commit'],terminal['base'])==
  (run['run_id'],run['execution_id'],run['source_commit'],str(base)),'terminal_receipt_identity')
 terminal_births=births();ownership=terminal_ownership()
 complete_updates=None;closed_attempts=None;driver=reward_window();counts=opportunity_counts()
 need(driver['recorded_complete_outers']==5 and counts['status']=='observed' and counts['cumulative_processed']==
  counts['cumulative_successful_adam']==100 and counts['cumulative_skip']==0 and counts['current_partial'] is None,'not_exact_cp5_prefix')
 need(driver['complete_prefix_sha256']==terminal['driver_prefix']['complete_prefix_sha256'] and
  [row['prefix_sha256'] for row in counts['per_rank']]==[row['prefix_sha256'] for row in terminal['four_rank_prefix']['per_rank']],
  'terminal_prefix_changed')
 cp=terminal['checkpoint5_metadata'];need(cp['source_run_manifest_sha256']==run_manifest_sha,'terminal_manifest_changed')
 prefix='checkpoints/search-g1-common8/actor/global_step_5'
 for name,expected_sha in (('complete.json',cp['complete_metadata_sha256']),('training_state/manifest.json',cp['manifest_metadata_sha256'])):
  _,info=document(prefix+'/'+name);need(info['scanned_sha256']==expected_sha,'terminal_checkpoint_metadata_changed')
 stage=request['stage_intent'];need(stage['role']=='a800_qprobe_prospective_intent' and stage['host']=='baai-a800','stage_intent_role')
 need(re.fullmatch('[0-9a-f]{32}',stage['deployment_id']) and re.fullmatch('[0-9a-f]{40}',stage['source_commit']),'stage_identity')
 source=Path(stage['source_base']);stage_repo=source/'repo'
 need(str(source)=='/data/shaoshitong/Search-G1-qprobe-source-'+stage['deployment_id'] and source.resolve()==source and
  source.stat().st_uid==os.getuid() and source!=base and stage['source_commit']!=run['source_commit'],'separate_source_stage')
 need(stage['launch_services'] is False and not stage.get('saved_checkpoint_restore') and not Path(stage['base']).exists(),'source_only_stage_required')
 need(json.loads(budget_bytes(source/'preparation/qprobe-intent.json',1024*1024))==stage,'remote_stage_intent_changed')
 staged=json.loads(budget_bytes(source/'logs/source-staging-complete.json',2*1024*1024))
 need(staged==request['stage_observation'] and staged['role']=='a800_published_source_staging_only' and
  staged['source_commit']==stage['source_commit'] and staged['source_sha256']==stage['source_sha256'] and
  staged['exact_published_source_verified'] is True and staged['git_tree_clean'] is True and
  staged['services_started'] is False and staged['trainer_started'] is False and staged['production_admission'] is False,'stage_receipt_changed')
 need(subprocess.check_output(['git','-C',str(stage_repo),'rev-parse','HEAD'],text=True,timeout=10).strip()==stage['source_commit'] and
  not subprocess.check_output(['git','-C',str(stage_repo),'status','--porcelain'],timeout=10).strip(),'stage_git_changed')
 paths=[p for p in stage_repo.rglob('*') if p.is_file() and (p.suffix in ('.py','.sh','.yaml','.yml','.json') or
  str(p.relative_to(stage_repo)) in ('verl/version/version','verl/single_controller/version/version')) and
  not {'.git','__pycache__','evidence'}.intersection(p.relative_to(stage_repo).parts)]
 actual={str(p.relative_to(stage_repo)):digest(budget_bytes(p,4*1024*1024)) for p in paths}
 need(actual==stage['source_sha256'] and 'search_g1/qprobe_saved_checkpoint.py' in actual,'stage_runtime_source_map_changed')
 qualification=Path(request['qualification_dir'])
 need(qualification.is_absolute() and qualification.resolve()==qualification and not qualification.exists() and
  re.fullmatch(r'/(data|mnt|backup)/[A-Za-z0-9._/-]{1,480}',str(qualification)) and '..' not in qualification.parts and
  not qualification.is_relative_to(base) and not base.is_relative_to(qualification) and
  not qualification.is_relative_to(stage_repo) and not stage_repo.is_relative_to(qualification),'new_remote_qualification_namespace')
 need(qualification.parent.is_dir() and qualification.parent.resolve()==qualification.parent and
  qualification.parent.stat().st_uid==os.getuid(),'qualification_parent_owner')
 token=hashlib.sha256(str(qualification).encode()).hexdigest()[:32]
 work=source/'logs'/('saved-cp5-qualification-'+token)
 claim=source/'logs'/('saved-cp5-qualification-'+token+'.claim.json')
 check();signal.alarm(0)
 immutable_once(claim,dict(role='single_saved_cp5_cpu_qualification_intent',run_id=run['run_id'],execution_id=run['execution_id'],
  parent_source_commit=run['source_commit'],qualifier_source_commit=stage['source_commit'],qualification_dir=str(qualification),
  terminal_receipt_sha256=request['terminal_sha256'],created_unix=time.time(),actual_cuda_restore_accepted=False,no_automatic_retry=True))
 work.mkdir(mode=0o700,exist_ok=False)
 for name in ('tmp','cache','cache/hf','cache/triton','cache/inductor'):(work/name).mkdir(mode=0o700,parents=True,exist_ok=True)
 environment=dict(os.environ)
 for key in tuple(environment):
  if key.startswith('SLURM_') or key in ('PYTHONHOME','RAY_ADDRESS','LD_LIBRARY_PATH','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY',
   'http_proxy','https_proxy','all_proxy'):environment.pop(key,None)
 environment.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
  PYTHONPATH=str(stage_repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',SEARCH_G1_PROFILE='common8',
  SEARCH_G1_BASE=str(base),SEARCH_G1_TRAINING_WORLD_SIZE='4',TMPDIR=str(work/'tmp'),HF_HOME=str(work/'cache/hf'),
  HF_HUB_CACHE=str(work/'cache/hf/hub'),HUGGINGFACE_HUB_CACHE=str(work/'cache/hf/hub'),
  HF_DATASETS_CACHE=str(work/'cache/hf/datasets'),TRANSFORMERS_CACHE=str(work/'cache/hf/transformers'),
  TORCH_HOME=str(work/'cache/torch'),XDG_CACHE_HOME=str(work/'cache'),TRITON_CACHE_DIR=str(work/'cache/triton'),
  TORCHINDUCTOR_CACHE_DIR=str(work/'cache/inductor'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 log=work/'cpu-qualification.log'
 command=['/data/search-g1/env/bin/python','-m','search_g1.qprobe_saved_checkpoint','--base',str(base),
  '--qualification-dir',str(qualification),'--expected-run-id',run['run_id'],'--expected-execution-id',run['execution_id'],
  '--expected-source-commit',run['source_commit']]
 with log.open('xb') as stream:
  child=subprocess.Popen(command,cwd=stage_repo,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  launch_handle=dict(pid=child.pid,expected_sid=child.pid,process=None,birth_status='not_yet_observed',log=str(log),
   work=str(work),launch_record=str(work/'launch.json'),process_record=str(work/'process.json'),qualification_dir=str(qualification),
   run_id=run['run_id'],execution_id=run['execution_id'],qualifier_source_commit=stage['source_commit'],actual_cuda_restore_accepted=False)
  result.update(process_handle=launch_handle,log=str(log),work=str(work),process_record=str(work/'process.json'),
   qualification_dir=str(qualification),qualifier_source_commit=stage['source_commit'])
  immutable_once(work/'launch.json',dict(role='saved_cp5_cpu_qualifier_launch',**launch_handle))
  print('SAVED_CP5_PROCESS_JSON:'+json.dumps(launch_handle),flush=True)
  observed_identity=current_identity(child.pid)
  need(observed_identity['sid']==child.pid and observed_identity['uid']==os.getuid(),'cpu_process_birth');identity=observed_identity
  launch_handle.update(process=identity,birth_status='exact_birth_observed')
  result['process']=identity
  immutable_once(work/'process.json',dict(role='saved_cp5_cpu_qualifier_process',**identity,log=str(log),qualification_dir=str(qualification)))
  try:exit_code=child.wait(timeout=1200)
  except subprocess.TimeoutExpired:exit_code=None
 result.update(status='CPU_PROCESS_STILL_RUNNING_OR_UNKNOWN' if exit_code is None else 'CPU_PROCESS_EXITED',
  process=identity,process_state=process(identity),cpu_exit_code=exit_code,log=str(log),qualification_dir=str(qualification),
  qualifier_source_commit=stage['source_commit'],terminal_recheck=dict(ownership=ownership,complete_outers=5,logical_successful_adam=100),
  preflight_bytes_read=spent,preflight_limits=dict(aggregate_read_bytes=TOTAL_LIMIT,ledger_bytes=LEDGER_LIMIT,
   process_ledger_bytes=PROCESS_LEDGER_LIMIT,deadline_seconds=65),observed_unix=time.time())
 with log.open('rb') as stream:
  stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536));tail=stream.read(65536)
 result['log_bytes']=size;result['diagnostics']=safe_trace(tail)
 if exit_code==0:
  receipt=qualification/'source-receipt.json';need(receipt.resolve()==receipt and receipt.stat().st_uid==os.getuid() and
   receipt.stat().st_size<=8*1024*1024,'qualified_receipt_metadata_limit')
  raw=receipt.read_bytes();value=json.loads(raw)
  need(value['status']=='QPROBE_SAVED_CHECKPOINT_FULL_SHA_CPU_STATES_PASS' and value['kind']=='question_probe_saved_checkpoint' and
   value['run_id']==run['run_id'] and value['source_execution_id']==run['execution_id'] and value['source_commit']==run['source_commit'] and
   value['source_base']==str(base) and value['step']==5 and value['world_size']==4 and value['attempt_progress']['cumulative_writes']==100 and
   value['actual_cpu_readback'] is True and value['actual_cuda_restore_accepted'] is False and value['completed_pair_published'] is False,
   'qualified_source_receipt_identity')
  result.update(status='SAVED_CP5_CPU_QUALIFIED',source_receipt=str(receipt),source_receipt_sha256=hashlib.sha256(raw).hexdigest(),
   actual_cpu_readback=True,post_evaluation_rng_persisted=value['post_evaluation_rng_persisted'],
   original_native_five_accepted=value['original_native_five_accepted'],completed_pair_published=False)
 immutable_once(work/'operator-result.json',result)
except (Unknown,AssertionError,KeyError,IndexError,TypeError,ValueError,OverflowError,OSError,subprocess.SubprocessError) as failure:
 result.update(status='UNKNOWN_OR_FAILED',reason=str(failure) if isinstance(failure,Unknown) else type(failure).__name__,observed_unix=time.time())
 if child is not None:
  result.update(process_handle=launch_handle,cpu_exit_code=child.poll(),log=str(log),work=str(work),
   process_record=str(work/'process.json'))
  if identity is not None:result.update(process=identity,process_state=process(identity))
  try:immutable_once(work/'operator-failure.json',result);result['failure_record_status']='persisted'
  except OSError:result['failure_record_status']='not_persisted'
finally:signal.alarm(0)
encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode())>128*1024:encoded=json.dumps(dict(role=result['role'],run_id=result['run_id'],execution_id=result['execution_id'],
 source_commit=result['source_commit'],status='UNKNOWN_OR_FAILED',reason='output_limit',actual_cuda_restore_accepted=False,
 process_handle=launch_handle,cpu_exit_code=result.get('cpu_exit_code'),log=str(log) if log is not None else None,
 work=str(work) if work is not None else None,process_record=str(work/'process.json') if work is not None else None),allow_nan=False)
print('SAVED_CP5_QUALIFIER_JSON:'+encoded,flush=True)
'''


def local_json(path, limit=8*1024*1024):
    assert path.is_absolute() and path.resolve()==path and path.is_file() and path.stat().st_size<=limit
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def load_reader():
    raw=METADATA.read_bytes();assert hashlib.sha256(raw).hexdigest()==METADATA_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',METADATA_COMMIT+':'+str(METADATA.relative_to(OPS))],timeout=30)==raw
    head=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True,timeout=30).strip()
    own=Path(__file__).resolve()
    assert subprocess.check_output(['git','-C',str(OPS),'show',head+':'+str(own.relative_to(OPS))],timeout=30)==own.read_bytes(), 'Publish exact operator first'
    spec=importlib.util.spec_from_file_location('published_saved_cp5_metadata',METADATA)
    metadata=importlib.util.module_from_spec(spec);spec.loader.exec_module(metadata)
    flex,commit,remote,common_sha=metadata.load_reader()
    marker='raw_payload=sys.stdin.read(1024*1024+1)';assert remote.count(marker)==1
    remote=remote.split(marker,1)[0]+'\n'+REMOTE_CPU;ast.parse(remote)
    return flex,head,remote,common_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('failed-handle','failed-runtime','terminal-metadata','stage-handle','stage-runtime','stage-receipt',
                 'stage-publication','stage-review','qualification-dir','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-run','expected-execution','terminal-metadata-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    assert all(re.fullmatch('[0-9a-f]{32}',v) for v in (args.expected_run,args.expected_execution))
    assert re.fullmatch('[0-9a-f]{64}',args.terminal_metadata_sha256)
    flex,commit,remote,common_sha=load_reader()
    failed=flex.read(args.failed_handle);authority_sha=flex.authority(args.failed_handle,failed,args.expected_run,args.expected_execution)
    transport=flex.runtime_module(args.failed_runtime,failed)
    terminal,terminal_sha=local_json(args.terminal_metadata)
    assert terminal_sha==args.terminal_metadata_sha256 and terminal['operator_sha256']==METADATA_SHA and terminal['exit_code']==0
    assert (terminal['run_id'],terminal['execution_id'],terminal['source_commit'],terminal['base'])==(
        args.expected_run,args.expected_execution,failed['source_commit'],failed['base'])
    observation=terminal['observation'];assert observation['status']=='observed' and observation['metadata_fields_complete'] is True
    ownership=observation['terminal_ownership'];counts=observation['four_rank_prefix']
    assert ownership['pipeline_stage']=='FAILED' and ownership['all_recorded_births_gone'] is True and ownership['all_project_namespace_trainers_gone'] is True
    assert observation['driver_prefix']['recorded_complete_outers']==5 and counts['status']=='observed' and counts['four_rank_complete_prefix_agreement'] is True
    assert counts['cumulative_processed']==counts['cumulative_successful_adam']==100 and counts['cumulative_skip']==0
    stage,stage_sha=local_json(args.stage_handle);staged,staged_sha=local_json(args.stage_receipt)
    publication,publication_sha=local_json(args.stage_publication);review,review_sha=local_json(args.stage_review)
    assert stage['source_commit']!=failed['source_commit'] and stage['launch_services'] is False and not stage.get('saved_checkpoint_restore')
    assert staged['action']=='stage' and staged['exit_code']==0 and len(staged['observations'])==1
    stage_observation=staged['observations'][0]
    assert stage_observation['role']=='a800_published_source_staging_only' and stage_observation['source_commit']==stage['source_commit']
    assert stage_observation['source_sha256']==stage['source_sha256'] and stage_observation['exact_published_source_verified'] is True
    assert publication['source_commit']==publication['remote_commit']==stage['source_commit'] and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['published_ref']==stage['published_ref']
    assert review['role']=='a800_qprobe_independent_code_review' and review['approved'] is True and review['independent'] is True and review['source_commit']==stage['source_commit']
    assert type(review['reviewer']) is str and review['reviewer']!=review['implementation_actor'] and not review['findings']
    assert review['source_sha256']==stage['source_sha256']
    assert review['source_map_sha256']==hashlib.sha256(json.dumps(stage['source_sha256'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert review['source_metadata_observation_sha256']==terminal_sha
    flex.runtime_module(args.stage_runtime,stage)
    assert publication['changed_files_verified']==len(publication['files'])==len(review['reviewed_changed_source_sha256'])
    assert {row['path']:row['sha256'] for row in publication['files']}==review['reviewed_changed_source_sha256']
    for row in publication['files']:
        relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
        path=args.stage_runtime/relative;assert path.resolve()==path
        raw=path.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']==stage['source_sha256'][str(relative)]
        assert subprocess.check_output(['git','-C',str(args.stage_runtime),'rev-parse',stage['source_commit']+':'+str(relative)],text=True,timeout=30).strip()==row['remote_blob']
    assert args.receipt.is_absolute() and args.receipt.resolve()==args.receipt and not args.receipt.exists()
    args.receipt.relative_to(ROOT/'preparation')
    route=transport.ssh_command(remote);assert route[-1].startswith('python3 -c ')
    route[-1]='/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    flex.authority(args.failed_handle,failed,args.expected_run,args.expected_execution)
    result=dict(role='single_saved_cp5_cpu_qualifier_receipt',operator_commit=commit,operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        run_id=args.expected_run,execution_id=args.expected_execution,source_commit=failed['source_commit'],base=failed['base'],
        qualifier_source_commit=stage['source_commit'],terminal_metadata_sha256=terminal_sha,stage_handle_sha256=stage_sha,
        stage_receipt_sha256=staged_sha,stage_publication_sha256=publication_sha,stage_review_sha256=review_sha,
        authority_sha256=authority_sha,reused_common_sha256=common_sha,remote_reader_sha256=hashlib.sha256(remote.encode()).hexdigest(),
        qualification_dir=str(args.qualification_dir),actual_cuda_restore_accepted=False,no_automatic_retry=True,observation=None,process_handle=None)
    flex.once(args.receipt.with_suffix('.claim.json'),dict(role='once_remote_cpu_saved_cp5_qualification_claim',
        run_id=args.expected_run,execution_id=args.expected_execution,operator_sha256=result['operator_sha256'],
        qualification_dir=str(args.qualification_dir),claimed_unix=time.time(),no_automatic_retry=True))
    payload=json.dumps(dict(intent=failed,expected_run=args.expected_run,expected_execution=args.expected_execution,after_step=0,
        terminal=observation,terminal_sha256=terminal_sha,stage_intent=stage,stage_observation=stage_observation,
        qualification_dir=str(args.qualification_dir)),allow_nan=False).encode()
    expected_work=Path(stage['source_base'])/'logs'/('saved-cp5-qualification-'+hashlib.sha256(str(args.qualification_dir).encode()).hexdigest()[:32])
    def checked_handle(handle,expected_pid=None):
        assert (handle['run_id'],handle['execution_id'],handle['qualifier_source_commit'])==(args.expected_run,args.expected_execution,stage['source_commit'])
        assert type(handle['pid']) is int and handle['pid']>0 and handle['expected_sid']==handle['pid'] and handle['actual_cuda_restore_accepted'] is False
        assert expected_pid is None or handle['pid']==expected_pid
        assert handle['qualification_dir']==str(args.qualification_dir) and handle['work']==str(expected_work)
        assert handle['log']==str(expected_work/'cpu-qualification.log') and handle['process_record']==str(expected_work/'process.json')
        assert handle['launch_record']==str(expected_work/'launch.json')
        birth=handle['process']
        if birth is None:assert handle['birth_status']=='not_yet_observed'
        else:
            assert handle['birth_status']=='exact_birth_observed' and birth['pid']==birth['sid']==handle['pid']
            assert all(type(birth[k]) is int and birth[k]>=(0 if k=='uid' else 1) for k in ('pid','start_ticks','sid','uid'))
        return handle
    with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        try:
            actual=subprocess.run(route,input=payload,stdout=out,stderr=err,timeout=1320)
            result.update(exit_code=actual.returncode,status='REMOTE_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            result.update(exit_code=None,status='TIMEOUT_OR_TRANSPORT_UNKNOWN',failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(),stderr_bytes=err.tell());out.seek(0);raw=out.read(128*1024+1)
        err.seek(0);stderr=err.read(64*1024);result['bounded_stderr_sha256']=hashlib.sha256(stderr).hexdigest()
        result['remote_exception_frames']=[dict(file=Path(m.group(1)).name[:160],line=int(m.group(2)),function=m.group(3)[:96])
            for m in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',stderr.decode(errors='replace'),re.MULTILINE)][-8:]
    if len(raw)<=128*1024:
        try:
            lines=raw.decode().splitlines()
            processes=[json.loads(line.split(':',1)[1]) for line in lines if line.startswith('SAVED_CP5_PROCESS_JSON:')]
            if processes:
                assert len(processes)==1
                result['process_handle']=checked_handle(processes[0])
            values=[json.loads(line.split(':',1)[1]) for line in lines if line.startswith('SAVED_CP5_QUALIFIER_JSON:')]
            if values:
                assert len(values)==1;value=values[0]
                assert (value['run_id'],value['execution_id'],value['source_commit'])==(args.expected_run,args.expected_execution,failed['source_commit'])
                assert value['role']=='remote_saved_cp5_cpu_qualification' and value['actual_cuda_restore_accepted'] is False
                final_handle=result['process_handle']
                if value.get('process_handle') is not None:
                    final_handle=checked_handle(value['process_handle'],final_handle['pid'] if final_handle else None)
                if value['status']=='SAVED_CP5_CPU_QUALIFIED':
                    assert final_handle is not None and final_handle['process'] is not None and value['actual_cpu_readback'] is True and value['cpu_exit_code']==0
                    assert value['source_receipt']==str(args.qualification_dir/'source-receipt.json') and re.fullmatch('[0-9a-f]{64}',value['source_receipt_sha256'])
                result.update(observation=value,process_handle=final_handle)
        except (AssertionError,KeyError,TypeError,ValueError,UnicodeError) as failure:
            result.update(status='SANITIZED_OUTPUT_REJECTED',failure_type=type(failure).__name__,observation=None)
    else:result['status']='OUTPUT_LIMIT_EXCEEDED'
    result['returned_unix']=time.time();flex.once(args.receipt,result)
    print(json.dumps(dict(receipt=str(args.receipt),status=result['status'],exit_code=result['exit_code'],
        observation=result['observation'],process_handle=result['process_handle'],actual_cuda_restore_accepted=False,no_automatic_retry=True),allow_nan=False))
    return 0 if result['status']=='REMOTE_RETURNED' and result['exit_code']==0 and result['observation'] is not None and result['observation']['status']=='SAVED_CP5_CPU_QUALIFIED' else 1


if __name__=='__main__':
    raise SystemExit(main())
