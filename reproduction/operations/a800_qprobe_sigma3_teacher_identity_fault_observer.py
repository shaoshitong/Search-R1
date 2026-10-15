"""One metadata-only diagnosis of the pinned sigma3 teacher-identity failure.

Questions are borrowed solely as a comparison placeholder and never recomputed,
returned or qualified. No model, tensor, QA, rollout, teacher chunk or raw log
is read. This operator only reads metadata and owned process/GPU identities.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'qprobe-a800-sigma3-fresh-start-fix'
PREP=ROOT/'preparation'
HANDLE=PREP/'qprobe-a800-sigma3-fresh-start-fix-deployment.json'
READY=PREP/'qprobe-sigma3-fresh-start-fix-reviewed-prepared-ready.json'
PUBLICATION=PREP/'qprobe-sigma3-fresh-start-fix-source-publication.json'
INSPECT=PREP/'qprobe-a800-sigma3-fresh-start-fix-deployment.inspect-1792050214040750000.json'
PINS={str(HANDLE):'934e5aaf4d94e06b467a821ca22825570c9d3c7ab28f938e1476f3248b02819b',
 str(READY):'6bf593ef1fc392569ed428dd580896ba4fdfd090ac0913ed9824a951507101e7',
 str(PUBLICATION):'a593f71bd8ff924cef7c288d830e3d0e07bd2c61e34b550366e4c999df1f801b',
 str(INSPECT):'a0ba75b1923327c754fa592bec5979995fc7acba48795d84d3c137e93f92ed68'}
SOURCE='aa56499c894bfd18e6fcf6c189d2aa2008403a14'
MAP_SHA='d45d360230fcb0d97ecd2663991c12db622030326be96266dd59f8c333ef20bf'
RUN='157362c09cfd4665b43019c09de99ac7'
EXECUTION='a75219f8b9d04a778b0f7594ca08de11'
BASE='/data/shaoshitong/Search-G1-a800-fresh200-f7fa3a82c8144c84b3ea300e287a07e8'
PID,TICKS=4116877,612760539
TRAINER_PID,TRAINER_TICKS=4123820,612807854
CACHE='/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v3-async-top512'
SUPPORT='reproduction/operations/a800_qprobe_sigma3_switch_stop.py'
SUPPORT_COMMIT='a521995b09d35d70dc27a6c8d0c5835e634c7e60'
SUPPORT_SHA='0a5734aa1358a18601f89b46f2ed7112c137467d622f6346af16660f2423adae'


def source_support():
    path=OPS/SUPPORT;raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==SUPPORT_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',SUPPORT_COMMIT+':'+SUPPORT],timeout=10)==raw
    spec=importlib.util.spec_from_file_location('pinned_teacher_fault_source_support',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


REMOTE=r'''
import ast,hashlib,json,os,re,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['GIT_OPTIONAL_LOCKS']='0'
p=json.load(sys.stdin);deadline=time.monotonic()+50

def check():
 if time.monotonic()>=deadline:raise TimeoutError('metadata deadline')
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def process(pid):
 check();root=Path('/proc')/str(pid);raw=(root/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=root.stat().st_uid,state=fields[0])
def live(record):
 try:actual=process(record['pid'])
 except (FileNotFoundError,ProcessLookupError):return False
 return actual['state'] not in ('Z','X') and all(actual[key]==record[key] for key in ('start_ticks','sid','uid'))
def summarized(value):
 if value is None or type(value) in (bool,int,float):return value
 if type(value) is str and re.fullmatch('[0-9a-f]{64}',value):return dict(sha256=value)
 return dict(type=type(value).__name__,sha256=digest(value))
def differences(current,cached,path=''):
 if type(current) is dict and type(cached) is dict:
  result=[]
  for name in sorted(set(current)|set(cached)):
   if name=='questions' and not path:continue
   location=path+'.'+name if path else name
   if name not in current or name not in cached:
    result.append(dict(keypath=location,current=summarized(current.get(name)),cached=summarized(cached.get(name)),
     current_present=name in current,cached_present=name in cached))
   else:result+=differences(current[name],cached[name],location)
  return result
 if type(current)!=type(cached) or current!=cached:return [dict(keypath=path,current=summarized(current),cached=summarized(cached))]
 return []
'''

REMOTE_READ=r'''
base=Path(p['base']);repo=base/'repo';intent=p['intent'];check()
assert sys.executable=='/data/search-g1/env/bin/python'
assert base==base.resolve() and base.stat().st_uid==os.getuid() and document(base/'preparation/qprobe-intent.json')==intent
run=document(base/'logs/run-manifest.json');assert (run['run_id'],run['execution_id'],run['source_commit'])==(p['run'],p['execution'],p['source'])
assert run['source_sha256']==intent['source_sha256'] and len(run['source_sha256'])==481
verify_source(repo,p['source'],run['source_sha256'],481,allow_hydra=True)
launch=document(base/'logs/deployment-launch-receipt.json');claim=document(base/'logs/pipeline-launch-claim.json')
producer=run['pipeline_identity'];keys=('pid','start_ticks','sid','uid')
assert producer==claim['pipeline_identity'] and tuple(producer[key] for key in keys)==(p['pid'],p['ticks'],p['pid'],os.getuid())
assert (launch['pid'],launch['start_ticks'],launch['run_id'],launch['execution_id'])==(p['pid'],p['ticks'],p['run'],p['execution'])
ledger=document(base/'logs/process-ledger.json',128*1024**2);assert ledger['run_id']==run['run_id']
trainer=ledger['jobs']['trainer'];assert trainer['state']=='STOPPED' and trainer['exit_code']==1
assert tuple(trainer['leader'][key] for key in keys)==(p['trainer_pid'],p['trainer_ticks'],p['trainer_pid'],os.getuid())
records=[producer];leaders=[producer];jobs=[]
for name,job in ledger['jobs'].items():
 assert job['state']=='STOPPED';leader=job['leader'];assert leader['pid']==leader['sid'] and leader['uid']==os.getuid()
 rows=[leader]+list(job['ledger'].values())
 assert all(type(row[key]) is int for row in rows for key in keys)
 assert all(row['uid']==leader['uid'] and row['sid']==leader['sid'] for row in rows)
 records+=rows;leaders.append(leader);jobs.append(dict(name=name,leader={key:leader[key] for key in keys},recorded_births=len(rows),state='STOPPED',exit_code=job.get('exit_code')))
assert not any(live(row) for row in records)
sessions={(row['sid'],row['uid']):row['start_ticks'] for row in leaders};survivors=[];reused_sessions=[]
for path in Path('/proc').iterdir():
 check()
 if not path.name.isdigit():continue
 try:actual=process(int(path.name))
 except (FileNotFoundError,ProcessLookupError):continue
 except PermissionError:
  try:assert path.stat().st_uid!=os.getuid()
  except FileNotFoundError:pass
  continue
 key=(actual['sid'],actual['uid'])
 if actual['state'] in ('Z','X') or key not in sessions:continue
 try:now_leader=process(actual['sid'])
 except (FileNotFoundError,ProcessLookupError):now_leader=None
 if now_leader is not None and now_leader['pid']==now_leader['sid'] and now_leader['start_ticks']!=sessions[key]:
  reused_sessions.append(actual['sid']);continue
 survivors.append(actual)
assert not survivors
cuda=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
assert cuda.returncode==0 and len(cuda.stdout.encode())<=128*1024
births={}
for row in records:births.setdefault(row['pid'],set()).add(tuple(row[key] for key in ('start_ticks','sid','uid')))
owned_cuda=[];reused_cuda=[]
for line in cuda.stdout.splitlines():
 fields=[field.strip() for field in line.split(',')];assert len(fields)==2
 pid=int(fields[1])
 if pid not in births:continue
 try:actual=process(pid)
 except (FileNotFoundError,ProcessLookupError):owned_cuda.append(pid);continue
 if tuple(actual[key] for key in ('start_ticks','sid','uid')) in births[pid]:owned_cuda.append(pid)
 else:reused_cuda.append(pid)
assert not owned_cuda
cleanup=document(base/'logs/cleanup-status.json');assert cleanup['run_id']==run['run_id'] and cleanup['execution_id']==run['execution_id'] and cleanup['cleaned'] is True
zero={}
for name in ('actual-updates.jsonl','actual-attempts.jsonl'):
 path=base/'logs'/name;zero[name]=not path.exists() and not path.is_symlink()
assert all(zero.values())
assert not any((base/'logs').glob('optimizer-rank-*')) and not any((base/'logs').glob('scheduler-rank-*'))
checkpoint_root=base/'checkpoints';assert checkpoint_root==checkpoint_root.resolve()
for path in checkpoint_root.rglob('*'):check();assert not path.is_symlink() and not path.is_file()
assert not (base/'artifacts/question-probe/teacher-warmup-ready.json').exists()
assert not (base/'artifacts/question-probe/teacher-warmup-head/initial.pt').exists()
resolved=document(base/'logs/resolved-training-config.json',8*1024**2)
assert (resolved['run_id'],resolved['execution_id'])==(run['run_id'],run['execution_id'])
config=resolved['resolved_config'];assert hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()==resolved['resolved_config_sha256']
features=document(base/'question-features/manifest.json',4*1024**2);assert digest(features['spec'])==features['spec_sha']
assert features['spec']['train_parquet_sha256']==run['input_sha256']['train.parquet']
cache=Path(p['cache']);assert str(cache)==run['question_probe']['teacher_warmup_cache']
cached=document(cache/'identity.json',4*1024**2)
scorer_names=('search_g1/mixed_runtime.py','search_g1/signals.py','search_r1/qa_protocol.py','verl/utils/reward_score/qa_em.py')
scorer={name:hashlib.sha256(sealed(repo/name)).hexdigest() for name in scorer_names}
assert all(scorer[name]==run['source_sha256'][name] for name in scorer)
rollout_named=('reproduction/common8/generation_behavior.py','reproduction/common8/behavior_runtime.py',
 'reproduction/common8/context_budget.py','reproduction/common8/sampling_capture.py','search_g1/async_rollout_runtime.py',
 'search_g1/rollout_phase_contract.py','verl/workers/fsdp_workers.py','verl/workers/sharding_manager/fsdp_vllm.py',
 'search_g1/qprobe_teacher_worker.py','search_g1/qprobe_warmup.py')
rollout={name:value for name,value in run['source_sha256'].items() if name.startswith(('search_r1/llm_agent/','verl/workers/rollout/vllm_rollout/')) or name in rollout_named}
# Verify the observed producer still uses this exact source predicate; never import its torch runtime.
syntax=ast.parse(sealed(repo/'search_g1/qprobe_warmup.py'))
function=next(node for node in syntax.body if isinstance(node,ast.FunctionDef) and node.name=='teacher_identity')
source_dict=next(node for node in ast.walk(function) if isinstance(node,ast.keyword) and node.arg=='rollout_sources').value
expected_predicate=ast.parse("{name:value for name,value in manifest['source_sha256'].items() if name.startswith(('search_r1/llm_agent/','verl/workers/rollout/vllm_rollout/')) or name in "+repr(rollout_named)+'}',mode='eval').body
assert ast.dump(source_dict,include_attributes=False)==ast.dump(expected_predicate,include_attributes=False)
current=dict(schema=2,model_revision=run['model_revision'],teacher_frozen_evidence_required=True,training_world_size=run.get('training_world_size',8),
 model_files_sha256=run['initial_model_sha256'],train_parquet_sha256=run['input_sha256']['train.parquet'],
 data_manifest_sha256=run['budget']['common_manifest_sha256'],retriever_version=run['retriever_version'],feature_spec_sha=features['spec_sha'],
 sampling_config=config['actor_rollout_ref']['rollout'],agent_config={key:config['data'][key] for key in ('max_start_length','max_prompt_length','max_response_length','max_obs_length')},
 max_turns=config['max_turns'],seed=config['question_probe']['seed'],questions=cached['questions'],scorer_sources=scorer,rollout_sources=rollout)
comparison=differences(current,cached);assert len(comparison)<=128
without_questions=lambda value:{key:row for key,row in value.items() if key!='questions'}
assert not any(name=='torch' or name.startswith(('torch.','transformers.','vllm.')) for name in sys.modules)
result=dict(role='a800_qprobe_sigma3_actual_teacher_identity_fault_metadata',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 fault_code='WARMUP_TEACHER_DATA_SAMPLING_OR_SCORING_NAMESPACE_CHANGED',pinned_inspect_sha256=p['inspect_sha256'],
 questions_not_recomputed=True,questions_placeholder_from_cache_only=True,cache_qualified=False,qualification_attempted=False,
 current_identity_from_launch_resolved_config=True,current_runtime_question_identity_not_saved=True,
 nonquestion_current_sha256=digest(without_questions(current)),nonquestion_cached_sha256=digest(without_questions(cached)),
 mismatches=comparison,mismatch_count=len(comparison),scorer_source_map_pairs=[dict(path=name,current_bound_sha256=run['source_sha256'][name],current_disk_sha256=scorer[name],cached_sha256=cached.get('scorer_sources',{}).get(name)) for name in scorer_names],
 rollout_source_keysets_equal=set(rollout)==set(cached.get('rollout_sources',{})),rollout_source_predicate_exact=True,
 actual_owned_cleanup=dict(accepted=True,original_pipeline_birth_live=False,all_recorded_births_exited=True,owned_session_survivors=0,
 owned_cuda_contexts_remaining=0,recorded_births=len(records),jobs=jobs,reused_session_ids=sorted(set(reused_sessions)),reused_cuda_pids=sorted(set(reused_cuda))),
 zero_formal_ledgers=zero,optimizer_rank_logs=0,scheduler_rank_logs=0,checkpoint_files=0,question_head_fitted=False,
 metadata_sha256=dict(run_manifest=hashlib.sha256(sealed(base/'logs/run-manifest.json')).hexdigest(),resolved_config=hashlib.sha256(sealed(base/'logs/resolved-training-config.json')).hexdigest(),
 feature_manifest=hashlib.sha256(sealed(base/'question-features/manifest.json')).hexdigest(),teacher_identity=hashlib.sha256(sealed(cache/'identity.json')).hexdigest()),
 no_remote_mutation=True,no_tensor_model_qa_or_rollout_read=True,no_cuda_initialization=True,checked_unix=time.time())
out=json.dumps(result,sort_keys=True,allow_nan=False);assert len(out.encode())<=120*1024
print('SIGMA3_TEACHER_FAULT_JSON:'+out,flush=True)
'''


def local_inputs(support):
    values={}
    for name,digest in PINS.items():
        path=Path(name);assert support.sha(path)==digest;values[name]=support.document(path)
    intent=values[str(HANDLE)];ready=values[str(READY)];publication=values[str(PUBLICATION)]
    assert intent['source_commit']==SOURCE and len(intent['source_sha256'])==481 and support.canonical_sha(intent['source_sha256'])==MAP_SHA
    assert intent['base']==BASE and intent.get('saved_checkpoint_restore') is None
    assert (ready['run_id'],ready['execution_id'],ready['source_commit'],ready['base'])==(RUN,EXECUTION,SOURCE,BASE)
    assert publication['source_commit']==publication['remote_commit']==SOURCE and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['source_sha256']==intent['source_sha256'] and publication['source_map_sha256']==MAP_SHA
    for key in ('handle','review','source_publication'):
        record=ready['files'][key];assert support.sha(Path(record['path']))==record['sha256']
    review=support.document(Path(ready['files']['review']['path']))
    assert review['approved'] is True and review['source_commit']==SOURCE and review['source_sha256']==intent['source_sha256']
    inspected=values[str(INSPECT)];assert inspected['action']=='inspect' and inspected['exit_code']==0
    for row in inspected['observations']:
        assert (row['run_id'],row['execution_id'],row['source_commit'],row['pid'],row['start_ticks'])==(RUN,EXECUTION,SOURCE,PID,TICKS)
        assert row['pipeline_alive'] is False and row['cleanup_status']['cleaned'] is True and row['actual_updates_ledger']['exists'] is False
        trainers=[job for job in row['owned_jobs'] if job['name']=='trainer'];assert len(trainers)==1
        assert trainers[0]['exit_code']==1 and trainers[0]['leader']['pid']==TRAINER_PID and trainers[0]['leader']['start_ticks']==TRAINER_TICKS
        assert any('WARMUP_TEACHER_DATA_SAMPLING_OR_SCORING_NAMESPACE_CHANGED' in line for line in trainers[0]['exception_diagnostic'])
    support.verify_source(RUNTIME,SOURCE,intent['source_sha256'],481)
    return dict(intent=intent,run=RUN,execution=EXECUTION,source=SOURCE,base=BASE,pid=PID,ticks=TICKS,trainer_pid=TRAINER_PID,trainer_ticks=TRAINER_TICKS,cache=CACHE,inspect_sha256=PINS[str(INSPECT)])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operator-publication',type=Path,required=True);args=parser.parse_args()
    support=source_support();path=Path(__file__).resolve();relative=str(path.relative_to(OPS));raw=support.sealed(path)
    commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True,timeout=10).strip()
    assert subprocess.check_output(['git','-C',str(OPS),'show',commit+':'+relative],timeout=10)==raw
    publication=support.document(args.operator_publication)
    assert publication['source_commit']==publication['remote_commit']==commit and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    selected=[row for row in publication['files'] if row['path']==relative];assert len(selected)==1 and selected[0]['sha256']==hashlib.sha256(raw).hexdigest()
    payload=local_inputs(support);code=REMOTE+support.SOURCE_CHECK+REMOTE_READ;ast.parse(code)
    spec=importlib.util.spec_from_file_location('pinned_teacher_fault_existing_route',RUNTIME/'reproduction/a800_common/deploy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert support.sha(Path(module.__file__))==payload['intent']['source_sha256']['reproduction/a800_common/deploy.py']
    route=module.ssh_command(code);assert route[-1].startswith('python3 -c ')
    route[-1]=shlex.quote('/data/search-g1/env/bin/python')+' -c '+shlex.quote(code)
    stem='qprobe-sigma3-teacher-identity-fault-read-once';claim=PREP/(stem+'.json');output=claim.with_name(claim.stem+'.result.json')
    request=dict(role='sigma3_teacher_identity_fault_one_metadata_read_claim',run_id=RUN,execution_id=EXECUTION,source_commit=SOURCE,
        operator_commit=commit,operator_sha256=hashlib.sha256(raw).hexdigest(),remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        pinned_evidence_sha256=PINS,no_remote_mutation=True,no_automatic_retry=True,claimed_unix=time.time())
    support.once(claim,request)
    with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
        try:
            completed=subprocess.run(route,input=json.dumps(payload,sort_keys=True,allow_nan=False).encode(),stdout=stdout,stderr=stderr,timeout=60)
            status=dict(exit_code=completed.returncode,status='READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:status=dict(exit_code=None,status='TRANSPORT_UNKNOWN_NO_RETRY',failure_type=type(failure).__name__)
        status.update(stdout_bytes=stdout.tell(),stderr_bytes=stderr.tell());stdout.seek(0);returned=stdout.read(128*1024+1);stderr.seek(0);error=stderr.read(64*1024)
    status['bounded_stderr_sha256']=hashlib.sha256(error).hexdigest();text=error.decode(errors='replace')
    status['remote_exception_frames']=[dict(file=Path(match.group(1)).name[:96],line=int(match.group(2))) for match in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+),',text,re.MULTILINE)][-8:]
    types=re.findall(r'^([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception))(?::|$)',text,re.MULTILINE);status['remote_exception_type']=types[-1].rsplit('.',1)[-1][:96] if types else None
    results=[]
    if status['exit_code']==0 and len(returned)<=128*1024:
        try:
            results=[json.loads(line.split(':',1)[1]) for line in returned.decode().splitlines() if line.startswith('SIGMA3_TEACHER_FAULT_JSON:')]
            assert len(results)==1 and (results[0]['run_id'],results[0]['execution_id'],results[0]['source_commit'])==(RUN,EXECUTION,SOURCE)
            assert results[0]['questions_not_recomputed'] is True and results[0]['qualification_attempted'] is False
        except (AssertionError,ValueError,KeyError,TypeError,UnicodeDecodeError) as failure:results=[];status.update(status='OUTPUT_REJECTED_NO_RETRY',failure_type=type(failure).__name__)
    elif len(returned)>128*1024:status['status']='OUTPUT_LIMIT_EXCEEDED_NO_RETRY'
    support.once(output,dict(request,**status,results=results,observed_unix=time.time()))
    print(json.dumps(dict(receipt=str(output),**status,results=results,no_automatic_retry=True),allow_nan=False))
    return 0 if status['exit_code']==0 and len(results)==1 else 1


if __name__=='__main__':raise SystemExit(main())
