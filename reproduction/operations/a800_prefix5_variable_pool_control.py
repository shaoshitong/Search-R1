"""One authorized full-state boundary switch to ordinary5120 / P10-update2560.

Staging never stops the healthy parent. Boundary handoff is a separate published
entry. CPU prepare qualifies the stopped saved state once, then links its exact
raw actor/probe/optimizer/RNG and candidate cursor into a distinct execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from . import a800_i1_i6_fresh_control as fresh

ROOT, PYTHON, MAX_OUTPUT = fresh.ROOT, fresh.PYTHON, fresh.MAX_OUTPUT
route = fresh.route
_prefix = fresh._prefix.replace("str(b)==PREFIX+h['run_id']", "str(b)==PREFIX+h['run_id']+'-resume-'+h['execution_id']")
COMMON = ("import json,pathlib,sys\nsys.dont_write_bytecode=True\np=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'\n"
          + _prefix + fresh._helpers + r'''
def source(require_receipt=True):
 assert read(b/'preparation/deployment.json')==h
 actual=set()
 for path in repo.rglob('*'):
  if path.is_dir():assert not path.is_symlink() and path.resolve()==path
  else:regular(path);actual.add(str(path.relative_to(repo)))
 assert actual==set(h['source_files'])
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest,'Published source differs: '+name
 assert sha(b/'preparation/human-intent.json')==h['human_intent_sha256']
 human=read(b/'preparation/human-intent.json')
 assert human['role']=='latest_direct_human_variable_candidate_schedule_authorization'
 if require_receipt:
  value=read(b/'preparation/source-receipt.json')
  assert value['source_commit']==h['source_commit'] and value['source_files_verified']==len(h['source_files'])
def emit(value):
 raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<=128*1024;print(raw)
def prepared():
 ended=read(b/'preparation/prepare-exit.json')
 assert ended['exit_code']==0 and ended['run_id']==h['run_id'] and ended['execution_id']==h['execution_id']
 assert not alive(native_identity(ended['identity']))
 c=read(b/'preparation/launch-config.json');assert sha(b/'preparation/launch-config.json')==ended['config_sha256']
 assert c['source_commit']==h['source_commit'] and c['resume_joint']['adapter']=='i1_i6_joint_v1'
 assert sha(b/'preparation/full-state-qualified.json')==ended['qualification_sha256']
 sys.path.insert(0,str(repo))
 from reproduction.prefix_selection200.i1_i6_joint_resume import validate_imported_boundary
 assert validate_imported_boundary(b,c)==ended['completed_outer']
 return c,ended
''')

_stage = fresh.previous.STAGE[len(fresh.previous.COMMON):]
_stage = _stage.replace("assert {'launch-config-template.json','old-actor-deployment.json'}<=set(p['side_files'])",
    "assert set(p['side_files'])=={'human-intent.json'}")
STAGE = COMMON + _stage.replace('A800_ACTORHIDDEN_PREFIX5_EXACT_SOURCE_STAGED', 'A800_VARIABLE_POOL_EXACT_SOURCE_STAGED')

HANDOFF = COMMON + r'''
source()
assert not (b/'preparation/handoff-intent.json').exists(),'Handoff already claimed; inspect its outcome'
parent=pathlib.Path(h['parent_handle']['base']);submit=read(parent/'preparation/submit-receipt.json')
assert alive(native_identity(submit['leader'])),'Parent is not confirmed live; reconcile its actual state'
write(b/'preparation/handoff-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],human_intent_sha256=h['human_intent_sha256'],requested_unix=time.time()),True)
command=['/data/search-g1/env/bin/python','-m','reproduction.operations.a800_prefix5_variable_pool_control','--handoff-worker','--handle',str(b/'preparation/deployment.json')]
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
with (b/'logs/boundary-handoff.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='ONCE_NATIVE_BOUNDARY_HANDOFF_WORKER_STARTED_NOT_ACCEPTED',run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],identity=birth(child.pid),checked_unix=time.time())
write(b/'preparation/handoff-launch.json',value,True);emit(value)
'''

PREPARE = COMMON + r'''
source()
assert not (b/'preparation/prepare-intent.json').exists(),'Prepare already claimed; inspect, never replay'
terminal=read(h['parent_terminal_path']);assert terminal['termination_cause']=='AUTHORIZED_BOUNDARY_HANDOFF'
assert terminal['human_intent_sha256']==h['human_intent_sha256'] and terminal['owned_writers_gone']
write(b/'preparation/prepare-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],checked_unix=time.time(),parent_terminal_sha256=sha(h['parent_terminal_path'])),True)
command=[PYTHON if 'PYTHON' in globals() else '/data/search-g1/env/bin/python','-m','reproduction.operations.a800_prefix5_variable_pool_control','--prepare-worker','--handle',str(b/'preparation/deployment.json')]
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
with (b/'logs/cpu-prepare.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='VARIABLE_POOL_CPU_PREPARE_STARTED_NOT_CUDA_ACCEPTANCE',run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],identity=birth(child.pid),checked_unix=time.time())
write(b/'preparation/prepare-launch.json',value,True);emit(value)
'''

LAUNCH = COMMON + r'''
source();c,ended=prepared()
assert not (b/'preparation/variable-pool-launch-intent.json').exists() and not (b/'preparation/submit-intent.json').exists(),'Launch already claimed'
assert not bound_processes(b,exclude=(os.getpid(),)),'Current namespace writer is still present'
from reproduction.prefix_selection200.launch import submit
write(b/'preparation/variable-pool-launch-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],completed_outer=ended['completed_outer'],effective_from_outer=c['candidate_schedule']['effective_from_outer'],optimizer_reset=False,requested_unix=time.time()),True)
value=submit(b);emit(dict(status='VARIABLE_POOL_FULL_STATE_ONCE_LAUNCHED_NOT_CUDA_ACCEPTANCE',submit=value,completed_outer=ended['completed_outer'],effective_from_outer=c['candidate_schedule']['effective_from_outer']))
'''

_body = fresh._observe
_guard = "assert joint['run_id']==h['run_id'] and joint['source_commit']==h['source_commit']"
assert _body.count(_guard) == 1
_body = _body.replace(_guard, "assert joint['run_id']==h['run_id']\n c=read(b/'preparation/launch-config.json')\n if joint['outer']<=c['resume_joint']['completed_outer']:\n  assert sha(b/'latest-joint.json')==c['resume_joint']['source_joint_sha256']\n else:assert joint['source_commit']==h['source_commit']")
_body = _body.removesuffix('\nemit(v)\n')
OBSERVE = COMMON + _body + r'''
v['handoff_launch']=document(b,'preparation/handoff-launch.json')
v['handoff_exit']=document(b,'preparation/handoff-exit.json')
v['handoff_progress']=document(b,'preparation/handoff-progress.json')
v['full_state_qualification']=None
q=document(b,'preparation/full-state-qualified.json')
if q:v['full_state_qualification']=project(q,('status','completed_outer','actual_adam_steps_per_rank','source_commit','source_execution_id','cuda_restore_verified'))
v['candidate_schedule']=None if c is None else c.get('candidate_schedule')
v['current_candidate_counts']={}
if type(current) is int and 1<=current<=200:
 for name in ('selection.json','prefix/complete.json','probe-score/selection.json','full-pool/complete.json','full/complete.json','probe-update/complete.json'):
  value=document(b/'outers'/('outer-%06d'%current),name)
  if value is not None:v['current_candidate_counts'][name]=project(value,('outer','questions','trajectories','count','candidate_questions','scheduled_candidate_questions','selected_questions','new_adam_steps_per_rank','status','actual_selected_mixed_fraction','candidate_mixed_questions','selected_mixed_questions','fallback_questions'))
v['cpu_qualification_is_not_cuda_restoration']=True
emit(v)
'''

def handoff_worker(handle_path):
    from . import a800_i1_i6_boundary_handoff as boundary
    from reproduction.prefix20_online.launch import once,publish,read
    h=read(handle_path);base=Path(h['base']);parent=Path(h['parent_handle']['base'])
    result=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],identity=fresh.previous._identity(),started_unix=time.time())
    try:
        deadline=time.time()+3600;previous=None
        while True:
            stage=read(parent/'logs/pipeline-status.json');name=stage['stage'];outer=stage['outer']
            submit=read(parent/'preparation/submit-receipt.json');who=boundary.identity(submit['leader'])
            actual=boundary.same(who)
            if actual is None or actual['state'] in ('Z','X'):
                raise RuntimeError('Parent exited before the requested clean boundary; preserve terminal state')
            eligible=(type(outer)is int and 1<=outer<200 and outer%10!=0 and outer not in (5,50,100,150)
                and name.startswith('outer-%06d-actor-'%outer))
            if eligible:
                ledger=read(parent/'logs/process-ledger.json');job=ledger.get('jobs',{}).get(name)
                if job and boundary.same(boundary.identity(job['leader'])):
                    spec={key:h['parent_handle'][key] for key in ('base','run_id','execution_id','source_commit')}
                    spec.update(outer=outer,actor_job_name=name,pipeline_identity=who,actor_identity=boundary.identity(job['leader']),
                        authorization='complete_current_ordinary_actor_then_native_cleanup_and_cpu_joint_commit',
                        human_intent_path=str(base/'preparation/human-intent.json'),human_intent_sha256=h['human_intent_sha256'])
                    boundary.guard(spec)
                    path=base/'preparation/boundary-spec.json';once(path,spec)
                    publish(base/'preparation/handoff-progress.json',dict(result,stage='AUTHENTICATED_ORDINARY_ACTOR_NATIVE_COMPLETION',outer=outer,actor_identity=spec['actor_identity']))
                    value=boundary.execute(path,Path(h['parent_terminal_path']).parent,1800)
                    result.update(exit_code=0,status=value['status'],completed_outer=outer,terminal_proof=value['terminal_proof'],terminal_proof_sha256=value['terminal_proof_sha256'],finished_unix=time.time())
                    break
            current=(outer,name)
            if current!=previous:
                publish(base/'preparation/handoff-progress.json',dict(result,stage='WAITING_ORDINARY_ACTOR_WITHOUT_INTERRUPTING_CURRENT_WORK',parent_stage=name,parent_outer=outer,checked_unix=time.time()));previous=current
            if time.time()>deadline:raise TimeoutError('No eligible ordinary actor boundary within one hour')
            time.sleep(5)
    except BaseException as error:
        result.update(exit_code=1,error_class=type(error).__name__,error=str(error),failed_unix=time.time(),same_onceclaim_must_not_be_replayed=True)
        once(base/'preparation/handoff-exit.json',result);traceback.print_exc();raise
    once(base/'preparation/handoff-exit.json',result)

def prepare_worker(handle_path):
    from reproduction.prefix_selection200 import i1_i6_joint_resume as resume
    from reproduction.prefix20_online.launch import once, publish, read, sha
    h=read(handle_path);base=Path(h['base']);parent=Path(h['parent_handle']['base'])
    result=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],identity=fresh.previous._identity(),started_unix=time.time())
    try:
        original=read(parent/'preparation/launch-config.json');terminal=read(h['parent_terminal_path']);n=terminal['stopped_after_complete_outer']
        if (original['run_id']!=h['run_id'] or original['execution_id']!=h['parent_handle']['execution_id'] or original['source_commit']!=h['parent_handle']['source_commit'] or
                terminal['human_intent_sha256']!=h['human_intent_sha256'] or sha(base/'preparation/human-intent.json')!=h['human_intent_sha256']):
            raise ValueError('Exact live parent and latest human handoff identity differ')
        schedule=dict(effective_from_outer=n+1,legacy_candidates=2560,ordinary_candidates=5120,update_candidates=2560,probe_update_period=10,probe_update_at_final=False,selected_questions=512)
        parent_manifest=resume.source_manifest(parent/'repo',original['source_commit'])
        target_manifest=resume.source_manifest(base/'repo',h['source_commit'],science_source_commit=original['source_commit'])
        delta=dict(candidate_schedule=schedule,human_intent_sha256=h['human_intent_sha256'],only_authorized_algorithm_delta='ordinary_candidate_pool_5120_periodic2560')
        proof=resume.compatibility_proof(parent_manifest,target_manifest,authorized_candidate_delta=delta)
        proof_path=base/'preparation/source-compatibility.json';once(proof_path,proof)
        qualification=base/'preparation/full-state-qualified.json'
        publish(base/'preparation/prepare-progress.json',dict(result,stage='ONCE_FULL_SHA_FINITE_SAVED_BOUNDARY_QUALIFICATION',completed_outer=n))
        from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import qualify
        qualify(parent,outer=n,terminal_proof=Path(h['parent_terminal_path']),
            terminal_sha=sha(h['parent_terminal_path']),output=qualification)
        c=dict(original,base=str(base),execution_id=h['execution_id'],source_commit=h['source_commit'],science_source_commit=original['source_commit'],
            source_compatibility=str(proof_path),source_compatibility_sha256=sha(proof_path),candidate_schedule=schedule,
            candidate_schedule_authorization=str(base/'preparation/human-intent.json'),candidate_schedule_authorization_sha256=h['human_intent_sha256'],
            pool_state_path=str(base/'question-pool/state.json'),prefix_probe_receipt=str(base/'probe/latest.json'),
            fresh_actor_optimizer_scheduler_rng_sampler=False,retained_warmup_model_only=False,full_state_resume=True)
        c['resume_joint']=dict(adapter=resume.ADAPTER,completed_outer=n,source_base=str(parent),source_execution_id=original['execution_id'],source_commit=original['source_commit'],
            source_config_sha256=sha(parent/'preparation/launch-config.json'),source_joint_sha256=sha(parent/'latest-joint.json'),
            terminal_proof=h['parent_terminal_path'],terminal_proof_sha256=sha(h['parent_terminal_path']),qualification_proof=str(qualification),qualification_proof_sha256=sha(qualification))
        c.pop('resume_import',None);c.pop('resume_pending_probe',None)
        mapping=fresh._idle_mapping(base,result)
        for key in ('physical_gpu_indices','physical_gpu_uuids','auxiliary_gpu_indices','auxiliary_gpu_uuids'):
            c[key]=mapping[key]
        # Imports are exact same-filesystem hardlinks; existing model/tensor bytes stay remote.
        once(base/'preparation/launch-config.json',c)
        imported=resume.import_complete(base)
        from reproduction.prefix_selection200.launch import validate_config
        validate_config(c,base)
        result.update(exit_code=0,status='VARIABLE_POOL_FULL_STATE_CPU_IMPORT_COMPLETE_NOT_CUDA_ACCEPTANCE',completed_outer=n,effective_from_outer=n+1,
            qualification_sha256=sha(qualification),config_sha256=sha(base/'preparation/launch-config.json'),import_receipt_sha256=sha(base/'preparation/resume-joint-receipt.json'),completed_unix=time.time())
    except BaseException as error:
        result.update(exit_code=1,error_class=type(error).__name__,error=str(error),failed_unix=time.time(),same_onceclaim_must_not_be_replayed=True)
        once(base/'preparation/prepare-exit.json',result);traceback.print_exc();raise
    once(base/'preparation/prepare-exit.json',result)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',choices=('stage','handoff','prepare','launch','observe'))
    parser.add_argument('--handle',type=Path,required=True);parser.add_argument('--payload',type=Path);parser.add_argument('--prepare-worker',action='store_true');parser.add_argument('--handoff-worker',action='store_true');args=parser.parse_args()
    if args.prepare_worker:prepare_worker(args.handle);return
    if args.handoff_worker:handoff_worker(args.handle);return
    if args.action is None or (args.action=='stage' and args.payload is None):parser.error('Action and stage payload are required')
    h=json.loads(args.handle.read_text());payload=dict(handle=h)
    if args.action=='stage':payload.update(json.loads(args.payload.read_text()))
    began=time.time();code=dict(stage=STAGE,handoff=HANDOFF,prepare=PREPARE,launch=LAUNCH,observe=OBSERVE)[args.action]
    try:
        p=subprocess.run(route(code),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
        receipt=dict(action=args.action,checked_unix=time.time(),elapsed_seconds=time.time()-began,exit_code=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode(),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'])
    except subprocess.TimeoutExpired as error:
        receipt=dict(action=args.action,checked_unix=time.time(),exit_code=None,timed_out=True,remote_action_outcome_unknown=True,mutation_must_not_be_replayed=True,stdout=(error.stdout or b'').decode(errors='replace'),stderr=(error.stderr or b'').decode(errors='replace'))
    out=ROOT/'preparation'/('a800-prefix5-variable-pool-'+args.action+'-'+str(time.time_ns())+'.json');out.write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt.get('timed_out') or receipt['exit_code']:raise RuntimeError('Inspect actual receipt before any retry: '+str(out))
    if len(p.stdout)>MAX_OUTPUT:raise ValueError('Bounded output exceeded')
    print(json.dumps(dict(receipt=str(out),result=json.loads(p.stdout))))

if __name__=='__main__':main()
