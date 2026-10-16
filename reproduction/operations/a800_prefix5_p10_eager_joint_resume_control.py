"""Exact-source, one-execution recovery of A800 native joint27 with eager rollout.

The parent actor/probe/optimizer/scheduler/RNG/pool is restored, never reset.
Qualified prefix/score28 are reused; failed continuation partials are preserved
at their parent paths. This controller cannot stop tasks or run evaluations.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from . import a800_actorhidden_prefix5_p10_control as original
from . import a800_actorhidden_prefix5_p10_observer as scalar_observer
from . import a800_prefix5_p10_pending_probe_control as pending

ROOT, PYTHON, MAX_OUTPUT = original.ROOT, original.PYTHON, original.MAX_OUTPUT
route = original.route
PARENT_RUN = '387d3f6e228a488f90c418e2ba47e28e'
PARENT_EXECUTION = 'cb12bf6ab10b46b0a72f1f0c2e753d78'
PARENT_SOURCE = '858381986d3332898c4e8aa8e5c31e966363ceea'
PARENT_CONFIG_SHA = '3c11e74d379b7a95163aacf86bd9554b7165aa8dd8bafb3bc2fc013f337e0e17'
PARENT_JOINT_SHA = '5fe53575eea47c2393b2925e0c108ba0164787429a0528f637bfeb34a2341679'
QUALIFIED_STATUS = 'NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS'

# Keep the original exact source manifest, old411c provenance and authenticated
# process checks. Only the direct recovery-parent pins/namespace differ.
COMMON = pending.COMMON[:pending.COMMON.rindex('def qualification():')]
for before, after in ((pending.PARENT_EXECUTION, PARENT_EXECUTION),
                      (pending.PARENT_SOURCE, PARENT_SOURCE),
                      (pending.PARENT_CONFIG_SHA, PARENT_CONFIG_SHA)):
    COMMON = COMMON.replace(before, after)
_parent_base_assert = "assert str(parent)==PREFIX+h['run_id'] and parent!=b and parent!=old"
assert COMMON.count(_parent_base_assert) == 1
COMMON = COMMON.replace(_parent_base_assert,
    "assert str(parent)==PREFIX+h['run_id']+'-resume-'+parent_h['execution_id'] and parent!=b and parent!=old")
COMMON += r'''
assert h['resume_parent_config_sha256']=='3c11e74d379b7a95163aacf86bd9554b7165aa8dd8bafb3bc2fc013f337e0e17'
def qualification():
 ended=read(b/'preparation/prepare-exit.json')
 assert ended['exit_code']==0 and ended['run_id']==h['run_id'] and ended['execution_id']==h['execution_id']
 assert ended['source_commit']==h['source_commit'] and not alive(native_identity(ended['identity']))
 qpath=b/'preparation/native-a800-joint-qualified.json'
 assert sha(qpath)==ended['qualification_sha256']
 q=read(qpath)
 assert q['status']=='NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS'
 assert q['run_id']==h['run_id'] and q['source_base']==str(parent)
 assert q['source_execution_id']==parent_h['execution_id'] and q['source_commit']==parent_h['source_commit']
 c=read(b/'preparation/launch-config.json')
 assert sha(b/'preparation/launch-config.json')==ended['config_sha256']
 assert c['run_id']==h['run_id'] and c['execution_id']==h['execution_id']
 assert c['source_commit']==h['source_commit'] and c['base']==str(b)
 assert c['rollout_enforce_eager'] is True and 'resume_pending_probe' not in c
 spec=c['resume_joint']
 assert spec['adapter']=='native_a800_joint_v1' and spec['completed_outer']==27
 assert spec['source_base']==str(parent) and spec['source_execution_id']==parent_h['execution_id']
 assert spec['source_commit']==parent_h['source_commit'] and spec['source_config_sha256']==h['resume_parent_config_sha256']
 assert spec['checkpoint_cpu_proof']==str(qpath) and spec['checkpoint_cpu_proof_sha256']==sha(qpath)
 assert spec['terminal_proof']==str(terminal_file) and spec['terminal_proof_sha256']==h['terminal_failure_sha256']
 assert spec['inherited_prefix_outer']==28
 assert spec['source_joint_sha256']=='5fe53575eea47c2393b2925e0c108ba0164787429a0528f637bfeb34a2341679'
 assert sha(b/'preparation/resume-joint-receipt.json')==ended['import_receipt_sha256']
 assert sha(b/'preparation/resume-import-receipt.json')==ended['phase_import_receipt_sha256']
 return q,c
'''

STAGE = COMMON + original.STAGE[len(original.COMMON):]
STAGE = STAGE.replace("assert {'launch-config-template.json','old-actor-deployment.json'}<=set(p['side_files'])",
    "assert {'launch-config-template.json','old-actor-deployment.json','resume-parent-deployment.json',terminal_file.name}<=set(p['side_files'])")
STAGE = STAGE.replace('A800_ACTORHIDDEN_PREFIX5_EXACT_SOURCE_STAGED', 'A800_NATIVE_JOINT27_EAGER_EXACT_SOURCE_STAGED')

PREPARE = COMMON + r'''
source();parent_terminal()
assert not (b/'preparation/prepare-intent.json').exists(),'Prepare onceclaim exists; inspect, never replay'
assert not (b/'preparation/prepare-exit.json').exists()
write(b/'preparation/prepare-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],checked_unix=time.time(),completed_outer=27,
 full_state_resume=True,old_actor_stop_requested=False),True)
command=['/data/search-g1/env/bin/python','-m','reproduction.operations.a800_prefix5_p10_eager_joint_resume_control',
 '--prepare-worker','--handle',str(b/'preparation/deployment.json')]
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
with (b/'logs/cpu-prepare.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
who=birth(child.pid);assert who['sid']==who['pid'] and who['uid']==os.getuid()
v=dict(status='NATIVE_JOINT27_CPU_PREPARE_ONCE_LAUNCHED_NOT_ACCEPTED',run_id=h['run_id'],
 execution_id=h['execution_id'],source_commit=h['source_commit'],identity=who,started_unix=time.time(),
 completed_outer=27,old_actor_stop_requested=False,rollout_enforce_eager=True)
write(b/'preparation/prepare-launch.json',v,True);emit(v)
'''

LAUNCH = COMMON + r'''
source();qualified,config=qualification();terminal=parent_terminal()
assert not (b/'preparation/eager-joint-launch-intent.json').exists(),'Launch intent exists; inspect, never replay'
assert not (b/'preparation/submit-intent.json').exists(),'Native once submit already claimed'
assert not bound_processes(b,exclude=(os.getpid(),)),'A new namespace writer remains active'
sys.path.insert(0,str(repo))
from reproduction.prefix_selection200.launch import validate_config
from reproduction.prefix_selection200.native_joint_resume import validate_imported_boundary
validate_config(config,b);assert validate_imported_boundary(b,config)==27
write(b/'preparation/eager-joint-launch-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],config_sha256=sha(b/'preparation/launch-config.json'),
 requested_unix=time.time(),retained_complete_outer=27,retained_actor_adam_per_rank=[540]*4,
 retained_probe_adam_per_rank=[100]*4,actor_updates_to_reexecute=0,
 inherited_prefix_outer=28,formal_actor_fresh0=False,optimizer_reset=False,
 rollout_enforce_eager=True,parent_terminal=terminal),True)
import fcntl
from reproduction.a800.owned_processes import identity
with (b/'preparation/submit-control.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (b/'preparation/submit-intent.json').exists()
 write(b/'preparation/submit-intent.json',dict(run_id=h['run_id'],source_commit=h['source_commit'],
  config_sha256=sha(b/'preparation/launch-config.json'),created_unix=time.time()),True)
 command=[str(pathlib.Path(config['environment_root'])/'bin/python'),'-m',
  'reproduction.prefix_selection200.launch','run','--base',str(b)]
 env=dict(os.environ,PYTHONPATH=str(repo),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
 with (b/'logs/native-pipeline.log').open('xb') as log:
  child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,
   stderr=subprocess.STDOUT,start_new_session=True)
 who=identity(child.pid);assert who['sid']==child.pid and who['uid']==os.getuid()
 value=dict(status='ONCE_LAUNCHED_NOT_TRAINING_ACCEPTANCE',leader=who,run_id=h['run_id'],
  source_commit=h['source_commit'],command=command,launched_unix=time.time(),
  config_sha256=sha(b/'preparation/launch-config.json'),execution_id=h['execution_id'],
  resume_kind='native_a800_joint_v1',retained_complete_outer=27,rollout_enforce_eager=True)
 write(b/'preparation/submit-receipt.json',value,True)
emit(dict(status='NATIVE_JOINT_SUBMIT_RETURNED_NOT_CUDA_RESTORE_ACCEPTANCE',submit=value,
 parent_terminal=terminal,checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit']))
'''

_OBSERVE_BODY = original.OBSERVE[len(original.COMMON):]
_OLD_SOURCE_GUARD = "assert joint['run_id']==h['run_id'] and joint['source_commit']==h['source_commit']"
assert _OBSERVE_BODY.count(_OLD_SOURCE_GUARD) == 1
_OBSERVE_BODY = _OBSERVE_BODY.replace(_OLD_SOURCE_GUARD, "assert joint['run_id']==h['run_id']\n if joint['outer']==27:\n  assert joint['source_commit']==parent_h['source_commit']\n  imported=read(b/'preparation/resume-joint-receipt.json')\n  assert sha(b/'latest-joint.json')==sha(parent/'latest-joint.json')=='5fe53575eea47c2393b2925e0c108ba0164787429a0528f637bfeb34a2341679'\n else:assert joint['outer']>=28 and joint['source_commit']==h['source_commit']")
_OBSERVE_BODY = _OBSERVE_BODY.removesuffix('\nemit(v)\n')
OBSERVE = COMMON + _OBSERVE_BODY + r'''
v['prepare_progress']=document(b,'preparation/prepare-progress.json')
qpath=b/'preparation/native-a800-joint-qualified.json'
q=document(b,'preparation/native-a800-joint-qualified.json')
v['native_joint_cpu_qualification']=None
qualified_metadata=False
sealed_files={};history_origins={};inherited=None
if q is not None:
 if q.get('status')=='NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS':
  assert q['run_id']==h['run_id'] and q['source_base']==str(parent)
  assert q['source_execution_id']==parent_h['execution_id'] and q['source_commit']==parent_h['source_commit']
  keys=('status','adapter','run_id','source_base','source_execution_id','source_commit','completed_outer',
   'actor_adam_per_rank','scheduler_completed_outer','probe_completed_outer','probe_adam_per_rank',
   'owned_births_all_gone','source_joint_sha256')
  qualified_metadata=True
  sealed_files={row['path']:row['sha256'] for row in q['verified_files']}
 else:
  keys=('status','run_id','error_type','error','elapsed_seconds','checked_unix','failed_unix')
 v['native_joint_cpu_qualification']={key:q[key] for key in keys if key in q}
if (b/'preparation/resume-joint-receipt.json').exists():
 inherited=read(b/'preparation/resume-joint-receipt.json')
 for row in inherited['history_provenance']:
  assert row['target'] not in history_origins,'Duplicate inherited history target'
  history_origins[row['target']]=row
 v['native_joint_import']={key:inherited[key] for key in ('status','run_id','execution_id','source_commit',
  'source_base','source_execution_id','completed_outer','actor_adam_per_rank','probe_adam_per_rank',
  'full_state_import','optimizer_reset','cuda_restore_verified') if key in inherited}
else:v['native_joint_import']=None
def inherited_origin(target,outer,value):
 assert qualified_metadata,'Inherited history lacks an accepted CPU proof'
 row=history_origins[str(target)]
 assert row['outer']==outer and row.get('run_id',row.get('run'))==h['run_id']
 origin=canonical_path(row['source']);assert str(origin) in sealed_files
 assert row['sha256']==sealed_files[str(origin)]==sha(origin)==sha(target)
 assert row['source_commit']==value['source_commit']
 assert value['run_id']==h['run_id'] and value['outer']==outer
 return origin
'''
_EXTRA = scalar_observer.EXTRA
_HISTORY_GUARD = "assert completed_joint['run_id']==h['run_id'] and completed_joint['source_commit']==h['source_commit']"
assert _EXTRA.count(_HISTORY_GUARD) == 1
_EXTRA = _EXTRA.replace(_HISTORY_GUARD, "assert completed_joint['run_id']==h['run_id']\n if outer<=27:inherited_origin(root/'complete.json',outer,completed_joint)\n else:assert completed_joint['source_commit']==h['source_commit']")
_QA_MARKER = "qa=document(root,'validation/complete.json')"
assert _EXTRA.count(_QA_MARKER) == 1
_EXTRA = _EXTRA.replace(_QA_MARKER, "qa=document(root,'validation/complete.json')\n if qa is not None and outer<=27:\n  target=root/'validation/complete.json'\n  assert qualified_metadata and str(target) in history_origins\n  origin=history_origins[str(target)]\n  assert origin['sha256']==sealed_files[origin['source']]==sha(origin['source'])==sha(target)")
OBSERVE += _EXTRA + r'''
if q is not None and not qualified_metadata and joint is None:
 v['complete_outer']=None;v['probe_clock_contract']=None
 v['recovery_boundary_status']='UNQUALIFIED_OR_FAILED_CPU_PROOF_BOUNDARY_UNKNOWN'
v['cpu_qualification_is_not_cuda_restoration']=True
v['operator_contract']='a800_p10_eager_native_joint_bounded_metadata_v1'
emit(v)
'''

_sha_file, _write_once, _publish = pending._sha_file, pending._write_once, pending._publish
_idle_mapping = pending._idle_mapping


def prepare_worker(handle_path):
    """CPU qualify/import first, then record actual idle mapping for one launch."""
    h = json.loads(Path(handle_path).read_text())
    base=Path(h['base']); parent_h=h['resume_parent_handle']; parent=Path(parent_h['base'])
    receipt=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
        identity=original._identity(),checked_unix=time.time(),retained_complete_outer=27,
        actor_updates_to_reexecute=0,optimizer_reset=False,warmup_checkpoint_read_again=False)
    try:
        prefix='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'
        if (h['run_id']!=PARENT_RUN or parent_h['execution_id']!=PARENT_EXECUTION
                or parent_h['source_commit']!=PARENT_SOURCE
                or str(base)!=prefix+PARENT_RUN+'-resume-'+h['execution_id']
                or str(parent)!=prefix+PARENT_RUN+'-resume-'+PARENT_EXECUTION
                or h['execution_id']==PARENT_EXECUTION or h['source_commit']==PARENT_SOURCE
                or h['resume_parent_config_sha256']!=PARENT_CONFIG_SHA):
            raise ValueError('Require exact same-run complete27 state and a new source/execution')
        source_config_path=parent/'preparation/launch-config.json'
        if _sha_file(source_config_path)!=PARENT_CONFIG_SHA:
            raise ValueError('Parent actual configuration bytes changed')
        source_config=json.loads(source_config_path.read_text())
        if any(source_config[key]!=value for key,value in dict(base=str(parent),run_id=PARENT_RUN,
                execution_id=PARENT_EXECUTION,source_commit=PARENT_SOURCE).items()):
            raise ValueError('Parent configuration identity differs')
        template=json.loads(Path(h['template_config_path']).read_text())
        allowed={'base','execution_id','source_commit','pool_state_path','prefix_probe_receipt',
            'resume_joint','resume_pending_probe','continue_reuse','rollout_enforce_eager',
            'physical_gpu_indices','physical_gpu_uuids','auxiliary_gpu_indices','auxiliary_gpu_uuids'}
        for key,value in template.items():
            if key not in allowed and value!=source_config.get(key):
                raise ValueError('Cannot change scientific/input field in recovery: '+key)
        if any(template.get(key)!=h[key] for key in ('base','execution_id','source_commit','run_id')):
            raise ValueError('Template must bind the selected recovery source/execution')
        if template.get('rollout_enforce_eager') is not True:
            raise ValueError('Only the authorized eager-rollout execution repair is allowed')
        terminal=Path(h['terminal_failure_file'])
        if _sha_file(terminal)!=h['terminal_failure_sha256']:
            raise ValueError('Actual parent terminal proof changed')
        spec=dict(template['resume_joint'])
        expected=dict(adapter='native_a800_joint_v1',completed_outer=27,source_base=str(parent),
            source_execution_id=PARENT_EXECUTION,source_commit=PARENT_SOURCE,source_config_sha256=PARENT_CONFIG_SHA,
            terminal_proof=str(terminal),terminal_proof_sha256=h['terminal_failure_sha256'],inherited_prefix_outer=28)
        if any(spec.get(key)!=value for key,value in expected.items()):
            raise ValueError('Native joint recovery specification differs from complete27 boundary')
        qualified_path=base/'preparation/native-a800-joint-qualified.json'
        if qualified_path.exists():
            raise FileExistsError('CPU qualifier once output exists; do not rerun')
        _publish(base/'preparation/prepare-progress.json',dict(receipt,stage='CPU_FULL_SHA_FINITE_NATIVE_ACTOR_PROBE_JOINT27_QUALIFICATION'))
        command=[PYTHON,'-m','reproduction.prefix_selection200.checkpoint_native_a800_joint',
            '--base',str(parent),'--expected-run',PARENT_RUN,'--expected-execution',PARENT_EXECUTION,
            '--expected-source',PARENT_SOURCE,'--expected-config-sha256',PARENT_CONFIG_SHA,
            '--terminal-proof',str(terminal),'--terminal-proof-sha256',h['terminal_failure_sha256'],
            '--output',str(qualified_path)]
        result=subprocess.run(command,cwd=base/'repo',env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
            PYTHONDONTWRITEBYTECODE='1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if result.returncode:
            print(result.stderr.decode(errors='replace'),flush=True)
            raise RuntimeError('Native joint CPU qualifier exited '+str(result.returncode))
        qualified=json.loads(qualified_path.read_text())
        if (qualified.get('status')!=QUALIFIED_STATUS or qualified.get('run_id')!=PARENT_RUN
                or qualified.get('source_base')!=str(parent) or qualified.get('source_execution_id')!=PARENT_EXECUTION
                or qualified.get('source_commit')!=PARENT_SOURCE or qualified.get('source_joint_sha256')!=PARENT_JOINT_SHA):
            raise ValueError('CPU qualification does not seal the actual native joint27 parent')
        spec.update(checkpoint_cpu_proof=str(qualified_path),checkpoint_cpu_proof_sha256=_sha_file(qualified_path),
            source_joint_sha256=PARENT_JOINT_SHA)
        config=copy.deepcopy(source_config)
        config.pop('resume_pending_probe',None)
        config.update(base=str(base),execution_id=h['execution_id'],source_commit=h['source_commit'],
            pool_state_path=str(base/'question-pool/state.json'),prefix_probe_receipt=str(base/'probe/latest.json'),
            rollout_enforce_eager=True,resume_joint=spec)
        whitelist=qualified.get('continuation_rank_whitelist')
        if not isinstance(whitelist,list):
            raise ValueError('CPU qualification must explicitly identify accepted continuation ranks')
        if (any(type(row.get('rank')) is not int or not 0<=row['rank']<4 for row in whitelist)
                or len({row['rank'] for row in whitelist})!=len(whitelist)):
            raise ValueError('Qualified continuation rank whitelist has duplicate or invalid ranks')
        config['continue_reuse']=dict(outer=28,source_base=str(parent),source_config_sha256=PARENT_CONFIG_SHA,
            imported_receipts_sha256={str(row['rank']):row['receipt_sha256'] for row in whitelist})
        from reproduction.prefix_selection200.launch import validate_config
        validate_config(config,base)
        config_path=base/'preparation/launch-config.json'
        _write_once(config_path,config)
        receipt['pre_gpu_config_sha256']=_sha_file(config_path)
        _publish(base/'preparation/prepare-progress.json',dict(receipt,stage='IMPORTING_FULL_JOINT27_AND_QUALIFIED_PREFIX_SCORE28'))
        from reproduction.prefix_selection200.native_joint_resume import import_complete,validate_imported_boundary
        imported=import_complete(base)
        assert validate_imported_boundary(base,config)==27
        # No CUDA model has been started. Only after CPU state import do we
        # choose actual idle cards; native_allocation rechecks under UUID lease.
        mapping=_idle_mapping(base,receipt)
        for key in ('physical_gpu_indices','physical_gpu_uuids','auxiliary_gpu_indices','auxiliary_gpu_uuids'):
            config[key]=mapping[key]
        validate_config(config,base)
        _publish(config_path,config)
        assert validate_imported_boundary(base,config)==27
        joint_receipt=base/'preparation/resume-joint-receipt.json'
        phase_receipt=base/'preparation/resume-import-receipt.json'
        receipt.update(exit_code=0,status='NATIVE_A800_JOINT27_EAGER_CPU_PREPARE_COMPLETE',
            qualification_sha256=_sha_file(qualified_path),config_sha256=_sha_file(config_path),
            import_receipt_sha256=_sha_file(joint_receipt),phase_import_receipt_sha256=_sha_file(phase_receipt),
            idle_mapping_sha256=_sha_file(base/'preparation/idle-mapping.json'),
            actor_adam_per_rank=[540]*4,probe_adam_per_rank=[100]*4,scheduler_completed_outer=27,
            inherited_prefix_outer=28,rollout_enforce_eager=True,checkpoint_cp5_deserializations=0,
            cpu_qualified_not_cuda_restored=True,completed_unix=time.time())
        _write_once(base/'preparation/prepare-complete.json',receipt)
        _publish(base/'preparation/prepare-progress.json',dict(receipt,stage='CPU_IMPORT_AND_IDLE_MAPPING_COMPLETE_WAITING_ONCE_LAUNCH'))
    except BaseException as error:
        receipt.update(exit_code=1,error_class=type(error).__name__,completed_unix=time.time(),
            failed_namespace_preserved=True,same_onceclaim_must_not_be_replayed=True)
        _write_once(base/'preparation/prepare-exit.json',receipt)
        traceback.print_exc();raise
    _write_once(base/'preparation/prepare-exit.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('stage','prepare','launch','observe'))
    parser.add_argument('--handle',type=Path,required=True)
    parser.add_argument('--payload',type=Path)
    parser.add_argument('--prepare-worker',action='store_true')
    args=parser.parse_args()
    if not args.handle.is_absolute():parser.error('--handle must be absolute')
    if args.prepare_worker:
        prepare_worker(args.handle);return
    if args.action is None or (args.action=='stage' and args.payload is None):
        parser.error('--action is required; stage also requires --payload')
    h=json.loads(args.handle.read_text());payload=dict(handle=h)
    if args.action=='stage':
        extra=json.loads(args.payload.read_text())
        if 'handle' in extra:raise ValueError('Stage payload cannot replace the selected deployment')
        payload.update(extra)
    code=dict(stage=STAGE,prepare=PREPARE,launch=LAUNCH,observe=OBSERVE)[args.action]
    started=time.time()
    try:
        result=subprocess.run(route(code),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,timeout=65)
        record=dict(action=args.action,checked_unix=time.time(),elapsed_seconds=time.time()-started,
            exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())
    except subprocess.TimeoutExpired as error:
        record=dict(action=args.action,checked_unix=time.time(),elapsed_seconds=time.time()-started,
            exit_code=None,timed_out=True,remote_action_outcome_unknown=True,mutation_must_not_be_replayed=True,
            stdout=(error.stdout or b'').decode(errors='replace'),stderr=(error.stderr or b'').decode(errors='replace'))
    record.update(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
        remote_mutations=args.action!='observe')
    out=ROOT/'preparation'/('a800-prefix5-p10-eager-joint-'+str(args.action)+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(record,indent=2)+'\n')
    if record.get('timed_out') or record.get('exit_code'):
        raise RuntimeError('Unknown or failed action; never repeat onceclaim. Receipt '+str(out))
    if len(result.stdout)>MAX_OUTPUT:raise ValueError('Output exceeds128KiB; retain unknown '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(result.stdout))))


if __name__=='__main__':main()
