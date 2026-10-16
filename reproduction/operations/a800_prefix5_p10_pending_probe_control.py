"""One isolated pending-probe recovery after saved actor10, without resetting it.

This controller has no stop action. CPU qualification and import precede one
new execution's admission/submission. Its observer reads existing metadata,
never checkpoint/tensor payloads and never triggers an evaluation.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from . import a800_actorhidden_prefix5_p10_control as previous
from . import a800_actorhidden_prefix5_p10_observer as scalar_observer

ROOT = previous.ROOT
PYTHON = previous.PYTHON
MAX_OUTPUT = previous.MAX_OUTPUT
CONTROL_MODULE = 'reproduction.operations.a800_prefix5_p10_pending_probe_control'
PARENT_RUN = '387d3f6e228a488f90c418e2ba47e28e'
PARENT_EXECUTION = '90aea944c40d41eab4cf84efac02b22a'
PARENT_SOURCE = '31f616264ceeb4f0e4d33b39bd85df1214afc15d'
PARENT_CONFIG_SHA = '0f9fd64fbc28a1d849f7716030dd4f960e990ed9d50fd9f2b0ffe0241e3d2011'
route = previous.route

_BASE_ASSERT = "assert str(b)==PREFIX+h['run_id'] and h.get('newbase',str(b))==str(b)"
assert previous.COMMON.count(_BASE_ASSERT) == 1
COMMON = previous.COMMON.replace(_BASE_ASSERT,
    "assert str(b)==PREFIX+h['run_id']+'-resume-'+h['execution_id'] and h.get('newbase',str(b))==str(b)")
COMMON += r'''
parent_h=h['resume_parent_handle'];parent=canonical_path(parent_h['base'])
assert h['run_id']==parent_h['run_id']=='387d3f6e228a488f90c418e2ba47e28e'
assert parent_h['execution_id']=='90aea944c40d41eab4cf84efac02b22a'
assert parent_h['source_commit']=='31f616264ceeb4f0e4d33b39bd85df1214afc15d'
assert h['execution_id']!=parent_h['execution_id'] and h['source_commit']!=parent_h['source_commit']
assert str(parent)==PREFIX+h['run_id'] and parent!=b and parent!=old
assert re.fullmatch('[0-9a-f]{64}',h['resume_parent_handle_sha256'])
assert re.fullmatch('[0-9a-f]{64}',h['terminal_failure_sha256'])
terminal_file=canonical_path(h['terminal_failure_file'])
assert terminal_file.parent==b/'preparation'
def verify_parent():
 staged=b/'preparation/resume-parent-deployment.json'
 assert sha(staged)==h['resume_parent_handle_sha256'] and read(staged)==parent_h
 # Remote deployment can be canonical JSON while the staged original is pretty.
 assert read(parent/'preparation/deployment.json')==parent_h
 assert sha(parent/'preparation/launch-config.json')=='0f9fd64fbc28a1d849f7716030dd4f960e990ed9d50fd9f2b0ffe0241e3d2011'
 assert sha(terminal_file)==h['terminal_failure_sha256']
 original=read(terminal_file)
 assert isinstance(original,dict),'Actual terminal proof must be a retained metadata object'
 return original
_previous_source=source
def source(require_receipt=True):
 _previous_source(require_receipt);verify_parent()
def parent_terminal():
 verify_parent()
 launched=read(parent/'preparation/submit-receipt.json')
 who=native_identity(launched['leader'])
 assert not alive(who),'Parent pipeline remains live; no recovery launch'
 assert not bound_processes(parent),'A parent-owned writer remains live'
 ledger=read(parent/'logs/process-ledger.json');assert ledger['run_id']==h['run_id']
 count=0
 for job in ledger.get('jobs',{}).values():
  leader=job.get('leader')
  if leader is not None:assert not alive(native_identity(leader)),'A parent job leader remains live'
  for item in job.get('ledger',{}).values():
   assert not alive(native_identity(item)),'A recorded parent birth remains live'
   count+=1
 cleanup=read(parent/'logs/cleanup-status.json')
 assert cleanup['run_id']==h['run_id'] and cleanup.get('cleaned') is True
 return dict(all_parent_owned_processes_gone=True,parent_run_id=h['run_id'],
  parent_execution_id=parent_h['execution_id'],parent_source_commit=parent_h['source_commit'],
  wrapper=who,recorded_owned_births_checked=count,terminal_proof=str(terminal_file),
  terminal_proof_sha256=h['terminal_failure_sha256'],cleanup=cleanup,checked_unix=time.time())
def qualification():
 ended=read(b/'preparation/prepare-exit.json')
 assert ended['exit_code']==0 and ended['run_id']==h['run_id'] and ended['execution_id']==h['execution_id']
 assert ended['source_commit']==h['source_commit'] and not alive(native_identity(ended['identity']))
 qpath=b/'preparation/native-pending-probe-qualified.json'
 assert sha(qpath)==ended['qualification_sha256']
 q=read(qpath)
 assert q['status']=='QUALIFIED_NATIVE_PENDING_PROBE' and q['run_id']==h['run_id']
 assert q['source_base']==str(parent) and q['source_execution_id']==parent_h['execution_id']
 assert q['source_commit']==parent_h['source_commit'] and q['owned_births_all_gone'] is True
 assert q['completed_joint_outer']==9 and q['pending_outer']==10 and q['actor_completed_outer']==10
 assert q['actor_adam_per_rank']==[200]*4 and q['scheduler_completed_outer']==10
 assert q['probe_completed_outer']==0 and q['probe_adam_per_rank']==[0]*4
 assert q['probe_update_new_adam_per_rank']==[0]*4
 c=read(b/'preparation/launch-config.json')
 assert sha(b/'preparation/launch-config.json')==ended['config_sha256']
 assert c['run_id']==h['run_id'] and c['execution_id']==h['execution_id']
 assert c['source_commit']==h['source_commit'] and c['base']==str(b)
 spec=c['resume_pending_probe']
 assert spec['source_base']==str(parent) and spec['source_execution_id']==parent_h['execution_id']
 assert spec['source_commit']==parent_h['source_commit']
 assert spec['source_config_sha256']=='0f9fd64fbc28a1d849f7716030dd4f960e990ed9d50fd9f2b0ffe0241e3d2011'
 assert spec['qualification_proof']==str(qpath) and spec['qualification_proof_sha256']==sha(qpath)
 assert spec['terminal_proof']==str(terminal_file) and spec['terminal_proof_sha256']==h['terminal_failure_sha256']
 assert spec['completed_joint_outer']==9 and spec['pending_outer']==10 and not c.get('resume_joint')
 imported=read(b/'preparation/resume-pending-probe-receipt.json')
 assert sha(b/'preparation/resume-pending-probe-receipt.json')==ended['import_receipt_sha256']
 assert imported['status']=='COMMITTED_ACTOR_PENDING_PROBE_IMPORTED_NO_RESET'
 assert imported['run_id']==h['run_id'] and imported['execution_id']==h['execution_id']
 assert imported['source_commit']==h['source_commit'] and imported['actor_adam_per_rank']==[200]*4
 assert imported['probe_adam_per_rank']==[0]*4 and imported['actor_updates_to_reexecute']==0
 assert imported['candidates_to_resample']==0 and imported['optimizer_reset'] is False
 return q,c
'''

# The staged archive extraction, exact source inventory/hash and old411c asset
# binding are unchanged. Only this recovery namespace's additional pins differ.
STAGE = COMMON + previous.STAGE[len(previous.COMMON):]
STAGE = STAGE.replace("assert {'launch-config-template.json','old-actor-deployment.json'}<=set(p['side_files'])",
    "assert {'launch-config-template.json','old-actor-deployment.json','resume-parent-deployment.json',terminal_file.name}<=set(p['side_files'])")
STAGE = STAGE.replace("A800_ACTORHIDDEN_PREFIX5_EXACT_SOURCE_STAGED", "A800_P10_PENDING_PROBE_EXACT_SOURCE_STAGED")

PREPARE = COMMON + r'''
source();parent_terminal()
assert not (b/'preparation/prepare-intent.json').exists(),'Prepare onceclaim exists; inspect, never replay'
assert not (b/'preparation/prepare-exit.json').exists()
write(b/'preparation/prepare-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],checked_unix=time.time(),pending_actor_outer=10,
 old_actor_stop_requested=False,full_state_resume=True),True)
command=['/data/search-g1/env/bin/python','-m','reproduction.operations.a800_prefix5_p10_pending_probe_control',
 '--prepare-worker','--handle',str(b/'preparation/deployment.json')]
env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
with (b/'logs/cpu-prepare.log').open('xb') as log:
 child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
who=birth(child.pid);assert who['sid']==who['pid'] and who['uid']==os.getuid()
v=dict(status='PENDING_PROBE_CPU_PREPARE_ONCE_LAUNCHED_NOT_ACCEPTED',run_id=h['run_id'],
 execution_id=h['execution_id'],source_commit=h['source_commit'],identity=who,started_unix=time.time(),
 actor_updates_to_reexecute=0,candidates_to_resample=0,old_actor_stop_requested=False)
write(b/'preparation/prepare-launch.json',v,True);emit(v)
'''

LAUNCH = COMMON + r'''
source();qualified,config=qualification();terminal=parent_terminal()
assert not (b/'preparation/pending-probe-launch-intent.json').exists(),'Launch intent exists; inspect, never replay'
assert not (b/'preparation/submit-intent.json').exists(),'Native once submit already claimed'
assert not bound_processes(b,exclude=(os.getpid(),)),'A new namespace writer remains active'
sys.path.insert(0,str(repo))
from reproduction.prefix_selection200.launch import validate_config
from reproduction.prefix_selection200.pending_probe_resume import validate_imported_pending
validate_config(config,b);validate_imported_pending(b,config)
write(b/'preparation/pending-probe-launch-intent.json',dict(run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit'],config_sha256=sha(b/'preparation/launch-config.json'),
 requested_unix=time.time(),pending_outer=10,committed_joint_outer=9,retained_actor_adam_per_rank=[200]*4,
 retained_probe_adam_per_rank=[0]*4,actor_updates_to_reexecute=0,candidates_to_resample=0,
 formal_actor_fresh0=False,optimizer_reset=False,parent_terminal=terminal),True)
# This is a full-state recovery submit. Deliberately do not call the fresh
# submit's CP5 hash/import gate or initialize any warmup/actor/probe/pool state.
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
  resume_kind='native_actor_committed_probe_pending')
 write(b/'preparation/submit-receipt.json',value,True)
emit(dict(status='PENDING_PROBE_SUBMIT_RETURNED_NOT_CUDA_RESTORE_ACCEPTANCE',submit=value,
 parent_terminal=terminal,checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],
 source_commit=h['source_commit']))
'''

# Keep the ordinary bounded actor/probe/phase ledger projection, but authenticate
# the imported parent joint9 explicitly rather than treating it as new-source10.
_OBSERVE_BODY = previous.OBSERVE[len(previous.COMMON):]
_OLD_JOINT_GUARD = "assert joint['run_id']==h['run_id'] and joint['source_commit']==h['source_commit']"
assert _OBSERVE_BODY.count(_OLD_JOINT_GUARD) == 1
_OBSERVE_BODY = _OBSERVE_BODY.replace(_OLD_JOINT_GUARD, "assert joint['run_id']==h['run_id']\n if joint['source_commit']==parent_h['source_commit']:\n  assert joint['outer']==9\n  imported=read(b/'preparation/resume-pending-probe-receipt.json')\n  assert imported['source_base']==str(parent) and imported['parent_source_commit']==parent_h['source_commit']\n  assert imported['joint_sha256']==sha(parent/'latest-joint.json')==sha(b/'latest-joint.json')\n else:assert joint['source_commit']==h['source_commit']")
_OBSERVE_BODY = _OBSERVE_BODY.removesuffix('\nemit(v)\n')
OBSERVE = COMMON + _OBSERVE_BODY + r'''
v['pending_probe_qualification']=document(b,'preparation/native-pending-probe-qualified.json')
if v['pending_probe_qualification'] is not None:
 q=v['pending_probe_qualification']
 if q.get('status')=='QUALIFIED_NATIVE_PENDING_PROBE':
  keys=('status','run_id','source_base','source_execution_id','source_commit','completed_joint_outer',
   'pending_outer','actor_completed_outer','actor_adam_per_rank','probe_completed_outer',
   'probe_adam_per_rank','probe_update_new_adam_per_rank','owned_births_all_gone')
 else:
  keys=('status','run_id','error_type','error','elapsed_seconds','checked_unix','failed_unix')
 v['pending_probe_qualification']={key:q[key] for key in keys if key in q}
v['prepare_progress']=document(b,'preparation/prepare-progress.json')
v['resume_pending_probe_import']=document(b,'preparation/resume-pending-probe-receipt.json')
if v['resume_pending_probe_import'] is not None:
 imported=v['resume_pending_probe_import']
 v['resume_pending_probe_import']={key:imported[key] for key in ('status','run_id','execution_id','source_commit',
  'source_base','source_execution_id','parent_source_commit','joint_completed_outer','pending_outer',
  'actor_adam_per_rank','scheduler_completed_outer','probe_adam_per_rank','actor_updates_to_reexecute',
  'candidates_to_resample','outcomes_to_rescore','probe_initialized_again','optimizer_reset',
  'full_state_import','cuda_restore_verified') if key in imported}
v['operator_contract']='a800_p10_pending_probe_bounded_metadata_v1'
v['cpu_qualification_is_not_cuda_restoration']=True
'''

# Parent1..9 source is accepted only through its immutable, full-SHA-qualified
# origin path and the identical imported metadata bytes. New10+ must carry the
# new source. This preserves the scientific source guard across the recovery.
_EXTRA = scalar_observer.EXTRA
_HISTORY_SOURCE_GUARD = "assert completed_joint['run_id']==h['run_id'] and completed_joint['source_commit']==h['source_commit']"
assert _EXTRA.count(_HISTORY_SOURCE_GUARD) == 1
_EXTRA = _EXTRA.replace(_HISTORY_SOURCE_GUARD, "assert completed_joint['run_id']==h['run_id']\n if outer<=9:\n  assert completed_joint['source_commit']==parent_h['source_commit']\n  origin=parent/'outers'/('outer-%06d'%outer)/'complete.json'\n  assert str(origin) in parent_history_seals,'Parent history lacks a full-SHA CPU seal'\n  assert sha(origin)==parent_history_seals[str(origin)]==sha(root/'complete.json')\n else:assert completed_joint['source_commit']==h['source_commit']")
_EXTRA = _EXTRA.replace("for outer in (5,50,100,150,200):\n root=b/'outers'/('outer-%06d'%outer)",
    "for outer in (5,50,100,150,200):\n root=(parent if outer<=min(n,9) else b)/'outers'/('outer-%06d'%outer)")
OBSERVE += r'''
parent_history_seals={}
qpath=b/'preparation/native-pending-probe-qualified.json'
if qpath.exists():
 q=read(qpath)
 if q.get('status')=='QUALIFIED_NATIVE_PENDING_PROBE':
  assert q['run_id']==h['run_id'] and q['source_base']==str(parent) and q['source_commit']==parent_h['source_commit']
  parent_history_seals={row['path']:row['sha256'] for row in q['verified_files']}
''' + _EXTRA + r'''
if qpath.exists() and q.get('status')!='QUALIFIED_NATIVE_PENDING_PROBE' and joint is None:
 v['complete_outer']=None
 v['probe_clock_contract']=None
 v['recovery_boundary_status']='UNQUALIFIED_OR_FAILED_CPU_PROOF_BOUNDARY_UNKNOWN'
v['operator_contract']='a800_p10_pending_probe_bounded_metadata_v1'
emit(v)
'''


def _sha_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def _write_once(path, value):
    previous._write_once(path, value)


def _publish(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _idle_mapping(base, receipt):
    """Wait locally for actual idle cards; never reserve or signal a GPU owner."""
    from reproduction.prefix_selection200.a800_runtime import gpu_inventory, compute_processes
    while True:
        inventory = gpu_inventory()
        used = {row['uuid'] for row in compute_processes()}
        idle = [index for index, row in sorted(inventory.items())
                if row['uuid'] not in used and row['memory_MiB'] <= 64]
        training = [index for index in idle if index not in (5, 6)]
        auxiliary_ready = all(index in idle for index in (5, 6))
        if auxiliary_ready and len(training) >= 4:
            chosen = training[:4]
            proof = dict(run_id=receipt['run_id'], execution_id=receipt['execution_id'],
                physical_gpu_indices=chosen, physical_gpu_uuids=[inventory[i]['uuid'] for i in chosen],
                auxiliary_gpu_indices=[5, 6], auxiliary_gpu_uuids=[inventory[i]['uuid'] for i in (5, 6)],
                actual_inventory=[inventory[i] for i in chosen + [5, 6]], checked_unix=time.time(),
                idle_memory_threshold_MiB=64, actual_idle_mapping=True,
                occupancy_reservation_started=False, foreign_processes_stopped=False,
                launch_native_idle_uuid_lease_admission_still_required=True)
            _write_once(base/'preparation/idle-mapping.json', proof)
            return proof
        _publish(base/'preparation/prepare-progress.json', dict(receipt,
            stage='WAITING_ACTUAL_IDLE_FOUR_TRAINING_AND_AUX5_6', checked_unix=time.time(),
            idle_training_count=len(training), auxiliary5_6_idle=auxiliary_ready,
            local_wait_seconds=30, no_foreign_signals=True, no_gpu_reservation=True))
        time.sleep(30)


def prepare_worker(handle_path):
    """Qualify/import the saved asymmetric boundary once, not a fresh init."""
    h = json.loads(Path(handle_path).read_text())
    base = Path(h['base']); parent_h = h['resume_parent_handle']; parent = Path(parent_h['base'])
    receipt = dict(run_id=h['run_id'], execution_id=h['execution_id'], source_commit=h['source_commit'],
        identity=previous._identity(), checked_unix=time.time(), actor_updates_to_reexecute=0,
        candidates_to_resample=0, warmup_checkpoint_read_again=False, optimizer_reset=False)
    try:
        if (h['run_id'] != PARENT_RUN or parent_h['execution_id'] != PARENT_EXECUTION
                or parent_h['source_commit'] != PARENT_SOURCE
                or base != Path(str(parent) + '-resume-' + h['execution_id'])
                or h['execution_id'] == PARENT_EXECUTION or h['source_commit'] == PARENT_SOURCE):
            raise ValueError('Require a new source/execution for the exact same pending-probe run')
        source_config_path = parent/'preparation/launch-config.json'
        if _sha_file(source_config_path) != PARENT_CONFIG_SHA:
            raise ValueError('Actual parent launch-config bytes differ')
        source_config = json.loads(source_config_path.read_text())
        if (source_config['base'] != str(parent) or source_config['run_id'] != PARENT_RUN
                or source_config['execution_id'] != PARENT_EXECUTION or source_config['source_commit'] != PARENT_SOURCE):
            raise ValueError('Parent configuration identity differs')
        template = json.loads(Path(h['template_config_path']).read_text())
        allowed = {'base','execution_id','source_commit','pool_state_path','prefix_probe_receipt',
                   'resume_pending_probe','physical_gpu_indices','physical_gpu_uuids',
                   'auxiliary_gpu_indices','auxiliary_gpu_uuids'}
        for key, value in template.items():
            if key not in allowed and value != source_config.get(key):
                raise ValueError('Recovery template cannot change existing scientific/input field: ' + key)
        if any(template.get(key) != h[key] for key in ('base','execution_id','source_commit','run_id')):
            raise ValueError('Template does not bind the selected recovery deployment')
        terminal = Path(h['terminal_failure_file'])
        if _sha_file(terminal) != h['terminal_failure_sha256']:
            raise ValueError('Staged original terminal proof changed')
        spec = dict(template['resume_pending_probe'])
        expected = dict(source_base=str(parent), source_execution_id=PARENT_EXECUTION,
            source_commit=PARENT_SOURCE, source_config_sha256=PARENT_CONFIG_SHA,
            completed_joint_outer=9, pending_outer=10, terminal_proof=str(terminal),
            terminal_proof_sha256=h['terminal_failure_sha256'])
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError('Pending resume specification differs from the exact parent boundary')
        qualification = base/'preparation/native-pending-probe-qualified.json'
        if qualification.exists():
            raise FileExistsError('Qualifier output exists; do not replay the once CPU qualification')
        _publish(base/'preparation/prepare-progress.json', dict(receipt,
            stage='CPU_FULL_SHA_FINITE_QUALIFICATION_SAVED_ACTOR10_PROBE0_POOL10'))
        command = [PYTHON, '-m', 'reproduction.prefix_selection200.checkpoint_native_pending_probe',
            '--base', str(parent), '--expected-run', PARENT_RUN, '--expected-execution', PARENT_EXECUTION,
            '--expected-source', PARENT_SOURCE, '--expected-config-sha256', PARENT_CONFIG_SHA,
            '--terminal-proof', str(terminal), '--terminal-proof-sha256', h['terminal_failure_sha256'],
            '--output', str(qualification)]
        result = subprocess.run(command, cwd=base/'repo', env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
            PYTHONDONTWRITEBYTECODE='1'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            print(result.stderr.decode(errors='replace'), flush=True)
            raise RuntimeError('Native pending-state CPU qualifier exited ' + str(result.returncode))
        qualified = json.loads(qualification.read_text())
        if (qualified.get('status') != 'QUALIFIED_NATIVE_PENDING_PROBE'
                or qualified.get('run_id') != PARENT_RUN or qualified.get('owned_births_all_gone') is not True):
            raise ValueError('CPU qualifier did not produce a valid full-state seal')
        spec.update(qualification_proof=str(qualification), qualification_proof_sha256=_sha_file(qualification))
        config = copy.deepcopy(source_config)
        config.update(base=str(base), execution_id=h['execution_id'], source_commit=h['source_commit'],
            pool_state_path=str(base/'question-pool/state.json'), prefix_probe_receipt=str(base/'probe/latest.json'),
            resume_pending_probe=spec)
        if config.get('resume_joint'):
            raise ValueError('A pending-probe asymmetric boundary cannot also be a joint resume')
        mapping = _idle_mapping(base, receipt)
        for key in ('physical_gpu_indices','physical_gpu_uuids','auxiliary_gpu_indices','auxiliary_gpu_uuids'):
            config[key] = mapping[key]
        from reproduction.prefix_selection200.launch import validate_config
        validate_config(config, base)
        config_path = base/'preparation/launch-config.json'
        _write_once(config_path, config)
        _publish(base/'preparation/prepare-progress.json', dict(receipt,
            stage='IMPORTING_QUALIFIED_SAVED_ACTOR10_UNCHANGED_PROBE0_NO_REGENERATION'))
        from reproduction.prefix_selection200.pending_probe_resume import import_pending
        imported = import_pending(base)
        import_path = base/'preparation/resume-pending-probe-receipt.json'
        if imported.get('status') != 'COMMITTED_ACTOR_PENDING_PROBE_IMPORTED_NO_RESET':
            raise ValueError('Expected the unique new pending-boundary import')
        receipt.update(exit_code=0, status='NATIVE_PENDING_PROBE_CPU_PREPARE_COMPLETE',
            qualification_sha256=_sha_file(qualification), config_sha256=_sha_file(config_path),
            import_receipt_sha256=_sha_file(import_path), idle_mapping_sha256=_sha_file(base/'preparation/idle-mapping.json'),
            actor_adam_per_rank=[200]*4, probe_adam_per_rank=[0]*4, scheduler_completed_outer=10,
            completed_joint_outer=9, pending_outer=10, cpu_qualified_not_cuda_restored=True,
            checkpoint_cp5_deserializations=0, completed_unix=time.time())
        _write_once(base/'preparation/prepare-complete.json', receipt)
        _publish(base/'preparation/prepare-progress.json', dict(receipt, stage='CPU_PREPARE_COMPLETE_WAITING_ONCE_LAUNCH'))
    except BaseException as error:
        receipt.update(exit_code=1, error_class=type(error).__name__, completed_unix=time.time(),
            failed_namespace_preserved=True, same_onceclaim_must_not_be_replayed=True)
        _write_once(base/'preparation/prepare-exit.json', receipt)
        traceback.print_exc()
        raise
    _write_once(base/'preparation/prepare-exit.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('stage','prepare','launch','observe'))
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--prepare-worker', action='store_true')
    args = parser.parse_args()
    if not args.handle.is_absolute():
        parser.error('--handle must be an absolute deployment JSON path')
    if args.prepare_worker:
        prepare_worker(args.handle)
        return
    if args.action is None or (args.action == 'stage' and args.payload is None):
        parser.error('--action is required; stage also requires --payload')
    h = json.loads(args.handle.read_text()); payload = dict(handle=h)
    if args.action == 'stage':
        staged = json.loads(args.payload.read_text())
        if 'handle' in staged:
            raise ValueError('Payload cannot replace the selected handle')
        payload.update(staged)
    code = dict(stage=STAGE, prepare=PREPARE, launch=LAUNCH, observe=OBSERVE)[args.action]
    started = time.time()
    try:
        result = subprocess.run(route(code), input=json.dumps(payload).encode(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=65)
        record = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time()-started,
            exit_code=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode())
    except subprocess.TimeoutExpired as error:
        record = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time()-started,
            exit_code=None, timed_out=True, remote_action_outcome_unknown=True, mutation_must_not_be_replayed=True,
            stdout=(error.stdout or b'').decode(errors='replace'), stderr=(error.stderr or b'').decode(errors='replace'))
    record.update(run_id=h['run_id'], execution_id=h['execution_id'], source_commit=h['source_commit'])
    record['remote_mutations'] = args.action != 'observe'
    out = ROOT/'preparation'/('a800-prefix5-p10-pending-probe-'+str(args.action)+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(record,indent=2)+'\n')
    if record.get('timed_out') or record.get('exit_code'):
        raise RuntimeError('Action outcome unknown or failed; do not repeat onceclaim. Receipt '+str(out))
    if len(result.stdout)>MAX_OUTPUT:
        raise ValueError('Output exceeds128KiB; retain unknown '+str(out))
    print(json.dumps(dict(receipt=str(out), result=json.loads(result.stdout))))


if __name__ == '__main__':
    main()
