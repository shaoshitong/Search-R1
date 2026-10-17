"""Fault-only full-state continuation; no actor/probe algorithm change.

Seal the actual stopped parent, preserve a failed uncommitted continuation,
and retain its accepted next-round prefix/selection through the existing joint
adapter. Never replay a launch claim or treat CPU qualification as CUDA restore.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from . import a800_prefix5_variable_pool_control as previous

ROOT = previous.ROOT
MODULE = 'reproduction.operations.a800_prefix5_fault_joint_recovery_control'
STAGE = previous.STAGE
PREPARE = previous.PREPARE.replace(
    'reproduction.operations.a800_prefix5_variable_pool_control', MODULE)
LAUNCH = previous.LAUNCH
OBSERVE = previous.OBSERVE

TERMINAL = previous.COMMON + r'''
import socket
source()
sys.path.insert(0,str(repo))
from reproduction.operations.a800_i1_i6_boundary_handoff import identity, same
c=read(b/'preparation/launch-config.json');joint=read(b/'latest-joint.json')
n=p['completed_outer'];pending=n+1
assert n==64 and joint['outer']==n and joint['actor']['completed_outer']==n
assert joint['actor']['actual_adam_steps_per_rank']==[1280]*4
assert joint['probe']['completed_outer']==60 and joint['probe']['actual_adam_steps_per_rank']==[300]*4
submit=read(b/'preparation/submit-receipt.json');assert same(submit['leader']) is None
cleanup=read(b/'logs/cleanup-status.json')
assert cleanup['cleaned'] is True and cleanup['owned_births_all_gone'] is True
ledger=read(b/'logs/process-ledger.json');births={}
for job in ledger['jobs'].values():
 assert job['state']=='STOPPED'
 for row in [job.get('leader'),*job.get('ledger',{}).values()]:
  if isinstance(row,dict) and all(k in row for k in ('pid','sid','uid')) and ('birth' in row or 'start_ticks' in row):
   who=identity(row);assert same(who) is None
   births[(who['pid'],who['start_ticks'],who['sid'],who['uid'])]=who
assert births and not bound_processes(b,exclude=(os.getpid(),))
failed=ledger['jobs']['outer-000065-continue'];assert failed['exit_code']==1
root=b/'outers/outer-000065'
assert not (root/'complete.json').exists() and not (root/'probe-update').exists()
assert not list((b/'actor/outers').glob('outer-0065-attempt-*'))
assert (root/'prefix/complete.json').is_file() and (root/'probe-score/complete.json').is_file()
terminal_path=b/'preparation/fault-outer65-terminal-20261018.json'
def terminal_summary(value):
 summary={k:v for k,v in value.items() if k!='owned_births'}
 raw=json.dumps(value['owned_births'],sort_keys=True,allow_nan=False).encode()
 summary.update(owned_births_count=len(value['owned_births']),
  owned_births_canonical_sha256=hashlib.sha256(raw).hexdigest(),
  owned_births_projection='all identities retained in immutable remote proof; exact count/hash returned')
 return summary
if terminal_path.exists():
 value=read(terminal_path)
 assert value['source_joint_sha256']==sha(b/'latest-joint.json') and value['process_ledger_sha256']==sha(b/'logs/process-ledger.json')
 emit(dict(status='EXISTING_EXACT_FAULT_TERMINAL',terminal_proof=str(terminal_path),terminal_proof_sha256=sha(terminal_path),value=terminal_summary(value)))
else:
 intent=b/'preparation/fault-outer65-archive-intent-20261018.json'
 assert not intent.exists(),'Reconcile an interrupted archive; do not replay'
 full=root/'full';regular(full/'rank-0.log')
 assert not (full/'complete.json').exists() and not list(full.glob('rank-*-complete.json'))
 files=[]
 for f in sorted(full.rglob('*')):
  assert not f.is_symlink()
  if f.is_file():regular(f);files.append(dict(relative_path=str(f.relative_to(full)),bytes=f.stat().st_size,sha256=sha(f)))
 assert files
 archive=b/'failed-sampling'/('outer-000065-continue-'+h['execution_id'])
 assert not archive.exists()
 value=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
  completed_outer=n,pending_outer=pending,source_directory=str(full),archive_directory=str(archive),files=files,
  accepted_full_phase=False,accepted_rank_receipts=0,actor_updates=0,probe_updates=0,checked_unix=time.time())
 write(intent,value,True);archive.parent.mkdir(exist_ok=True);os.rename(full,archive)
 assert not full.exists() and all(sha(archive/row['relative_path'])==row['sha256'] for row in files)
 archive_receipt=b/'preparation/fault-outer65-archive-complete-20261018.json'
 write(archive_receipt,dict(value,status='FAILED_UNCOMMITTED_CONTINUATION_RETAINED',archive_verified=True),True)
 terminal=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],base=str(b),
  host=socket.gethostname(),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
  termination_cause='FAULT_EXIT',completed_outer=n,pending_outer=pending,pipeline_alive=False,
  owned_writers_gone=True,actual_births_present=[],owned_births=list(births.values()),
  process_ledger_sha256=sha(b/'logs/process-ledger.json'),source_joint_sha256=sha(b/'latest-joint.json'),
  failed_job='outer-000065-continue',failed_job_exit_code=1,
  fault_error_class='CUDA_ILLEGAL_MEMORY_ACCESS_VLLM_SAMPLER',
  failed_log_sha256=sha(archive/'rank-0.log'),failed_sampling_archive=str(archive),
  failed_sampling_archive_receipt=str(archive_receipt),failed_sampling_archive_receipt_sha256=sha(archive_receipt),
  uncommitted_after_joint=dict(actor_actual_writes_per_rank=[0]*4,probe_actual_writes_per_rank=[0]*4,
   optimizer_work_in_flight=False,all_updates_accounted=True),
  accepted_prefix_and_score_preserved=True,failed_rollout_artifacts_deleted=False,checked_unix=time.time())
 write(terminal_path,terminal,True)
 emit(dict(status='ACTUAL_FAULT_TERMINAL_AND_FAILED_SAMPLING_PRESERVED',terminal_proof=str(terminal_path),
  terminal_proof_sha256=sha(terminal_path),value=terminal_summary(terminal)))
'''


def prepare_worker(handle_path):
    from reproduction.prefix_selection200 import i1_i6_joint_resume as resume
    from reproduction.prefix20_online.launch import once, publish, read, sha
    h = read(handle_path)
    base = Path(h['base'])
    parent = Path(h['parent_handle']['base'])
    result = dict(run_id=h['run_id'], execution_id=h['execution_id'],
                  source_commit=h['source_commit'], identity=previous.fresh.previous._identity(),
                  started_unix=time.time())
    try:
        original = read(parent/'preparation/launch-config.json')
        terminal = read(h['parent_terminal_path'])
        n = h['resume_completed_outer']
        if (n != 64 or terminal['termination_cause'] != 'FAULT_EXIT'
                or terminal['completed_outer'] != n
                or original['run_id'] != h['run_id']
                or original['execution_id'] != h['parent_handle']['execution_id']
                or original['source_commit'] != h['parent_handle']['source_commit']
                or sha(h['parent_terminal_path']) != h['parent_terminal_sha256']
                or sha(base/'preparation/human-intent.json') != h['human_intent_sha256']
                or original['candidate_schedule'] != h['retained_candidate_schedule']):
            raise ValueError('Require the exact stopped fault boundary and unchanged candidate schedule')
        science = original.get('science_source_commit', original['source_commit'])
        parent_manifest = resume.source_manifest(parent/'repo', original['source_commit'],
                                                  science_source_commit=science)
        deployment = read(parent/'preparation/deployment.json')
        expected = {name: digest for name, digest in deployment['source_files'].items()
                    if Path(name).suffix in ('.py', '.sh')}
        if parent_manifest['files_sha256'] != expected:
            raise ValueError('Parent source no longer matches its actual immutable deployment')
        target_manifest = resume.source_manifest(base/'repo', h['source_commit'], science_source_commit=science)
        proof = resume.compatibility_proof(parent_manifest, target_manifest)
        proof_path = base/'preparation/source-compatibility.json'
        once(proof_path, proof)
        qualification = base/'preparation/full-state-qualified.json'
        publish(base/'preparation/prepare-progress.json', dict(result,
            stage='ONCE_FULL_SHA_FINITE_JOINT64_AND_ACCEPTED_PENDING65_QUALIFICATION', completed_outer=n))
        from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import qualify
        qualify(parent, outer=n, terminal_proof=Path(h['parent_terminal_path']),
                terminal_sha=sha(h['parent_terminal_path']), output=qualification)
        c = dict(original, base=str(base), execution_id=h['execution_id'], source_commit=h['source_commit'],
                 science_source_commit=science, source_compatibility=str(proof_path),
                 source_compatibility_sha256=sha(proof_path), pool_state_path=str(base/'question-pool/state.json'),
                 prefix_probe_receipt=str(base/'probe/latest.json'), fresh_actor_optimizer_scheduler_rng_sampler=False,
                 retained_warmup_model_only=False, full_state_resume=True)
        c['resume_joint'] = dict(adapter=resume.ADAPTER, completed_outer=n, source_base=str(parent),
            source_execution_id=original['execution_id'], source_commit=original['source_commit'],
            source_config_sha256=sha(parent/'preparation/launch-config.json'),
            source_joint_sha256=sha(parent/'latest-joint.json'), terminal_proof=h['parent_terminal_path'],
            terminal_proof_sha256=h['parent_terminal_sha256'], qualification_proof=str(qualification),
            qualification_proof_sha256=sha(qualification))
        c.pop('resume_import', None)
        c.pop('resume_pending_probe', None)
        mapping = previous.fresh._idle_mapping(base, result)
        for key in ('physical_gpu_indices', 'physical_gpu_uuids', 'auxiliary_gpu_indices', 'auxiliary_gpu_uuids'):
            c[key] = mapping[key]
        once(base/'preparation/launch-config.json', c)
        resume.import_complete(base)
        from reproduction.prefix_selection200.launch import validate_config
        validate_config(c, base)
        result.update(exit_code=0, status='VARIABLE_POOL_FULL_STATE_CPU_IMPORT_COMPLETE_NOT_CUDA_ACCEPTANCE',
            completed_outer=n, effective_from_outer=c['candidate_schedule']['effective_from_outer'],
            qualification_sha256=sha(qualification), config_sha256=sha(base/'preparation/launch-config.json'),
            import_receipt_sha256=sha(base/'preparation/resume-joint-receipt.json'), completed_unix=time.time())
    except BaseException as error:
        result.update(exit_code=1, error_class=type(error).__name__, error=str(error),
                      failed_unix=time.time(), same_onceclaim_must_not_be_replayed=True)
        once(base/'preparation/prepare-exit.json', result)
        traceback.print_exc()
        raise
    once(base/'preparation/prepare-exit.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('terminal', 'stage', 'prepare', 'launch', 'observe'))
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--after-step', type=int)
    parser.add_argument('--prepare-worker', action='store_true')
    args = parser.parse_args()
    if args.prepare_worker:
        prepare_worker(args.handle)
        return
    if args.action is None or (args.action == 'stage' and args.payload is None):
        parser.error('Choose an action and provide its stage payload when required')
    h = json.loads(args.handle.read_text())
    payload = dict(handle=h)
    if args.action == 'stage':
        payload.update(json.loads(args.payload.read_text()))
    if args.action == 'terminal':
        payload['completed_outer'] = 64
    if args.action == 'observe':
        anchor = json.loads((ROOT/'progress-checks/hourly-anchor.json').read_text())
        if args.after_step is not None:
            payload['after_step'] = args.after_step
        elif anchor.get('delivered') is True and all(anchor.get(k) == h[k] for k in ('run_id', 'execution_id')):
            payload['after_step'] = anchor['complete_outers']
        else:
            payload['after_step'] = h['resume_completed_outer']
    began = time.time()
    code = dict(terminal=TERMINAL, stage=STAGE, prepare=PREPARE, launch=LAUNCH, observe=OBSERVE)[args.action]
    try:
        p = subprocess.run(previous.route(code), input=json.dumps(payload).encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=65)
        receipt = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time()-began,
            exit_code=p.returncode, stdout=p.stdout.decode(), stderr=p.stderr.decode(),
            run_id=h['run_id'], execution_id=h['execution_id'], source_commit=h['source_commit'])
    except subprocess.TimeoutExpired as error:
        receipt = dict(action=args.action, checked_unix=time.time(), exit_code=None, timed_out=True,
            remote_action_outcome_unknown=True, mutation_must_not_be_replayed=True,
            stdout=(error.stdout or b'').decode(errors='replace'), stderr=(error.stderr or b'').decode(errors='replace'))
    out = ROOT/'preparation'/('a800-prefix5-fault65-'+args.action+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    if receipt.get('timed_out') or receipt['exit_code']:
        raise RuntimeError('Inspect actual receipt; never replay a mutation: '+str(out))
    print(json.dumps(dict(receipt=str(out), result=json.loads(receipt['stdout']))))


if __name__ == '__main__':
    main()
