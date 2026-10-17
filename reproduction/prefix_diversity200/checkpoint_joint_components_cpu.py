"""Once-only full CPU joint-component audit for actual150/200; no training or GPU operations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from . import checkpoint_cpu_once as core

STATUS = 'JOINT_MILESTONE_FULL_SHA_CPU_COMPONENTS_VERIFIED_EXPLICIT_RECOVERY_BINDINGS'
LEGACY_BASE = '/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261016-f3ee1435e501-resume3'
LEGACY_POOL_SHA = 'a09e649610c7d586cfa0121f38e86f7b174494d3928708e94d9bca3b5efb9b40'
LEGACY_PROBE_SHA = 'e300ccbcc09ac24889f1c70a92e92550aaa114b5eab438413e69942eb5070fd0'


def identity(config):
    base=Path(config['base']); outer=config['outer']
    core.need(type(outer) is int and outer in (150,200), 'Only150/200 are authorized by this entry')
    core.need(base.parent==Path('/mnt/data/shaoshitong') and base.name.startswith('search-g1-prefix-diversity200-')
              and base.resolve()==base, 'Unsupported canonical run base')
    path=base/'preparation/launch-config.json'; actual=core.read(path)
    core.need(core.hash_file(path)['sha256']==config['runtime_config_sha256']
              and actual['run_id']==config['run_id']
              and actual['source_commit']==config['runtime_source_commit']
              and actual['base']==str(base) and actual['world_size']==8 and actual['outers']==200,
              'Pinned run/source/config identity changed; publish an explicit new control config')
    return base,outer,actual


def readiness(config, stability_seconds=2):
    """Small receipts/stat only. Missing or active writer never creates an audit intent."""
    base,outer,actual=identity(config); root=base/'outers'/('outer-%06d'%outer); path=root/'complete.json'
    if not path.is_file():
        return dict(status='PENDING_ACTUAL_JOINT_NO_AUDIT_INTENT',outer=outer,missing=str(path))
    joint=core.read(path); actor,probe=joint['actor'],joint['probe']
    core.need(joint['run_id']==actual['run_id'] and joint['outer']==outer
              and joint['source_commit']==actual['source_commit']
              and actor['completed_outer']==probe['completed_outer']==outer
              and actor['actual_adam_steps_per_rank']==probe['actual_adam_steps_per_rank']==[20*outer]*8
              and actor['scheduler_completed_outer']==outer and actor['milestone'] is True,
              'Joint receipt is not the actual complete milestone')
    ledger=core.read(base/'logs/process-ledger.json'); core.need(ledger['run_id']==actual['run_id'],'Foreign writer ledger')
    attempt=Path(actor['state_path']).name.split('-attempt-',1)[1]
    names=['outer-%06d-actor-%s'%(outer,attempt),'outer-%06d-probe-update'%outer]
    writers={name:ledger['jobs'].get(name) for name in names}
    if not all(row and row.get('state')=='STOPPED' and row.get('exit_code')==0 for row in writers.values()):
        return dict(status='PENDING_ACTUAL_CHECKPOINT_WRITER_NO_AUDIT_INTENT',outer=outer,writers=writers)
    paths=[path,Path(actor['state_path'])/'complete.json',root/'probe-update/complete.json',
           Path(joint['pool_snapshot']),Path(joint['probe_snapshot']),Path(actor['driver_state']['path'])]
    paths += [Path(row['path']) for row in actor['rank_states']]
    paths += [Path(actor['hf_model_dir'])/name for name in actor['hf_files']]
    core.need(core.read(Path(actor['state_path'])/'complete.json')==actor
              and core.read(root/'probe-update/complete.json')['checkpoint']==probe,
              'Original producer receipts differ from joint')
    def snapshot():
        result={}
        for item in paths:
            core.owned(base,item);info=item.stat();result[str(item)]=[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns]
        return result
    before=snapshot();time.sleep(stability_seconds);core.need(before==snapshot(),'Checkpoint files are not immutable yet')
    return dict(status='ACTUAL_JOINT_WRITERS_STOPPED_AND_FILES_STABLE',outer=outer,
        joint_complete_sha256=core.hash_file(path)['sha256'],writers=writers,files=before,
        stability_seconds=stability_seconds,observed_unix=time.time(),audit_intent_created=False)


def driver_components(driver, actor, config, joint, outer, pool, manifest):
    """Bind recovery to real joint state; classify legacy descriptors without repairing them."""
    base=Path(config['base']); previous_path=core.owned(base,base/'outers'/('outer-%06d'%(outer-1))/'complete.json')
    previous=core.read(previous_path); previous_probe=previous['probe']
    core.need(previous['run_id']==config['run_id'] and previous['outer']==previous_probe['completed_outer']==outer-1
              and previous_probe['actual_adam_steps_per_rank']==[20*(outer-1)]*8
              and previous_probe['sha256']==manifest['probe_before_checkpoint_sha256'],
              'Actual selection-time preceding probe receipt is missing or wrong')
    core.need(driver['run_id']==config['run_id'] and driver['completed_outer']==outer
              and driver['actor_version']==actor['actor_version'], 'Actual driver identity differs')
    dp=driver['candidate_pool_state']; pp=driver['prefix_probe_receipt']; pv=pp['value']
    core.need(dp['sha256']==hashlib.sha256(core.canonical(dp['value'])+b'\n').hexdigest(),
              'Driver internal pool descriptor hash differs')
    pool_matches=(dp['value']==pool)
    probe_matches=(pv['run_id']==config['run_id'] and pv['completed_outer']==outer-1
                   and pv['actual_adam_steps_per_rank']==[20*(outer-1)]*8
                   and pv['sha256']==previous_probe['sha256'])
    if not pool_matches:
        core.need(dp['path']==LEGACY_BASE+'/question-pool/state.json' and dp['path']==config['pool_state_path']
                  and dp['sha256']==LEGACY_POOL_SHA and dp['value']['consumed_candidates']==271360
                  and dp['value']['next_outer']==54, 'Unknown driver pool defect: no unconditional component PASS')
    if not probe_matches:
        core.need(pp['path']==LEGACY_BASE+'/probe/latest.json' and pp['path']==config['prefix_probe_receipt']
                  and pv['run_id']==config['run_id'] and pv['completed_outer']==52
                  and pv['actual_adam_steps_per_rank']==[1040]*8 and pv['sha256']==LEGACY_PROBE_SHA,
                  'Unknown driver probe defect: no unconditional component PASS')
    return dict(standalone_driver_status='DESCRIPTORS_MATCH_JOINT' if pool_matches and probe_matches
                    else 'FAIL_KNOWN_STALE_RESUME3_DESCRIPTORS_RETAINED',
        driver_candidate_pool_matches_joint=pool_matches,driver_previous_probe_matches_joint=probe_matches,
        observed_driver_pool_path=dp['path'],observed_driver_pool_next_outer=dp['value']['next_outer'],
        observed_driver_previous_probe_path=pp['path'],observed_driver_previous_probe_outer=pv['completed_outer'],
        actual_selection_time_probe_receipt=previous_probe,actual_selection_time_joint_file=core.hash_file(previous_path),
        saved_driver_rng_checked_by_shared_reader=True,
        recovery_bindings=dict(actor_hf_dir=actor['hf_model_dir'],actor_rank_states=actor['rank_states'],
            authoritative_candidate_pool_snapshot=joint['pool_snapshot'],candidate_pool_sha256=joint['pool_sha256'],
            authoritative_current_probe_snapshot=joint['probe_snapshot'],current_probe_receipt=joint['probe'],
            selection_time_previous_probe_receipt_path=str(previous_path),next_outer=outer+1,
            require_explicit_joint_bundle_binding=True,legacy_driver_data_probe_aliases_are_not_authoritative=True,
            native_cuda_executor_restore_performed=False))


def source_and_splits(base, actual, control):
    result={}
    for name,expected in control['runtime_source_files_sha256'].items():
        path=base/'repo'/name;proof=core.hash_file(path);core.need(proof['sha256']==expected,'Frozen runtime source changed: '+name)
        result[name]=proof
    for key in ('canonical_train','validation_questions','final_questions'):
        proof=core.hash_file(Path(actual[key]));core.need(proof['sha256']==actual[key+'_sha256'],'Pinned split changed: '+key)
        result[key]=proof
    return result


def run(control_path):
    core.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run only with empty CUDA visibility')
    control=core.read(control_path);base,outer,actual=identity(control)
    directory=core.owned(base,base/control['control_name']);proof_path=directory/'proof.json';intent=directory/'intent.json'
    with (directory/'reader.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if proof_path.exists():
            previous=core.read(proof_path)
            return dict(status='EXISTING_JOINT_COMPONENT_PROOF_NO_REPLAY',proof_status=previous['status'],
                        proof=core.hash_file(proof_path),exit_code=previous['exit_code'],large_files_read_this_call=False)
        core.need(not intent.exists(),'Existing incomplete intent requires explicit reconciliation, never blind replay')
        core.need(not list((base/'preparation').glob('joint-milestone-%04d-independent-full-sha-cpu-once-v1*.json'%outer)),
                  'Prior strict audit intent/proof prevents duplicate large-file read')
        ready=readiness(control)
        if ready['status']!='ACTUAL_JOINT_WRITERS_STOPPED_AND_FILES_STABLE':
            return dict(**ready,exit_code=2,large_files_read_this_call=False)
        core._once(intent,dict(run_id=actual['run_id'],outer=outer,started_unix=time.time(),pid=os.getpid(),
            proc_stat=Path('/proc/self/stat').read_text(),config_sha256=control['runtime_config_sha256'],readiness=ready))
        started=time.monotonic()
        try:
            pins=source_and_splits(base,actual,control)
            print(json.dumps(dict(stage='ACTUAL_SOURCE_SPLITS_AND_READY_JOINT_VERIFIED',outer=outer,unix=time.time())),flush=True)
            joint=core.read(base/'outers'/('outer-%06d'%outer)/'complete.json')
            details=core._audit_complete(base,actual,joint,outer,driver_validator=driver_components)
            core.need(details['metadata']['joint']['sha256']==ready['joint_complete_sha256']
                      ==core.hash_file(base/'outers'/('outer-%06d'%outer)/'complete.json')['sha256'],
                      'Immutable joint changed between readiness and completed read')
            core.need(core.hash_file(base/'preparation/launch-config.json')['sha256']==control['runtime_config_sha256'],
                      'Actual config changed during reader')
            result=dict(status=STATUS,exit_code=0,details=details,source_and_splits=pins,
                        full_current_milestone_sha_cpu_performed=True)
        except BaseException as error:
            result=dict(status='JOINT_COMPONENT_CPU_AUDIT_FAILED_TRAINING_UNCHANGED',exit_code=1,
                        error_type=type(error).__name__,error=str(error),full_current_milestone_sha_cpu_performed=False)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result.update(run_id=actual['run_id'],outer=outer,source_commit=actual['source_commit'],
            config_sha256=control['runtime_config_sha256'],joint_complete_sha256=ready['joint_complete_sha256'],
            elapsed_seconds=time.monotonic()-started,cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,
            max_rss_kib=usage.ru_maxrss,finished_unix=time.time(),readiness=ready,
            entry_source_sha256=core.hash_file(__file__)['sha256'],shared_reader_source_sha256=core.hash_file(core.__file__)['sha256'],
            training_modified=False,checkpoint_files_rewritten=False,gpu_forward_performed=False,cuda_rng_applied=False,
            old_milestone5_50_100_weights_read=False,native_cuda_executor_exact_resume_claimed=False,
            independent_backup_performed=False,large_payload_transferred_to_Mac=False)
        core._once(proof_path,result)
        return dict(status=result['status'],exit_code=result['exit_code'],error=result.get('error'),proof=core.hash_file(proof_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--control-config',required=True)
    args=parser.parse_args();result=run(args.control_config);print(json.dumps(result,sort_keys=True),flush=True)
    raise SystemExit(result['exit_code'])


if __name__=='__main__':main()
