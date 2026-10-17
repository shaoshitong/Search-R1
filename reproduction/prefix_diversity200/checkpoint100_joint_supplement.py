"""Append-only joint100 component audit; retain standalone-driver FAIL and never reread actor weights."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from .checkpoint_cpu_once import (canonical, digest, hash_file, load_torch_cpu, need,
    owned, read, signature, validate_pool_state, validate_probe_state, validate_rng)

CORE_SHA = '5ebc33355a6b6516bfe06cf04cb970887a95ec93e86eb0be4e3d7614865d8211'
FAILED_SHA = 'f917d16cf75a658e1fa066775d60f9053929bd7d099cc7a5c40d18e2579cbdcf'
CONFIG_SHA = '384cf0f516e7bf9ce3623bdf48e87a078f8eceaa5af5e03f238e04e86ae59ad8'
STATUS = 'JOINT100_COMPONENT_CPU_VERIFICATION_PASS_STANDALONE_DRIVER_FAIL_RETAINED'


def once(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False); handle.flush(); os.fsync(handle.fileno())


def verify_original_actor_read(base, root, joint, control):
    """Require completed stages, checked source control flow and unchanged immutable inodes."""
    previous = base/'acceptance100-cpu-control-v1'
    failure_path = base/'preparation/joint-milestone-0100-independent-full-sha-cpu-once-v1.json'
    failed_file = hash_file(failure_path)
    need(failed_file['sha256'] == FAILED_SHA, 'Original failed once proof changed')
    failed = read(failure_path)
    need(failed['status'] == 'PREFIX_SELECTION200_CPU_MILESTONE_AUDIT_FAILED_TRAINING_UNCHANGED'
         and failed['error'] == 'Actual actor driver state did not preserve the real candidate boundary'
         and failed['reader_source_sha256'] == CORE_SHA
         and failed['joint_complete_sha256'] == hash_file(root/'complete.json')['sha256'],
         'Unsupported original failed audit; no blind replay or inferred PASS')
    core_path = previous/'reproduction/prefix_diversity200/checkpoint_cpu_once.py'
    need(hash_file(core_path)['sha256'] == CORE_SHA, 'Original executed core source changed')
    start, terminal = read(previous/'reader-start.json'), read(previous/'reader-exit.json')
    need(start['CUDA_VISIBLE_DEVICES'] == '' and terminal['exit_code'] == 1
         and terminal['proof_sha256'] == FAILED_SHA and terminal['error'] == failed['error'],
         'Original actual reader identity/exit does not match the failed proof')
    expected_log = control/'evidence/ORIGINAL_READER.log'
    log_file = hash_file(previous/'reader.log')
    need(log_file['sha256'] == hash_file(expected_log)['sha256'], 'Original completed-stage log changed')
    rows = [json.loads(line) for line in (previous/'reader.log').read_text().splitlines()]
    events = [row for row in rows if 'stage' in row]
    actor = joint['actor']; hfdir = actor['hf_model_dir']
    need(len(events) == 20 and events[0]['stage'] == 'HF_FULL_SHA_CPU_BEGIN'
         and events[0]['path'] == hfdir and events[1]['stage'] == 'HF_FULL_SHA_CPU_COMPLETE'
         and events[1]['tensors'] == 435, 'Original HF full read never completed')
    for rank, item in enumerate(actor['rank_states']):
        begin, end = events[2+2*rank:4+2*rank]
        need(begin['stage'] == 'TORCH_FULL_SHA_CPU_BEGIN' and end['stage'] == 'TORCH_FULL_SHA_CPU_COMPLETE'
             and begin['path'] == end['path'] == item['path'] and end['bytes'] == item['bytes'],
             'Original rank CPU/SHA stage lacks an actual completed event')
    need(events[-2]['stage'] == 'TORCH_FULL_SHA_CPU_BEGIN'
         and events[-1]['stage'] == 'TORCH_FULL_SHA_CPU_COMPLETE'
         and events[-1]['path'] == actor['driver_state']['path']
         and rows[-1]['status'] == failed['status'] and rows[-1]['error'] == failed['error'],
         'Original reader did not reach the driver after all rank validations')
    ready = read(control/'evidence/READINESS.json')
    partial = read(control/'evidence/PARTIAL_ACTOR_RECEIPTS.json')
    need(ready['joint_complete_sha256'] == partial['joint_complete_sha256'] == failed['joint_complete_sha256']
         and ready['writer_ledger_all_stopped'] is True, 'Original readiness receipt belongs to another checkpoint')
    files = []
    for path in [Path(item['path']) for item in actor['rank_states']] + [Path(hfdir)/name for name in actor['hf_files']]:
        owned(base, path); info = path.stat(); now = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
        need(now == ready['files'][str(path)] == partial['current_file_stats'][str(path)],
             'Actor file changed since original completed full read; no cached evidence reuse')
        files.append(dict(path=str(path), signature=now, reread=False))
    # In this pinned core, each rank's full Adam/scheduler/RNG/ledger validation
    # occurs after its COMPLETE event and before the next BEGIN (driver for rank7).
    return dict(original_failed_proof=failed_file, original_completed_log=log_file,
        pinned_executed_core_sha256=CORE_SHA, original_reader_start=start, original_reader_exit=terminal,
        inherited_hf435_and_all8_full_sha_cpu_validation=True, completion_and_control_flow_verified=True,
        unchanged_actor_assets=files, large_actor_weight_files_read_this_call=False)


def supplement(base, control):
    import torch
    torch.set_num_threads(1)
    root = base/'outers/outer-000100'; cp = base/'preparation/launch-config.json'
    need(hash_file(cp)['sha256'] == CONFIG_SHA, 'Actual source/config binding changed')
    c, joint = read(cp), read(root/'complete.json')
    need(c['base'] == str(base) and c['source_commit'] == joint['source_commit']
         == 'e3463cfddca71a22ac684d75c875dbe123e8bc43'
         and c['run_id'] == joint['run_id'] == 'f3ee1435e5014ba1969387c91878893a'
         and joint['outer'] == joint['actor']['completed_outer'] == joint['probe']['completed_outer'] == 100
         and joint['actor']['actual_adam_steps_per_rank'] == joint['probe']['actual_adam_steps_per_rank'] == [2000]*8
         and joint['actor']['scheduler_completed_outer'] == 100, 'Actual joint100 scientific/counter identity differs')
    original = verify_original_actor_read(base, root, joint, control)
    pool_path = owned(base, joint['pool_snapshot']); pool_file = hash_file(pool_path)
    need(pool_file['sha256'] == joint['pool_sha256'], 'Actual joint pool snapshot producer SHA differs')
    pool, reservation = read(pool_path), read(root/'pool-reservation.json')
    binding = read(base/'question-pool/binding.json'); source_path = Path(c['source_question_pool'])
    source_file = hash_file(source_path)
    need(source_file['sha256'] == binding['sha256'], 'Canonical pool source changed')
    pool_summary = validate_pool_state(pool, reservation, binding, read(source_path), c, 100)
    manifest_path = root/'probe-score/manifest.json'; manifest = read(manifest_path)
    previous_joint = read(base/'outers/outer-000099/complete.json'); previous_probe = previous_joint['probe']
    need(previous_joint['run_id'] == c['run_id'] and previous_joint['outer'] == previous_probe['completed_outer'] == 99
         and previous_probe['actual_adam_steps_per_rank'] == [1980]*8
         and previous_probe['sha256'] == manifest['probe_before_checkpoint_sha256'],
         'Actual scoring-time probe99 receipt does not bind the original Top1024 scoring')
    print(json.dumps(dict(stage='ACTUAL_JOINT_POOL_AND_PROBE99_RECEIPT_VERIFIED', unix=time.time())), flush=True)
    driver, driver_file = load_torch_cpu(owned(base, joint['actor']['driver_state']['path']), joint['actor']['driver_state'])
    need(driver['run_id'] == c['run_id'] and driver['completed_outer'] == 100
         and driver['actor_version'] == joint['actor']['actor_version'], 'Small actor driver identity differs')
    driver_rng = validate_rng(driver['rng'])
    stale_pool = driver['candidate_pool_state']; stale_probe = driver['prefix_probe_receipt']
    need(stale_pool['path'] == c['pool_state_path'] and stale_probe['path'] == c['prefix_probe_receipt']
         and stale_pool['value'] != pool and stale_pool['value']['consumed_candidates'] == 271360
         and stale_pool['value']['next_outer'] == 54
         and stale_probe['value']['completed_outer'] == 52
         and stale_probe['value']['actual_adam_steps_per_rank'] == [1040]*8,
         'Stale driver condition changed; original failure must remain separately diagnosed')
    stale = dict(driver_file=driver_file, driver_rng=driver_rng,
        standalone_driver_status='FAIL_ACTUAL_STALE_POOL_AND_PREVIOUS_PROBE_DESCRIPTOR',
        saved_pool_path=stale_pool['path'], saved_pool_next_outer=54, saved_pool_consumed_candidates=271360,
        saved_previous_probe_path=stale_probe['path'], saved_previous_probe_outer=52)
    del driver; gc.collect()
    receipt = joint['probe']; probe_path = owned(base, base/'probe/head-outer-0100.pt')
    need(signature(probe_path.stat()) == signature(owned(base,joint['probe_snapshot']).stat()), 'Actual probe100 alias changed')
    probe, probe_file = load_torch_cpu(probe_path, receipt)
    rollout = read(root/'full/complete.json')
    probe_summary = validate_probe_state(probe, receipt, c, 100, rollout['actor_version'])
    score_file = hash_file(manifest_path); own_file = hash_file(root/'full/own-em.jsonl')
    need(probe['scored_features_manifest_sha256'] == score_file['sha256']
         and probe['own_em_sha256'] == own_file['sha256'] == rollout['own_em']['sha256'],
         'CPU probe100 lost actual ALL1024 own-prefix/outcome binding')
    test_path = Path(c['fixed_test_features'])/'manifest.json'; test_file = hash_file(test_path); test = read(test_path)
    need(test_file['sha256'] == c['fixed_test_manifest_sha256']
         and probe['binding']['encoder_files_sha256'] == manifest['encoder_files_sha256'] == test['spec']['encoder_files_sha256']
         and test['spec_sha'] == digest(test['spec']), 'Actual probe100 frozen encoder/fixed diagnostic differs')
    del probe; gc.collect()
    need(hash_file(cp)['sha256'] == CONFIG_SHA, 'Launch config changed during supplemental read')
    return dict(original_actor_read_evidence=original, pool=pool_summary, pool_file=pool_file,
        source_pool_file=source_file, actual_scoring_probe99=previous_probe,
        previous_probe99_receipt_cpu_weight_read=False, driver=stale,
        actual_probe100=dict(file=probe_file, summary=probe_summary), score_manifest_file=score_file,
        own_em_file=own_file, fixed_test_manifest_file=test_file,
        recovery_bindings=dict(actor_hf=joint['actor']['hf_model_dir'], actor_rank_state_receipt=joint['actor'],
            authoritative_candidate_pool_snapshot=str(pool_path), authoritative_current_probe_snapshot=joint['probe_snapshot'],
            selection_time_probe99_receipt_path=str(base/'outers/outer-000099/complete.json'), next_outer=101,
            ignore_legacy_driver_candidate_pool_and_prefix_probe_descriptors=True,
            require_explicit_joint_bundle_recovery_binding=True, native_cuda_executor_restore_performed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--control', required=True)
    args = parser.parse_args(); base, control = Path(args.base), Path(args.control)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and base.parent == Path('/mnt/data/shaoshitong')
         and base.name == 'search-g1-prefix-diversity200-20261016-f3ee1435e501-resume4'
         and control == base/'acceptance100-joint-supplement-v1', 'Unsupported CPU-only supplemental scope')
    with (control/'reader.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not (control/'intent.json').exists() and not (control/'proof.json').exists(), 'Existing supplement must not be replayed')
        started=time.monotonic(); once(control/'intent.json',dict(pid=os.getpid(),proc_stat=Path('/proc/self/stat').read_text(),started_unix=time.time()))
        try:
            details=supplement(base,control); result=dict(status=STATUS, details=details, exit_code=0)
        except BaseException as error:
            result=dict(status='JOINT100_SUPPLEMENT_FAILED_ORIGINAL_FAIL_RETAINED',error_type=type(error).__name__,error=str(error),exit_code=1)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result.update(finished_unix=time.time(),elapsed_seconds=time.monotonic()-started,cpu_user_seconds=usage.ru_utime,
            cpu_system_seconds=usage.ru_stime,max_rss_kib=usage.ru_maxrss,original_standalone_fail_overwritten=False,
            standalone_driver_pass_claimed=False,training_modified=False,gpu_forward_performed=False,
            native_cuda_executor_exact_resume_claimed=False,independent_backup_performed=False,
            helper_source_sha256=hash_file(__file__)['sha256'])
        once(control/'proof.json',result); print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'],
            proof=hash_file(control/'proof.json'),error=result.get('error'))),flush=True)
        raise SystemExit(result['exit_code'])


if __name__ == '__main__': main()
