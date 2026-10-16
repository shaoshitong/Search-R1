"""Final200 native4 acceptance from copied target bytes, never the train loop."""
import argparse
import copy
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.prefix20_joint4.restore_only_assets import (
    ROLE, prepare, publish, regular_under, sha, verify_target_copy,
)


def invariant_actor(row):
    return {key:row[key] for key in ('rank', 'actual_cuda_uuid', 'actual_optimizer_writes',
        'actual_counters', 'scheduler_last_epoch', 'scheduler_step_count',
        'model_hf_exact_tensors_checked_on_rank0', 'model_equal', 'optimizer_equal',
        'scheduler_equal', 'boundary_rng_equal', 'forbidden_call_counts')} | {
        'process_birth':{key:row['process_identity'][key] for key in ('pid', 'start_ticks', 'uid', 'sid')}}


def validate_actor_readbacks(rows, allocation, expected_writes):
    if [row['rank'] for row in rows] != list(range(4)):
        raise ValueError('RESTORE_ONLY_ACTUAL_FOUR_RANK_READBACKS_REQUIRED')
    if {row['actual_cuda_uuid'] for row in rows} != set(allocation['actual_cuda_runtime_uuids']):
        raise ValueError('RESTORE_ONLY_ACTUAL_GPU_UUIDS_DIFFER')
    births = {(row['process_identity']['pid'], row['process_identity']['start_ticks']) for row in rows}
    if len(births) != 4:
        raise ValueError('RESTORE_ONLY_DISTINCT_ACTUAL_WORKER_BIRTHS_REQUIRED')
    for row in rows:
        if (row['actual_optimizer_writes'] != expected_writes or row['actual_counters']['completed_outer'] != 200 or
                row['scheduler_last_epoch'] != 200 or row['scheduler_step_count'] != 201 or
                row['forbidden_call_counts'] or not all(row[name] for name in
                ('model_equal', 'optimizer_equal', 'scheduler_equal', 'boundary_rng_equal'))):
            raise ValueError('RESTORE_ONLY_ACTUAL_STATE_OR_COUNTER_DIFFERS')


def driver_task(config):
    """Runs in a separate CPU Ray driver; actor RNG is never driver RNG."""
    import ray
    import torch
    from verl.utils import hf_tokenizer
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.trainer.main_ppo_eco import EcoRewardManager
    from search_g1.qprobe_checkpoint import restore_driver
    from search_g1.prefix20_restore_only import assert_target_receipt, check_pending_membership
    from search_g1.prefix20_runtime import Prefix20Runtime
    from search_g1.prefix20_data import Prefix20CandidateStream
    from search_g1.training_state import compact_stream_state, driver_rng
    from search_g1.full_state_restore import equal_tree
    base = Path(os.environ['SEARCH_G1_BASE']).resolve()
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    receipt_path = run['full_state_restore']['source_receipt']
    _, receipt, _, _, _ = assert_target_receipt(receipt_path)
    tokenizer = hf_tokenizer(run['initial_model'], trust_remote_code=False)
    # No reference/retriever/reward GPU service is required to restore the
    # requested actor/probe states. Actor FSDP/Adam layout is unchanged.
    mapping = {Role.ActorRollout:'restore_pool'}
    manager = ResourcePoolManager(resource_pool_spec={'restore_pool':[4]}, mapping=mapping)
    trainer = RayPPOTrainer(config=config, tokenizer=tokenizer,
        role_worker_mapping={Role.ActorRollout:ray.remote(ActorRolloutRefWorker)},
        resource_pool_manager=manager, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=EcoRewardManager(tokenizer, num_examine=0),
        val_reward_fn=EcoRewardManager(tokenizer, num_examine=0, reward_mode='eval'))
    # This loads actual copied HF tensors, then existing per-rank Adam,
    # scheduler and boundary RNG. It performs no policy or data iteration.
    trainer.init_workers()
    restore_driver(trainer, receipt_path)
    stream = Prefix20CandidateStream(trainer.train_dataloader, config.prefix20_probe.eligible_pool_path,
                                     restored_state=trainer._restored_candidate_state)
    runtime = Prefix20Runtime(config.prefix20_probe, base, trainer.actor_rollout_wg)
    state = runtime.restore(receipt['question_probe_state_path'], receipt['actor_snapshot_id'])
    saved_driver = torch.load(receipt['boundary_driver']['path'], map_location='cpu', weights_only=False)
    from reproduction.common8.recoverable_attempt import AttemptProgress
    progress = AttemptProgress(run)
    for key in ('completed_outer', 'cumulative_writes', 'attempt_index', 'aborted_attempts', 'full_budget_outer'):
        setattr(progress, key, saved_driver['attempt_progress'][key])
    trainer.global_steps = 200
    trainer.inline_attempt_progress, trainer.mixed_stream, trainer.qprobe_runtime = progress, stream, runtime
    if not runtime.completed_attempts <= set(saved_driver['used_attempt_ids']):
        raise ValueError('RESTORE_ONLY_PROBE_ATTEMPTS_OUTSIDE_ACTOR_HISTORY')
    saved_probe = [torch.load(Path(receipt['question_probe_state_path']).parent / Path(row['path']).name,
                  map_location='cpu', weights_only=False) for row in state['rank_files']]
    expected_writes = receipt['attempt_progress']['cumulative_writes']
    allocation = json.loads((base / 'logs/native-allocation-admission.json').read_text())
    first = None
    for phase in ('A', 'B'):
        rows = trainer.actor_rollout_wg.readback_mixed_full_state(receipt_path, phase)
        validate_actor_readbacks(rows, allocation, expected_writes)
        probes = runtime.transport('readback', [{} for _ in range(4)])
        if [row['rank'] for row in probes] != list(range(4)):
            raise ValueError('RESTORE_ONLY_PROBE_RANKS_DIFFER')
        for saved, actual in zip(saved_probe, probes):
            equal_tree(saved, actual['state'])
            if (actual['state']['actual_steps'] != 2000 or actual['state']['completed_outer'] != 200 or
                    not actual['private_rng_activated_and_equal'] or not actual['actor_rng_unchanged'] or actual['forbidden_call_counts']):
                raise ValueError('RESTORE_ONLY_PROBE_ACTUAL_READBACK_DIFFERS')
        equal_tree(state['selection_rng'], runtime.selection_rng.bit_generator.state)
        equal_tree(state['candidate_state'], compact_stream_state(stream.state()))
        equal_tree(saved_driver['candidate_stream'], compact_stream_state(stream.state()))
        equal_tree(saved_driver['rng'], driver_rng())
        equal_tree(saved_driver['attempt_progress'], trainer._restored_attempt_progress)
        actual_progress = {key:getattr(progress, key) for key in
            ('completed_outer', 'cumulative_writes', 'attempt_index', 'aborted_attempts', 'full_budget_outer')}
        equal_tree(saved_driver['attempt_progress'], actual_progress)
        if (trainer.global_steps != 200 or progress.pending is not None or
                runtime.completed_outer != 200 or runtime.pending is not None or runtime.completed_attempts != set(state['completed_attempts'])):
            raise ValueError('RESTORE_ONLY_DRIVER_PROBE_COUNTERS_DIFFER')
        membership = check_pending_membership(stream)
        equal_tree(saved_driver['rng'], driver_rng())
        actual = dict(actors=[invariant_actor(row) for row in rows], candidate=copy.deepcopy(compact_stream_state(stream.state())),
                      selection_rng=copy.deepcopy(runtime.selection_rng.bit_generator.state),
                      driver_rng=copy.deepcopy(driver_rng()), attempt_progress=copy.deepcopy(actual_progress), global_steps=trainer.global_steps)
        if first is None:
            first = actual
        else:
            equal_tree(first, actual)
        from reproduction.a800.owned_processes import identity as process_identity
        proof = dict(role=ROLE, phase=phase, source_step=200, source_run_id=receipt['run_id'],
            driver_process_identity=process_identity(os.getpid()),
            source_receipt_sha256=sha(receipt_path), actual_actor_readbacks=rows,
            actual_probe_readbacks=[{key:value for key,value in row.items() if key != 'state'} | {
                'completed_outer':row['state']['completed_outer'], 'actual_steps':row['state']['actual_steps'],
                'model_optimizer_rng_exact_equal':True} for row in probes],
            candidate_dataequal=True, selector_rng_equal=True, driver_rng_equal=True,
            partial_pending_membership=membership, source_attempt_progress=saved_driver['attempt_progress'],
            optimizer_write_increment=0, scheduler_advance_increment=0, probe_update_increment=0,
            candidate_consumption_increment=0, generation_calls=0, training_checkpoint_save_calls=0,
            independent_second_readback_equal=(phase == 'B'), checked_unix=time.time())
        publish(base / 'logs' / ('restore-only-readback-' + phase + '.json'), proof)
        del probes
    forbidden = [base / 'logs' / name for name in ('training-launch-claim.json', 'forbidden-training.log')]
    forbidden += list((base / 'logs').glob('optimizer-rank-*.jsonl')) + list((base / 'logs').glob('scheduler-rank-*.jsonl'))
    if any(path.exists() for path in forbidden) or list((base / 'checkpoints').rglob('global_step_201')):
        raise ValueError('RESTORE_ONLY_TRAINING_OR_201_ARTIFACT_APPEARED')
    return dict(role=ROLE, completed_outer=200, accepted=True, actual_native4_cuda_restore=True,
                actual_two_independent_readbacks=True, all_update_consumption_increments=0,
                no_update201=True, checked_unix=time.time())


def execute(base):
    from reproduction.prefix20_joint4.control import own_base, bind_repo
    from reproduction.common8.native_allocation import admit
    from reproduction.common8.configuration import resolved
    from reproduction.common8.native_ray import root_for, identity
    base = own_base(base)
    if os.environ.get('SEARCH_G1_BASE') != str(base) or os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE') != '4':
        raise ValueError('RESTORE_ONLY_EXPLICIT_REMOTE_NATIVE4_ENV_REQUIRED')
    proof = verify_target_copy(base, base / 'preparation/restore-copy-complete.json')
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    bind_repo(base, run['source_commit'])
    if Path(__file__).resolve() != base / 'repo/reproduction/prefix20_joint4/restore_only.py':
        raise ValueError('RESTORE_ONLY_ENTRY_MUST_LOAD_BOUND_TARGET_REPO')
    if run.get('restore_only_role') != ROLE or not os.environ.get('SLURM_JOB_ID', '').isdecimal():
        raise ValueError('RESTORE_ONLY_ACTUAL_SLURM_ROLE_REQUIRED')
    lock = (base / 'restore-only.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish(base / 'preparation/restore-execution-claim.json', dict(role=ROLE, job=os.environ['SLURM_JOB_ID'],
        copy_receipt_sha256=sha(base / 'preparation/restore-copy-complete.json'), checked_unix=time.time()))
    allocation = admit(base)
    if allocation['training_world_size'] != 4:
        raise ValueError('RESTORE_ONLY_NATIVE4_ALLOCATION_REQUIRED')
    from reproduction.common8.native_allocation import fields
    from reproduction.common8.stage_retrieval import memory_bytes
    job = fields(allocation['actual_allocation_detail'])
    expected = dict(UserId='jxu957(14090)', Account='jxu957', QOS='formal-user',
                    Partition='emergency_acd', JobState='RUNNING', NumCPUs='48')
    if (os.getuid() != 14090 or any(job.get(key) != value for key,value in expected.items()) or
            memory_bytes(job['TRES']) < 512 * 2**30):
        raise ValueError('RESTORE_ONLY_ACTUAL_OWNER_ACCOUNT_CPU_MEMORY_DIFFERS')
    _, config, _ = resolved(base)
    private_ray = root_for(run['native_ray_execution_id'])
    private_ray.mkdir(mode=0o700, exist_ok=False)
    publish(base / 'preparation/restore-ray-binding.json', dict(path=str(private_ray), identity=identity(private_ray)))
    os.environ['RAY_TMPDIR'] = str(private_ray)
    os.environ['ECO_NO_SELF'] = '1'
    import ray
    if ray.is_initialized():
        raise ValueError('RESTORE_ONLY_PRIVATE_FRESH_RAY_REQUIRED')
    try:
        ray.init(address='local', num_gpus=4, _temp_dir=str(private_ray),
                 runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'true', 'NCCL_DEBUG':'WARN'}})
        result = ray.get(ray.remote(num_cpus=1)(driver_task).remote(config))
    finally:
        ray.shutdown()
    from reproduction.a800.owned_processes import identity as process_identity
    readback = json.loads((base / 'logs/restore-only-readback-B.json').read_text())
    births = [row['process_identity'] for row in readback['actual_actor_readbacks']] + [readback['driver_process_identity']]
    def alive(birth):
        try:
            current = process_identity(birth['pid'])
            return current['state'] != 'Z' and all(current[key] == birth[key] for key in ('start_ticks', 'uid', 'sid'))
        except (FileNotFoundError, ProcessLookupError):
            return False
    deadline = time.monotonic() + 30
    while any(alive(birth) for birth in births):
        if time.monotonic() >= deadline:
            raise RuntimeError('RESTORE_ONLY_OWNED_WORKERS_REMAIN_AFTER_RAY_SHUTDOWN')
        time.sleep(.25)
    deadline = time.monotonic() + 30
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True, timeout=30)
        assigned = set(allocation['actual_cuda_runtime_uuids'])
        applications = [[value.strip() for value in row] for row in csv.reader(io.StringIO(raw)) if row]
        occupied = [row for row in applications if row[0] in assigned]
        if not occupied:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('RESTORE_ONLY_ASSIGNED_CUDA_CONTEXTS_REMAIN: ' + repr(occupied))
        time.sleep(.25)
    publish(base / 'logs/restore-only-owned-release.json', dict(role=ROLE,
            recorded_process_births=births, all_recorded_processes_terminal=True,
            assigned_gpu_contexts_released=True, actual_compute_apps_stdout=raw, checked_unix=time.time()))
    # Original and copied checkpoint inventories must still match after readback.
    verify_target_copy(base, base / 'preparation/restore-copy-complete.json')
    result = dict(result, copied_target_file_count=len(proof['files']), copied_target_bytes=proof['copied_bytes'],
                  actual_slurm_job_id=allocation['slurm_job_id'])
    publish(base / 'logs/restore-only-complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('copy', 'execute'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source-receipt', type=Path)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.mode == 'copy':
        if args.source_receipt is None or args.commit is None:
            parser.error('copy requires --source-receipt and --commit')
        result = dict(copy_receipt=str(prepare(args.base, args.source_receipt, args.commit)),
                      actual_cuda_restore_accepted=False)
    else:
        result = execute(args.base)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
