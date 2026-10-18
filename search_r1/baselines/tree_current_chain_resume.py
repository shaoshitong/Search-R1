"""Explicit migration of job3339's completed native6 boundary, never generic recovery.

Only native state of this current Fresh200 chain is accepted. File hashes and
loaded/live state comparisons execute at training startup; no executor/vLLM
restart equivalence or previously observed reload result is claimed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket

from .training_state import (atomic_save, cpu_copy, digest, driver_snapshot,
    prepare_cold_fsdp_layout, restore_driver, restore_worker, verified_state_file,
    worker_snapshot)

PROTOCOL = 'current_fresh200_chain_fault_repair_continuation_v1'
PARENT_JOB = '3339'
PARENT_RUN = 'tree-current200-continue-3339'
PARENT_COMMIT = '62271081ba1bb74d8fc3f4958d623758efa74fa5'
PARENT_CONFIG_SHA256 = '98a88882c09d8bbba89908b04d1f72930b4fccccc2bbcb5b6eb6886584655f3b'
PARENT_MANIFEST_SHA256 = '2421b7f1f8f8db0d5b73e342e9de1a9c8b61a5fe08cb6596da492a0d3a04fbc3'
PARENT_COMPLETED_STEP = 6
PARENT_OPTIMIZER_CALLS = 120
PARENT_CURSOR = 3072
PARENT_NEXT_STEP = PARENT_COMPLETED_STEP + 1
PARENT_MANIFEST = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs')/PARENT_RUN/\
    'checkpoints/training_state/rolling-completed/global_step_6/manifest.json'
OFFICIAL_MODEL = '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725'
ENV_PREFIX = 'TREE_CURRENT_CHAIN_RESUME_'
ENV_FIELDS = {'MANIFEST', 'MANIFEST_SHA256', 'AUTHORITY'}
COUNTS = {str(rank): PARENT_OPTIMIZER_CALLS for rank in range(8)}
PARENT_MIGRATION_RECEIPT = PARENT_MANIFEST.parents[4]/'output/current-chain-migrated.json'
PARENT_MIGRATION_SHA256 = '6384b8350550ef767c8d1e0b5c27705842913517c642d8556d59b37058a98c3e'


def enabled(environ=None):
    environ = os.environ if environ is None else environ
    from .recursive_a_schedule_resume import enabled as recursive_enabled
    return recursive_enabled(environ) or any(key.startswith(ENV_PREFIX) for key in environ)


def require(condition, message):
    if not condition:
        raise ValueError('Current-chain3339 migration: ' + message)


def validate_entry(config=None, environ=None):
    environ = os.environ if environ is None else environ
    from .recursive_a_schedule_resume import enabled as recursive_enabled, validate_entry as recursive_validate
    if recursive_enabled(environ): return recursive_validate(config,environ)
    if not enabled(environ):
        return False
    require(not any(key.startswith('TREE_RECOVERY_') for key in environ),
            'old recovery environment cannot be mixed into this chain')
    supplied = {key[len(ENV_PREFIX):] for key in environ if key.startswith(ENV_PREFIX)}
    require(supplied == ENV_FIELDS, 'all three explicit entry fields are required')
    require(environ[ENV_PREFIX+'AUTHORITY'] == PROTOCOL, 'continuation scope differs')
    require(environ[ENV_PREFIX+'MANIFEST_SHA256'] == PARENT_MANIFEST_SHA256,
            'parent manifest SHA differs from the pinned native6 boundary')
    require(Path(environ[ENV_PREFIX+'MANIFEST']) == PARENT_MANIFEST,
            'only the byte-pinned preserved3339 native6 manifest is permitted')
    require(socket.gethostname().split('.')[0] == 'h100-10',
            'approved retired-node saved state must load only on retained18 h100-10')
    if config is not None:
        from .tree_fresh200 import validate_fresh200_config
        validate_fresh200_config(config)
        actor, rollout, ref = (config.actor_rollout_ref[k] for k in ('actor', 'rollout', 'ref'))
        require(config.actor_rollout_ref.model.path == ref.model_path == OFFICIAL_MODEL,
                'cold actor and reference must initialize from the official base')
        require(actor.get('training_state_enabled') is True
                and config.trainer.get('training_state_enabled') is True,
                'native training state must remain enabled')
        require(actor.optim.get('warmup_style') == 'constant'
                and config.baselines.seed == 42
                and actor.policy_safety.kl_log_bound == 30.
                and actor.policy_safety.monitor_rows_per_rank == 32
                and actor.policy_safety.negative_tail_limit == .01
                and actor.policy_safety.sampling_probability_correction.initial_actor_repeat_max_abs_delta == 0.0,
                'scheduler/seed/guard configuration differs')
        require((rollout.temperature, rollout.top_p, rollout.top_k, rollout.tensor_model_parallel_size)
                == (1., 1., -1, 1), 'sampling or tensor-parallel configuration differs')
    return True


def parent_manifest():
    require(validate_entry(), 'explicit current-chain entry is absent')
    raw = PARENT_MANIFEST.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_MANIFEST_SHA256,
            'actual parent manifest bytes changed')
    manifest = json.loads(raw)
    identity = manifest['identity']
    # Native rolling identity includes run_id; bind it alongside the exact bytes.
    expected = dict(job_id=PARENT_JOB, run_id=PARENT_RUN, code_commit=PARENT_COMMIT,
        config_sha256=PARENT_CONFIG_SHA256, completed_step=PARENT_COMPLETED_STEP,
        actual_update_limit=200)
    require(manifest['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED'
            and manifest['snapshot_kind'] == 'rolling_completed_outer'
            and all(identity.get(k) == v for k, v in expected.items()),
            'parent status/source/config/completed boundary differs')
    for key, variable in [('data_identity', 'SEARCH_R1_BASELINE_DATA_IDENTITY'),
                          ('model_identity', 'SEARCH_R1_MODEL_IDENTITY')]:
        require(identity.get(key) == os.environ[variable], 'parent ' + key + ' differs')
    from .tree_guard_continue import PROTOCOL as GUARD_PROTOCOL
    expected_guard = dict(protocol=GUARD_PROTOCOL, training_attempts=PARENT_COMPLETED_STEP,
        completed_outer_updates=PARENT_COMPLETED_STEP, refused_attempts=0, guard_refused_attempts=0,
        partial_optimizer_calls_per_rank={}, completed_optimizer_calls_per_rank=COUNTS,
        total_optimizer_calls_per_rank=COUNTS, partial_evidence=[])
    require(identity.get('guard_continue_accounting') == expected_guard,
            'parent complete/partial/attempt accounting differs')
    workers = manifest['worker_states']
    require(len(workers) == 8 and {r['identity']['rank'] for r in workers} == set(range(8)),
            'parent native rank coverage differs')
    for record in workers:
        rank_identity = record['identity']
        require(rank_identity == dict(identity, rank=rank_identity['rank'], world_size=8,
                    cuda_visible_devices=rank_identity.get('cuda_visible_devices'))
                and isinstance(rank_identity.get('cuda_visible_devices'), str)
                and bool(rank_identity['cuda_visible_devices'])
                and record['actual_total_optimizer_calls'] == PARENT_OPTIMIZER_CALLS
                and record['scheduler_last_epoch'] == PARENT_COMPLETED_STEP
                and record['full_worker_state_finite'] is True
                and record['observed_adam_steps'] == [PARENT_OPTIMIZER_CALLS]
                and record['exact_same_world_reload_verified'] is False,
                'parent native worker identity/counters differ')
    require(manifest['driver']['next_step'] == PARENT_NEXT_STEP
            and manifest['driver']['exact_reload_verified'] is False, 'parent driver next outer differs')
    return manifest


def state_record(record, filename):
    """Use a saved path when present; support only its exact parent sibling."""
    result = dict(record)
    expected = PARENT_MANIFEST.parent/filename
    path = Path(result.get('path', str(expected)))
    require(path == expected and not path.is_symlink(), 'state path leaves pinned parent boundary')
    result['path'] = str(path)
    if 'bytes' in result:
        require(path.stat().st_size == result['bytes'], 'native state byte count changed')
    return result


def model_section(state):
    return {key: state[key] for key in ('local_shards', 'parameters', 'buffers')}


def load_worker(worker):
    from .recursive_a_schedule_resume import enabled as recursive_enabled, load_worker as recursive_load
    if recursive_enabled(): return recursive_load(worker)
    require(worker._is_actor and worker.config.actor.get('training_state_enabled', False),
            'native actor state is disabled')
    require(worker.world_size == 8 and not worker.actor_optimizer.state
            and worker.actor_lr_scheduler.last_epoch == 0
            and worker.actor._fresh_total_committed_optimizer_calls == 0,
            'cold worker is not untouched official initialization')
    manifest = parent_manifest()
    record = next(r for r in manifest['worker_states'] if r['identity']['rank'] == worker.rank)
    record = state_record(record, f'rank-{worker.rank:02d}.pt')
    saved = verified_state_file(record)
    logical = digest(saved)
    require(logical == record['logical_state_sha256'] and saved['identity'] == record['identity'],
            'actual serialized worker state or identity changed')
    require(saved['optimizer']['state']
            and all(int(value['step']) == PARENT_OPTIMIZER_CALLS
                    for value in saved['optimizer']['state'].values())
            and saved['scheduler']['last_epoch'] == PARENT_COMPLETED_STEP,
            'actual serialized Adam/scheduler counters differ')
    model_hash = digest(model_section(saved))
    model_bytes = sum(t.numel()*t.element_size() for values in model_section(saved).values()
                      for t in values.values())
    prepare_cold_fsdp_layout(worker.actor_module_fsdp, worker._is_offload_param, worker._is_offload_grad)
    restore_worker(worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler,
                   saved, record['identity'], cold=True)
    del saved
    worker.actor._fresh_total_committed_optimizer_calls = PARENT_OPTIMIZER_CALLS
    actual = worker_snapshot(worker.actor_module_fsdp, worker.actor_optimizer,
                             worker.actor_lr_scheduler, record['identity'], cuda=True)
    live_logical = digest(actual)
    live_model_hash = digest(model_section(actual))
    require(live_logical == logical and live_model_hash == model_hash,
            'actual cold-loaded model/Adam/scheduler/RNG differs from saved parent')
    del actual
    adam_steps = sorted({int(value['step']) for value in worker.actor_optimizer.state.values()})
    require(adam_steps == [PARENT_OPTIMIZER_CALLS]
            and worker.actor_lr_scheduler.last_epoch == PARENT_COMPLETED_STEP,
            'actual cold-loaded Adam/scheduler counters differ')
    return dict(status='COLD_CURRENT_CHAIN_WORKER_STATE_MIGRATED', protocol=PROTOCOL,
        rank=worker.rank, world_size=worker.world_size, parent_identity=record['identity'],
        parent_manifest_sha256=PARENT_MANIFEST_SHA256, state_path=record['path'],
        state_file_sha256=record['sha256'], state_file_bytes=Path(record['path']).stat().st_size,
        actual_file_sha256_verified=True, serialized_logical_state_sha256=logical,
        actual_loaded_logical_state_sha256=live_logical,
        loaded_model_section_sha256=live_model_hash, loaded_model_tensor_bytes=model_bytes,
        model_tensor_byte_scope='Stored local_shards+parameters+buffers; sections may duplicate storage, not unique allocation bytes',
        actual_cold_native_state_equal=True, adam_steps=adam_steps,
        scheduler_step=worker.actor_lr_scheduler.last_epoch,
        actual_total_optimizer_calls=PARENT_OPTIMIZER_CALLS,
        child_code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        child_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        executor_equivalence=False, automatic_resume_supported=False)


def migrate_trainer(trainer):
    from .recursive_a_schedule_resume import enabled as recursive_enabled, migrate_trainer as recursive_migrate
    if recursive_enabled(): return recursive_migrate(trainer)
    require(validate_entry(trainer.config), 'explicit current-chain entry is absent')
    require(len(trainer.train_dataset) == 169028 and len(trainer.val_dataset) == 512,
            'fixed training/dev split sizes differ')
    child_commit = os.environ['SEARCH_R1_BASELINE_COMMIT']
    require(re.fullmatch('[0-9a-f]{40}', child_commit) is not None
            and child_commit != PARENT_COMMIT, 'new source commit must identify the current continuation binding')
    manifest = parent_manifest()
    workers = trainer.actor_rollout_wg.load_current_chain_training_state()
    require(len(workers) == 8 and {r['rank'] for r in workers} == set(range(8))
            and all(r['status'] == 'COLD_CURRENT_CHAIN_WORKER_STATE_MIGRATED'
                and r['actual_cold_native_state_equal'] is True and r['actual_file_sha256_verified'] is True
                and r['parent_manifest_sha256'] == PARENT_MANIFEST_SHA256
                and r['scheduler_step'] == PARENT_COMPLETED_STEP
                and r['adam_steps'] == [PARENT_OPTIMIZER_CALLS]
                and r['actual_total_optimizer_calls'] == PARENT_OPTIMIZER_CALLS
                and r['child_code_commit'] == child_commit
                for r in workers), 'actual cold worker receipts are incomplete')
    record = state_record(manifest['driver'], 'driver.pt')
    saved = verified_state_file(record)
    require(digest(saved) == record['logical_state_sha256'] and saved['identity'] == manifest['identity'],
            'actual serialized driver state or identity changed')
    require(saved['completed_step'] == saved['actual_outer_updates'] == PARENT_COMPLETED_STEP
            and saved['next_step'] == PARENT_NEXT_STEP
            and saved['sampler']['batch_size'] == trainer.config.data.train_batch_size == 512
            and saved['sampler']['cursor'] == PARENT_CURSOR == PARENT_COMPLETED_STEP * trainer.config.data.train_batch_size
            and saved['sampler']['epoch'] == 0, 'actual saved counters/cursor differ')
    restore_driver(trainer, saved, manifest['identity'])
    require(digest(driver_snapshot(trainer, manifest['identity'])) == record['logical_state_sha256'],
            'actual cold-loaded driver/sampler/Tree RNG differs from parent')
    saved_parent_provenance = cpu_copy(saved.get('recovery_provenance'))
    require(isinstance(saved_parent_provenance, dict)
            and saved_parent_provenance.get('protocol') == PROTOCOL
            and saved_parent_provenance.get('same_fresh200_chain') is True
            and saved_parent_provenance.get('child_job_id') == PARENT_JOB
            and saved_parent_provenance.get('child_run_id') == PARENT_RUN
            and saved_parent_provenance.get('child_code_commit') == PARENT_COMMIT
            and saved_parent_provenance.get('child_config_sha256') == PARENT_CONFIG_SHA256,
            'saved parent continuation provenance differs')
    parent_receipt_raw = PARENT_MIGRATION_RECEIPT.read_bytes()
    require(hashlib.sha256(parent_receipt_raw).hexdigest() == PARENT_MIGRATION_SHA256,
            'saved parent full ancestry receipt bytes changed')
    parent_receipt = json.loads(parent_receipt_raw)
    require(parent_receipt['status'] == 'COLD_CURRENT_CHAIN_STATE_MIGRATED'
            and saved_parent_provenance == parent_receipt['provenance'],
            'saved parent complete ancestry differs from actual3339 migration receipt')
    provenance = dict(protocol=PROTOCOL, parent_identity=manifest['identity'],
        parent_manifest_path=str(PARENT_MANIFEST), parent_manifest_sha256=PARENT_MANIFEST_SHA256,
        parent_completed_step=PARENT_COMPLETED_STEP, parent_cursor=PARENT_CURSOR,
        parent_epoch=0, new_batch_size=512, saved_parent_provenance=saved_parent_provenance,
        same_fresh200_chain=True, child_code_commit=child_commit,
        child_config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        child_job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(), child_run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        parent_failed_attempt=dict(attempt_id=PARENT_NEXT_STEP, completed_outer_credit=0,
            optimizer_writes=None, published_rank_optimizer_record=None,
            error_type='HumanRequestedStop', failure_component='manual_stop_slow3339',
            scope='User stopped slow3339 after completed6; interrupted7 partial writes remain unknown; only completed native6 is migrated'),
        automatic_resume_supported=False, executor_equivalence=False, exact_executor_resume=False)
    # The existing snapshot format preserves this provenance without old env,
    # sampler migration, old300 protocol or a ten-write restriction.
    trainer.recovery_provenance = cpu_copy(provenance)
    trainer.current_chain_resume_provenance = cpu_copy(provenance)
    trainer.acceptance_released = True
    receipt = dict(status='COLD_CURRENT_CHAIN_STATE_MIGRATED', protocol=PROTOCOL,
        provenance=provenance, workers=workers, driver=dict(state_path=record['path'],
            state_file_sha256=record['sha256'], state_file_bytes=Path(record['path']).stat().st_size,
            actual_file_sha256_verified=True, actual_cold_driver_state_equal=True,
            completed_outer_updates=trainer.actual_outer_updates, next_outer_update=trainer.global_steps+1,
            sampler_cursor=trainer.checkpoint_sampler.cursor, sampler_epoch=trainer.checkpoint_sampler.epoch,
            training_attempts=trainer.training_attempts,
            completed_optimizer_calls_per_rank=trainer.completed_optimizer_calls_per_rank),
        cold_official_initialization=trainer.fresh_initialization_provenance,
        historical_parent_automatic_resume_supported=manifest.get('automatic_resume_supported', False),
        executor_equivalence=False, initial_evaluation_added=False, step5_approval_hold_added=False)
    atomic_save(receipt, Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'current-chain-migrated.json',
                json_format=True)
    print('TREE_COLD_CURRENT_CHAIN_STATE_MIGRATED=' + str(trainer.actual_outer_updates), flush=True)
    return receipt
