"""Independent final200 native4 state reload; no training, generation, or update201."""
import argparse
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

ROLE = 'easy2560_final200_restore_only'
STATUS = 'EASY2560_FINAL200_ACTUAL_NATIVE4_RESTORE_ONLY_PASS'


def need(condition, message):
    if not condition: raise ValueError('EASY_FINAL_RESTORE_' + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def host_identity():
    return dict(hostname=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid_namespace=os.readlink('/proc/self/ns/pid'))


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def source(path, *, rehash_checkpoint=False):
    from search_g1.easy2560_final_checkpoint import source as final_source
    result = final_source(path, rehash_checkpoint=rehash_checkpoint)
    _, receipt, _, _, parent = result
    from reproduction.common8.precision_contract import easy2560_native4
    need(receipt['step'] == 200 and receipt['world_size'] == 4 and
         type(receipt['hf_saved_tensor_count']) is int and receipt['hf_saved_tensor_count'] > 0 and
         easy2560_native4(parent) and
         parent.get('question_probe') is None, 'FINAL200_EASY_NATIVE4_SOURCE_REQUIRED')
    return result


def target_run(parent, base, commit, source_map, path, receipt, checkpoint, origin):
    run = copy.deepcopy(parent)
    for key in ('full_state_restore', 'ray_tmp_dir', 'ray_tmp_dir_identity', 'ray_tmp_retention',
                'initial_reward_probe_source_base', 'slurm_job_id', 'shared_gpu_cohosts', 'shared_gpu_policy'):
        run.pop(key, None)
    run.update(source_commit=commit, source_sha256=source_map, restore_only_role=ROLE,
        completed_updates=0, native_ray_execution_id=uuid.uuid4().hex,
        checkpoint_dir=str(base/'checkpoints/search-g1-common8'),
        train_log=str(base/'logs/forbidden-training.log'), pretraining_baseline_policy='reuse_historical',
        initial_probe_version=receipt['reward_probe_version'],
        full_state_restore=dict(kind=ROLE, source_receipt=str(path), source_receipt_sha256=sha(path),
            source_checkpoint=str(checkpoint), source_base=str(origin), source_job=receipt['source_job'],
            source_commit=receipt['source_commit'], completed_outer=200,
            actual_optimizer_writes_per_rank=receipt['attempt_progress']['cumulative_writes'],
            attempt_progress=copy.deepcopy(receipt['attempt_progress']),
            candidate_source_cursor=receipt['driver_state']['source_consumed'],
            reward_probe_version=receipt['reward_probe_version'], reward_probe_round=receipt['reward_probe_round']))
    return run


def prepare_flash3_runtime(base, run):
    """Keep the source FA3 bytes, with paths owned by this new restore target."""
    from search_g1.easy7b_infra import flags
    if not flags(run).get('flash_attention_3_actor', False):
        return None
    from .flash3_runtime import prepare
    original = run.get('infra_flash_attention_3')
    private = prepare(base)
    need(isinstance(original, dict) and set(original) == set(private) and
         all(original[role + '_sha256'] == private[role + '_sha256']
             for role in ('interface', 'extension', 'cudart')),
         'SAVED_FLASH3_RUNTIME_SHA_DIFFERS')
    run['infra_flash_attention_3'] = private
    return private


def ray_env_vars(run):
    """Pass the preloaded runtime explicitly to the driver and all four actors."""
    from search_g1.easy7b_infra import flags
    values = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}
    if flags(run).get('flash_attention_3_actor', False):
        preload = os.environ.get('LD_PRELOAD', '')
        loaded = {str(Path(path).resolve()) for path in preload.split(':') if path}
        need(str(Path(run['infra_flash_attention_3']['cudart_path']).resolve()) in loaded,
             'PRIVATE_FLASH3_CUDART_NOT_PRELOADED')
        values['LD_PRELOAD'] = preload
    return values


def prepare(base, receipt_path, commit):
    """Reuse source CP200; copy saved heads and the optional private FA3 runtime."""
    from .control import bootstrap, own_base
    from reproduction.common8.source_files import runtime_files
    base = own_base(base); path, receipt, checkpoint, origin, parent = source(receipt_path)
    need(base != origin and not (base/'logs/run-manifest.json').exists(), 'NEW_DISTINCT_TARGET_REQUIRED')
    bootstrap(base, commit, str(origin))
    publish(base/'preparation/final-restore-prepare-claim.json', dict(role=ROLE,
        source_receipt=record(path), source_checkpoint=str(checkpoint), tensor_copies=0, checked_unix=time.time()))
    for name in ('calibration-300.parquet', 'audit.json'):
        destination = base/'data'/name; need(not destination.exists(), 'TARGET_DERIVATIVE_ALREADY_EXISTS')
        destination.symlink_to((origin/'data'/name).resolve())
    run = target_run(parent, base, commit,
        {str(p.relative_to(base/'repo')):sha(p) for p in runtime_files(base/'repo')},
        path, receipt, checkpoint, origin)
    prepare_flash3_runtime(base, run)
    live = read(receipt['reward_probe_manifest']['path'])
    need(live['run_id'] == run['run_id'] and live['round'] == receipt['reward_probe_round'] == 150 and
         live['version'] == receipt['reward_probe_version'], 'SAVED_ROUND150_GROUNDING_REQUIRED')
    heads = base/'artifacts/final-restored-grounding/heads'; heads.mkdir(parents=True)
    source_heads = Path(live['commit']).parent
    for name, key in (('commit_probe.pkl', 'commit'), ('boundary_probe.pkl', 'boundary')):
        original = Path(live[key]); need(sha(original) == live['head_sha256'][name], 'SAVED_GROUNDING_SHA_DIFFERS')
        with original.open('rb') as src, (heads/name).open('xb') as dst:
            import shutil
            shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
        need(sha(heads/name) == live['head_sha256'][name], 'COPIED_GROUNDING_SHA_DIFFERS')
        live[key] = str(heads/name)
    if live.get('calibration_report_sha256'):
        original = source_heads/'calibration-report.json'
        need(sha(original) == live['calibration_report_sha256'], 'SAVED_CALIBRATION_REPORT_SHA_DIFFERS')
        with (heads/'calibration-report.json').open('xb') as stream:
            stream.write(original.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    publish(heads.parent/'version-manifest.json', live)
    publish(base/'artifacts/live/manifest.json', live)
    publish(base/'logs/run-manifest.json', run)
    return dict(role=ROLE, base=str(base), source_receipt=record(path), source_checkpoint=str(checkpoint),
                tensor_copies=0, actual_cuda_restore_accepted=False)


def blocked(worker, label):
    def reject(*args, **kwargs):
        worker._easy_final_forbidden_calls[label] = worker._easy_final_forbidden_calls.get(label, 0) + 1
        raise RuntimeError('EASY_FINAL_RESTORE_MUTATION_FORBIDDEN: ' + label)
    return reject


def install_guards(worker):
    worker._easy_final_forbidden_calls = {}
    worker.actor_optimizer.step = blocked(worker, 'actor_optimizer_step')
    scheduler = worker.actor_lr_scheduler
    scheduler.__class__ = type('EasyFinalReadOnly' + type(scheduler).__name__, (type(scheduler),),
                              {'step': blocked(worker, 'actor_scheduler_step')})
    worker.rollout.inference_engine.generate = blocked(worker, 'rollout_generate')
    worker.rollout.inference_engine.llm_engine.add_request = blocked(worker, 'vllm_add_request')


def collective(function, label):
    import torch.distributed as dist
    value = None; failure = None
    try: value = function()
    except Exception as error: failure = type(error).__name__ + ': ' + str(error)[:512]
    failures = [None] * dist.get_world_size(); dist.all_gather_object(failures, failure)
    need(not any(failures), 'COLLECTIVE_' + label + ': ' + repr(failures))
    return value


def readback_actor(worker, receipt_path, phase):
    """Collective actual FSDP readback, without another load or any optimizer step."""
    import torch
    from safetensors import safe_open
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad, offload_fsdp_param_and_grad
    from search_g1.training_state import cpu_tree, worker_rng
    from search_g1.full_state_restore import equal_tree
    need(phase in ('A', 'B'), 'TWO_EXPLICIT_PHASES_REQUIRED')
    worker._assert_generation_phase_inactive()
    def inputs():
        path, receipt, checkpoint, _, _ = source(receipt_path)
        row = next(row for row in receipt['files'] if row['relative_path'] == 'training_state/rank-%d.pt' % worker.rank)
        need(sha(row['path']) == row['full_read_sha256'], 'ACTUAL_RANK_STATE_SHA_DIFFERS')
        saved = torch.load(row['path'], map_location='cpu', weights_only=False)
        need(saved['rank'] == worker.rank and saved['world_size'] == 4 and saved['completed_outer'] == 200,
             'SAVED_NATIVE4_COUNTER_DIFFERS')
        return receipt, checkpoint, saved, row, sha(path)
    receipt, checkpoint, saved, state_record, receipt_sha = collective(inputs, 'INPUTS')
    if worker._is_offload_param:
        load_fsdp_param_and_grad(worker.actor_module_fsdp, device_id=torch.cuda.current_device(), load_grad=False)
    try:
        with FSDP.state_dict_type(worker.actor_module_fsdp, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            model = worker.actor_module_fsdp.state_dict()
        def compare_model():
            checked = 0
            if worker.rank == 0:
                weights = read(checkpoint/'model.safetensors.index.json')['weight_map']
                need(set(model) == set(weights), 'ACTUAL_MODEL_TENSOR_SET_DIFFERS')
                for name in sorted(set(weights.values())):
                    with safe_open(str(checkpoint/name), framework='pt', device='cpu') as shard:
                        for key in shard.keys(): equal_tree(model[key], shard.get_tensor(key)); checked += 1
                need(checked == len(weights) == receipt['hf_saved_tensor_count'], 'ACTUAL_MODEL_TENSOR_COUNT_DIFFERS')
            return checked
        checked = collective(compare_model, 'MODEL'); del model
    finally:
        if worker._is_offload_param:
            offload_fsdp_param_and_grad(worker.actor_module_fsdp, offload_grad=worker._is_offload_grad)
    def compare_local():
        equal_tree(cpu_tree(worker.actor_optimizer.state_dict()), saved['optimizer'])
        equal_tree(cpu_tree(worker.actor_lr_scheduler.state_dict()), saved['scheduler'])
        equal_tree(worker_rng(worker), saved['rng'])
        counters = dict(actual_optimizer_writes=worker.actor._optimizer_writes,
            completed_outer=worker.actor._complete_outers, last_attempt_index=worker.actor._last_attempt_index,
            used_attempt_ids=sorted(worker.actor._used_attempt_ids), policy_rejections=worker.actor._policy_rejections,
            full_budget_outers=worker.actor._full_budget_outers, optimizer_calls=worker.actor._optimizer_calls,
            optimizer_rollbacks=worker.actor._optimizer_rollbacks)
        for key, value in counters.items(): equal_tree(value, saved[key])
        need(not worker._easy_final_forbidden_calls, 'FORBIDDEN_CALL_ATTEMPTED')
        return counters
    counters = collective(compare_local, 'LOCAL_STATE')
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.native_allocation import cuda_uuids
    return dict(rank=worker.rank, phase=phase, process_identity=identity(os.getpid()),
        run_id=receipt['run_id'], slurm_job_id=os.environ['SLURM_JOB_ID'], source_step=200,
        source_receipt_sha256=receipt_sha, source_checkpoint=str(checkpoint),
        source_state_full_sha256=state_record['full_read_sha256'],
        actual_cuda_uuid=cuda_uuids(torch.cuda.device_count())[torch.cuda.current_device()],
        actual_optimizer_writes=counters['actual_optimizer_writes'], actual_counters=counters,
        scheduler_last_epoch=worker.actor_lr_scheduler.last_epoch, scheduler_step_count=worker.actor_lr_scheduler._step_count,
        model_hf_exact_tensors_checked_on_rank0=checked, model_equal=True, optimizer_equal=True,
        scheduler_equal=True, saved_rng_equal=True, forbidden_call_counts={}, checked_unix=time.time())


def worker_class():
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import Dispatch, register
    class EasyFinalRestoreWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model(); install_guards(self)
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def readback_easy_final(self, receipt_path, phase): return readback_actor(self, receipt_path, phase)
    # Deny mutating RPCs before they can invoke a model, sampler, or optimizer.
    for name in ('update_actor', 'generate_sequences', 'begin_generation_phase', 'advance_generation_phase',
                 'end_generation_phase', 'save_checkpoint', 'capture_mixed_state',
                 'prefix20_probe_operation', 'qprobe_operation'):
        def forbidden(self, *args, _name=name, **kwargs): return blocked(self, _name)(*args, **kwargs)
        setattr(EasyFinalRestoreWorker, name, register(dispatch_mode=Dispatch.ONE_TO_ALL)(forbidden))
    return EasyFinalRestoreWorker


def restore_loader(trainer):
    """Construct the same canonical training dataset; do not draw from its sampler."""
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from omegaconf import OmegaConf, open_dict
    config = trainer.config
    trainer.train_dataset = RLHFDataset(parquet_files=config.data.train_files, tokenizer=trainer.tokenizer,
        prompt_key=config.data.prompt_key, max_prompt_length=config.data.max_prompt_length,
        filter_prompts=True, return_raw_chat=config.data.get('return_raw_chat', False), truncation='error')
    trainer.train_dataloader = DataLoader(trainer.train_dataset, batch_size=config.data.train_batch_size,
        shuffle=config.data.shuffle_train_dataloader, drop_last=True, collate_fn=collate_fn)
    trainer.val_dataset = trainer.val_dataloader = None; trainer.total_training_steps = 201
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        # The SDK validates this200 then constructs temporary empty Adam at0,
        # before the actual four saved states are loaded. No scheduler replay.
        config.actor_rollout_ref.actor.optim.total_training_steps = 201
        config.actor_rollout_ref.actor.optim.completed_updates = 200


def validate_actors(rows, allocation, receipt):
    need([row['rank'] for row in rows] == list(range(4)), 'FOUR_ACTUAL_RANKS_REQUIRED')
    need({row['actual_cuda_uuid'] for row in rows} == set(allocation['actual_cuda_runtime_uuids']) and
         len({(row['process_identity']['pid'], row['process_identity']['start_ticks']) for row in rows}) == 4,
         'ACTUAL_GPU_OR_WORKER_BIRTHS_DIFFER')
    writes = receipt['attempt_progress']['cumulative_writes']
    for row in rows:
        need(row['actual_optimizer_writes'] == writes and row['actual_counters']['completed_outer'] == 200 and
             row['scheduler_last_epoch'] == 200 and row['scheduler_step_count'] == 201 and
             not row['forbidden_call_counts'] and all(row[key] is True for key in
                 ('model_equal', 'optimizer_equal', 'scheduler_equal', 'saved_rng_equal')), 'ACTUAL_STATE_DIFFERS')
    expected_tensors = receipt['hf_saved_tensor_count']
    need(type(expected_tensors) is int and expected_tensors > 0 and
         rows[0]['model_hf_exact_tensors_checked_on_rank0'] == expected_tensors,
         'ALL_SAVED_ACTUAL_MODEL_TENSORS_REQUIRED')


def invariant(row):
    result = {key:value for key, value in row.items() if key not in ('phase', 'checked_unix', 'process_identity')}
    result['process_birth'] = {key:row['process_identity'][key] for key in ('pid', 'start_ticks', 'uid', 'sid')}
    return result


def pending_membership(stream):
    """Validate saved partial rows against canonical eligible records without draws."""
    from search_g1.qprobe_features import original_question_and_id
    from search_g1.training_state import compact_stream_state
    state = compact_stream_state(stream.state()); pending = state['pending_source']
    order, cursor, epoch, offset = (state[key] for key in ('candidate_order', 'candidate_cursor', 'epoch', 'pending_offset'))
    consumed = epoch * len(order) + cursor; checked = 0
    if pending is not None:
        need(0 <= offset < len(pending) and cursor >= len(pending), 'PENDING_CURSOR_DIFFERS')
        for i, pool_index in enumerate(order[cursor-len(pending):cursor]):
            question, qid = original_question_and_id(stream.loader.dataset.dataframe.iloc[pool_index],
                                                    stream.loader.dataset.prompt_key)
            expected = dict(qprobe_source_index=pool_index, pool_index=pool_index,
                canonical_source_index=stream.loader.dataset.indices[pool_index], pool_cycle=epoch)
            need(all(int(pending.batch[key][i]) == value for key, value in expected.items()) and
                 pending.non_tensor_batch['candidate_stable_id'][i] == qid and
                 pending.non_tensor_batch['candidate_original_question'][i] == question, 'PENDING_CANONICAL_MEMBERSHIP_DIFFERS')
            checked += 1
        consumed -= len(pending) - offset
    need(consumed == state['consumed_source_rows'], 'ACTUAL_SOURCE_CONSUMPTION_DIFFERS')
    return dict(pending_rows_verified=checked, pending_offset=offset, candidate_cursor=cursor, epoch=epoch,
                consumed_source_rows=state['consumed_source_rows'],
                ordered_membership_sha256=stream.binding['ordered_membership_sha256'])


def no_training(base):
    logs = base/'logs'
    need(not any((logs/name).exists() for name in ('training-launch-claim.json', 'forbidden-training.log',
         'actual-updates.jsonl', 'actual-attempts.jsonl')) and not list(logs.glob('optimizer-rank-*.jsonl')) and
         not list(logs.glob('scheduler-rank-*.jsonl')) and
         not list((base/'checkpoints').rglob('global_step_201')), 'TRAINING_OR_201_ARTIFACT_APPEARED')


def source_no201(origin, checkpoint):
    ledger = origin/'logs/actual-updates.jsonl'; rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    need(rows and rows[-1]['step'] == 200 and all(row['step'] <= 200 for row in rows) and
         not (checkpoint.parent/'global_step_201').exists(), 'SOURCE_FINAL200_OR_NO201_DIFFERS')
    paths = ['logs/actual-updates.jsonl', 'logs/actual-attempts.jsonl'] + [
        'logs/' + role + '-rank-' + str(rank) + '.jsonl' for role in ('optimizer', 'scheduler') for rank in range(4)]
    return dict(source_ledger=record(ledger), source_ledgers={Path(name).name:record(origin/name) for name in paths},
                last_completed_outer=200, update201_present=False,
                checkpoint201_present=False, checked_unix=time.time())


def driver_task(config):
    import ray
    import torch
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from search_g1.full_state_restore import equal_tree, restore_driver
    from search_g1.training_state import compact_stream_state, driver_rng
    from reproduction.common8.recoverable_attempt import AttemptProgress
    from reproduction.a800.owned_processes import identity
    base = Path(os.environ['SEARCH_G1_BASE']).resolve(); run = read(base/'logs/run-manifest.json')
    receipt_path = run['full_state_restore']['source_receipt']; _, receipt, _, _, _ = source(receipt_path)
    class ReadOnlyTrainer(RayPPOTrainer):
        def _create_dataloader(self): restore_loader(self)
        def fit(self): raise RuntimeError('EASY_FINAL_RESTORE_TRAINING_ENTRY_FORBIDDEN')
    trainer = ReadOnlyTrainer(config=config, tokenizer=hf_tokenizer(run['initial_model'], trust_remote_code=False),
        role_worker_mapping={Role.ActorRollout:ray.remote(worker_class())},
        resource_pool_manager=ResourcePoolManager(resource_pool_spec={'restore_pool':[4]}, mapping={Role.ActorRollout:'restore_pool'}),
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=type('ReadOnlyRewardBinding', (), {'no_self':True, 'expected_probe_version':None})(), val_reward_fn=None)
    trainer.init_workers()  # Actual model/FSDP followed by all4 restore_worker RPCs.
    restore_driver(trainer, receipt_path)
    need(sha(receipt['driver_file']['path']) == receipt['driver_file']['full_read_sha256'], 'DRIVER_SHA_DIFFERS')
    saved = torch.load(receipt['driver_file']['path'], map_location='cpu', weights_only=False)
    stream = trainer.mixed_stream; progress = AttemptProgress(run)
    for key, value in saved['attempt_progress'].items(): setattr(progress, key, value)
    trainer.global_steps = 200; trainer.inline_attempt_progress = progress
    allocation = read(base/'logs/native-allocation-admission.json'); first = None
    for phase in ('A', 'B'):
        rows = trainer.actor_rollout_wg.readback_easy_final(receipt_path, phase)
        validate_actors(rows, allocation, receipt)
        equal_tree(compact_stream_state(stream.state()), compact_stream_state(saved['candidate_stream']))
        equal_tree(driver_rng(), saved['rng'])
        actual_progress = {key:getattr(progress, key) for key in saved['attempt_progress']}
        equal_tree(actual_progress, saved['attempt_progress'])
        need(trainer.global_steps == 200 and progress.pending is None and
             trainer._restored_reward_refresh_pending is False, 'ACTUAL_DRIVER_COUNTER_DIFFERS')
        # The source factory validates canonical eligible membership; identical
        # pending tensors/order/cursor and current driver RNG are read back here.
        equal_tree(stream.binding, saved['candidate_stream']['prefix20_pool_binding'])
        membership = pending_membership(stream)
        need(trainer.reward_fn.expected_probe_version == saved['probe_version'] and
             saved['probe_round'] == 150, 'ACTUAL_GROUNDING_BINDING_DIFFERS')
        actual = dict(actors=[invariant(row) for row in rows], stream=copy.deepcopy(compact_stream_state(stream.state())),
                      rng=copy.deepcopy(driver_rng()), attempt_progress=actual_progress)
        if first is None: first = actual
        else: equal_tree(first, actual)
        proof = dict(schema_version=1, role=ROLE, phase=phase, run_id=run['run_id'],
            host_identity=host_identity(),
            source_receipt_sha256=sha(receipt_path), driver_process_identity=identity(os.getpid()),
            actual_actor_readbacks=rows, driver=dict(completed_outer=200,
                actual_driver_rng_readback_equal=True, actual_candidate_stream_readback_equal=True,
                actual_candidate_source_cursor=stream.state()['consumed_source_rows'], attempt_progress=actual_progress,
                source_driver_full_sha256=receipt['driver_file']['full_read_sha256'],
                actual_probe_version=trainer.reward_fn.expected_probe_version, actual_probe_round=saved['probe_round'],
                reward_refresh_pending=trainer._restored_reward_refresh_pending, question_probe=None,
                eligible_mapping_equal=True, pending_offset=stream.state()['pending_offset'], membership=membership),
            optimizer_write_increment=0, scheduler_advance_increment=0, candidate_consumption_increment=0,
            generation_calls=0, training_checkpoint_save_calls=0, forbidden_call_counts={},
            independent_second_readback_equal=phase == 'B', checked_unix=time.time())
        publish(base/'logs'/('final-restore-only-readback-' + phase + '.json'), proof)
        no_training(base)
    return dict(role=ROLE, run_id=run['run_id'], source_receipt_sha256=sha(receipt_path),
                driver_process_identity=identity(os.getpid()),
                actual_all4_model_optimizer_scheduler_rng_dataequal=True, actual_cuda_restore_accepted=True,
                actual_two_independent_readbacks=True, no_update201=True)


def execute(base):
    from .control import own_base
    from reproduction.common8.native_allocation import admit
    from reproduction.common8.configuration import resolved
    from reproduction.common8.native_ray import root_for
    base = own_base(base); run = read(base/'logs/run-manifest.json'); binding = run['full_state_restore']
    need(run.get('restore_only_role') == binding['kind'] == ROLE and
         os.environ.get('SEARCH_G1_BASE') == str(base) and os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE') == '4' and
         os.environ.get('SLURM_JOB_ID', '').isdecimal(), 'EXPLICIT_NATIVE4_RESTORE_ROLE_REQUIRED')
    lock = (base/'preparation/final-restore-execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish(base/'preparation/final-restore-execution-claim.json', dict(role=ROLE,
        job=os.environ['SLURM_JOB_ID'], source_receipt=record(binding['source_receipt']), checked_unix=time.time()))
    need(Path(__file__).resolve() == base/'repo/reproduction/easy2560_joint4/final_restore_only.py',
         'ENTRY_MUST_USE_BOUND_TARGET_REPO')
    path, receipt, checkpoint, origin, parent = source(binding['source_receipt'], rehash_checkpoint=True)
    need(sha(path) == binding['source_receipt_sha256'], 'BOUND_RECEIPT_SHA_DIFFERS')
    no_training(base)
    before = source_no201(origin, checkpoint); publish(base/'preparation/source-no201-before.json', before)
    allocation = admit(base); need(allocation['training_world_size'] == 4, 'ACTUAL_NATIVE4_REQUIRED')
    _, config, _ = resolved(base)  # Resolves actor checkpoint; never calls training from_config/bind.
    ray_root = root_for(run['native_ray_execution_id']); ray_root.mkdir(mode=0o700, exist_ok=False)
    os.environ['RAY_TMPDIR'] = str(ray_root); os.environ['ECO_NO_SELF'] = '1'
    import ray
    need(not ray.is_initialized(), 'PRIVATE_RAY_REQUIRED')
    from .native_ray_resources import raise_self_soft
    raise_self_soft()
    try:
        ray.init(address='local', num_gpus=4, num_cpus=16, object_store_memory=8 * 1024**3,
                 _temp_dir=str(ray_root), runtime_env={'env_vars':ray_env_vars(run)})
        result = ray.get(ray.remote(num_cpus=1)(driver_task).remote(config))
        completion = dict(role=ROLE, run_id=run['run_id'], source_receipt_sha256=sha(path),
            host_identity=host_identity(),
            driver_process_identity=result['driver_process_identity'], ray_get_returned=True, task_result=result,
            task_result_sha256=hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            checked_unix=time.time())
        publish(base/'logs/final-restore-execution-completion.json', completion)
    finally: ray.shutdown()
    from reproduction.a800.owned_processes import identity
    last = read(base/'logs/final-restore-only-readback-B.json')
    births = [row['process_identity'] for row in last['actual_actor_readbacks']] + [last['driver_process_identity']]
    def alive(birth):
        try:
            actual = identity(birth['pid'])
            return actual['state'] != 'Z' and all(actual[key] == birth[key] for key in ('start_ticks', 'uid', 'sid'))
        except (FileNotFoundError, ProcessLookupError): return False
    deadline = time.monotonic() + 30
    while any(alive(birth) for birth in births):
        need(time.monotonic() < deadline, 'OWNED_WORKERS_REMAIN_AFTER_SHUTDOWN'); time.sleep(.25)
    terminal = []
    for birth in births:
        try:
            observed = identity(birth['pid'])
            state = 'zombie' if observed['state'] == 'Z' else 'birth_replaced'
        except (FileNotFoundError, ProcessLookupError): observed = None; state = 'absent'
        terminal.append(dict(process_identity=birth, terminal=True, observation=state, observed_process_identity=observed))
    deadline = time.monotonic() + 30
    while True:
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                                      text=True, timeout=30)
        occupied = [row for row in csv.reader(io.StringIO(raw)) if row and row[0].strip() in allocation['actual_cuda_runtime_uuids']]
        if not occupied: break
        need(time.monotonic() < deadline, 'ASSIGNED_CUDA_CONTEXTS_REMAIN'); time.sleep(.25)
    publish(base/'logs/final-restore-owned-release.json', dict(role=ROLE, recorded_process_births=births,
        host_identity=host_identity(),
        terminal_observations=terminal, assigned_gpu_applications_after=[], actual_compute_apps_stdout=raw,
        execution_completion=record(base/'logs/final-restore-execution-completion.json'),
        all_recorded_processes_terminal=True, assigned_gpu_contexts_released=True, checked_unix=time.time()))
    no_training(base); after = source_no201(origin, checkpoint)
    need(before['source_ledgers'] == after['source_ledgers'], 'SOURCE_CHANGED_DURING_READBACK')
    publish(base/'preparation/source-no201-after.json', after)
    result = dict(result, schema_version=1, status=STATUS, role=ROLE, kind=ROLE, restore_only_role=ROLE,
        target_base=str(base), target_source_commit=run['source_commit'], source_base=str(origin),
        source_commit=receipt['source_commit'], source_job=receipt['source_job'], source_checkpoint=str(checkpoint),
        source_step=200, run_id=run['run_id'], source_receipt=record(path),
        allocation=record(base/'logs/native-allocation-admission.json'),
        readbacks={phase:record(base/'logs'/('final-restore-only-readback-' + phase + '.json')) for phase in ('A', 'B')},
        owned_release=record(base/'logs/final-restore-owned-release.json'),
        execution_completion=record(base/'logs/final-restore-execution-completion.json'),
        source_no201_before=record(base/'preparation/source-no201-before.json'),
        source_no201_after=record(base/'preparation/source-no201-after.json'),
        attempt_progress=receipt['attempt_progress'], actual_optimizer_writes_per_rank=receipt['attempt_progress']['cumulative_writes'],
        reward_probe_round=receipt['reward_probe_round'], reward_probe_version=receipt['reward_probe_version'],
        new_updates=0, optimizer_step_calls=0, scheduler_step_calls=0, generation_calls=0, training_checkpoint_save_calls=0,
        candidate_source_cursor=receipt['driver_state']['source_consumed'], checked_unix=time.time())
    publish(base/'logs/final-restore-only-complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('prepare', 'execute'))
    parser.add_argument('--base', type=Path, required=True); parser.add_argument('--source-receipt'); parser.add_argument('--commit')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.source_receipt or not args.commit: parser.error('prepare requires --source-receipt and --commit')
        result = prepare(args.base, args.source_receipt, args.commit)
    else: result = execute(args.base)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
