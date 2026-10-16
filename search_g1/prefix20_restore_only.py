"""Pure actual-state readbacks for final200; no fit/generation/update/save API."""
import copy
import json
import os
from pathlib import Path
import time

ROLE = 'prefix20_final200_restore_only'


def active():
    base = os.environ.get('SEARCH_G1_BASE')
    if not base:
        return False
    path = Path(base) / 'logs/run-manifest.json'
    return path.is_file() and json.loads(path.read_text()).get('restore_only_role') == ROLE


def deny_training():
    if active():
        raise RuntimeError('RESTORE_ONLY_TRAINING_ENTRY_FORBIDDEN')


def assert_target_receipt(receipt_path):
    from reproduction.prefix20_joint4.restore_only_assets import regular_under, sha
    from search_g1.qprobe_checkpoint import read_source
    base = Path(os.environ['SEARCH_G1_BASE']).resolve()
    if not active():
        raise ValueError('RESTORE_ONLY_EXPLICIT_ROLE_REQUIRED')
    path = regular_under(Path(receipt_path), base)
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    binding = run['full_state_restore']
    if str(path) != binding['source_receipt'] or sha(path) != binding['source_receipt_sha256']:
        raise ValueError('RESTORE_ONLY_BOUND_TARGET_RECEIPT_DIFFERS')
    result = read_source(path)
    _, receipt, checkpoint, origin, _ = result
    if receipt['step'] != 200 or receipt['world_size'] != 4 or origin != base or not checkpoint.is_relative_to(base):
        raise ValueError('RESTORE_ONLY_FINAL200_TARGET_REQUIRED')
    for row in receipt['files'] + receipt['boundary_worker_files'] + [receipt['boundary_driver']]:
        regular_under(Path(row['path']), base)
    return result


def install_mutation_guards(worker):
    """Install after temporary optimizer/scheduler construction, before reload."""
    if not active():
        return
    if getattr(worker, '_restore_only_forbidden_calls', None) is not None:
        raise ValueError('RESTORE_ONLY_MUTATION_GUARDS_ALREADY_INSTALLED')
    worker._restore_only_forbidden_calls = {}
    def block(label):
        def forbidden(*args, **kwargs):
            counts = worker._restore_only_forbidden_calls
            counts[label] = counts.get(label, 0) + 1
            raise RuntimeError('RESTORE_ONLY_MUTATION_FORBIDDEN: ' + label)
        return forbidden
    worker._restore_only_block = block
    if worker._is_actor:
        worker.actor_optimizer.step = block('actor_optimizer_step')
        # An instance .step attribute would leak into scheduler.state_dict(),
        # invalidating exact restored-state equality. Override on a subclass.
        scheduler = worker.actor_lr_scheduler
        scheduler.__class__ = type('RestoreOnly' + type(scheduler).__name__, (type(scheduler),),
                                   {'step':block('actor_scheduler_step')})
    if worker._is_rollout:
        worker.rollout.inference_engine.generate = block('rollout_engine_generate')
        worker.rollout.inference_engine.llm_engine.add_request = block('vllm_add_request')


def guard_probe_operation(worker, operation):
    if active() and operation not in ('init', 'load_state_dict', 'state_dict', 'readback'):
        counts = worker._restore_only_forbidden_calls
        counts['probe_' + operation] = counts.get('probe_' + operation, 0) + 1
        raise RuntimeError('RESTORE_ONLY_PROBE_OPERATION_FORBIDDEN')


def guard_probe_optimizer(worker, head):
    if active():
        head.optimizer.step = worker._restore_only_block('probe_optimizer_step')


def _collective(function, label):
    import torch.distributed as dist
    result, failure = None, None
    try:
        result = function()
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)[:512]
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    if any(value is not None for value in failures):
        raise RuntimeError('RESTORE_ONLY_COLLECTIVE_' + label + ': ' + repr(failures))
    return result


def readback_actor(worker, receipt_path, phase):
    """Read twice without reloading; collective FSDP model + all local states."""
    import torch
    from safetensors import safe_open
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad, offload_fsdp_param_and_grad
    from search_g1.training_state import cpu_tree, worker_rng
    from search_g1.full_state_restore import equal_tree
    from search_g1.qprobe_checkpoint import boundary_worker_rng, validate_rank_state
    if phase not in ('A', 'B'):
        raise ValueError('RESTORE_ONLY_TWO_EXPLICIT_READBACK_PHASES')
    worker._assert_generation_phase_inactive()
    def inputs():
        _, receipt, checkpoint, _, _ = assert_target_receipt(receipt_path)
        row = next(row for row in receipt['files'] if row['relative_path'] == 'training_state/rank-%d.pt' % worker.rank)
        from reproduction.prefix20_joint4.restore_only_assets import sha
        if sha(row['path']) != row['full_read_sha256']:
            raise ValueError('RESTORE_ONLY_TARGET_ACTOR_STATE_SHA_DIFFERS')
        saved = torch.load(row['path'], map_location='cpu', weights_only=False)
        validate_rank_state(saved, worker.rank, 4, 200,
                            expected_writes=receipt['attempt_progress']['cumulative_writes'])
        return receipt, checkpoint, saved
    receipt, checkpoint, saved = _collective(inputs, 'INPUTS')
    if worker._is_offload_param:
        load_fsdp_param_and_grad(worker.actor_module_fsdp, device_id=torch.cuda.current_device(), load_grad=False)
    try:
        with FSDP.state_dict_type(worker.actor_module_fsdp, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            model = worker.actor_module_fsdp.state_dict()
        def compare_model():
            checked = 0
            if worker.rank == 0:
                weights = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
                if set(model) != set(weights):
                    raise ValueError('RESTORE_ONLY_MODEL_TENSOR_SET_DIFFERS')
                for name in sorted(set(weights.values())):
                    with safe_open(str(checkpoint / name), framework='pt', device='cpu') as shard:
                        for key in shard.keys():
                            equal_tree(model[key], shard.get_tensor(key))
                            checked += 1
                if checked != receipt['hf_saved_tensor_count']:
                    raise ValueError('RESTORE_ONLY_MODEL_TENSOR_COUNT_DIFFERS')
            return checked
        checked = _collective(compare_model, 'MODEL')
        del model
    finally:
        if worker._is_offload_param:
            offload_fsdp_param_and_grad(worker.actor_module_fsdp, offload_grad=worker._is_offload_grad)
    def compare_local():
        equal_tree(cpu_tree(worker.actor_optimizer.state_dict()), saved['optimizer'])
        equal_tree(cpu_tree(worker.actor_lr_scheduler.state_dict()), saved['scheduler'])
        equal_tree(worker_rng(worker), boundary_worker_rng(receipt, worker.rank, saved))
        counters = dict(actual_optimizer_writes=worker.actor._optimizer_writes,
                        completed_outer=worker.actor._complete_outers,
                        last_attempt_index=worker.actor._last_attempt_index,
                        used_attempt_ids=sorted(worker.actor._used_attempt_ids),
                        policy_rejections=worker.actor._policy_rejections,
                        full_budget_outers=worker.actor._full_budget_outers,
                        optimizer_calls=worker.actor._optimizer_calls,
                        optimizer_rollbacks=worker.actor._optimizer_rollbacks)
        for key, value in counters.items():
            equal_tree(value, saved[key])
        if worker._restore_only_forbidden_calls:
            raise ValueError('RESTORE_ONLY_FORBIDDEN_CALL_ATTEMPTED')
        return counters
    counters = _collective(compare_local, 'LOCAL_STATES')
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.native_allocation import cuda_uuids
    uuid = cuda_uuids(torch.cuda.device_count())[torch.cuda.current_device()]
    return dict(rank=worker.rank, phase=phase, process_identity=identity(os.getpid()),
                actual_cuda_uuid=uuid, actual_optimizer_writes=counters['actual_optimizer_writes'],
                actual_counters=counters, scheduler_last_epoch=worker.actor_lr_scheduler.last_epoch,
                scheduler_step_count=worker.actor_lr_scheduler._step_count,
                model_hf_exact_tensors_checked_on_rank0=checked, model_equal=True,
                optimizer_equal=True, scheduler_equal=True, boundary_rng_equal=True,
                forbidden_call_counts={}, checked_unix=time.time())


def readback_probe(worker, head):
    from search_g1.training_state import cpu_tree
    from search_g1.prefix20_probe_worker import rng_state, probe_rng
    from search_g1.full_state_restore import equal_tree
    if not active() or head.pending_outer is not None:
        raise ValueError('RESTORE_ONLY_PROBE_IDLE_ROLE_REQUIRED')
    actor_rng = rng_state(head.device)
    with probe_rng(head):
        equal_tree(head.rng, rng_state(head.device))
    equal_tree(actor_rng, rng_state(head.device))
    if worker._restore_only_forbidden_calls:
        raise ValueError('RESTORE_ONLY_FORBIDDEN_CALL_ATTEMPTED')
    return dict(rank=worker.rank, state=dict(rank=worker.rank, completed_outer=head.completed_outer,
                actual_steps=head.actual_steps, model=cpu_tree(head.model.state_dict()),
                optimizer=cpu_tree(head.optimizer.state_dict()), rng=cpu_tree(head.rng)),
                private_rng_activated_and_equal=True, actor_rng_unchanged=True, forbidden_call_counts={})


def check_pending_membership(stream):
    """Check partial pending rows against the actual canonical source, no draws."""
    from search_g1.qprobe_features import original_question_and_id
    from search_g1.training_state import compact_stream_state
    state = compact_stream_state(stream.state())
    pending, offset = state['pending_source'], state['pending_offset']
    order, cursor, epoch = state['candidate_order'], state['candidate_cursor'], state['epoch']
    expected_consumed = epoch * len(order) + cursor
    checked = 0
    if pending is not None:
        if not 0 <= offset < len(pending) or cursor < len(pending):
            raise ValueError('RESTORE_ONLY_PENDING_OFFSET_OR_CURSOR_DIFFERS')
        indices = order[cursor-len(pending):cursor]
        for i, pool_index in enumerate(indices):
            source = stream.loader.dataset.dataframe.iloc[pool_index]
            question, qid = original_question_and_id(source, stream.loader.dataset.prompt_key)
            expected = dict(qprobe_source_index=pool_index, pool_index=pool_index,
                            canonical_source_index=stream.loader.dataset.indices[pool_index], pool_cycle=epoch)
            if any(int(pending.batch[key][i]) != value for key,value in expected.items()):
                raise ValueError('RESTORE_ONLY_PENDING_INDEX_MEMBERSHIP_DIFFERS')
            if pending.non_tensor_batch['candidate_stable_id'][i] != qid or pending.non_tensor_batch['candidate_original_question'][i] != question:
                raise ValueError('RESTORE_ONLY_PENDING_CANONICAL_QUESTION_DIFFERS')
            checked += 1
        expected_consumed -= len(pending) - offset
    if state['consumed_source_rows'] != expected_consumed:
        raise ValueError('RESTORE_ONLY_DATA_CONSUMPTION_IDENTITY_DIFFERS')
    return dict(pending_rows_verified=checked, pending_offset=offset,
                consumed_source_rows=state['consumed_source_rows'], candidate_cursor=cursor, epoch=epoch,
                ordered_membership_sha256=stream.binding['ordered_membership_sha256'])
