"""Isolated torchrun4 CP1-model numerical diagnostic; never a training entry.

This loads CP1 HF actor weights into the ordinary four-rank FSDP actor, with a
temporary *empty* Adam/scheduler. It intentionally does not restore Adam, saved
RNG, data cursor, grounding state, or a vLLM manager. Production must subsequently
start in a fresh process and perform its original complete state restoration.

The old pool's binary-EM advantages are qualification-only test inputs, not the
production grounding reward. No sampling, optimizer/scheduler step, clipping,
checkpoint save, or formal restoration receipt occurs here.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

# Keep the isolated entry's imports from creating repo-local bytecode artifacts.
sys.dont_write_bytecode = True


def _need(condition, message):
    if not condition:
        raise ValueError('SPARSE_MODEL_DIAGNOSTIC_' + message)


def prepare_micro_batch(raw, rank, *, world_size=4, physical_micro=8):
    """Select the first32 mixed-group trajectories, then split rank8 on CPU.

    All five original EM labels determine each group's sample standard
    deviation before the first32 rows are selected. A partial final group is
    therefore never re-normalized from only its selected members.
    """
    import torch
    _need(type(rank) is int and 0 <= rank < world_size and world_size == 4 and
          physical_micro == 8, 'ORIGINAL_FOUR_BY_EIGHT_GEOMETRY_REQUIRED')
    labels = raw['strict_correctness']
    _need(isinstance(labels, (list, tuple)) and len(labels) > 0 and
          all(isinstance(group, (list, tuple)) and len(group) == 5 and
              all(type(value) in (int, bool) and value in (0, 1) for value in group)
              for group in labels), 'ACTUAL_FIVE_BINARY_EM_LABELS_REQUIRED')
    scores = torch.tensor(labels, dtype=torch.float32, device='cpu')
    counts = scores.sum(dim=1)
    mixed = (counts > 0) & (counts < 5)
    selected = [row for group in mixed.nonzero(as_tuple=False).flatten().tolist()
                for row in range(group * 5, group * 5 + 5)][:world_size * physical_micro]
    _need(len(selected) == 32, 'AT_LEAST_32_ACTUAL_MIXED_GROUP_TRAJECTORIES_REQUIRED')
    batch = raw['output'].batch
    keys = ('input_ids', 'attention_mask', 'position_ids', 'responses',
            'sampling_log_probs', 'info_mask')
    _need(all(name in batch and isinstance(batch[name], torch.Tensor) and
              batch[name].device.type == 'cpu' and batch[name].ndim == 2 and
              batch[name].shape[0] == scores.numel() for name in keys),
          'ALIGNED_ORIGINAL_CPU_POOL_TENSORS_REQUIRED')
    response_width = batch['responses'].shape[1]
    sequence_shape = batch['input_ids'].shape
    _need(0 < response_width < sequence_shape[1] and
          all(batch[name].shape == sequence_shape for name in
              ('attention_mask', 'position_ids', 'info_mask')) and
          batch['sampling_log_probs'].shape == batch['responses'].shape,
          'ORIGINAL_RESPONSE_AND_CONTEXT_SHAPES_DIFFER')
    local_rows = selected[rank * physical_micro:(rank + 1) * physical_micro]
    indices = torch.tensor(local_rows, dtype=torch.long)
    micro = {name: batch[name].index_select(0, indices).clone()
             for name in ('input_ids', 'attention_mask', 'position_ids', 'responses')}
    # Same state-masking boundary as the ordinary trainer. Tool tokens remain
    # in decoder context and are excluded only from the policy loss.
    micro['loss_mask'] = batch['info_mask'].index_select(0, indices)[:, -response_width:].clone()
    micro['old_log_probs'] = batch['sampling_log_probs'].index_select(0, indices).clone()
    mask = micro['loss_mask']
    _need(bool(((mask == 0) | (mask == 1)).all()) and bool(mask.bool().any(dim=1).all()) and
          bool((mask <= micro['attention_mask'][:, -response_width:]).all()),
          'AUTHORED_LOSS_MASK_EMPTY_OR_INCLUDES_PADDING')
    _need(bool(torch.isfinite(micro['old_log_probs'][mask.bool()]).all()),
          'AUTHORED_OLD_LOG_PROBABILITIES_NONFINITE')
    standardized = (scores - scores.mean(dim=1, keepdim=True)) / (
        scores.std(dim=1, correction=1, keepdim=True) + 1e-6)
    row_advantages = standardized.flatten().index_select(0, indices)
    micro['advantages'] = row_advantages[:, None].expand(-1, response_width).clone() * mask
    _need(bool(row_advantages.abs().gt(0).all()), 'MIXED_TEST_ADVANTAGE_MUST_BE_NONZERO')
    ordinals = raw.get('ordinals', list(range(len(labels))))
    _need(len(ordinals) == len(labels), 'QUESTION_ORDINAL_COUNT_DIFFERS')
    identity = raw.get('identity', {})
    rows = [dict(pool_row=row, question_index=row // 5,
                 question_ordinal=int(ordinals[row // 5]), trajectory_index=row % 5,
                 binary_em=int(scores.flatten()[row]), correct_count=int(counts[row // 5]),
                 qualification_only_advantage=float(standardized.flatten()[row]),
                 authored_tokens=int(mask[local].sum()))
            for local, row in enumerate(local_rows)]
    proof = dict(rank=rank, selected_pool_rows=local_rows, global_selected_pool_rows=selected,
        original_questions=len(labels), original_trajectories=scores.numel(),
        original_mixed_questions=int(mixed.sum()), response_width=response_width,
        context_width=sequence_shape[1], rows=rows,
        raw_pool_identity={key: copy.deepcopy(identity[key]) for key in
            ('run_id', 'logical_update', 'policy_snapshot_id', 'attempt_id', 'attempt_index')
            if key in identity},
        loss_mask='Original info_mask[:, -response_width:]; no response/tool token reassembly',
        old_log_probabilities='Original saved sampling_log_probs; no extra policy forward',
        advantages=dict(role='qualification_only_binary_EM_test_advantages_NOT_production_grounding_rewards',
            formula='(EM - mean(original five EMs))/(sample_std(original five EMs) + 1e-6)',
            all_five_labels_used_before_selection=True, virtual_zero=False))
    return micro, proof


def block_updates(worker):
    """Block actual update APIs after their ordinary constructor initialization."""
    counters = dict(optimizer_step=0, scheduler_step=0)
    def block(name):
        def reject(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('SPARSE_MODEL_DIAGNOSTIC_UPDATE_FORBIDDEN: ' + name)
        return reject
    worker.actor_optimizer.step = block('optimizer_step')
    scheduler = worker.actor_lr_scheduler
    scheduler.__class__ = type('SparseDiagnostic' + type(scheduler).__name__, (type(scheduler),),
                              {'step': block('scheduler_step')})
    return counters


def _publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def _paths(base, raw_pool_file, output):
    base = Path(base).resolve(strict=True)
    raw_pool_file = Path(raw_pool_file).resolve(strict=True)
    output = Path(output).absolute()
    _need(output.resolve().is_relative_to(base / 'diagnostics') and not output.exists(),
          'NEW_OUTPUT_UNDER_OWN_DIAGNOSTICS_REQUIRED')
    return base, raw_pool_file, output


def run(base, raw_pool_file, output):
    import torch
    import torch.distributed as dist
    from omegaconf import open_dict
    from reproduction.common8.configuration import resolved
    from search_g1.easy7b_infra import flags
    from verl.utils.weight_continuation import model_initialization_path

    base, raw_pool_file, output = _paths(base, raw_pool_file, output)
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    _need(int(os.environ['WORLD_SIZE']) == 4, 'TORCHRUN_FOUR_RANKS_REQUIRED')
    torch.cuda.set_device(local_rank)
    os.environ['SEARCH_G1_BASE'] = str(base)
    os.environ['SEARCH_G1_PROFILE'] = 'common8'
    os.environ['SEARCH_G1_TRAINING_WORLD_SIZE'] = '4'
    # Direct torchrun uses its own rendezvous. An inherited Ray worker prefix
    # would otherwise make Worker.__new__ replace rank0's MASTER_ADDR/PORT.
    os.environ.pop('WG_PREFIX', None)
    os.environ.pop('WG_BACKEND', None)
    os.environ.pop('DISABLE_WORKER_INIT', None)
    run_manifest = json.loads((base / 'logs/run-manifest.json').read_text())
    os.environ['SEARCH_G1_SOURCE_COMMIT'] = run_manifest['source_commit']
    # init_model's tokenizer/config/model reads are from the same local CP1.
    # No separate tokenizer/model, download, or writable HF cache is needed.
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    bound, config, _ = resolved(base)
    binding = bound.get('full_state_restore', {})
    _need(binding.get('completed_outer') == 1 and flags(bound) ==
          dict(selected_policy_logits=True, exact_retrieval_cache=True),
          'EXPLICIT_CP1_SPARSE_BASE_REQUIRED')
    checkpoint = Path(model_initialization_path(config.actor_rollout_ref.model, actor=True)).resolve()
    _need(checkpoint == Path(binding['source_checkpoint']).resolve() and
          raw_pool_file.is_relative_to(Path(binding['source_base']).resolve()),
          'BOUND_CP1_MODEL_AND_ORIGINAL_SOURCE_POOL_REQUIRED')
    # Mirror the original trainer's in-memory optimizer construction settings;
    # optimizer_construction_config deliberately constructs temporary empty
    # objects here. Their production saved states are NOT loaded by this tool.
    with open_dict(config.actor_rollout_ref.actor.optim):
        config.actor_rollout_ref.actor.optim.total_training_steps = config.trainer.total_training_steps
        config.actor_rollout_ref.actor.optim.completed_updates = binding['completed_outer']
    raw = torch.load(raw_pool_file, map_location='cpu', weights_only=False)
    _need(raw['identity']['run_id'] == bound['run_id'] and
          raw['identity']['logical_update'] == 1, 'ORIGINAL_OUTER1_POOL_IDENTITY_DIFFERS')
    micro, selection = prepare_micro_batch(raw, rank)
    del raw
    gc.collect()

    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad
    from search_g1.sparse_policy_qualification import qualify_sparse_policy
    started = time.time()
    worker = ActorRolloutRefWorker(config.actor_rollout_ref, role='actor')
    worker.init_model()
    _need(worker._is_actor and not worker._is_rollout and not worker._is_ref and
          not hasattr(worker, 'rollout') and not hasattr(worker, 'ref_policy'),
          'ACTOR_ONLY_INITIALIZATION_REQUIRED')
    _need(len(worker.actor_optimizer.state) == 0, 'DIAGNOSTIC_ADAM_MUST_BE_EMPTY')
    scheduler_before = copy.deepcopy(worker.actor_lr_scheduler.state_dict())
    calls = block_updates(worker)
    load_fsdp_param_and_grad(worker.actor_module_fsdp,
                            device_id=torch.cuda.current_device(), load_grad=False)
    device = torch.device('cuda', torch.cuda.current_device())
    micro = {key: value.to(device) for key, value in micro.items()}
    result = qualify_sparse_policy(worker.actor, micro,
                                  float(config.actor_rollout_ref.rollout.temperature))
    _need(calls == dict(optimizer_step=0, scheduler_step=0) and
          len(worker.actor_optimizer.state) == 0 and
          worker.actor_lr_scheduler.state_dict() == scheduler_before,
          'DIAGNOSTIC_OPTIMIZER_OR_SCHEDULER_CHANGED')
    local = dict(rank=rank, local_rank=local_rank, hostname=socket.gethostname(),
        selection=selection, update_calls=calls, fresh_adam_state_entries=0,
        scheduler_unchanged=True, numerical_comparison=result)
    ranks = [None] * 4
    dist.all_gather_object(ranks, local)
    accepted = all(row['numerical_comparison']['accepted'] for row in ranks)
    report = dict(format='easy7b_cp1_model_only_sparse_cuda_qualification_v1', accepted=accepted,
        diagnostic_only=True, normal_training_auto_call=False, source_commit=bound['source_commit'],
        base=str(base), cp1_hf_model=str(checkpoint), raw_pool_file=str(raw_pool_file),
        raw_pool_bytes=raw_pool_file.stat().st_size, run_id=bound['run_id'], world_size=4,
        model_source_completed_outer=1, model_initialized_from_cp1=True,
        actual_optimizer_state_restored=False, actual_scheduler_state_restored=False,
        saved_training_rng_restored=False, data_cursor_restored=False,
        formal_actual_restore_receipt_written=False, production_full_state_restore_still_required=True,
        rollout_or_reference_model_initialized=False, sampling_calls=0, optimizer_calls=0,
        scheduler_calls_after_construction=0, physical_micro_per_rank=8,
        logical_micro_per_rank=4, ranks=ranks, started_unix=started, completed_unix=time.time(),
        scope='CP1 model forward/backward only on32 original outer1 mixed-EM trajectories; '
              'binary-EM test advantages are NOT production grounding rewards; '
              'no whole-training equivalence or throughput claim')
    if rank == 0:
        _publish(output, report)
        print('SEARCH_G1_SPARSE_MODEL_DIAGNOSTIC=' + json.dumps(dict(
            accepted=accepted, output=str(output), model_only=True,
            production_full_state_restore_still_required=True)), flush=True)
    dist.barrier()
    del micro, worker
    gc.collect(); torch.cuda.empty_cache()
    dist.destroy_process_group()
    if not accepted:
        raise RuntimeError('SPARSE_MODEL_DIAGNOSTIC_NUMERICAL_THRESHOLDS_REJECTED; see ' + str(output))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--raw-pool-file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        run(args.base, args.raw_pool_file, args.output)
    except BaseException as error:
        # Preserve only a small diagnostic; never attempt recovery or run a
        # production worker in this process after OOM/collective failure.
        rank = os.environ.get('RANK', 'unknown')
        failure = Path(args.output).with_name(Path(args.output).stem + '-rank-' + rank + '-failure.json')
        try:
            base = Path(args.base).resolve()
            if failure.resolve().is_relative_to(base / 'diagnostics') and not failure.exists():
                _publish(failure, dict(diagnostic_only=True, rank=rank, accepted=False,
                    error_type=type(error).__name__, error=str(error)[:2048], checked_unix=time.time(),
                    worker_reuse_permitted=False, formal_actual_restore_receipt_written=False))
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
