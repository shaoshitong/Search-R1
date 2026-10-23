"""Probe operations inside the four already allocated actor Ray workers."""
from pathlib import Path

import torch
import torch.distributed as dist

from .group_probe_adapter import (GroupProbeAdapter, WARMUP_SHA256, cpu_tree,
    file_sha256, pack_group, probe_rng, restore_rng, rng_state)
from .probe_update import ProbeUpdater, save_probe_checkpoint


def _verify_once(path, expected):
    result = None
    if dist.get_rank() == 0:
        try:
            actual = file_sha256(path)
            result = dict(ok=expected is None or actual == expected, sha256=actual)
        except OSError as error:
            result = dict(ok=False, error=str(error))
    values = [result]
    dist.broadcast_object_list(values, src=0)
    if not values[0]['ok']:
        raise ValueError('Probe checkpoint unavailable or SHA differs: %s' % values[0])
    return values[0]['sha256']


def _idle_actor(worker):
    worker._assert_generation_phase_inactive()
    # Native4 already offloads these at normal phase boundaries. Repeating
    # the configured offload here is cheap and makes the probe's GPU ownership
    # explicit; actor update/generation restores the existing FSDP shards.
    from verl.utils.fsdp_utils import offload_fsdp_param_and_grad, offload_fsdp_optimizer
    if worker._is_offload_param:
        offload_fsdp_param_and_grad(worker.actor_module_fsdp,
                                   offload_grad=worker._is_offload_grad)
    if worker._is_offload_optimizer:
        offload_fsdp_optimizer(worker.actor_optimizer)


def worker_operation(worker, operation, payload):
    if not worker._is_actor or dist.get_world_size() != 4:
        raise ValueError('DynRPO7B probe runs on the four allocated actor ranks')
    _idle_actor(worker)
    rank, device = dist.get_rank(), torch.device('cuda', torch.cuda.current_device())
    if operation == 'init':
        if getattr(worker, '_dynrpo7b_probe', None) is not None:
            head = worker._dynrpo7b_probe
            return dict(rank=rank, initialized=True, already_initialized=True,
                        completed_update_outer=head.completed_update_outer,
                        online_updates=head.online_updates)
        path = payload['checkpoint_path']
        expected = payload.get('checkpoint_sha256', WARMUP_SHA256)
        sha = _verify_once(path, expected)
        actor_rng = rng_state(device)
        try:
            head = GroupProbeAdapter.from_warmup(payload['model_dir'], path,
                checkpoint_sha256=None, device=device, rank=rank,
                encoder_attn_implementation=payload.get('encoder_attn_implementation', 'flash_attention_2'))
            head.source_checkpoint = dict(path=str(path), sha256=sha)
            head.offload()
            worker._dynrpo7b_probe = head
        finally:
            restore_rng(actor_rng, device)
        return dict(rank=rank, initialized=True, warmup_updates=head.warmup_cursor,
                    online_updates=head.online_updates,
                    completed_update_outer=head.completed_update_outer,
                    current_prefix=head.current_prefix, selection_state='raw',
                    official_frozen_encoder=True, extra_GPU_allocation=False,
                    RNG_expansion=head.RNG_expansion)
    head = getattr(worker, '_dynrpo7b_probe', None)
    if head is None:
        raise ValueError('Initialize the DynRPO7B probe before use')
    if operation in ('predict', 'score'):
        all_groups = payload.get('groups')
        if all_groups is None:
            shards = payload['groups_by_rank']
            all_groups = [g for shard in shards for g in shard]
            groups = shards[rank]
        else:
            # The driver transmits compact tokens once through ONE_TO_ALL;
            # each rank does one quarter of the actual encoder work.
            groups = all_groups[rank::4]
        prefix = payload.get('prefix', head.current_prefix)
        required = max((pack_group(g, head.separator, prefix)['valid_positions']
                        for g in all_groups), default=1)
        previous_capacity = head.probe.group_head.positions.num_embeddings
        with probe_rng(head):
            head.ensure_position_capacity(required)
            if head.probe.group_head.positions.num_embeddings != previous_capacity:
                head.activate(training=True)
                dist.broadcast(head.probe.group_head.positions.weight.data, src=0)
                dist.broadcast(head.ema.group_head.positions.weight.data, src=0)
                head.offload()
            scores = head.score_groups(groups, prefix=prefix,
                                       batch_size=payload.get('batch_size', 8), state='raw')
        return dict(rank=rank, prefix=prefix, completed_update_outer=head.completed_update_outer,
                    online_updates=head.online_updates, selection_state='raw',
                    position_capacity=head.probe.group_head.positions.num_embeddings,
                    position_capacity_changed=head.probe.group_head.positions.num_embeddings != previous_capacity,
                    predictions=[dict(qid=g['qid'], score=score,
                                      predicted_variance=score / 4)
                                 for g, score in zip(groups, scores)])
    if operation in ('fit', 'update'):
        return ProbeUpdater(head, global_batch=payload.get('global_batch', 128),
            micro=payload.get('micro', 8)).fit(payload['groups'],
                completed_actor_outer=payload['completed_actor_outer'],
                rollout_actor_outer=payload['rollout_actor_outer'],
                actor_checkpoint_ref=payload['actor_checkpoint_ref'],
                output_dir=payload['output_dir'], epochs=payload.get('epochs', 3))
    if operation == 'state_dict':
        path = payload.get('path')
        if path is None:
            path = str(Path(payload['output_dir']) / 'probe-state.pt')
        reference = save_probe_checkpoint(head, path)
        return dict(rank=rank, reference=reference, completed_update_outer=head.completed_update_outer,
                    online_updates=head.online_updates, current_prefix=head.current_prefix)
    if operation == 'load_state_dict':
        reference = payload.get('reference', payload.get('checkpoint'))
        if reference is None:
            reference = payload['state'].get('reference', payload['state'])
        _verify_once(reference['path'], reference['sha256'])
        saved = torch.load(reference['path'], map_location='cpu', weights_only=False)
        head.load_state_dict(saved)
        return dict(rank=rank, restored=True, reference=reference,
                    completed_update_outer=head.completed_update_outer,
                    online_updates=head.online_updates, current_prefix=head.current_prefix,
                    pending_update=head.pending_update)
    if operation == 'readback':
        return dict(rank=rank, warmup_updates=head.warmup_cursor,
                    online_updates=head.online_updates, EMA_updates=head.EMA_updates,
                    completed_update_outer=head.completed_update_outer,
                    pending_update=head.pending_update, current_prefix=head.current_prefix,
                    raw_parameters=sum(p.numel() for p in head.probe.parameters()),
                    frozen_encoder_parameters=sum(p.numel() for p in head.encoder.parameters()),
                    extra_GPU_allocation=False)
    raise ValueError('Unknown DynRPO7B probe operation: ' + operation)
