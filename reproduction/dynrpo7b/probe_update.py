"""Three inherited probe epochs on all 2560 completed own-G5 groups."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .group_probe_adapter import cpu_tree, file_sha256, probe_rng, target4variance
from .prefix_schedule import DEFAULT_SCHEDULE


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n')
    temporary.replace(path)


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _ranks():
    return (dist.get_rank(), dist.get_world_size()) if _distributed() else (0, 1)


def save_probe_checkpoint(adapter, path):
    """One common remote tensor file, including every rank's probe RNG."""
    rank, world = _ranks()
    rngs = [None] * world
    if _distributed():
        dist.all_gather_object(rngs, cpu_tree(adapter.rng))
    else:
        rngs[0] = cpu_tree(adapter.rng)
    reference = None
    if rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
        torch.save(adapter.state_dict(rngs), temporary)
        temporary.replace(path)
        reference = dict(path=str(path), bytes=path.stat().st_size,
                         sha256=file_sha256(path), warmup_updates=adapter.warmup_cursor,
                         online_updates=adapter.online_updates,
                         completed_update_outer=adapter.completed_update_outer,
                         current_prefix=adapter.current_prefix,
                         pending_update=copy.deepcopy(adapter.pending_update))
        atomic_json(path.with_suffix(path.suffix + '.json'), reference)
    if _distributed():
        value = [reference]
        dist.broadcast_object_list(value, src=0)
        reference = value[0]
    return reference


def _shuffle_groups(groups, outer, epoch, seed):
    """Same authentic G5 in every epoch; only q/trajectory ordering changes."""
    ordered = list(range(len(groups)))
    rng = random.Random('%s:%s:%s:question-order' % (seed, outer, epoch))
    rng.shuffle(ordered)
    for index in ordered:
        original = groups[index]
        order = list(range(5))
        random.Random('%s:%s:%s:%s:trajectory-order' %
                      (seed, outer, epoch, original['qid'])).shuffle(order)
        row = dict(original)
        row['policy_token_ids'] = [original['policy_token_ids'][i] for i in order]
        row['strict_em'] = [original['strict_em'][i] for i in order]
        if 'trajectory_ids' in original:
            row['trajectory_ids'] = [original['trajectory_ids'][i] for i in order]
        yield row


def _saved_epoch_receipts(output_dir, outer):
    result = {}
    for epoch in range(1, 4):
        path = output_dir / ('probe-epoch-%02d.pt.json' % epoch)
        if not path.exists():
            continue
        reference = json.loads(path.read_text())
        pending = reference.get('pending_update')
        if ((pending and pending['completed_actor_outer'] == outer
             and pending['completed_epochs'] == epoch)
                or epoch == 3 and reference.get('completed_update_outer') == outer):
            result[epoch] = reference
    return result


def _restore_epoch(adapter, reference):
    rank, _ = _ranks()
    status = None
    if rank == 0:
        try:
            status = dict(ok=file_sha256(reference['path']) == reference['sha256'])
        except OSError as error:
            status = dict(ok=False, error=str(error))
    if _distributed():
        received = [status]
        dist.broadcast_object_list(received, src=0)
        status = received[0]
    if not status['ok']:
        raise ValueError('Saved probe epoch is unavailable or its SHA differs')
    adapter.load_state_dict(torch.load(reference['path'], map_location='cpu', weights_only=False))


class ProbeUpdater:
    def __init__(self, adapter, *, global_batch=128, micro=8, seed=2026102327):
        self.adapter, self.global_batch, self.micro, self.seed = adapter, global_batch, micro, seed

    def fit(self, groups, *, completed_actor_outer, rollout_actor_outer,
            actor_checkpoint_ref, output_dir, epochs=3):
        adapter = self.adapter
        rank, world = _ranks()
        binding = DEFAULT_SCHEDULE.update_binding(completed_actor_outer)
        if (rollout_actor_outer != binding['rollout_actor_outer'] or not actor_checkpoint_ref
                or epochs != 3 or len(groups) != 2560
                or len({str(g['qid']) for g in groups}) != 2560):
            raise ValueError('Probe fit requires saved actor t and all2560 own-G5 from actor t-1')
        if self.global_batch % world or len(groups) % self.global_batch:
            raise ValueError('Probe global128 batch must divide the rank and data layouts')
        for group in groups:
            target4variance(group['strict_em'])
        dataset_sha = hashlib.sha256(json.dumps([
            (g['qid'], g.get('trajectory_ids'), g['prompt_token_ids'], g['strict_em'],
             [t[:binding['prefix']] for t in g['policy_token_ids']])
            for g in groups], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_epochs = _saved_epoch_receipts(output_dir, completed_actor_outer)
        completed_epochs = (3 if adapter.completed_update_outer == completed_actor_outer
                            else (adapter.pending_update or {}).get('completed_epochs', 0))
        if saved_epochs and max(saved_epochs) > completed_epochs:
            _restore_epoch(adapter, saved_epochs[max(saved_epochs)])
            completed_epochs = max(saved_epochs)
        summaries = []
        for epoch in range(1, completed_epochs + 1):
            summary_file = output_dir / ('epoch-%02d.json' % epoch)
            if summary_file.exists():
                summaries.append(json.loads(summary_file.read_text()))
            elif epoch in saved_epochs:
                summaries.append(dict(**binding, epoch=epoch, checkpoint=saved_epochs[epoch],
                                      recovered_from_checkpoint_receipt=True))
        if adapter.completed_update_outer == completed_actor_outer:
            if adapter.last_update is not None and adapter.last_update['dataset_sha256'] != dataset_sha:
                raise ValueError('Published probe phase used different completed trajectories')
            reference = saved_epochs.get(3)
            if reference is None:
                raise ValueError('Published probe is missing its durable epoch3 receipt')
            if rank == 0:
                atomic_json(output_dir / 'published.json', dict(
                    status='PROBE_UPDATE_PUBLISHED', **binding, checkpoint=reference,
                    dataset_sha256=dataset_sha, online_updates=adapter.online_updates,
                    effective_state='raw', score_semantics='4*predicted_population_variance'))
            return dict(rank=rank, already_published=True, published=True,
                        completed_update_outer=completed_actor_outer,
                        checkpoint=reference, epochs=summaries, binding=binding,
                        online_updates=adapter.online_updates, current_prefix=adapter.current_prefix)
        if adapter.pending_update is None:
            adapter.pending_update = dict(**binding, dataset_sha256=dataset_sha,
                actor_checkpoint_ref=actor_checkpoint_ref, completed_epochs=0,
                execution_phase='actor_saved_probe_pending')
        elif (adapter.pending_update['completed_actor_outer'] != completed_actor_outer
              or adapter.pending_update['dataset_sha256'] != dataset_sha):
            raise ValueError('Pending probe phase belongs to different completed trajectories')
        local_batch = self.global_batch // world
        # Position capacity is agreed once from the entire supplied pool, so
        # every rank retains identical parameter shapes for gradient reduction.
        from .group_probe_adapter import pack_group
        required = max(pack_group(g, adapter.separator, binding['prefix'])['valid_positions'] for g in groups)
        with probe_rng(adapter):
            adapter.ensure_position_capacity(required)
        adapter.activate(training=True)
        if _distributed():
            dist.broadcast(adapter.probe.group_head.positions.weight.data, src=0)
            dist.broadcast(adapter.ema.group_head.positions.weight.data, src=0)
        try:
            with probe_rng(adapter):
                for epoch in range(adapter.pending_update['completed_epochs'] + 1, epochs + 1):
                    plan = list(_shuffle_groups(groups, completed_actor_outer, epoch, self.seed))
                    adapter.probe.train()
                    start, loss_sum, seen = time.monotonic(), 0., 0
                    for offset in range(0, len(plan), self.global_batch):
                        batch = plan[offset + rank * local_batch:offset + (rank + 1) * local_batch]
                        adapter.optimizer.zero_grad(set_to_none=True)
                        total = 0.
                        for micro_start in range(0, len(batch), self.micro):
                            rows = batch[micro_start:micro_start + self.micro]
                            targets = torch.tensor([target4variance(g['strict_em']) for g in rows],
                                dtype=torch.float32, device=adapter.device)
                            data = adapter.capture(rows, binding['prefix'])
                            with adapter.autocast():
                                logits = adapter.probe(data)
                            loss = (torch.sigmoid(logits).float() - targets).square().sum() / len(batch)
                            loss.backward()
                            total += float(loss.detach())
                        if _distributed():
                            for parameter in adapter.probe.parameters():
                                if parameter.grad is not None:
                                    dist.all_reduce(parameter.grad)
                                    parameter.grad.div_(world)
                        torch.nn.utils.clip_grad_norm_(adapter.probe.parameters(), 1., error_if_nonfinite=True)
                        adapter.optimizer.step()
                        with torch.no_grad():
                            for parameter, average in zip(adapter.probe.parameters(), adapter.ema.parameters()):
                                average.mul_(.9995).add_(parameter, alpha=.0005)
                        adapter.online_updates += 1
                        adapter.EMA_updates += 1
                        loss_sum += total * len(batch)
                        seen += len(batch)
                    adapter.pending_update['completed_epochs'] = epoch
                    adapter.pending_update['execution_phase'] = 'probe_epoch_complete'
                    if epoch == epochs:
                        adapter.last_update = dict(**binding, dataset_sha256=dataset_sha,
                                                   completed_epochs=epochs)
                        adapter.completed_update_outer = completed_actor_outer
                        adapter.current_prefix = binding['prefix']
                        adapter.pending_update = None
                    # Store the RNG after the completed epoch, before restoring
                    # the actor streams at the enclosing context boundary.
                    from .group_probe_adapter import rng_state
                    adapter.rng = rng_state(adapter.device)
                    reference = save_probe_checkpoint(adapter, output_dir / ('probe-epoch-%02d.pt' % epoch))
                    counts = torch.tensor([loss_sum, seen], device=adapter.device, dtype=torch.float64)
                    if _distributed():
                        dist.all_reduce(counts)
                    summary = dict(**binding, epoch=epoch,
                        target4variance_MSE=float(counts[0] / counts[1]),
                        actual_questions=int(counts[1]), global_batch=self.global_batch,
                        actual_optimizer_updates=20, online_updates=adapter.online_updates,
                        warmup_updates=adapter.warmup_cursor, EMA_decay=.9995,
                        learning_rate=adapter.optimizer.param_groups[0]['lr'],
                        seconds=time.monotonic() - start, checkpoint=reference,
                        dataset_sha256=dataset_sha, generated_extra_trajectories=0,
                        G5_reused_this_epoch=True)
                    summaries.append(summary)
                    if rank == 0:
                        atomic_json(output_dir / ('epoch-%02d.json' % epoch), summary)
                        atomic_json(output_dir / 'latest.json', reference)
                        print(json.dumps(dict(event='dynrpo7b_probe_epoch_complete', **summary)), flush=True)
                if rank == 0:
                    atomic_json(output_dir / 'published.json', dict(
                        status='PROBE_UPDATE_PUBLISHED', **binding, checkpoint=reference,
                        dataset_sha256=dataset_sha, online_updates=adapter.online_updates,
                        effective_state='raw', score_semantics='4*predicted_population_variance'))
        finally:
            adapter.offload()
        return dict(rank=rank, published=True, binding=binding, epochs=summaries,
                    checkpoint=reference, completed_update_outer=adapter.completed_update_outer,
                    online_updates=adapter.online_updates, current_prefix=adapter.current_prefix)
