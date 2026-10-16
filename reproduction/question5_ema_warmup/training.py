"""Whole-event G5 minibatches, exact-once epoch plans and actual-Adam EMA."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import random

import torch
import torch.distributed as dist

from .ema import unwrap
from .loss import joint_loss
from .metrics import GroupMetrics


@dataclass
class TrainCounters:
    optimizer_steps: int = 0
    ema_updates: int = 0
    complete_epochs: int = 0
    global_groups_seen: int = 0
    global_trajectories_seen: int = 0
    rejected_nonfinite_updates: int = 0

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        result = cls(**state)
        if any(type(value) is not int or value < 0 for value in result.state_dict().values()):
            raise ValueError("Saved counters must be real nonnegative integers")
        if result.ema_updates != result.optimizer_steps:
            raise ValueError("Saved successful Adam/EMA counters differ")
        if result.global_trajectories_seen != 5 * result.global_groups_seen:
            raise ValueError("Saved group/trajectory exposure counters differ")
        return result


@dataclass(frozen=True)
class GroupStep:
    """One global optimizer opportunity, partitioned only along whole G5 events."""
    group_indices: tuple
    rank_group_indices: tuple

    @property
    def global_groups(self):
        return len(self.group_indices)

    def state_dict(self):
        return dict(group_indices=list(self.group_indices),
                    rank_group_indices=[list(row) for row in self.rank_group_indices],
                    global_groups=self.global_groups)


def build_epoch_plan(group_count, *, epoch, seed, world=1,
                     global_group_pattern=(51, 51, 51, 51, 52)):
    """Shuffle every event once; 51/52 groups average 256 trajectories/update.

The default pattern matches the reference complete-G5 approximation to batch
256. The final update contains its actual remainder, including zero-data ranks
when fewer than ``world`` events remain. No drop_last, duplication or replacement.
The sampler owns a private Python RNG and never changes global RNG state.
"""
    if (type(group_count) is not int or group_count < 1 or type(epoch) is not int or epoch < 1
            or type(seed) is not int or type(world) is not int or world < 1
            or not global_group_pattern
            or any(type(n) is not int or n < 1 for n in global_group_pattern)):
        raise ValueError("Invalid exact-once epoch/group/rank/batch configuration")
    order = list(range(group_count))
    random.Random(seed + epoch).shuffle(order)
    plan, offset, extras_start = [], 0, 0
    while offset < group_count:
        size = min(global_group_pattern[len(plan) % len(global_group_pattern)], group_count - offset)
        indices = tuple(order[offset:offset + size])
        base, extras = divmod(size, world)
        counts = [base] * world
        for i in range(extras):
            counts[(extras_start + i) % world] += 1
        by_rank, cursor = [], 0
        for count in counts:
            by_rank.append(indices[cursor:cursor + count])
            cursor += count
        plan.append(GroupStep(indices, tuple(by_rank)))
        offset += size
        extras_start = (extras_start + extras) % world
    flattened = [i for step in plan for i in step.group_indices]
    if len(flattened) != group_count or set(flattened) != set(range(group_count)):
        raise RuntimeError("Exact-once epoch coverage was not preserved")
    return plan


def _distributed():
    return dist.is_available() and dist.is_initialized()


def actual_adam_step(optimizer):
    """Read each optimizer parameter's real saved Adam step, not a call counter."""
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Optimizer parameter set is empty or duplicated")
    if not optimizer.state:
        return 0
    if set(optimizer.state) != set(parameters):
        raise RuntimeError("Adam state is partial or contains a foreign parameter")
    steps = []
    for parameter in parameters:
        state = optimizer.state[parameter]
        if not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise RuntimeError("Adam state lacks actual step/moments")
        value = state["step"]
        value = value.item() if isinstance(value, torch.Tensor) else value
        if int(value) != value or value < 0:
            raise RuntimeError("Invalid actual Adam parameter step")
        steps.append(int(value))
    if len(set(steps)) != 1:
        raise RuntimeError("Adam parameters have unequal actual update counters")
    return steps[0]


def _event_id(group):
    identity = group.get("event_group_id", group.get("event_id"))
    if identity is None:
        raise ValueError("Every complete G5 must retain its sampling-event identity")
    if "event_id" in group and "event_group_id" in group and group["event_id"] != group["event_group_id"]:
        raise ValueError("Conflicting G5 event identities")
    return str(identity)


def validate_groups(groups):
    """Call once on the full split, before producing the first epoch plan."""
    events, tids = set(), set()
    for group in groups:
        event = _event_id(group)
        ids, labels = group["trajectory_ids"], group["labels"]
        if event in events or len(ids) != 5 or len(set(ids)) != 5 or len(labels) != 5:
            raise ValueError("Duplicate event or incomplete same-event G5")
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError("Every trajectory must retain its own binary strict EM")
        if tids.intersection(ids):
            raise ValueError("A trajectory event is duplicated across G5 groups")
        tids.update(ids)
        events.add(event)
    if not events:
        raise ValueError("No complete sampling events in this split")
    return dict(groups=len(events), trajectories=len(tids))


def _batch(reader, groups, device):
    ids = [tid for group in groups for tid in group["trajectory_ids"]]
    labels = torch.tensor([group["labels"] for group in groups], device=device, dtype=torch.float32)
    batch = reader.get_batch(ids, device=device)
    if batch.get("trajectory_ids") != ids:
        raise ValueError("Provider reordered or replaced original trajectory events")
    if "labels" in batch and not torch.equal(batch["labels"].to(labels).reshape(-1), labels.reshape(-1)):
        raise ValueError("Provider strict EM differs from the bound original event labels")
    return batch, labels


def _autocast(device, dtype):
    device_type = torch.device(device).type
    if dtype is None:
        return nullcontext()
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Use BF16/FP16 autocast or explicitly disable for CPU arithmetic checks")
    return torch.autocast(device_type=device_type, dtype=dtype)


def _all_finite(flag, device):
    value = torch.as_tensor(flag, device=device, dtype=torch.int32)
    if _distributed():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value.item())


def train_update(model, optimizer, ema, counters, groups, step, *, rank, reader,
                 micro_groups, device, autocast_dtype=torch.bfloat16):
    """One returned Adam and one EMA update after all complete local G5 micros.

``model`` may be ordinary or DDP (construct DDP with broadcast_buffers=False).
DDP averages rank gradients, so every micro's mean is multiplied by
world * micro_groups / actual_global_groups. Unequal rank/micro/remainder
sizes therefore equal one direct global mean of both loss terms.
No gradient scaler, schedule, class weights, clipping, or extra evaluation is
introduced here. The caller sets the authorized LR before this call.
"""
    world = len(step.rank_group_indices)
    if (type(rank) is not int or not 0 <= rank < world or type(micro_groups) is not int
            or micro_groups < 1 or step.global_groups < 1):
        raise ValueError("Invalid complete-G5 update coordinates")
    partitions = [i for row in step.rank_group_indices for i in row]
    if partitions != list(step.group_indices) or len(set(partitions)) != step.global_groups:
        raise ValueError("Rank partition lost or repeated a complete event")
    if _distributed():
        if dist.get_world_size() != world or dist.get_rank() != rank or not hasattr(model, "no_sync"):
            raise ValueError("Distributed ranks and bound complete-G5 plan differ")
    elif world != 1 or rank != 0:
        raise ValueError("Multi-rank plan requires initialized DDP")
    raw = unwrap(model)
    if any(parameter.dtype != torch.float32 for parameter in raw.parameters()):
        raise ValueError("Trainable block/Linear master parameters must stay FP32")
    actual_before = actual_adam_step(optimizer)
    if actual_before != counters.optimizer_steps or ema.updates != actual_before or counters.ema_updates != actual_before:
        raise RuntimeError("Raw Adam/EMA/persisted counters disagree before this update")
    if {id(p) for p in raw.parameters() if p.requires_grad} != {
            id(p) for g in optimizer.param_groups for p in g["params"]}:
        raise ValueError("Adam must cover exactly this fresh trainable probe")
    local_indices = step.rank_group_indices[rank]
    micros = [local_indices[start:start + micro_groups] for start in range(0, len(local_indices), micro_groups)]
    # Empty final rank uses a real DDP zero graph, not an invented sample.
    if not micros:
        micros = [()]
    optimizer.zero_grad(set_to_none=True)
    model.train()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    loss_finite = torch.ones((), dtype=torch.bool, device=device)
    for micro_number, indices in enumerate(micros):
        sync = nullcontext() if micro_number == len(micros) - 1 or not hasattr(model, "no_sync") else model.no_sync()
        with sync:
            if indices:
                micro = [groups[index] for index in indices]
                batch, labels = _batch(reader, micro, device)
                with _autocast(device, autocast_dtype):
                    logits = model(batch).reshape(len(micro), 5)
                losses = joint_loss(logits, labels)
                backward = losses["loss"] * (world * len(micro) / step.global_groups)
                totals[0] += losses["trajectory_bce"].detach().double() * len(micro)
                totals[1] += losses["group_bce"].detach().double() * len(micro)
                totals[2] += len(micro)
                totals[3] += len(micro) * 5
            else:
                with _autocast(device, autocast_dtype):
                    backward = model(None)
            loss_finite &= torch.isfinite(backward.detach())
            backward.backward()
    gradients = [p.grad for p in raw.parameters() if p.requires_grad]
    all_gradients_present = all(value is not None for value in gradients)
    finite_grads = torch.stack([torch.isfinite(value).all() for value in gradients if value is not None]).all()
    if not _all_finite(loss_finite & finite_grads & all_gradients_present, device):
        optimizer.zero_grad(set_to_none=True)
        counters.rejected_nonfinite_updates += 1
        raise FloatingPointError("Nonfinite/missing complete-G5 loss or gradient; no Adam/EMA performed")
    optimizer.step()
    actual_after = actual_adam_step(optimizer)
    if actual_after != actual_before + 1:
        raise RuntimeError("Returned Adam did not advance every parameter exactly once")
    # Record the retained actual raw update before a post-step/EMA error can
    # occur. Such an error is partial state, never a claimed automatic rollback.
    counters.optimizer_steps = actual_after
    counters.global_groups_seen += step.global_groups
    counters.global_trajectories_seen += step.global_groups * 5
    post_finite = torch.stack([torch.isfinite(p).all() for p in raw.parameters()]).all()
    if not _all_finite(post_finite, device):
        raise FloatingPointError("Nonfinite retained post-Adam raw parameters; EMA not advanced")
    ema.update(raw, actual_after)
    counters.ema_updates = ema.updates
    if _distributed():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        boundaries = [None] * world
        dist.all_gather_object(boundaries, (actual_after, ema.updates))
        if boundaries != [(actual_after, actual_after)] * world:
            raise RuntimeError("Ranks disagree on returned Adam/EMA boundaries")
    if int(totals[2]) != step.global_groups or int(totals[3]) != step.global_groups * 5:
        raise RuntimeError("Actual completed group/trajectory exposure count differs")
    return dict(actual_adam_step=actual_after, ema_updates=ema.updates,
                global_groups=step.global_groups, global_trajectories=step.global_groups * 5,
                local_groups=len(local_indices), local_trajectories=len(local_indices) * 5,
                online_raw_trajectory_bce=float(totals[0] / totals[2]),
                online_raw_group_bce=float(totals[1] / totals[2]),
                online_raw_total_loss=float((totals[0] + totals[1]) / totals[2]),
                metric_semantics="raw_pre_own_update_complete_G5_actual_group_weighted",
                actual_optimizer_learning_rates=[float(g["lr"]) for g in optimizer.param_groups],
                skipped_adam=False)


@torch.no_grad()
def evaluate_groups(model, groups, *, rank, world, reader, micro_groups, device,
                    weight_kind="ema0.99", autocast_dtype=torch.bfloat16):
    """Same fixed Head whole-split diagnostics; each event assigned to one rank.

Use ``ema.model`` for both train and fixed test. The raw model is not swapped,
and no train/Adam/EMA/RNG step is performed. Only small scalar sums are reduced.
"""
    if not 0 <= rank < world or micro_groups < 1:
        raise ValueError("Invalid diagnostic group/rank coordinates")
    if _distributed():
        if dist.get_world_size() != world or dist.get_rank() != rank:
            raise ValueError("Diagnostic distributed rank/world differs")
    elif world != 1:
        raise ValueError("Distributed evaluation requires initialized process group")
    target = unwrap(model)
    was_training = target.training
    target.eval()
    metrics = GroupMetrics()
    assigned = range(rank, len(groups), world)
    try:
        for start in range(0, len(assigned), micro_groups):
            selected = [groups[index] for index in assigned[start:start + micro_groups]]
            batch, labels = _batch(reader, selected, device)
            with _autocast(device, autocast_dtype):
                logits = target(batch).reshape(len(selected), 5)
            metrics.add(logits, labels)
        metrics.reduce_distributed(device)
        if metrics.groups != len(groups):
            raise RuntimeError("Whole-split same-head diagnostic coverage differs")
        return metrics.result(weight_kind=weight_kind)
    finally:
        target.train(was_training)
