"""Durable raw/EMA/Adam/RNG state at a real complete minibatch boundary."""

import hashlib
import os
from pathlib import Path
import random

import torch

from .ema import unwrap
from .training import TrainCounters, actual_adam_step


def capture_rng(*, rank=0, device=None, sampler_state=None):
    import numpy as np
    state = dict(rank=rank, python=random.getstate(), numpy=np.random.get_state(),
                 torch_cpu=torch.get_rng_state(), sampler_state=sampler_state)
    if device is not None and torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, *, rank=0, device=None):
    import numpy as np
    if state.get("rank") != rank:
        raise ValueError("Saved RNG belongs to a different rank")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device is not None and torch.device(device).type == "cuda":
        if "torch_cuda" not in state:
            raise ValueError("CUDA continuation requires this rank's saved CUDA RNG")
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    return state.get("sampler_state")


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def make_checkpoint(model, optimizer, ema, counters, *, binding, epoch,
                    next_update_in_epoch, epoch_plan_sha256, rank_rng, completed_epoch=False):
    """The binding must include immutable data/source/model/architecture identities.

``next_update_in_epoch`` is zero-based; when a complete epoch was committed the
next epoch starts at zero. ``rank_rng`` includes each private sampler state.
Storage/cleanup policy and checkpoint publication remain the caller's work.
"""
    steps = actual_adam_step(optimizer)
    if counters.optimizer_steps != steps or counters.ema_updates != steps or ema.updates != steps:
        raise ValueError("Checkpoint requires a complete synchronized raw Adam/EMA boundary")
    if (type(epoch) is not int or epoch < 1 or type(next_update_in_epoch) is not int
            or next_update_in_epoch < 0 or not isinstance(epoch_plan_sha256, str)
            or len(epoch_plan_sha256) != 64 or not binding or not rank_rng):
        raise ValueError("Checkpoint requires real epoch, sampler cursor, plan SHA, binding and RNG")
    if [row.get("rank") for row in rank_rng] != list(range(len(rank_rng))):
        raise ValueError("All rank RNG records must be present in rank order")
    raw = unwrap(model)
    raw_persistent = raw.state_dict()
    raw_nonpersistent = {name: value for name, value in raw.named_buffers() if name not in raw_persistent}
    return dict(schema_version=1, role="fresh_question20_joint200_ema099",
                binding=_cpu_copy(binding), raw_model=_cpu_copy(raw_persistent),
                raw_nonpersistent_buffers=_cpu_copy(raw_nonpersistent),
                ema=_cpu_copy(ema.state_dict()), optimizer=_cpu_copy(optimizer.state_dict()),
                counters=counters.state_dict(), rank_rng=_cpu_copy(rank_rng),
                epoch=epoch, completed_epoch=bool(completed_epoch),
                next_update_in_epoch=next_update_in_epoch, epoch_plan_sha256=epoch_plan_sha256,
                actual_adam_steps_per_rank=[steps] * len(rank_rng),
                old_probe_imported=False, old_optimizer_imported=False, actor_updates=0)


def restore_checkpoint(saved, model, optimizer, ema, *, expected_binding, device=None):
    """Restore only the same new experiment, never a prior reference head/state."""
    if saved.get("role") != "fresh_question20_joint200_ema099" or saved.get("binding") != expected_binding:
        raise ValueError("Checkpoint run/source/model/data/recipe binding differs")
    if saved.get("old_probe_imported") is not False or saved.get("old_optimizer_imported") is not False:
        raise ValueError("This fresh experiment cannot import a prior probe or optimizer")
    counters = TrainCounters.from_state_dict(saved["counters"])
    boundaries = saved["actual_adam_steps_per_rank"]
    if not boundaries or boundaries != [counters.optimizer_steps] * len(saved["rank_rng"]):
        raise ValueError("Saved rank Adam boundaries differ")
    raw = unwrap(model)
    raw.load_state_dict(saved["raw_model"], strict=True)
    persistent = raw.state_dict()
    buffers = {name: value for name, value in raw.named_buffers() if name not in persistent}
    saved_buffers = saved["raw_nonpersistent_buffers"]
    if set(buffers) != set(saved_buffers):
        raise ValueError("Saved raw nonpersistent native buffer identities differ")
    with torch.no_grad():
        for name, target in buffers.items():
            value = saved_buffers[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError("Saved raw native buffer shape/dtype differs: " + name)
            target.copy_(value.to(target.device))
    optimizer.load_state_dict(saved["optimizer"])
    if device is not None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
    if actual_adam_step(optimizer) != counters.optimizer_steps:
        raise ValueError("Restored actual Adam state differs from the saved counter")
    ema.load_state_dict(saved["ema"], actual_adam_step=counters.optimizer_steps)
    # Restore RNG separately after all constructors/DDP setup have finished.
    return counters


def save_checkpoint(path, saved):
    """Atomic fsync + SHA receipt. Call only with a real remote artifact path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError("Existing partial checkpoint must be preserved for explicit recovery")
    with partial.open("xb") as handle:
        torch.save(saved, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest(),
                actual_adam_steps_per_rank=saved["actual_adam_steps_per_rank"],
                ema_updates=saved["ema"]["updates"], epoch=saved["epoch"],
                completed_epoch=saved["completed_epoch"], actor_updates=0)
