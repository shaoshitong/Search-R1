"""A separately stored EMA model, initialized as an exact fresh-model copy."""

import copy
from collections import defaultdict
import torch


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def _named_tensors(model):
    # named_buffers includes native RoPE buffers marked persistent=False, which
    # ordinary module.state_dict intentionally omits.
    result = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if set(result).intersection(buffers):
        raise ValueError("Model parameter and buffer names overlap")
    result.update(buffers)
    return result


class ParameterEMA:
    """ema <- .99 * ema + .01 * raw, exactly once after each returned Adam.

The raw optimization model is never replaced for evaluation. Nonfloating
buffers are copied exactly; floating buffers follow the same EMA rule.
The caller saves this object along with raw parameters, Adam and RNG.
"""
    def __init__(self, model, decay=0.99):
        if float(decay) != 0.99:
            raise ValueError("This experiment requires EMA decay exactly 0.99")
        self.decay = 0.99
        self.updates = 0
        self.model = copy.deepcopy(unwrap(model)).requires_grad_(False).eval()
        raw = _named_tensors(unwrap(model))
        if not all(torch.equal(value, raw[name]) for name, value in _named_tensors(self.model).items()):
            raise RuntimeError("Initial EMA must exactly copy the new raw model")

    @torch.no_grad()
    def update(self, model, actual_adam_step):
        if type(actual_adam_step) is not int or actual_adam_step != self.updates + 1:
            raise ValueError("EMA update must follow exactly the next successful actual Adam")
        raw = _named_tensors(unwrap(model))
        shadow = _named_tensors(self.model)
        if set(raw) != set(shadow):
            raise ValueError("Raw/EMA state identities differ")
        # Validate all state before mutating any EMA tensor.
        floating = []
        for name, value in raw.items():
            dest = shadow[name]
            if value.shape != dest.shape or value.dtype != dest.dtype:
                raise ValueError("Raw/EMA tensor shape or dtype differs: " + name)
            if value.is_floating_point():
                floating.append(value)
        if floating and not bool(torch.stack([torch.isfinite(value).all() for value in floating]).all()):
            raise FloatingPointError("Nonfinite post-Adam raw state")
        grouped = defaultdict(lambda: ([], []))
        for name, value in raw.items():
            dest = shadow[name]
            value = value.detach().to(device=dest.device)
            if dest.is_floating_point():
                destinations, sources = grouped[(dest.device, dest.dtype)]
                destinations.append(dest)
                sources.append(value)
            else:
                dest.copy_(value)
        for destinations, sources in grouped.values():
            torch._foreach_mul_(destinations, self.decay)
            torch._foreach_add_(destinations, sources, alpha=1.0 - self.decay)
        self.updates = actual_adam_step
        self.model.eval()

    def state_dict(self):
        persistent = self.model.state_dict()
        nonpersistent = {name: value for name, value in self.model.named_buffers() if name not in persistent}
        return dict(decay=self.decay, updates=self.updates, model=persistent,
                    nonpersistent_buffers=nonpersistent)

    def load_state_dict(self, state, *, actual_adam_step):
        if (state.get("decay") != 0.99 or type(state.get("updates")) is not int
                or state["updates"] != actual_adam_step):
            raise ValueError("Saved EMA decay/update counter differs from saved actual Adam")
        self.model.load_state_dict(state["model"], strict=True)
        persistent = self.model.state_dict()
        buffers = {name: value for name, value in self.model.named_buffers() if name not in persistent}
        saved_buffers = state["nonpersistent_buffers"]
        if set(buffers) != set(saved_buffers):
            raise ValueError("Saved EMA nonpersistent native buffer identities differ")
        with torch.no_grad():
            for name, target in buffers.items():
                value = saved_buffers[name]
                if target.shape != value.shape or target.dtype != value.dtype:
                    raise ValueError("Saved EMA native buffer shape/dtype differs: " + name)
                target.copy_(value.to(target.device))
        self.model.requires_grad_(False).eval()
        self.updates = actual_adam_step
