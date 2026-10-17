"""Keep objective normalization independent of the physical forward batch.

The historical actor objective gives every fixed logical micro equal weight,
then gives every authored token *within* that micro equal weight. A longer
trajectory must not silently change another logical micro's total weight when
several micros share a single forward/backward. This module does not forward,
step an optimizer, or catch distributed/CUDA failures.
"""
import torch


def logical_slices(rows, logical_micro_batch_size):
    assert type(rows) is int and rows > 0
    assert type(logical_micro_batch_size) is int and logical_micro_batch_size > 0
    assert rows % logical_micro_batch_size == 0, 'Physical batch must contain whole logical micros'
    return [slice(start, start + logical_micro_batch_size)
            for start in range(0, rows, logical_micro_batch_size)]


def logical_token_weights(mask, logical_micro_batch_size):
    """Unscaled token weights: each logical micro has total mass one.

    Monitor ratios divide by their total mass; the actor separately averages
    over logical micros, including the micros in other physical backwards.
    """
    assert mask.ndim == 2
    logical_slices(mask.shape[0], logical_micro_batch_size)
    return torch.cat([part / part.sum().clamp_min(1)
                      for part in mask.split(logical_micro_batch_size)])


def logical_masked_mean(values, mask, logical_micro_batch_size):
    assert values.shape == mask.shape
    weights = logical_token_weights(mask, logical_micro_batch_size)
    # Sanitize external observations/padding before multiplication: 0 * NaN
    # would otherwise contaminate the objective despite a zero target mask.
    authored = torch.where(mask.bool(), values, torch.zeros_like(values))
    return (authored * weights).sum() / (mask.shape[0] // logical_micro_batch_size)


def logical_policy_loss(compute_policy_loss, *, logical_micro_batch_size, **kwargs):
    """Apply the unmodified PPO/DualClip formula and average logical means.

    Slicing the already-computed log probabilities creates no model forwards.
    A physical batch can therefore perform one backward while retaining the
    loss and gradients of the original smaller logical batches.
    """
    slices = logical_slices(kwargs['log_prob'].shape[0], logical_micro_batch_size)
    rows = []
    for part in slices:
        local = {name: value[part] if name in ('old_log_prob', 'log_prob', 'advantages', 'eos_mask')
                 else value for name, value in kwargs.items()}
        rows.append(compute_policy_loss(**local))
    return tuple(torch.stack(values).mean() for values in zip(*rows))
