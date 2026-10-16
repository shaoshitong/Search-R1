"""Joint proper trajectory BCE and tied G5 mixed-event BCE.

Each row is one complete local G5 with its own strict binary EM labels. The
group target includes K=1: a group is mixed exactly when0<sum(labels)<5.
The mixed and extreme event masses are computed from all32 Bernoulli patterns
in the log domain. No probability floor, clamp, detached group graph, or
cross-rank gather is used.
"""
import math

import torch
import torch.nn.functional as F


def joint_loss(logits, labels, group_weight=1.0):
    """Return two independent means and their weighted sum for logits[B,5].

    ``labels`` must contain each trajectory's own0/1 EM, in the same G5 order.
    Low precision logits are promoted to FP32; FP64 is preserved for analyses.
    The caller keeps each complete G5 on its rank so this graph remains local.
    """
    if logits.ndim != 2 or logits.shape[1] != 5 or logits.shape[0] < 1:
        raise ValueError('joint_loss requires nonempty complete local G5 logits[B,5]')
    if labels.shape != logits.shape:
        raise ValueError('Each G5 logit needs its own trajectory EM label')
    if not logits.is_floating_point():
        raise TypeError('Correctness logits must be floating point')
    weight = float(group_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('The configured group BCE weight must be finite and nonnegative')
    z = logits if logits.dtype == torch.float64 else logits.float()
    target = labels.to(device=z.device, dtype=z.dtype)
    trajectory_bce = F.binary_cross_entropy_with_logits(z, target, reduction='mean')

    # Patterns0 and31 are allwrong/allcorrect; patterns1..30 are mixed.
    numbers = torch.arange(32, dtype=torch.long, device=z.device).unsqueeze(1)
    shifts = torch.arange(5, dtype=torch.long, device=z.device).unsqueeze(0)
    bits = torch.bitwise_and(torch.bitwise_right_shift(numbers, shifts), 1).bool()
    log_patterns = torch.where(bits.unsqueeze(0), F.logsigmoid(z).unsqueeze(1),
                               F.logsigmoid(-z).unsqueeze(1)).sum(dim=-1)
    log_mixed_mass = torch.logsumexp(log_patterns[:, 1:31], dim=1)
    log_extreme_mass = torch.logsumexp(torch.stack(
        (log_patterns[:, 0], log_patterns[:, 31]), dim=1), dim=1)
    # The event logit is log P(mixed)/P(extreme). Taking their ratio is the
    # log-domain normalization of the32 enumerated patterns, not a probability
    # clamp. Algebraically sigmoid(logit)=1-prod(p)-prod(1-p).
    mixed_event_logit = log_mixed_mass - log_extreme_mass
    k = target.sum(dim=1)
    mixed_label = ((k > 0) & (k < 5)).to(dtype=z.dtype)
    group_bce = F.binary_cross_entropy_with_logits(mixed_event_logit, mixed_label, reduction='mean')
    mixed_probability = torch.sigmoid(mixed_event_logit)
    total = trajectory_bce + weight * group_bce
    return dict(loss=total, total_loss=total, trajectory_bce=trajectory_bce,
                group_bce=group_bce, mixed_probability=mixed_probability,
                mixed_label=mixed_label)
