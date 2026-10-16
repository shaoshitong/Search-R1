"""Equal-weight proper trajectory BCE and same-event complete-G5 mixed BCE."""

import torch
import torch.nn.functional as F


def _logits(logits):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape[1] != 5 or logits.shape[0] < 1):
        raise ValueError("Require nonempty complete G5 logits [groups, 5]")
    if not logits.is_floating_point():
        raise TypeError("Correctness logits must be floating point")
    # Keep FP64 for tiny mathematical checks; production BF16/FP16 is promoted.
    return logits if logits.dtype == torch.float64 else logits.float()


def mixed_event_logits(logits):
    """Compute log(P(mixed)/P(extreme)) without subtracting probabilities.

The 30 mixed Bernoulli patterns and the two extreme patterns partition all
32 possible G5 outcomes. Their logsumexp ratio is algebraically equivalent
to h = 1 - prod(sigmoid(z)) - prod(1 - sigmoid(z)), including extreme logits.
No clamp, probability floor, detach or cross-event grouping is used.
"""
    z = _logits(logits)
    numbers = torch.arange(32, device=z.device).unsqueeze(1)
    shifts = torch.arange(5, device=z.device).unsqueeze(0)
    bits = ((numbers >> shifts) & 1).bool()
    log_patterns = torch.where(bits.unsqueeze(0), F.logsigmoid(z).unsqueeze(1),
                               F.logsigmoid(-z).unsqueeze(1)).sum(dim=-1)
    mixed = torch.logsumexp(log_patterns[:, 1:31], dim=1)
    extreme = torch.logsumexp(log_patterns[:, (0, 31)], dim=1)
    return mixed - extreme


def joint_loss(logits, labels):
    """Return both independent means and their sum; both weights are exactly 1.

Labels are each trajectory's original strict EM, not a question-wide label.
One correct response out of five is a mixed group (as are K=2,3,4).
"""
    z = _logits(logits)
    if not isinstance(labels, torch.Tensor) or labels.shape != z.shape:
        raise ValueError("Every G5 logit requires its own aligned EM label")
    target = labels.to(device=z.device, dtype=z.dtype)
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("Strict EM labels must be exactly binary")
    k = target.sum(dim=1)
    mixed_label = ((k > 0) & (k < 5)).to(z.dtype)
    group_logits = mixed_event_logits(z)
    trajectory_bce = F.binary_cross_entropy_with_logits(z, target, reduction="mean")
    group_bce = F.binary_cross_entropy_with_logits(group_logits, mixed_label, reduction="mean")
    total = trajectory_bce + group_bce
    return dict(loss=total, total_loss=total, trajectory_bce=trajectory_bce,
                group_bce=group_bce, mixed_event_logits=group_logits,
                mixed_probability=torch.sigmoid(group_logits), mixed_label=mixed_label)
