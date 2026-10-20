"""Exact three-event G5 probabilities using a three-state log-space recursion.

This is algebraically the same 32-pattern probability as group_event_scores.
It reduces temporary size for repeated CPU calibration; no independence or
label definition changes. The old primitive and all live runtimes are intact.
"""
import torch
import torch.nn.functional as F


def group_event_scores_dp(logits):
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != 2 or logits.shape[1] != 5
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Finite floating logits[N,5] required')
    z = logits if logits.dtype == torch.float64 else logits.float()
    yes, no = F.logsigmoid(z), F.logsigmoid(-z)
    correct, wrong = yes[:, 0], no[:, 0]
    mixed = torch.logaddexp(wrong + yes[:, 1], correct + no[:, 1])
    correct, wrong = correct + yes[:, 1], wrong + no[:, 1]
    for j in range(2, 5):
        mixed = torch.logaddexp(mixed, torch.logaddexp(
            wrong + yes[:, j], correct + no[:, j]))
        correct, wrong = correct + yes[:, j], wrong + no[:, j]
    extreme = torch.logaddexp(wrong, correct)
    event_logit = mixed - extreme
    return dict(log_allwrong=wrong, log_allcorrect=correct, log_mixed=mixed,
                mixed_event_logit=event_logit,
                mixed_probability=event_logit.sigmoid())
