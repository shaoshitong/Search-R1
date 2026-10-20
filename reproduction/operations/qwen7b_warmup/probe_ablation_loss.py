"""Pure complete-global-batch G5 losses for controlled probe ablations.

No collectives, detached gathered logits, inference labels, or implicit class
reweighting occur here. A distributed trainer must construct an autograd-correct
global batch and reproduce these reductions before claiming equivalent grads.
Inference is ALWAYS q=1-prod(p)-prod(1-p), independent of every label/loss flag.
"""
import math
import torch
import torch.nn.functional as F


def _logits(logits):
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError('Floating-point logits required')
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] != 5:
        raise ValueError('Nonempty complete global G5 logits[B,5] required')
    z = logits if logits.dtype == torch.float64 else logits.float()
    if not bool(torch.isfinite(z).all()):
        raise ValueError('Finite logits required')
    return z


def group_event_scores(logits):
    """Label-free, stable scores; G5 probabilities are enumerated in log space."""
    z = _logits(logits)
    numbers = torch.arange(32, dtype=torch.long, device=z.device)[:, None]
    shifts = torch.arange(5, dtype=torch.long, device=z.device)[None, :]
    bits = ((numbers >> shifts) & 1).bool()
    log_patterns = torch.where(bits[None], F.logsigmoid(z)[:, None],
                               F.logsigmoid(-z)[:, None]).sum(-1)
    log_wrong, log_correct = log_patterns[:, 0], log_patterns[:, 31]
    log_mixed = torch.logsumexp(log_patterns[:, 1:31], dim=1)
    log_extreme = torch.logsumexp(torch.stack((log_wrong, log_correct), 1), 1)
    event_logit = log_mixed - log_extreme
    return dict(mixed_event_logit=event_logit,
                mixed_probability=event_logit.sigmoid(),
                log_allwrong=log_wrong, log_mixed=log_mixed,
                log_allcorrect=log_correct)


def probe_ablation_loss(
        logits, labels, *, trajectory_weight=1., binary_mixed_weight=0.,
        three_event_weight=0., within_rank_weight=0., inter_rank_weight=0.,
        stat_mean_weight=0., stat_variance_weight=0., within_pair_mask=None,
        within_margin=1., inter_margin=1., inter_hard_negative_k=None):
    """Return independently reported components and their explicit weighted sum.

    Reductions: trajectory BCE over B*5 trajectories; event losses and the two
    moment MSEs over B questions. Within ranking first averages valid own-EM
    positive-negative pairs per question, then averages ONLY eligible questions.
    Caller supplies the input-distinguishability mask; logits cannot infer it.
    Inter ranking averages all actual Mixed/nonMixed pairs. Hard-negative mode
    selects up to k highest-scored nonMixed questions, shared by all positives;
    ties use lower global-batch index first. No eligible pairs => connected zero.
    """
    z = _logits(logits)
    if not isinstance(labels, torch.Tensor) or labels.shape != z.shape:
        raise ValueError('Each logit needs its own EM label in the same G5 order')
    y = labels.to(device=z.device, dtype=z.dtype)
    if not bool(torch.isfinite(y).all()) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError('Strict binary own-trajectory EM labels required')
    weights = dict(trajectory=float(trajectory_weight),
                   binary_mixed=float(binary_mixed_weight),
                   three_event=float(three_event_weight),
                   within_rank=float(within_rank_weight),
                   inter_rank=float(inter_rank_weight),
                   stat_mean=float(stat_mean_weight),
                   stat_variance=float(stat_variance_weight))
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Every component weight must be finite and nonnegative')
    if weights['binary_mixed'] and weights['three_event']:
        raise ValueError('Binary Mixed and three-event objectives are alternatives')
    if any(not math.isfinite(float(v)) or float(v) < 0
           for v in (within_margin, inter_margin)):
        raise ValueError('Finite nonnegative ranking margins required')
    if inter_hard_negative_k is not None and (
            isinstance(inter_hard_negative_k, bool) or
            not isinstance(inter_hard_negative_k, int) or inter_hard_negative_k < 1):
        raise ValueError('Hard-negative k must be a positive integer or None')

    scores = group_event_scores(z)
    p = z.sigmoid()
    k = y.sum(1)
    mixed = (k > 0) & (k < 5)
    zero = z.sum() * 0.
    trajectory = F.binary_cross_entropy_with_logits(z, y, reduction='mean')
    binary = F.binary_cross_entropy_with_logits(
        scores['mixed_event_logit'], mixed.to(z.dtype), reduction='mean')
    event_nll = -torch.where(k == 0, scores['log_allwrong'],
                            torch.where(k == 5, scores['log_allcorrect'],
                                        scores['log_mixed'])).mean()

    expected = (y[:, :, None] == 1) & (y[:, None, :] == 0)
    if within_pair_mask is None:
        if weights['within_rank']:
            raise ValueError('Enabled within ranking requires distinguishability mask')
        valid = torch.zeros_like(expected)
    else:
        if (not isinstance(within_pair_mask, torch.Tensor) or
                within_pair_mask.dtype != torch.bool or
                within_pair_mask.shape != expected.shape):
            raise ValueError('Boolean within mask[B,5,5] required')
        valid = within_pair_mask.to(z.device)
        if bool((valid & ~expected).any()):
            raise ValueError('Within mask must select own positive-negative pairs')
    differences = z[:, :, None] - z[:, None, :]
    counts = valid.sum((1, 2))
    eligible = counts > 0
    within_per_group = (F.softplus(float(within_margin)-differences)*valid).sum((1, 2)) / counts.clamp_min(1)
    within = within_per_group[eligible].mean() if bool(eligible.any()) else zero

    positives = torch.nonzero(mixed, as_tuple=False).flatten()
    negatives = torch.nonzero(~mixed, as_tuple=False).flatten()
    selected_negatives = negatives
    event_score = scores['mixed_event_logit']
    if inter_hard_negative_k is not None and negatives.numel():
        order = torch.argsort(event_score[negatives], descending=True, stable=True)
        selected_negatives = negatives[order[:inter_hard_negative_k]]
    n_inter = positives.numel() * selected_negatives.numel()
    if n_inter:
        inter_differences = event_score[positives, None] - event_score[selected_negatives][None, :]
        inter = F.softplus(float(inter_margin)-inter_differences).mean()
        inter_accuracy = (inter_differences > 0).to(z.dtype).mean()
    else:
        inter = zero
        inter_accuracy = zero

    predicted_mean, target_mean = p.mean(1), y.mean(1)
    predicted_variance = p.var(1, unbiased=False)
    target_variance = y.var(1, unbiased=False)
    mean_mse = (predicted_mean-target_mean).square().mean()
    variance_mse = (predicted_variance-target_variance).square().mean()
    components = dict(trajectory=trajectory, binary_mixed=binary,
                      three_event=event_nll, within_rank=within,
                      inter_rank=inter, stat_mean=mean_mse,
                      stat_variance=variance_mse)
    total = sum((weights[name]*value for name, value in components.items()), zero)
    return dict(loss=total, total_loss=total, trajectory_bce=trajectory,
                group_bce=binary, group_event_nll=event_nll,
                within_rank_loss=within, inter_rank_loss=inter,
                stat_mean_mse=mean_mse, stat_variance_mse=variance_mse,
                weights=weights, mixed_label=mixed.to(z.dtype),
                true_counts=k, predicted_mean=predicted_mean,
                target_mean=target_mean,
                predicted_population_variance=predicted_variance,
                target_population_variance=target_variance,
                within_eligible_groups=int(eligible.sum()),
                within_eligible_pairs=int(counts.sum()),
                within_pairwise_accuracy=((differences > 0) & valid).sum().to(z.dtype)/counts.sum().clamp_min(1),
                inter_positive_groups=positives.numel(),
                inter_negative_groups=negatives.numel(),
                inter_selected_negative_groups=selected_negatives.numel(),
                inter_selected_negative_indices=selected_negatives,
                inter_actual_pairs=n_inter, inter_pairwise_accuracy=inter_accuracy,
                denominators=dict(trajectory=z.numel(), question=z.shape[0],
                                  within_groups=int(eligible.sum()), inter_pairs=n_inter),
                **scores)
