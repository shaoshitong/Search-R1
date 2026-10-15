"""Scalar, in-sample probe diagnostics; never a held-out quality claim."""
import torch
from search_g1.qprobe_model import S_MAX


def binary_mixed_training_metrics(prediction, counts, *, top_k=448):
    """Use P(1 <= K <= 4), with stable input-order ties and random expectation.

    The random baseline is the population expectation for sampling without
    replacement, not a new sampled or held-out evaluation. Top-k is retrospective
    on these labels; it does not prove future policy-generation mixed yield.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError('QPROBE_METRICS_TOP_K_INVALID')
    counts = torch.as_tensor(counts).detach().cpu()
    if counts.ndim != 1 or not len(counts) or counts.dtype == torch.bool or counts.is_floating_point():
        raise ValueError('QPROBE_METRICS_COUNTS_INVALID')
    if not ((counts >= 0) & (counts <= 5)).all():
        raise ValueError('QPROBE_METRICS_COUNTS_INVALID')
    p = prediction['mixed_probability'].detach().cpu().double()
    sigma = prediction['expected_std'].detach().cpu().double()
    if p.shape != counts.shape or sigma.shape != counts.shape or not torch.isfinite(p).all() or not torch.isfinite(sigma).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('QPROBE_METRICS_PREDICTIONS_INVALID')
    truth = (counts >= 1) & (counts <= 4)
    selected = p >= .5
    tp = int((truth & selected).sum()); fp = int((~truth & selected).sum())
    tn = int((~truth & ~selected).sum()); fn = int((truth & ~selected).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    effective_k = min(top_k, len(counts))
    order = torch.argsort(p, descending=True, stable=True)
    hits = int(truth[order[:effective_k]].sum())
    sigma_order = torch.argsort(sigma, descending=True, stable=True)
    sigma_hits = int(truth[sigma_order[:effective_k]].sum())
    random_rate = float(truth.double().mean())
    target = truth.double(); safe_p = p.clamp(1e-12, 1 - 1e-12)
    true_sigma = (counts.double() * (5 - counts.double()) / 20).sqrt()
    sigma_mse = float(((sigma - true_sigma) ** 2).mean())
    return dict(
        metric_scope='same_training_questions_in_sample', binary_target='1<=K<=4',
        binary_threshold=.5, training_groups=len(counts), mixed_groups=int(truth.sum()),
        binary_tp=tp, binary_fp=fp, binary_tn=tn, binary_fn=fn,
        binary_majority_accuracy_baseline=max(random_rate,1-random_rate),
        p_mix_min=float(p.min()), p_mix_max=float(p.max()),
        p_mix_mean=float(p.mean()), p_mix_std=float(p.std(unbiased=False)),
        binary_accuracy=(tp + tn) / len(counts), binary_precision=precision,
        binary_recall=recall, binary_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
        binary_brier=float(((p - target) ** 2).mean()),
        binary_bce=float(-(target * safe_p.log() + (1 - target) * (1 - safe_p).log()).mean()),
        ranking_score='mixed_probability', ranking_tie_policy='stable_training_input_order',
        top_requested=top_k, top_effective=effective_k, top_mixed_hits=hits,
        top_mixed_rate=hits / effective_k, random_mixed_rate_expected=random_rate,
        random_mixed_hits_expected=effective_k * random_rate,
        top_mixed_rate_lift=hits / effective_k - random_rate,
        p_mix_top_mixed_hits=hits, p_mix_top_mixed_rate=hits / effective_k,
        sigma_top_mixed_hits=sigma_hits, sigma_top_mixed_rate=sigma_hits / effective_k,
        sigma_top_mixed_rate_lift=sigma_hits / effective_k - random_rate,
        sigma_mse=sigma_mse, sigma_normalized_mse=sigma_mse / S_MAX ** 2,
        sigma_mae=float((sigma - true_sigma).abs().mean()),
    )
