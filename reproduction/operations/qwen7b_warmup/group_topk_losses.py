"""Group-level Mixed ranking losses; no trajectory correctness objective.

The finite-pool bounds concern the score vector supplied to these functions.
A minibatch is not the future2048-question candidate pool. Scores are logits,
not sigmoid probabilities; a finite unit logit margin can be satisfied.
"""


def reference_regret(scores, labels, k):
    selected = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return min(k, sum(labels)) - sum(labels[i] for i in selected)


def reference_topk_ramp(scores, labels, k, margin=1.0):
    b = min(k, sum(labels))
    positive = sorted((s for s, y in zip(scores, labels) if y), reverse=True)
    negative = sorted((s for s, y in zip(scores, labels) if not y), reverse=True)
    oracle = sum(positive[:b]) + sum(negative[:k-b])
    augmented = sorted((s-margin*y for s, y in zip(scores, labels)), reverse=True)
    return max(0.0, margin*b + sum(augmented[:k]) - oracle)/(margin*k)


def reference_pairwise_hinge_sum(scores, labels, margin=1.0):
    return sum(max(0.0, 1.0-(scores[i]-scores[j])/margin)
               for i, y in enumerate(labels) if y
               for j, z in enumerate(labels) if not z)


def precision_at_k_ramp(logits, labels, k, margin=1.0):
    """Upper-bound empirical lost Mixed count/K, including fewer thanK positives.

    b=min(K,M), C*=best score sum over label-optimal sizeK selections.
    L=relu(gamma*b+TopK(s-gamma*y).sum()-C*)/(gamma*K).
    This is a generally nonconvex, piecewise-linear loss. ReLU selects zero
    gradient at an exact zero. No BCE or variance anchor is added here.
    """
    import torch
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError('One complete score/label vector is required')
    if not 1 <= k <= logits.numel() or margin <= 0:
        raise ValueError('Require1<=K<=N and positive logit margin')
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Actual binary group Mixed labels required')
    scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    y = labels.to(device=scores.device, dtype=scores.dtype)
    pos = scores[y == 1]
    neg = scores[y == 0]
    b = min(k, pos.numel())
    oracle = scores.sum()*0.0
    if b:
        oracle = oracle + torch.topk(pos, b).values.sum()
    if k-b:
        oracle = oracle + torch.topk(neg, k-b).values.sum()
    augmented = torch.topk(scores-margin*y, k).values.sum()
    return torch.relu(margin*b + augmented-oracle)/(margin*k)


def group_pairwise_hinge(logits, labels, margin=1.0):
    """Mean over all positive/negative QUESTION-GROUP pairs in this vector."""
    import torch
    pos, neg = logits[labels == 1], logits[labels == 0]
    if not pos.numel() or not neg.numel():
        return logits.sum()*0.0
    return torch.relu(1.0-(pos[:, None]-neg[None, :])/margin).mean()


def squared_epsilon_regression(prediction, target, epsilon):
    """Optional dead-zone regression; unused in the first pure ramp comparison."""
    import torch
    if epsilon < 0:
        raise ValueError('epsilon must be nonnegative')
    return torch.relu((prediction-target).abs()-epsilon).square().mean()
