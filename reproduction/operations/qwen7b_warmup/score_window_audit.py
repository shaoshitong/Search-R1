"""Read-only score-drift audit of one frozen detached-cache candidate window.

The caller invokes this on every participating rank. ``score_fn`` owns online
capture/probe evaluation and any all-gather, and returns the complete supplied
window in precisely its supplied order. Its inputs contain identities only,
never EM labels, cached scores, targets or answers. No cache refresh is applied.
"""
from collections import Counter
import hashlib
import json
import math

from .group_topk_losses import precision_at_k_ramp


def _sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _snapshot(entries, current_step):
    result = []
    for entry in entries:
        qid = entry['qid']
        tids = list(entry['trajectory_ids'])
        labels = list(entry['labels'])
        if not isinstance(qid, str) or not tids or len(tids) != len(labels) or len(set(tids)) != len(tids):
            raise ValueError('Own question identity and distinct own trajectory identities required')
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError('Actual selected-subset strict EM binary labels required')
        mixed = 0 < sum(labels) < len(labels)
        if 'true_mixed' in entry and bool(entry['true_mixed']) != mixed:
            raise ValueError('Mixed target must describe the selected own subset, not its parent G10')
        score = float(entry['cached_score'])
        scored_step = entry['scored_step']
        if not math.isfinite(score) or type(scored_step) is not int or not 0 <= scored_step <= current_step:
            raise ValueError('Finite raw logit and actual nonfuture optimizer update counter required')
        if entry.get('score_model', 'raw') != 'raw':
            raise ValueError('This audit compares cached raw scores with current raw scores')
        result.append(dict(qid=qid, trajectory_ids=tids, labels=labels, true_mixed=mixed,
                           cached_score=score, scored_step=scored_step,
                           scored_batch_attempt=entry.get('scored_batch_attempt')))
    return result


def _top_indices(scores, k):
    # The reference regret function uses supplied vector order to break ties.
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:k]


def audit_score_window(entries, *, current_step, score_fn, k, gamma=1.0,
                       loss_fn=precision_at_k_ramp):
    """Return aggregates only, using the exact current training ramp objective.

    ``entries`` have qid, trajectory_ids, labels, cached_score and scored_step;
    score_model must be raw. ``score_fn`` receives fresh identity-only dicts
    {qid, trajectory_ids, M}. It must be collectively called on all ranks when
    it performs collectives, and must return one globally ordered logit vector.
    It should use the current raw model's ordinary inference path. The auditor
    wraps it in torch.no_grad, does not change train/eval mode, never steps an
    optimizer and never replaces cached scores. Caller-owned mode/buffer policy
    and full-window rank gathering stay the caller's responsibility.

    With b=min(K,totalMixed), the training objective upper-bounds b/K-P@K.
    Each TopK sum and the label-optimal oracle is K-Lipschitz in supnorm, so
    L(fresh)<=L(cache)+2*max_abs_score_drift/gamma, including when totalMixed<K.
    This is a finite-window certificate, not a guarantee about a future pool.
    """
    import torch
    if type(current_step) is not int or current_step < 0:
        raise ValueError('Actual nonnegative optimizer update counter required')
    if not math.isfinite(float(gamma)) or gamma <= 0:
        raise ValueError('Positive finite logit margin required')
    if loss_fn is not precision_at_k_ramp:
        raise ValueError('Use the exact imported group_topk_losses.precision_at_k_ramp')
    snapshot = _snapshot(entries, current_step)
    n = len(snapshot)
    if n == 0:
        return dict(status='EMPTY_CACHE_WINDOW_NO_FORWARD', window_entries=0, k=0,
                    cache_refreshed=False, final_pool_read=False)
    if type(k) is not int or not 1 <= k <= n:
        raise ValueError('Require1<=K<=complete supplied window size')
    identities = [dict(qid=row['qid'], trajectory_ids=list(row['trajectory_ids']),
                       M=len(row['trajectory_ids'])) for row in snapshot]
    cached_scores = [row['cached_score'] for row in snapshot]
    targets = [int(row['true_mixed']) for row in snapshot]
    before = _sha(snapshot)
    with torch.no_grad():
        fresh = score_fn(identities)
        if torch.is_tensor(fresh):
            fresh_values = fresh.detach().reshape(-1).cpu().tolist()
        else:
            fresh_values = list(fresh)
        fresh_scores = [float(score) for score in fresh_values]
        if len(fresh_scores) != n or any(not math.isfinite(score) for score in fresh_scores):
            raise ValueError('One finite current raw logit per matched supplied window entry required')
        # Only small CPU logit/label vectors enter the audit loss; no model tensors are copied.
        cached_tensor = torch.tensor(cached_scores, dtype=torch.float64, device='cpu')
        fresh_tensor = torch.tensor(fresh_scores, dtype=torch.float64, device='cpu')
        label_tensor = torch.tensor(targets, dtype=torch.float64, device='cpu')
        cached_L = float(loss_fn(cached_tensor, label_tensor, k, margin=gamma).item())
        fresh_L = float(loss_fn(fresh_tensor, label_tensor, k, margin=gamma).item())
    if _sha(_snapshot(entries, current_step)) != before:
        raise RuntimeError('Caller-owned cache changed during the frozen-window audit')
    if not math.isfinite(cached_L) or not math.isfinite(fresh_L):
        raise ValueError('Finite exact training loss values required')
    delta = [abs(a-b) for a,b in zip(cached_scores, fresh_scores)]
    max_delta = max(delta)
    mean_delta = math.fsum(delta)/n
    cached_top = _top_indices(cached_scores, k)
    fresh_top = _top_indices(fresh_scores, k)
    cached_hits = sum(targets[index] for index in cached_top)
    fresh_hits = sum(targets[index] for index in fresh_top)
    oracle_hits = min(k, sum(targets))
    oracle_precision = oracle_hits/k
    drift_bound = 2*max_delta/gamma
    fresh_bound = cached_L+drift_bound
    fresh_regret = (oracle_hits-fresh_hits)/k
    ages = [current_step-row['scored_step'] for row in snapshot]
    attempts = [row['scored_batch_attempt'] for row in snapshot if row['scored_batch_attempt'] is not None]
    qids = [row['qid'] for row in snapshot]
    view_hashes = [_sha(identity) for identity in identities]
    tolerance = 1e-10*(1+abs(cached_L)+abs(fresh_L)+drift_bound)
    return dict(status='FROZEN_RAW_SCORE_WINDOW_RECOMPUTED_AGGREGATES_ONLY',
        window_entries=n, k=k, current_actual_optimizer_updates=current_step,
        score_model='raw', score_units='unbounded_raw_logit', gamma=float(gamma),
        max_abs_score_drift=max_delta, mean_abs_score_drift=mean_delta,
        delta_max=max_delta, delta_mean=mean_delta,
        cached_topk_Mixed=cached_hits, fresh_topk_Mixed=fresh_hits,
        cached_P_at_K=cached_hits/k, fresh_P_at_K=fresh_hits/k,
        old_fresh_TopK_overlap_count=len(set(cached_top)&set(fresh_top)),
        old_fresh_TopK_overlap_fraction=len(set(cached_top)&set(fresh_top))/k,
        TopK_tie_policy='descending_logit_then_supplied_window_index',
        total_Mixed=sum(targets), total_nonMixed=n-sum(targets),
        oracle_Mixed=oracle_hits, oracle_precision=oracle_precision,
        cached_attainable_regret=(oracle_hits-cached_hits)/k,
        fresh_attainable_regret=fresh_regret,
        cached_L=cached_L, cached_bound=cached_L, fresh_L=fresh_L,
        fresh_bound=fresh_bound, two_delta_over_gamma=drift_bound,
        fresh_bound_Lcache_plus_2delta_over_gamma=fresh_bound,
        fresh_P_at_K_lower_bound=max(0.0, oracle_precision-fresh_bound),
        fresh_attainable_regret_upper_bound=min(oracle_precision, fresh_bound),
        loss_absolute_drift=abs(fresh_L-cached_L),
        loss_drift_bound_holds=abs(fresh_L-cached_L) <= drift_bound+tolerance,
        cached_ramp_bounds_measured_regret=(oracle_hits-cached_hits)/k <= cached_L+tolerance,
        fresh_ramp_bounds_measured_regret=fresh_regret <= fresh_L+tolerance,
        fresh_regret_bound_holds=fresh_regret <= fresh_bound+tolerance,
        loss_definition='exact group_topk_losses.precision_at_k_ramp; label-optimal C_star oracle retained',
        certificate_scope='this exact matched finite window only; not a future or final pool guarantee',
        age_actual_updates_histogram={str(age):count for age,count in sorted(Counter(ages).items())},
        age_actual_updates_min=min(ages), age_actual_updates_max=max(ages),
        age_actual_updates_mean=math.fsum(ages)/n,
        scored_batch_attempt_min=min(attempts) if attempts else None,
        scored_batch_attempt_max=max(attempts) if attempts else None,
        ordered_view_coverage_sha256=_sha(identities),
        unordered_unique_view_coverage_sha256=_sha(sorted(set(view_hashes))),
        ordered_qid_coverage_sha256=_sha(qids), unique_qid_coverage_sha256=_sha(sorted(set(qids))),
        unique_qids=len(set(qids)), duplicate_qid_entries=n-len(set(qids)),
        unique_views=len(set(view_hashes)), duplicate_view_entries=n-len(set(view_hashes)),
        frozen_cache_metadata_sha256=before, callbacks_ran_no_grad=True,
        labels_or_cached_scores_passed_to_model_callback=False, cache_refreshed=False,
        auditor_updates_parameters=False, model_mode_changed_by_auditor=False, final_pool_read=False)
