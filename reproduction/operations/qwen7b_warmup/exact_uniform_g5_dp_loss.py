"""Bounded, exact uniform G5 loss on authentic G10/G20; three-state DP.

For Q original questions the objective is 1/Q times the sum of the source-G
trajectory BCE and the mean three-event NLL over all C(G,5) distinct subsets.
This preserves the old 32-pattern mathematical objective, not its arithmetic
execution order. Each chunk is differentiated then discarded. Only one [Q,G]
first-order VJP survives; no all-15504-view autograd graph or pattern expansion.
The published old implementation and native runners are deliberately intact.
"""
import itertools
import torch
import torch.nn.functional as F
from .g5_event_scores_dp import group_event_scores_dp
from .exact_uniform_g5_loss import (choose, validate_source, weights,
    mixed_combinatorial_mass, _ExactFirstOrderVJP)


def exact_uniform_g5_dp_loss(logits, source_groups, *, trajectory_weight=1.,
        three_event_weight=1., combination_chunk=256, progress_callback=None):
    """Exact finite objective and first gradient, with bounded vectorized chunks.

    Floating-point reduction order may differ with chunk size. ``loss`` is the
    custom first-order VJP; component values and confusion counts are detached.
    Inference scores are the same label-blind five-Bernoulli event probability.
    Labels only select which of the three event log probabilities is the NLL.
    """
    g = validate_source(source_groups)
    a, b = weights(trajectory_weight, three_event_weight)
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or tuple(logits.shape) != (len(source_groups), g)
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Finite authentic own logits[Q,G] required')
    if type(combination_chunk) is not int or not 1 <= combination_chunk <= 4096:
        raise ValueError('Explicit bounded combination_chunk1..4096 required')
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError('Progress callback must be callable or None')
    q, total = len(source_groups), choose(g, 5)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    z = logits.detach().to(dtype).requires_grad_(True)
    labels = torch.tensor([x['labels'] for x in source_groups],
                          dtype=dtype, device=z.device)
    with torch.no_grad():
        trajectory = F.binary_cross_entropy_with_logits(z, labels)
        gradient = a * (z.sigmoid() - labels) / (q * g)
        event = torch.zeros((), dtype=dtype, device=z.device)
    count_tensor = torch.zeros(4, dtype=torch.long, device=z.device)
    combinations = itertools.combinations(range(g), 5)
    seen, maximum = 0, 0
    while True:
        chunk = list(itertools.islice(combinations, combination_chunk))
        if not chunk:
            break
        n = len(chunk)
        seen += n
        maximum = max(maximum, q * n)
        index = torch.tensor(chunk, dtype=torch.long, device=z.device)
        with torch.enable_grad():
            own = z[:, index].reshape(q * n, 5)
            k = labels[:, index].reshape(q * n, 5).sum(1)
            scores = group_event_scores_dp(own)
            nll = -torch.where(k == 0, scores['log_allwrong'],
                torch.where(k == 5, scores['log_allcorrect'], scores['log_mixed'])).sum() / (q * total)
            if b:
                chunk_gradient = torch.autograd.grad(nll, z,
                    create_graph=False, retain_graph=False)[0]
                with torch.no_grad():
                    gradient.add_(chunk_gradient, alpha=b)
                del chunk_gradient
            with torch.no_grad():
                event.add_(nll.detach())
                truth = (k > 0) & (k < 5)
                pred = scores['mixed_probability'] >= .5
                count_tensor.add_(torch.stack([(pred & truth).sum(),
                    (~pred & truth).sum(), (pred & ~truth).sum(), (~pred & ~truth).sum()]))
        del own, k, scores, nll, index, chunk
        if progress_callback is not None:
            progress_callback(dict(combinations_done=seen, combinations_per_question=total,
                original_questions=q, event_views_done=q * seen,
                maximum_resident_G5_rows=maximum))
    if seen != total:
        raise AssertionError('Incomplete exact all-combinations coverage')
    with torch.no_grad():
        value = a * trajectory + b * event
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
            raise ValueError('Nonfinite exact DP value or gradient')
    loss = _ExactFirstOrderVJP.apply(logits, value, gradient.to(logits.dtype))
    counts = dict(zip(('tp', 'fn', 'fp', 'tn'), count_tensor.cpu().tolist()))
    capacities = [mixed_combinatorial_mass(x['labels']) for x in source_groups]
    exact_mixed = sum(x['Mixed'] for x in capacities)
    if counts['tp'] + counts['fn'] != exact_mixed:
        raise AssertionError('Enumerated Mixed labels differ from source combinatorics')
    return dict(loss=loss, total_loss=loss,
        components=dict(trajectory=trajectory.detach(), three_event=event.detach()),
        weights=dict(trajectory=a, three_event=b), original_questions=q,
        source_group_size=g, virtual_views=q * total, combinations_per_question=total,
        combination_chunk=combination_chunk, maximum_resident_G5_rows=maximum,
        maximum_resident_pattern_rows=0, maximum_resident_event_states=3 * maximum,
        event_evaluation='three_state_log_space_DP_no_32_pattern_expansion',
        first_order_only=True, retained_loss_gradient_elements=q * g,
        all_combinations_exactly_once=True,
        per_original_question_total_weight_exactly_one=True, view_weight='1/%d' % total,
        weighted_Mixed_fraction=exact_mixed / (q * total),
        source_combinatorial_class_mass=capacities,
        confusion_integer_all_views=counts,
        question_weighted_confusion={k: v / total for k, v in counts.items()},
        denominators=dict(question=q, source_trajectory=q * g, event_view=q * total),
        trajectory_reduction='uniform all-G5 trajectory BCE equals mean over own source G',
        event_reduction='1/Q sum_question 1/C(G,5) sum_all_real_G5 three-event NLL',
        inter_view_pairs_allocated=0, per_view_python_identity_metadata_allocated=0,
        native_model_forwards_performed_by_loss=0, actor_updates=0, supports_G20=g == 20)
