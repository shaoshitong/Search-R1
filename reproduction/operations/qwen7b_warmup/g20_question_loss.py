"""Explicit authentic G20 -> G5 original-question-weighted probe objective.

Shares the already tested G10 per-view mathematics, not its source validator:
exactly20 independently accepted trajectory IDs/own EM are mandatory. There is
no trajectory duplication, altered G5 label, or nominal G20 conversion of G10.
Source payload/seed/causal acceptance remains the real-bank loader's duty.
"""
from .g20_grouping import validate
from .g10_question_loss import _question_loss,_validate_views


def validate_views(source_groups,views):
    return _validate_views(source_groups,views,source_size=20,source_validator=validate)


def g20_question_loss(
        logits,source_groups,views,*,trajectory_weight=1.,binary_mixed_weight=0.,
        three_event_weight=0.,within_rank_weight=0.,inter_rank_weight=0.,
        stat_mean_weight=0.,stat_variance_weight=0.,distinguishable_pair_mask=None,
        within_margin=1.,inter_margin=1.):
    return _question_loss(logits,source_groups,views,source_size=20,source_validator=validate,
        trajectory_weight=trajectory_weight,binary_mixed_weight=binary_mixed_weight,
        three_event_weight=three_event_weight,within_rank_weight=within_rank_weight,
        inter_rank_weight=inter_rank_weight,stat_mean_weight=stat_mean_weight,
        stat_variance_weight=stat_variance_weight,distinguishable_pair_mask=distinguishable_pair_mask,
        within_margin=within_margin,inter_margin=inter_margin)
