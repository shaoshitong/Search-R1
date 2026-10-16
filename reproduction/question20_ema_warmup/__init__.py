"""Fresh question+policy20 probe: joint G5 supervision and post-Adam EMA.

Dataset construction, scientific identities, remote admission and launch are
deliberately outside this isolated core package. Importing it starts no jobs.
"""

from .loss import joint_loss, mixed_event_logits
from .ema import ParameterEMA
from .metrics import GroupMetrics
from .training import (TrainCounters, GroupStep, build_epoch_plan,
                       train_update, evaluate_groups)

__all__ = ["joint_loss", "mixed_event_logits", "ParameterEMA", "GroupMetrics",
           "TrainCounters", "GroupStep", "build_epoch_plan", "train_update",
           "evaluate_groups"]
