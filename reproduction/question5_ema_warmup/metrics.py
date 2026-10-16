"""Streaming same-head diagnostics; G5 mixed BA and trajectory BA stay distinct."""

import torch
from .loss import joint_loss


class GroupMetrics:
    # Counts are TP, FP, FN, TN separately for complete groups and trajectories.
    def __init__(self):
        self.groups = 0
        self.trajectory_bce_sum = 0.0
        self.group_bce_sum = 0.0
        self.mixed_probability_sum = 0.0
        self.group_confusion = [0, 0, 0, 0]
        self.trajectory_confusion = [0, 0, 0, 0]

    @staticmethod
    def _confusion(predicted, actual):
        return [int((predicted & actual).sum()), int((predicted & ~actual).sum()),
                int((~predicted & actual).sum()), int((~predicted & ~actual).sum())]

    @torch.no_grad()
    def add(self, logits, labels):
        values = joint_loss(logits.detach(), labels.detach())
        if not bool(torch.isfinite(values["loss"])):
            raise FloatingPointError("Nonfinite diagnostic logits/loss")
        n = logits.shape[0]
        self.groups += n
        self.trajectory_bce_sum += float(values["trajectory_bce"]) * n
        self.group_bce_sum += float(values["group_bce"]) * n
        self.mixed_probability_sum += float(values["mixed_probability"].double().sum())
        group = self._confusion(values["mixed_event_logits"] >= 0, values["mixed_label"].bool())
        trajectory = self._confusion(logits.detach() >= 0, labels.detach().bool())
        self.group_confusion = [a + b for a, b in zip(self.group_confusion, group)]
        self.trajectory_confusion = [a + b for a, b in zip(self.trajectory_confusion, trajectory)]

    def merge(self, other):
        self.groups += other.groups
        self.trajectory_bce_sum += other.trajectory_bce_sum
        self.group_bce_sum += other.group_bce_sum
        self.mixed_probability_sum += other.mixed_probability_sum
        self.group_confusion = [a + b for a, b in zip(self.group_confusion, other.group_confusion)]
        self.trajectory_confusion = [a + b for a, b in zip(self.trajectory_confusion, other.trajectory_confusion)]

    def reduce_distributed(self, device):
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return self
        values = torch.tensor([self.groups, self.trajectory_bce_sum, self.group_bce_sum,
                               self.mixed_probability_sum] + self.group_confusion
                              + self.trajectory_confusion, dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values = values.cpu().tolist()
        self.groups = int(values[0])
        self.trajectory_bce_sum, self.group_bce_sum, self.mixed_probability_sum = values[1:4]
        self.group_confusion = [int(value) for value in values[4:8]]
        self.trajectory_confusion = [int(value) for value in values[8:12]]
        return self

    @staticmethod
    def _summary(confusion):
        tp, fp, fn, tn = confusion
        positive = tp + fn
        negative = tn + fp
        tpr = tp / positive if positive else None
        tnr = tn / negative if negative else None
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, positive=positive, negative=negative,
                    accuracy=(tp + tn) / (positive + negative) if positive + negative else None,
                    balanced_accuracy=(tpr + tnr) / 2 if tpr is not None and tnr is not None else None)

    def result(self, *, weight_kind, semantics="same_fixed_head_eval_no_grad"):
        if weight_kind not in ("raw", "ema0.99"):
            raise ValueError("Diagnostic weight identity must be raw or ema0.99")
        n = self.groups
        if sum(self.group_confusion) != n or sum(self.trajectory_confusion) != n * 5:
            raise RuntimeError("Diagnostic complete-G5 count differs")
        return dict(weight_kind=weight_kind, metric_semantics=semantics, groups=n, trajectories=n * 5,
                    trajectory_bce=self.trajectory_bce_sum / n if n else None,
                    group_bce=self.group_bce_sum / n if n else None,
                    total_loss=(self.trajectory_bce_sum + self.group_bce_sum) / n if n else None,
                    mean_mixed_probability=self.mixed_probability_sum / n if n else None,
                    mixed_group=self._summary(self.group_confusion),
                    trajectory_correctness=self._summary(self.trajectory_confusion),
                    mixed_formula="1-prod(p_j)-prod(1-p_j)", true_mixed="0<K<5", threshold=0.5)
