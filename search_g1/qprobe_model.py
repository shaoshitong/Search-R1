"""CPU-only online question probe. Gold labels never enter ``predict``.

The six-class distribution describes K, the number of strictly correct answers
in a complete group of five. It induces E[mean] and E[sample std], rather than
incorrectly using sqrt(E[mean] * (1 - E[mean])) for within-group diversity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = 1
GROUP_SIZE = 5
S_MAX = math.sqrt(0.3)
LEGACY_LOSS_MODE = 'count6_moments_v1'
SIGMA3_LOSS_MODE = 'sigma3_std_aux6_v1'


@dataclass(frozen=True)
class QProbeConfig:
    mean_loss_weight: float = 0.1
    std_loss_weight: float = 0.1
    head_learning_rate: float = 3e-4
    head_batch_size: int = 256
    head_epochs_per_outer: int = 2
    replay_window_outer: int = 16
    replay_half_life_outer: float = 4.0
    bootstrap_max_groups_per_outer: int = 512
    hidden_size: int = 64
    seed: int = 1729
    # Missing fields in historical checkpoints retain the original CE6 loss.
    loss_mode: str = LEGACY_LOSS_MODE
    count_aux_loss_weight: float = 0.1
    bootstrap_head_epochs: int | None = None

    def __post_init__(self):
        for name in ('head_batch_size', 'head_epochs_per_outer',
                     'replay_window_outer', 'bootstrap_max_groups_per_outer', 'hidden_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'QPROBE_INVALID_CONFIG: {name} must be a positive integer')
        if (self.bootstrap_head_epochs is not None and
            (type(self.bootstrap_head_epochs) is not int or self.bootstrap_head_epochs <= 0)):
            raise ValueError('QPROBE_INVALID_CONFIG: bootstrap_head_epochs must be None or a positive integer')
        for name in ('head_learning_rate', 'replay_half_life_outer'):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'QPROBE_INVALID_CONFIG: {name} must be positive and finite')
        for name in ('mean_loss_weight', 'std_loss_weight', 'count_aux_loss_weight'):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'QPROBE_INVALID_CONFIG: {name} must be nonnegative and finite')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('QPROBE_INVALID_CONFIG: seed must be a nonnegative int64')
        if self.loss_mode not in (LEGACY_LOSS_MODE, SIGMA3_LOSS_MODE):
            raise ValueError('QPROBE_INVALID_CONFIG: unsupported loss_mode')
        if self.loss_mode == SIGMA3_LOSS_MODE and self.mean_loss_weight != 0:
            raise ValueError('QPROBE_INVALID_CONFIG: sigma3 loss requires mean_loss_weight=0')

    @property
    def loss_identity(self):
        std_term = (f'{self.std_loss_weight:g}*'
                    '((E[sample_std]-sample_std(K))/sqrt(0.3))^2')
        if self.loss_mode == SIGMA3_LOSS_MODE:
            return (f'{self.loss_mode}:CE3(K->min(K,5-K))+{std_term}'
                    f'+{self.count_aux_loss_weight:g}*CE6')
        return f'{self.loss_mode}:CE6+{self.mean_loss_weight:g}*mean_MSE+{std_term}'

    @classmethod
    def from_mapping(cls, values: Mapping | None):
        # Runtime selector/feature options share the config namespace. Only the
        # explicitly named head options are consumed here; no actor knobs are read.
        values = {} if values is None else dict(values)
        if 'head_hidden_size' in values:
            if 'hidden_size' in values and values['hidden_size'] != values['head_hidden_size']:
                raise ValueError('QPROBE_CONFLICTING_HEAD_HIDDEN_SIZE')
            values['hidden_size'] = values['head_hidden_size']
        allowed = {x.name for x in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in allowed})


class CountMomentProbe(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int = 64):
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1:
            raise ValueError('QPROBE_INVALID_FEATURE_DIM')
        self.feature_dim = feature_dim
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_size),
                                 nn.GELU(), nn.Linear(hidden_size, 6))
        # No learned labels => no invented question ranking. Uniform K yields
        # equal predictions; selector's independent tie RNG supplies exploration.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        k = torch.arange(6, dtype=torch.float32)
        self.register_buffer('means', k / GROUP_SIZE)
        self.register_buffer('variances', k * (GROUP_SIZE - k) / 20)
        self.register_buffer('stds', self.variances.sqrt())

    def forward(self, features: torch.Tensor):
        features = validate_features(features, self.feature_dim)
        logits = self.net(features)
        if not torch.isfinite(logits).all():
            raise ValueError('QPROBE_NONFINITE_LOGITS')
        return self.moments(logits)

    def moments(self, logits: torch.Tensor):
        if logits.ndim != 2 or logits.shape[1] != 6 or not torch.isfinite(logits).all():
            raise ValueError('QPROBE_INVALID_LOGITS')
        p = logits.float().softmax(-1)
        return dict(logits=logits, probabilities=p, mean=p @ self.means,
                    expected_std=p @ self.stds, expected_variance=p @ self.variances,
                    mixed_probability=p[:, 1:5].sum(-1))


def validate_features(features, feature_dim):
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError('QPROBE_FEATURE_SHAPE_MISMATCH')
    if features.shape[0] == 0 or not torch.isfinite(features).all():
        raise ValueError('QPROBE_EMPTY_OR_NONFINITE_FEATURES')
    return features.detach().to(device='cpu', dtype=torch.float32)


def count_moment_group_loss(prediction, counts, config: QProbeConfig):
    """The single per-group objective used for both Adam and loss reporting."""
    if not isinstance(counts, torch.Tensor) or counts.dtype != torch.long or counts.ndim != 1:
        raise ValueError('QPROBE_COUNTS_MUST_BE_INT64_VECTOR')
    logits = prediction['logits'].float()
    if logits.ndim != 2 or logits.shape[1] != 6 or not torch.isfinite(logits).all():
        raise ValueError('QPROBE_INVALID_LOGITS')
    if len(counts) != len(logits) or not len(counts) or bool(((counts < 0) | (counts > 5)).any()):
        raise ValueError('QPROBE_INVALID_COUNTS')
    stds = (counts.float() * (5 - counts.float()) / 20).sqrt()
    ce = F.cross_entropy(logits, counts, reduction='none')
    std_mse = ((prediction['expected_std'] - stds) / S_MAX).square()
    std_contribution = config.std_loss_weight * std_mse
    if config.loss_mode == SIGMA3_LOSS_MODE:
        # Softmax over these paired logsumexp logits is exactly
        # q=(p0+p5, p1+p4, p2+p3), including extreme logits where p underflows.
        sigma_logits = torch.stack([torch.logsumexp(logits[:, [k, 5-k]], dim=-1)
                                    for k in range(3)], dim=-1)
        sigma_classes = torch.minimum(counts, 5-counts)
        sigma_ce = F.cross_entropy(sigma_logits, sigma_classes, reduction='none')
        count_contribution = config.count_aux_loss_weight * ce
        loss = sigma_ce + std_contribution + count_contribution
        parts = dict(sigma3_ce=sigma_ce, count_ce=ce, std_mse=std_mse,
                     normalized_sample_std_mse=std_mse, sigma3_ce_contribution=sigma_ce,
                     std_mse_contribution=std_contribution, count_ce_contribution=count_contribution)
    else:
        mean_mse = (prediction['mean'] - counts.float()/5).square()
        mean_contribution = config.mean_loss_weight * mean_mse
        loss = ce + mean_contribution + std_contribution
        parts = dict(count_ce=ce, mean_mse=mean_mse, std_mse=std_mse,
                     count_ce_contribution=ce, mean_mse_contribution=mean_contribution,
                     std_mse_contribution=std_contribution)
    if not torch.isfinite(loss).all():
        raise ValueError('QPROBE_NONFINITE_TRAINING_LOSS')
    return loss, parts


def count_moment_loss(prediction, counts, config: QProbeConfig, weights=None):
    loss, parts = count_moment_group_loss(prediction, counts, config)
    if weights is None:
        weights = torch.ones_like(loss)
    if (not isinstance(weights, torch.Tensor) or weights.shape != loss.shape or
        not torch.isfinite(weights).all() or bool((weights <= 0).any())):
        raise ValueError('QPROBE_INVALID_REPLAY_WEIGHTS')
    weighted = lambda x: (x * weights).sum() / weights.sum()
    return weighted(loss), {key: weighted(value) for key, value in parts.items()}


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value, key):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('QPROBE_MISSING_LABEL_PROVENANCE: ' + key)
    return value


def validate_label(label, *, outer, run_id, attempt_id, policy_snapshot_id,
                   sampling_contract_sha, strict_scoring_contract_sha, retriever_version):
    if not isinstance(label, dict):
        raise ValueError('QPROBE_LABEL_MUST_BE_DICT')
    value = dict(label)
    if value.get('complete') is not True:
        raise ValueError('QPROBE_INCOMPLETE_GROUP_IS_NOT_EM_ZERO')
    em = value.get('strict_em')
    if not isinstance(em, (list, tuple)) or len(em) != 5 or any(type(x) is not int or x not in (0, 1) for x in em):
        raise ValueError('QPROBE_REQUIRES_FIVE_STRICT_EM_LABELS')
    for key in ('qid', 'question_hash', 'selection_source'):
        _required_text(value.get(key), key)
    digest = value['question_hash']
    if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
        raise ValueError('QPROBE_INVALID_QUESTION_HASH')
    expected = dict(run_id=run_id, attempt_id=attempt_id, source_outer=outer,
                    policy_snapshot_id=policy_snapshot_id, sampling_contract_sha=sampling_contract_sha,
                    strict_scoring_contract_sha=strict_scoring_contract_sha, retriever_version=retriever_version)
    for key, expected_value in expected.items():
        if key != 'source_outer':
            _required_text(expected_value, key)
        if key in value and value[key] != expected_value:
            raise ValueError('QPROBE_LABEL_PROVENANCE_MISMATCH: ' + key)
        value[key] = expected_value
    if 'correct_count' in value and value['correct_count'] != sum(em):
        raise ValueError('QPROBE_LABEL_COUNT_MISMATCH')
    value['strict_em'] = list(em)
    value['correct_count'] = sum(em)
    return value


class OnlineQProbe:
    """One head fit per successful outer; its own RNG cannot perturb actor RNG.

    Caller publishes only after the complete actor outer succeeds. Labels refer
    to its *sampling* actor; save's actor_snapshot_id binds the *updated* actor.
    Failed fit rolls the head, optimizer, RNG and replay back atomically in memory.
    """
    def __init__(self, feature_dim: int, config=None, feature_spec_sha: str = ''):
        self.config = config if isinstance(config, QProbeConfig) else QProbeConfig.from_mapping(config)
        self.feature_dim = feature_dim
        self.feature_spec_sha = feature_spec_sha
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.model = CountMomentProbe(feature_dim, self.config.hidden_size).cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.head_learning_rate)
        self.rng = torch.Generator(device='cpu').manual_seed(self.config.seed + 1)
        self.replay_labels = []
        self.replay_features = torch.empty((0, feature_dim), dtype=torch.float32)
        self.completed_outer = 0
        self.update_count = 0
        self.completed_attempts = {}
        self.protocol = None
        self.actor_snapshot_id = None

    @property
    def version(self):
        digest = hashlib.sha256(self.feature_spec_sha.encode())
        for key, value in sorted(self.model.state_dict().items()):
            digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        return f'qprobe:{self.completed_outer}:{self.update_count}:{digest.hexdigest()[:16]}'

    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            return {k: v.detach().clone() for k, v in self.model(features).items()}

    def _adam_steps(self):
        steps = [int(state['step'].item()) if isinstance(state['step'], torch.Tensor) else int(state['step'])
                 for state in self.optimizer.state.values() if 'step' in state]
        if steps and len(set(steps)) != 1:
            raise ValueError('QPROBE_INCONSISTENT_HEAD_ADAM_STEPS')
        return steps[0] if steps else 0

    def _training_summary(self, prediction, counts, weights, *, epoch, epochs, updates):
        """Full training replay statistics, using the same per-group objective."""
        objective, parts = count_moment_loss(prediction, counts, self.config, weights)
        size = len(counts)
        probabilities = prediction['probabilities']
        sigma_probabilities = torch.stack([probabilities[:, k] + probabilities[:, 5-k]
                                           for k in range(3)], dim=-1)
        sigma_classes = torch.minimum(counts, 5-counts)
        true_mixed = (counts > 0) & (counts < 5)
        predicted_mixed = prediction['mixed_probability'] >= .5
        tp = int((true_mixed & predicted_mixed).sum())
        tn = int((~true_mixed & ~predicted_mixed).sum())
        fp = int((~true_mixed & predicted_mixed).sum())
        fn = int((true_mixed & ~predicted_mixed).sum())
        ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.
        recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
        recalls = ([recall] if tp + fn else []) + ([specificity] if tn + fp else [])
        sigma_counts = torch.bincount(sigma_classes, minlength=3)
        count_counts = torch.bincount(counts, minlength=6)
        sigma_predicted, count_predicted = sigma_probabilities.argmax(-1), probabilities.argmax(-1)
        class_accuracy = lambda truth, predicted, classes: [
            ratio(int(((truth == k) & (predicted == k)).sum()), int((truth == k).sum()))
            for k in range(classes)]
        target_std = (counts.float() * (5-counts.float()) / 20).sqrt()
        return dict(dataset_scope='training_replay_not_independent_holdout', replay_groups=size,
            epoch=epoch, fit_epochs=epochs, head_batch_size=self.config.head_batch_size,
            head_learning_rate=float(self.optimizer.param_groups[0]['lr']),
            head_optimizer_updates=updates, head_adam_steps=self._adam_steps(),
            replay_weighted_loss=float(objective), total_loss=float(objective),
            **{key: float(value) for key, value in parts.items()},
            binary_mixed=dict(threshold=.5, true_positive=tp, true_negative=tn,
                false_positive=fp, false_negative=fn, accuracy=ratio(tp + tn, size),
                balanced_accuracy=sum(recalls) / len(recalls), precision=ratio(tp, tp + fp),
                recall=recall, specificity=specificity, f1=ratio(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy_definition='mean recall over classes present in training replay',
                zero_division=0.),
            true_mixed_fraction=float(true_mixed.float().mean()),
            predicted_mixed_fraction=float(predicted_mixed.float().mean()),
            sigma3_accuracy=float((sigma_predicted == sigma_classes).float().mean()),
            count6_accuracy=float((count_predicted == counts).float().mean()),
            sigma3_majority_baseline_accuracy=float(sigma_counts.max()) / size,
            count6_majority_baseline_accuracy=float(count_counts.max()) / size,
            sigma3_majority_class=int(sigma_counts.argmax()), count6_majority_class=int(count_counts.argmax()),
            sigma3_label_counts=sigma_counts.tolist(), count6_label_counts=count_counts.tolist(),
            sigma3_per_class_accuracy=class_accuracy(sigma_classes, sigma_predicted, 3),
            count6_per_class_accuracy=class_accuracy(counts, count_predicted, 6),
            sigma_mae=float((prediction['expected_std'] - target_std).abs().mean()))

    def _fit(self, outer, *, epochs=None, log_warmup=False):
        if not self.replay_labels:
            raise ValueError('QPROBE_CANNOT_FIT_WITHOUT_REAL_LABELS')
        epochs = self.config.head_epochs_per_outer if epochs is None else epochs
        if type(epochs) is not int or epochs <= 0:
            raise ValueError('QPROBE_INVALID_FIT_EPOCHS')
        counts = torch.tensor([x['correct_count'] for x in self.replay_labels], dtype=torch.long)
        weights = torch.tensor([2 ** (-(outer - x['source_outer']) / self.config.replay_half_life_outer)
                                for x in self.replay_labels], dtype=torch.float32)
        # Use a global denominator. Per-mini normalization cancels recency
        # weights in homogeneous mini-batches and changes the intended objective.
        weights = weights * (len(weights) / weights.sum())
        updates = 0
        adam_before = self._adam_steps()
        epoch_history, minibatch_history = [], []
        self.model.train()

        def summarize(epoch):
            training = self.model.training
            try:
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(self.replay_features)
                    return self._training_summary(prediction, counts, weights,
                        epoch=epoch, epochs=epochs, updates=updates)
            finally:
                self.model.train(training)

        if log_warmup:
            epoch_history.append(summarize(0))
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(counts), generator=self.rng)
            for batch_index, offset in enumerate(range(0, len(order), self.config.head_batch_size)):
                index = order[offset:offset + self.config.head_batch_size]
                prediction = self.model(self.replay_features[index])
                group_loss, group_parts = count_moment_group_loss(prediction, counts[index], self.config)
                # Preserve the global age-weighted estimator. Normalizing only
                # within this mini-batch would cancel homogeneous recency weights.
                loss = (group_loss * weights[index]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('QPROBE_NONFINITE_TRAINING_LOSS')
                minibatch = dict(epoch=epoch, minibatch_index=batch_index, batch_size=len(index),
                    head_update_ordinal=updates + 1, head_adam_steps_before=self._adam_steps(),
                    head_learning_rate=float(self.optimizer.param_groups[0]['lr']),
                    pre_step_weighted_loss=float(loss.detach()),
                    weight_reduction='mean(loss_i * globally_normalized_replay_weight_i)',
                    loss_parts={key: float((value.detach() * weights[index]).mean())
                                for key, value in group_parts.items()})
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise ValueError('QPROBE_NONFINITE_GRADIENT')
                self.optimizer.step(); updates += 1
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    raise ValueError('QPROBE_NONFINITE_PARAMETERS')
                minibatch['head_adam_steps_after'] = self._adam_steps()
                minibatch_history.append(minibatch)
            if log_warmup:
                epoch_history.append(summarize(epoch))
        if log_warmup:
            summary = epoch_history[-1]
            self.model.eval()
        else:
            # Reuse the existing final replay forward; formal fits add no eval forwards.
            pred = self.predict(self.replay_features)
            summary = self._training_summary(pred, counts, weights,
                epoch=epochs, epochs=epochs, updates=updates)
        parts = {key: summary[key] for key in group_parts}
        return dict(replay_groups=len(counts), head_optimizer_updates=updates,
                    replay_weighted_loss=summary['replay_weighted_loss'], loss_mode=self.config.loss_mode,
                    loss_identity=self.config.loss_identity, **parts,
                    fit_epochs=epochs, head_batch_size=self.config.head_batch_size,
                    head_learning_rate=float(self.optimizer.param_groups[0]['lr']),
                    head_adam_steps_before=adam_before, head_adam_steps_after=self._adam_steps(),
                    training_summary=summary, epoch_history=epoch_history, minibatch_history=minibatch_history)

    def update_completed_outer(self, outer: int, labels: Sequence[dict], features: torch.Tensor, *,
                               policy_snapshot_id: str, sampling_contract_sha: str,
                               strict_scoring_contract_sha: str, retriever_version: str,
                               run_id: str, attempt_id: str):
        if type(outer) is not int or outer <= 0:
            raise ValueError('QPROBE_INVALID_COMPLETED_OUTER')
        attempt_key = f'{run_id}:{outer}'
        if attempt_key in self.completed_attempts:
            raise ValueError('QPROBE_OUTER_ALREADY_UPDATED')
        if outer <= self.completed_outer:
            raise ValueError('QPROBE_OLD_OR_FUTURE_HEAD_ACTOR_PAIR')
        features = validate_features(features, self.feature_dim).clone()
        if len(labels) != len(features) or not labels:
            raise ValueError('QPROBE_LABEL_FEATURE_COUNT_MISMATCH')
        valid = [validate_label(x, outer=outer, run_id=run_id, attempt_id=attempt_id,
                                policy_snapshot_id=policy_snapshot_id, sampling_contract_sha=sampling_contract_sha,
                                strict_scoring_contract_sha=strict_scoring_contract_sha, retriever_version=retriever_version)
                 for x in labels]
        if len({x['qid'] for x in valid}) != len(valid):
            raise ValueError('QPROBE_DUPLICATE_QUESTION_WITHIN_OUTER')
        protocol = dict(sampling_contract_sha=sampling_contract_sha,
                        strict_scoring_contract_sha=strict_scoring_contract_sha, retriever_version=retriever_version)
        if self.protocol is not None and self.protocol != protocol:
            raise ValueError('QPROBE_PROTOCOL_CHANGE_REQUIRES_NEW_NAMESPACE')
        before = self.state_dict()
        started = time.monotonic()
        try:
            self.protocol = protocol
            self.replay_labels += valid
            self.replay_features = torch.cat([self.replay_features, features], dim=0)
            keep = [i for i, item in enumerate(self.replay_labels)
                    if item['source_outer'] > outer - self.config.replay_window_outer]
            self.replay_labels = [self.replay_labels[i] for i in keep]
            self.replay_features = self.replay_features[keep].clone()
            result = self._fit(outer)
            self.completed_outer = outer; self.update_count += 1
            self.completed_attempts[attempt_key] = attempt_id
            # A new head is not implicitly bound to the old sampling actor.
            self.actor_snapshot_id = None
            result.update(version=self.version, completed_outer=outer, update_count=self.update_count,
                          elapsed_seconds=time.monotonic()-started, label_policy_snapshot_id=policy_snapshot_id)
            return result
        except Exception:
            self._restore_state(before)
            raise

    def bootstrap(self, labels, features, *, actor_outer: int, actor_snapshot_id: str):
        """Fit complete all-candidate history without pretending it is new-policy truth.

        Prepare CLI additionally verifies pool completeness/lineage. Historical
        groups are uniformly capped per outer before fitting, independent of K.
        """
        if self.update_count or self.replay_labels or self.completed_outer:
            raise ValueError('QPROBE_BOOTSTRAP_ONLY_ON_FRESH_HEAD')
        if type(actor_outer) is not int or actor_outer < 0:
            raise ValueError('QPROBE_INVALID_BOOTSTRAP_ACTOR_OUTER')
        _required_text(actor_snapshot_id, 'actor_snapshot_id')
        features = validate_features(features, self.feature_dim)
        if len(labels) != len(features):
            raise ValueError('QPROBE_LABEL_FEATURE_COUNT_MISMATCH')
        valid = []
        protocol = None
        groups = {}
        seen = set()
        for index, label in enumerate(labels):
            source_outer = label.get('source_outer')
            if type(source_outer) is not int or not 0 <= source_outer <= actor_outer:
                raise ValueError('QPROBE_BOOTSTRAP_FUTURE_OR_UNKNOWN_POLICY_LABEL')
            value = validate_label(label, outer=source_outer, **{k:label.get(k) for k in (
                'run_id', 'attempt_id', 'policy_snapshot_id', 'sampling_contract_sha',
                'strict_scoring_contract_sha', 'retriever_version')})
            if value['selection_source'] != 'bootstrap_all_candidate':
                raise ValueError('QPROBE_BOOTSTRAP_MUST_INCLUDE_ALL_CANDIDATE_CLASSES')
            candidate_key = (value['run_id'], value['attempt_id'], source_outer, value['qid'])
            if candidate_key in seen:
                raise ValueError('QPROBE_DUPLICATE_BOOTSTRAP_LABEL')
            seen.add(candidate_key)
            candidate_protocol = {k:value[k] for k in (
                'sampling_contract_sha', 'strict_scoring_contract_sha', 'retriever_version')}
            if protocol is not None and candidate_protocol != protocol:
                raise ValueError('QPROBE_PROTOCOL_CHANGE_REQUIRES_NEW_NAMESPACE')
            protocol = candidate_protocol
            if source_outer > actor_outer - self.config.replay_window_outer:
                groups.setdefault(source_outer, []).append((index, value))
        for source_outer in sorted(groups):
            rows = groups[source_outer]
            permutation = torch.randperm(len(rows), generator=self.rng).tolist()
            valid += [rows[i] for i in permutation[:self.config.bootstrap_max_groups_per_outer]]
        if not valid:
            raise ValueError('QPROBE_NO_RECENT_COMPLETE_BOOTSTRAP_LABELS')
        self.replay_labels = [x[1] for x in valid]
        self.replay_features = features[[x[0] for x in valid]].clone()
        self.protocol = protocol
        metrics = self._fit(actor_outer, epochs=self.config.bootstrap_head_epochs,
                            log_warmup=self.config.bootstrap_head_epochs is not None)
        self.completed_outer = actor_outer
        self.actor_snapshot_id = actor_snapshot_id
        metrics.update(bootstrap_groups=len(valid), version=self.version, completed_outer=actor_outer)
        return metrics

    def state_dict(self):
        return copy.deepcopy(dict(schema_version=SCHEMA_VERSION, config=asdict(self.config),
            loss_identity=self.config.loss_identity,
            feature_dim=self.feature_dim, feature_spec_sha=self.feature_spec_sha,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            rng=self.rng.get_state(), replay_labels=self.replay_labels,
            replay_features=self.replay_features, completed_outer=self.completed_outer,
            update_count=self.update_count, completed_attempts=self.completed_attempts,
            protocol=self.protocol, actor_snapshot_id=self.actor_snapshot_id))

    def _restore_state(self, state):
        if state.get('schema_version') != SCHEMA_VERSION or state['feature_dim'] != self.feature_dim:
            raise ValueError('QPROBE_CHECKPOINT_SCHEMA_MISMATCH')
        if state.get('loss_identity', self.config.loss_identity) != self.config.loss_identity:
            raise ValueError('QPROBE_CHECKPOINT_LOSS_IDENTITY_MISMATCH')
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.set_state(state['rng'])
        self.replay_features = state['replay_features'].clone().float().cpu()
        self.replay_labels = copy.deepcopy(state['replay_labels'])
        self.completed_outer = state['completed_outer']; self.update_count = state['update_count']
        self.completed_attempts = dict(state['completed_attempts'])
        self.protocol = copy.deepcopy(state['protocol']); self.actor_snapshot_id = state['actor_snapshot_id']

    def load_state_dict(self, state):
        """In-memory transaction rollback; file restoration should use ``load``."""
        # Old serialized configs omit loss_mode/count_aux_loss_weight. Applying
        # their legacy defaults is compatible only with the same legacy config.
        if (not isinstance(state.get('config'), Mapping) or
            QProbeConfig.from_mapping(state['config']) != self.config or
            state.get('feature_spec_sha') != self.feature_spec_sha):
            raise ValueError('QPROBE_ROLLBACK_NAMESPACE_MISMATCH')
        self._restore_state(copy.deepcopy(state))

    def save(self, path, *, actor_snapshot_id: str):
        _required_text(actor_snapshot_id, 'actor_snapshot_id')
        if not self.feature_spec_sha:
            raise ValueError('QPROBE_CHECKPOINT_REQUIRES_FEATURE_SPEC_SHA')
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        # The checkpoint is immutable. Hard-link publication is atomic and
        # fails if a competing writer already published this exact target.
        fd, temporary = tempfile.mkstemp(prefix=path.name+'.partial-', dir=path.parent)
        os.close(fd)
        try:
            state = self.state_dict(); state['actor_snapshot_id'] = actor_snapshot_id
            torch.save(state, temporary)
            with open(temporary, 'rb') as f: os.fsync(f.fileno())
            sha = _sha_file(temporary)
            os.link(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        self.actor_snapshot_id = actor_snapshot_id
        return dict(path=str(path), checkpoint_sha=sha, sha256=sha, bytes=path.stat().st_size,
                    actor_snapshot_id=actor_snapshot_id,
                    version=self.version, completed_outer=self.completed_outer,
                    feature_spec_sha=self.feature_spec_sha, update_count=self.update_count,
                    head_epochs_per_outer=self.config.head_epochs_per_outer,
                    bootstrap_head_epochs=self.config.bootstrap_head_epochs,
                    loss_mode=self.config.loss_mode, loss_identity=self.config.loss_identity)

    @classmethod
    def load(cls, path, expected_actor_snapshot_id=None, expected_feature_spec_sha=None,
             expected_checkpoint_sha=None):
        if expected_checkpoint_sha and _sha_file(path) != expected_checkpoint_sha:
            raise ValueError('QPROBE_CHECKPOINT_SHA_MISMATCH')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('QPROBE_CHECKPOINT_SCHEMA_MISMATCH')
        if expected_actor_snapshot_id is not None and state.get('actor_snapshot_id') != expected_actor_snapshot_id:
            raise ValueError('QPROBE_ACTOR_BINDING_MISMATCH')
        if expected_feature_spec_sha is not None and state.get('feature_spec_sha') != expected_feature_spec_sha:
            raise ValueError('QPROBE_FEATURE_SPEC_MISMATCH')
        if not state.get('feature_spec_sha') or not state.get('actor_snapshot_id'):
            raise ValueError('QPROBE_UNBOUND_CHECKPOINT')
        instance = cls(state['feature_dim'], state['config'], state['feature_spec_sha'])
        instance._restore_state(state)
        if len(instance.replay_labels) != len(instance.replay_features):
            raise ValueError('QPROBE_REPLAY_COUNT_MISMATCH')
        if (type(instance.completed_outer) is not int or instance.completed_outer < 0 or
            type(instance.update_count) is not int or instance.update_count < 0):
            raise ValueError('QPROBE_INVALID_CHECKPOINT_COUNTERS')
        for label in instance.replay_labels:
            source_outer = label.get('source_outer')
            if type(source_outer) is not int or not 0 <= source_outer <= instance.completed_outer:
                raise ValueError('QPROBE_CHECKPOINT_CONTAINS_FUTURE_LABELS')
            validate_label(label, outer=source_outer, **{k:label.get(k) for k in (
                'run_id', 'attempt_id', 'policy_snapshot_id', 'sampling_contract_sha',
                'strict_scoring_contract_sha', 'retriever_version')})
            protocol = {key:label[key] for key in ('sampling_contract_sha','strict_scoring_contract_sha','retriever_version')}
            if protocol != instance.protocol:
                raise ValueError('QPROBE_CHECKPOINT_REPLAY_PROTOCOL_MISMATCH')
        if instance.replay_features.numel(): validate_features(instance.replay_features, instance.feature_dim)
        if any(not torch.isfinite(value).all() for value in instance.model.state_dict().values()):
            raise ValueError('QPROBE_NONFINITE_CHECKPOINT')
        for optimizer_state in instance.optimizer.state.values():
            if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                   for value in optimizer_state.values()):
                raise ValueError('QPROBE_NONFINITE_OPTIMIZER_CHECKPOINT')
        return instance
