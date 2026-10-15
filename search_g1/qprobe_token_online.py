"""Online full-token binary head with compact, immutable replay identities.

The caller owns actor/head publication transactions. A failed fit retains its
actual partial Adam writes until that caller explicitly restores a snapshot.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping

import torch
from torch.nn import functional as F

from search_g1.qprobe_model import _required_text, _sha_file, validate_label
from search_g1.qprobe_token_model import (
    TokenBinaryQProbeConfig, TokenBinaryQuestionProbe, TokenBinaryWarmup,
)


SCHEMA_VERSION = 1
HEAD_KIND = 'token_binary_v1'
PROTOCOL_KEYS = ('sampling_contract_sha', 'strict_scoring_contract_sha', 'retriever_version')
LABEL_KEYS = ('run_id', 'attempt_id', 'policy_snapshot_id', *PROTOCOL_KEYS)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('TOKEN_QPROBE_INVALID_SHA: ' + name)
    return value


@dataclass(frozen=True)
class TokenOnlineQProbeConfig:
    head_kind: str = HEAD_KIND
    ranking: str = 'binary_mixed_probability'
    max_tokens: int = 244
    feature_dim: int = 2048
    head_hidden_size: int = 128
    attention_heads: int = 4
    bootstrap_head_epochs: int = 10
    head_epochs_per_outer: int = 2
    head_batch_size: int = 256
    head_learning_rate: float = 3e-4
    std_loss_weight: float = .3
    target_statistic: str = 'sample_std'
    replay_window_outer: int = 16
    replay_half_life_outer: float = 4.
    seed: int = 1729

    def __post_init__(self):
        self.token_config()
        if (self.head_kind != HEAD_KIND or self.ranking != 'binary_mixed_probability' or
                self.target_statistic != 'sample_std' or self.std_loss_weight != .3):
            raise ValueError('TOKEN_QPROBE_ONLINE_OBJECTIVE_CHANGED')
        for name in ('head_epochs_per_outer', 'replay_window_outer'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('TOKEN_QPROBE_INVALID_CONFIG: ' + name)
        if not math.isfinite(self.replay_half_life_outer) or self.replay_half_life_outer <= 0:
            raise ValueError('TOKEN_QPROBE_INVALID_REPLAY_HALF_LIFE')

    def token_config(self):
        return TokenBinaryQProbeConfig(max_tokens=self.max_tokens, feature_dim=self.feature_dim,
            hidden_size=self.head_hidden_size, attention_heads=self.attention_heads,
            epochs=self.bootstrap_head_epochs, batch_size=self.head_batch_size,
            learning_rate=self.head_learning_rate, auxiliary_weight=self.std_loss_weight,
            target_statistic=self.target_statistic, top_k=512, seed=self.seed)

    @property
    def loss_identity(self):
        return self.token_config().loss_identity

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        # Runtime's seed belongs to question selection; head_seed overrides it.
        if 'head_seed' in values:
            values['seed'] = values['head_seed']
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


def token_binary_group_loss(prediction, counts, config):
    """Unreduced, identical BCE + bounded sample-std objective for age weights."""
    counts = torch.as_tensor(counts, device=prediction['binary_logit'].device)
    if (counts.dtype != torch.long or counts.ndim != 1 or len(counts) != len(prediction['binary_logit']) or
            not len(counts) or bool(((counts < 0) | (counts > 5)).any())):
        raise ValueError('TOKEN_QPROBE_INVALID_COUNTS')
    target = (counts.float() * (5-counts.float()) / 20).sqrt()
    bce = F.binary_cross_entropy_with_logits(prediction['binary_logit'],
                                             ((counts > 0) & (counts < 5)).float(), reduction='none')
    raw = (prediction['predicted_statistic']-target).square()
    normalized = ((prediction['predicted_statistic']-target)/math.sqrt(.3)).square()
    contribution = config.std_loss_weight * normalized
    loss = bce + contribution
    if not torch.isfinite(loss).all():
        raise ValueError('TOKEN_QPROBE_NONFINITE_TRAINING_LOSS')
    return loss, dict(binary_bce=bce, regression_mse_raw=raw,
                      regression_mse_normalized=normalized, regression_contribution=contribution)


class OnlineTokenBinaryQProbe:
    def __init__(self, feature_dim=2048, config=None, feature_spec_sha='', *,
                 feature_provider=None, feature_cache=None, device='cpu'):
        self.config = config if isinstance(config, TokenOnlineQProbeConfig) else TokenOnlineQProbeConfig.from_mapping(config)
        if feature_dim != self.config.feature_dim:
            raise ValueError('TOKEN_QPROBE_FEATURE_DIM_MISMATCH')
        self.feature_dim, self.feature_spec_sha = feature_dim, feature_spec_sha
        self.device = torch.device(device)
        # Use the exact warmup architecture without seeding any actor CUDA RNG.
        self._engine = TokenBinaryWarmup.__new__(TokenBinaryWarmup)
        self._engine.config, self._engine.device = self.config.token_config(), self.device
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(self.config.seed).get_state())
            self.model = TokenBinaryQuestionProbe(self._engine.config)
        self.model.to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.head_learning_rate)
        self.rng = torch.Generator(device='cpu').manual_seed(self.config.seed+1)
        self._engine.model, self._engine.optimizer, self._engine.rng = self.model, self.optimizer, self.rng
        self._engine.update_count, self._engine.fitted = 0, False
        self.replay_labels, self.replay_features = [], []  # features are ordered qids, never tensors
        self.completed_outer, self.update_count, self.completed_attempts = 0, 0, {}
        self.protocol, self.actor_snapshot_id = None, None
        self._initialization_provenance, self._fit_failed = {}, False
        self.feature_provider = None
        if feature_provider is not None and feature_cache is not None and feature_provider is not feature_cache:
            raise ValueError('TOKEN_QPROBE_CONFLICTING_FEATURE_PROVIDERS')
        provider = feature_provider if feature_provider is not None else feature_cache
        if provider is not None:
            self.attach_feature_provider(provider)

    @property
    def initialization_provenance(self):
        return copy.deepcopy(self._initialization_provenance)

    @property
    def version(self):
        digest = hashlib.sha256(self.feature_spec_sha.encode())
        for key, value in sorted(self.model.state_dict().items()):
            digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        return f'qprobe-token:{self.completed_outer}:{self.update_count}:{digest.hexdigest()[:16]}'

    @property
    def replay_identity_sha256(self):
        return _identity(dict(labels=self.replay_labels, qids=self.replay_features))

    def attach_feature_provider(self, provider):
        spec = getattr(provider, 'spec', {})
        if (getattr(provider, 'spec_sha', None) != _sha(self.feature_spec_sha, 'feature_spec_sha') or
                getattr(provider, 'feature_dim', None) != self.feature_dim or
                spec.get('global_max_tokens') != self.config.max_tokens or
                not hasattr(provider, 'get_batch') or not isinstance(getattr(provider, 'index', None), Mapping)):
            raise ValueError('TOKEN_QPROBE_FEATURE_PROVIDER_NAMESPACE_MISMATCH')
        self._resolve(self.replay_features, self.replay_labels, provider=provider, allow_empty=True)
        self.feature_provider = provider
        return self

    def _resolve(self, qids, labels=None, *, provider=None, allow_empty=False):
        if isinstance(qids, (str, bytes, torch.Tensor)):
            raise ValueError('TOKEN_QPROBE_REQUIRES_ORDERED_QIDS')
        qids = list(qids)
        if (not qids and not allow_empty) or any(not isinstance(qid, str) or not qid for qid in qids):
            raise ValueError('TOKEN_QPROBE_REQUIRES_ORDERED_QIDS')
        if labels is not None and (len(labels) != len(qids) or any(row['qid'] != qid for row, qid in zip(labels, qids))):
            raise ValueError('TOKEN_QPROBE_LABEL_QID_ORDER_MISMATCH')
        provider = self.feature_provider if provider is None else provider
        if provider is None:
            raise ValueError('TOKEN_QPROBE_FEATURE_PROVIDER_NOT_ATTACHED')
        lengths = []
        for index, qid in enumerate(qids):
            row = provider.index[qid]
            length = row.get('length')
            if type(length) is not int or not 0 < length <= self.config.max_tokens:
                raise ValueError('TOKEN_QPROBE_LENGTH_EXCEEDS_GLOBAL_BMAX')
            if labels is not None and row.get('question_hash') != labels[index]['question_hash']:
                raise ValueError('TOKEN_QPROBE_QUESTION_HASH_MISMATCH')
            lengths.append(length)
        return qids, torch.tensor(lengths, dtype=torch.long)

    def _prediction(self, qids):
        qids, lengths = self._resolve(qids)
        previous = self.model.training
        result = {key: [] for key in ('binary_logit', 'mixed_probability', 'predicted_statistic')}
        try:
            self.model.eval()
            with torch.no_grad():
                for index in torch.arange(len(qids)).split(self.config.head_batch_size):
                    features, sizes = self._engine._batch(self.feature_provider, index, lengths, qids)
                    prediction = self.model(features, sizes)
                    for key in result:
                        result[key].append(prediction[key].detach().cpu().clone())
        finally:
            self.model.train(previous)
        return {key: torch.cat(values) for key, values in result.items()}

    def predict(self, qids):
        prediction = self._prediction(qids)
        return dict(mixed_probability=prediction['mixed_probability'],
                    predicted_sample_std=prediction['predicted_statistic'])

    def _adam_steps(self):
        return self._engine._adam_steps()

    def _training_summary(self, prediction, counts, weights, epochs, updates):
        group_loss, parts = token_binary_group_loss(prediction, counts, self.config)
        weighted = lambda value: float((value * weights).mean())
        truth, positive = (counts > 0) & (counts < 5), prediction['mixed_probability'] >= .5
        tp, tn, fp, fn = [int(value.sum()) for value in
                         (truth & positive, ~truth & ~positive, ~truth & positive, truth & ~positive)]
        ratio = lambda n, d: n/d if d else 0.
        recall, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
        recalls = ([recall] if tp+fn else []) + ([specificity] if tn+fp else [])
        target = (counts.float()*(5-counts.float())/20).sqrt()
        return dict(dataset_scope='training_replay_not_independent_holdout', replay_groups=len(counts),
            epoch=epochs, fit_epochs=epochs, head_optimizer_updates=updates, head_adam_steps=self._adam_steps(),
            total_loss=weighted(group_loss), replay_weighted_loss=weighted(group_loss),
            **{key: weighted(value) for key, value in parts.items()},
            binary_mixed=dict(threshold=.5, true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn,
                accuracy=ratio(tp+tn,len(counts)), precision=ratio(tp,tp+fp), recall=recall, specificity=specificity,
                balanced_accuracy=sum(recalls)/len(recalls), f1=ratio(2*tp,2*tp+fp+fn), zero_division=0.),
            true_mixed_fraction=float(truth.float().mean()), predicted_mixed_fraction=float(positive.float().mean()),
            sigma_mae=float((prediction['predicted_statistic']-target).abs().mean()))

    def _fit(self, outer):
        if self._fit_failed:
            raise ValueError('TOKEN_QPROBE_FAILED_FIT_REQUIRES_EXPLICIT_TRANSACTION_RESTORE')
        qids, lengths = self._resolve(self.replay_features, self.replay_labels)
        counts = torch.tensor([row['correct_count'] for row in self.replay_labels], dtype=torch.long)
        weights = torch.tensor([2**(-(outer-row['source_outer'])/self.config.replay_half_life_outer)
                                for row in self.replay_labels], dtype=torch.float32)
        weights *= len(weights)/weights.sum()  # Global denominator, never per-mini normalization.
        before, history = self._adam_steps(), []
        previous = self.model.training
        try:
            self.model.train()
            for epoch in range(1, self.config.head_epochs_per_outer+1):
                order = torch.randperm(len(qids), generator=self.rng)
                for batch_index, index in enumerate(order.split(self.config.head_batch_size)):
                    features, sizes = self._engine._batch(self.feature_provider, index, lengths, qids)
                    prediction = self.model(features, sizes)
                    groups, parts = token_binary_group_loss(prediction, counts[index], self.config)
                    local_weights = weights[index].to(self.device)
                    loss = (groups*local_weights).mean()
                    row = dict(epoch=epoch, minibatch_index=batch_index, batch_size=len(index),
                        head_update_ordinal=len(history)+1, head_adam_steps_before=self._adam_steps(),
                        head_learning_rate=float(self.optimizer.param_groups[0]['lr']),
                        pre_step_weighted_loss=float(loss.detach()),
                        weight_reduction='mean(loss_i * globally_normalized_replay_weight_i)',
                        loss_parts={key: float((value.detach()*local_weights).mean()) for key,value in parts.items()})
                    self.optimizer.zero_grad(set_to_none=True); loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                        raise ValueError('TOKEN_QPROBE_NONFINITE_GRADIENT')
                    self.optimizer.step(); self._engine.update_count += 1
                    if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                        raise ValueError('TOKEN_QPROBE_NONFINITE_PARAMETERS')
                    row['head_adam_steps_after'] = self._adam_steps(); history.append(row)
            summary = self._training_summary(self._prediction(qids), counts, weights,
                                             self.config.head_epochs_per_outer, len(history))
            self._engine.fitted = True
            return dict(replay_groups=len(qids), head_optimizer_updates=len(history),
                replay_weighted_loss=summary['replay_weighted_loss'], total_loss=summary['total_loss'],
                loss_mode=HEAD_KIND, loss_identity=self.config.loss_identity,
                fit_epochs=self.config.head_epochs_per_outer, head_batch_size=self.config.head_batch_size,
                head_learning_rate=float(self.optimizer.param_groups[0]['lr']),
                head_adam_steps_before=before, head_adam_steps_after=self._adam_steps(),
                training_summary=summary, epoch_history=[], minibatch_history=history)
        except BaseException:
            self._fit_failed = True
            raise
        finally:
            self.model.train(previous)

    def update_completed_outer(self, outer, labels, features, *, policy_snapshot_id,
            sampling_contract_sha, strict_scoring_contract_sha, retriever_version, run_id, attempt_id):
        if type(outer) is not int or outer <= self.completed_outer:
            raise ValueError('QPROBE_OLD_OR_FUTURE_HEAD_ACTOR_PAIR')
        attempt_key = f'{run_id}:{outer}'
        if attempt_key in self.completed_attempts or attempt_id in self.completed_attempts.values():
            raise ValueError('QPROBE_OUTER_ALREADY_UPDATED')
        valid = [copy.deepcopy(validate_label(row, outer=outer, run_id=run_id, attempt_id=attempt_id,
            policy_snapshot_id=policy_snapshot_id, sampling_contract_sha=sampling_contract_sha,
            strict_scoring_contract_sha=strict_scoring_contract_sha, retriever_version=retriever_version)) for row in labels]
        qids, _ = self._resolve(features, valid)
        if len(set(qids)) != len(qids):
            raise ValueError('QPROBE_DUPLICATE_QUESTION_WITHIN_OUTER')
        protocol = {key: value for key, value in zip(PROTOCOL_KEYS,
                    (sampling_contract_sha, strict_scoring_contract_sha, retriever_version))}
        if self.protocol is not None and self.protocol != protocol:
            raise ValueError('QPROBE_PROTOCOL_CHANGE_REQUIRES_NEW_NAMESPACE')
        started = time.monotonic()
        self.protocol = protocol
        combined = self.replay_labels + valid
        self.replay_labels = [row for row in combined if row['source_outer'] > outer-self.config.replay_window_outer]
        self.replay_features = [row['qid'] for row in self.replay_labels]
        result = self._fit(outer)
        self.completed_outer = outer; self.update_count += 1
        self.completed_attempts[attempt_key] = attempt_id; self.actor_snapshot_id = None
        result.update(version=self.version, completed_outer=outer, update_count=self.update_count,
                      elapsed_seconds=time.monotonic()-started, label_policy_snapshot_id=policy_snapshot_id)
        return result

    def _bootstrap_labels(self, labels, qids):
        if len(labels) != 5120:
            raise ValueError('TOKEN_QPROBE_BOOTSTRAP_REQUIRES_ALL_5120_GROUPS')
        valid = []
        for row in labels:
            if row.get('source_outer') != 0 or row.get('selection_source') != 'bootstrap_all_candidate':
                raise ValueError('TOKEN_QPROBE_BOOTSTRAP_REQUIRES_LOGICAL_ZERO_ALL_CANDIDATES')
            valid.append(copy.deepcopy(validate_label(row, outer=0, **{key:row.get(key) for key in LABEL_KEYS})))
        qids = list(qids)
        if len(qids) != 5120 or len(set(qids)) != 5120 or [row['qid'] for row in valid] != qids:
            raise ValueError('TOKEN_QPROBE_BOOTSTRAP_QID_ORDER_OR_COMPLETENESS')
        protocols = [{key:row[key] for key in PROTOCOL_KEYS} for row in valid]
        if any(protocol != protocols[0] for protocol in protocols):
            raise ValueError('QPROBE_PROTOCOL_CHANGE_REQUIRES_NEW_NAMESPACE')
        if self.feature_provider is not None:
            self._resolve(qids, valid)
        return valid, qids, protocols[0]

    def bootstrap(self, labels, features, *, actor_outer, actor_snapshot_id):
        if self.replay_labels or self.update_count or self._adam_steps() or actor_outer != 0:
            raise ValueError('TOKEN_QPROBE_BOOTSTRAP_ONLY_ON_FRESH_LOGICAL_ZERO')
        _required_text(actor_snapshot_id, 'actor_snapshot_id')
        valid, qids, protocol = self._bootstrap_labels(labels, features)
        _, lengths = self._resolve(qids, valid)
        result = self._engine.fit(self.feature_provider, lengths,
                                  [row['correct_count'] for row in valid], qids)
        self.replay_labels, self.replay_features, self.protocol = valid, qids, protocol
        self.actor_snapshot_id = actor_snapshot_id
        self._initialization_provenance = dict(kind='fresh_bootstrap_fit', adopted_at_completed_outer=0,
                                               warmup_head_adam_steps=self._adam_steps(), no_refit=False)
        result.update(bootstrap_groups=5120, version=self.version, completed_outer=0)
        return result

    @classmethod
    def adopt_warmup(cls, path, labels, qids, *, expected_checkpoint_sha,
            expected_feature_spec_sha, provenance, config=None, feature_cache=None,
            feature_provider=None, device='cpu'):
        digest = _sha(expected_checkpoint_sha, 'warmup_checkpoint_sha')
        if _sha_file(path) != digest:
            raise ValueError('QPROBE_CHECKPOINT_SHA_MISMATCH')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('token_feature_spec_sha256') != _sha(expected_feature_spec_sha, 'feature_spec_sha'):
            raise ValueError('QPROBE_FEATURE_SPEC_MISMATCH')
        raw = state.get('head', {})
        instance = cls((config.feature_dim if isinstance(config, TokenOnlineQProbeConfig) else
                        dict(config or {}).get('feature_dim', 2048)), config, expected_feature_spec_sha,
                       feature_cache=feature_cache, feature_provider=feature_provider, device=device)
        if (raw.get('schema_version') != 1 or raw.get('config') != asdict(instance.config.token_config()) or
                raw.get('fitted') is not True or raw.get('update_count') != 200 or
                raw.get('loss_identity') != instance.config.loss_identity or
                instance.config.bootstrap_head_epochs != 10 or instance.config.head_batch_size != 256):
            raise ValueError('TOKEN_QPROBE_UNQUALIFIED_WARMUP_STATE')
        provenance = copy.deepcopy(dict(provenance))
        actor_id = _required_text(provenance.get('actor_snapshot_id'), 'actor_snapshot_id')
        valid, ordered, protocol = instance._bootstrap_labels(labels, qids)
        instance.model.load_state_dict(raw['model'], strict=True)
        instance.optimizer.load_state_dict(raw['optimizer']); instance.rng.set_state(raw['private_rng'])
        instance._engine.update_count, instance._engine.fitted = 200, True
        instance.replay_labels, instance.replay_features, instance.protocol = valid, ordered, protocol
        instance.actor_snapshot_id = actor_id
        instance._initialization_provenance = dict(provenance, kind='qualified_warmup_checkpoint_adoption',
            warmup_checkpoint_sha256=digest, warmup_head_adam_steps=200, adopted_at_completed_outer=0,
            bootstrap_groups=5120, no_refit=True, original_label_producer_provenance=copy.deepcopy(state.get('label_producer_provenance')))
        instance._validate_state()
        return instance

    def state_dict(self):
        return copy.deepcopy(dict(schema_version=SCHEMA_VERSION, head_kind=HEAD_KIND,
            config=asdict(self.config), loss_identity=self.config.loss_identity,
            feature_dim=self.feature_dim, feature_spec_sha=self.feature_spec_sha,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), rng=self.rng.get_state(),
            replay_labels=self.replay_labels, replay_features=self.replay_features,
            replay_identity_sha256=self.replay_identity_sha256,
            completed_outer=self.completed_outer, update_count=self.update_count,
            head_adam_updates=self._engine.update_count, fitted=self._engine.fitted,
            fit_failed=self._fit_failed, completed_attempts=self.completed_attempts,
            protocol=self.protocol, actor_snapshot_id=self.actor_snapshot_id,
            initialization_provenance=self._initialization_provenance))

    def _validate_state(self):
        if (type(self.completed_outer) is not int or self.completed_outer < 0 or
                type(self.update_count) is not int or self.update_count < 0 or
                type(self._engine.update_count) is not int or self._engine.update_count < 0 or
                self._adam_steps() != self._engine.update_count):
            raise ValueError('TOKEN_QPROBE_INVALID_CHECKPOINT_COUNTERS')
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError('QPROBE_NONFINITE_CHECKPOINT')
        parameters = list(self.model.parameters())
        if self._adam_steps() and set(self.optimizer.state) != set(parameters):
            raise ValueError('TOKEN_QPROBE_INCOMPLETE_ADAM_STATE')
        if len(self.optimizer.param_groups) != 1 or self.optimizer.param_groups[0]['lr'] != self.config.head_learning_rate:
            raise ValueError('TOKEN_QPROBE_OPTIMIZER_CONFIG_MISMATCH')
        for parameter, state in self.optimizer.state.items():
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and (not torch.isfinite(value).all() or
                        name in ('exp_avg', 'exp_avg_sq') and value.shape != parameter.shape):
                    raise ValueError('QPROBE_NONFINITE_OR_INVALID_OPTIMIZER_CHECKPOINT')
        if len(self.replay_labels) != len(self.replay_features):
            raise ValueError('QPROBE_REPLAY_COUNT_MISMATCH')
        identities, groups = {}, set()
        for row, qid in zip(self.replay_labels, self.replay_features):
            outer = row.get('source_outer')
            if (type(outer) is not int or not 0 <= outer <= self.completed_outer or
                    outer <= self.completed_outer-self.config.replay_window_outer or row.get('qid') != qid):
                raise ValueError('QPROBE_CHECKPOINT_CONTAINS_FUTURE_OR_EXPIRED_LABELS')
            checked = validate_label(row, outer=outer, **{key:row.get(key) for key in LABEL_KEYS})
            if checked != row or {key:row[key] for key in PROTOCOL_KEYS} != self.protocol:
                raise ValueError('QPROBE_CHECKPOINT_REPLAY_PROTOCOL_MISMATCH')
            key = (row['run_id'], row['attempt_id'], outer, qid)
            if key in groups or qid in identities and identities[qid] != row['question_hash']:
                raise ValueError('TOKEN_QPROBE_DUPLICATE_OR_CHANGED_REPLAY_IDENTITY')
            groups.add(key); identities[qid] = row['question_hash']
        if self.feature_provider is not None:
            self._resolve(self.replay_features, self.replay_labels, allow_empty=True)

    def load_state_dict(self, state):
        if (state.get('schema_version') != SCHEMA_VERSION or state.get('head_kind') != HEAD_KIND or
                TokenOnlineQProbeConfig.from_mapping(state.get('config')) != self.config or
                state.get('feature_dim') != self.feature_dim or state.get('feature_spec_sha') != self.feature_spec_sha or
                state.get('loss_identity') != self.config.loss_identity or
                state.get('replay_identity_sha256') != _identity(dict(labels=state['replay_labels'],qids=state['replay_features']))):
            raise ValueError('QPROBE_ROLLBACK_NAMESPACE_OR_REPLAY_MISMATCH')
        self.model.load_state_dict(state['model'], strict=True); self.optimizer.load_state_dict(state['optimizer'])
        self.rng.set_state(state['rng'])
        for key in ('replay_labels','replay_features','completed_outer','update_count','completed_attempts','protocol','actor_snapshot_id'):
            setattr(self, key, copy.deepcopy(state[key]))
        self._engine.update_count, self._engine.fitted = state['head_adam_updates'], state['fitted']
        self._fit_failed = state['fit_failed']
        self._initialization_provenance = copy.deepcopy(state['initialization_provenance'])
        self._validate_state()

    def save(self, path, *, actor_snapshot_id):
        _required_text(actor_snapshot_id, 'actor_snapshot_id'); _sha(self.feature_spec_sha, 'feature_spec_sha')
        if self._fit_failed:
            raise ValueError('TOKEN_QPROBE_CANNOT_PUBLISH_FAILED_FIT')
        self._validate_state()
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name+'.partial-', dir=path.parent); os.close(fd)
        try:
            state = self.state_dict(); state['actor_snapshot_id'] = actor_snapshot_id
            torch.save(state, temporary)
            with open(temporary, 'rb') as stream: os.fsync(stream.fileno())
            digest = _sha_file(temporary); os.link(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        self.actor_snapshot_id = actor_snapshot_id
        return dict(path=str(path), checkpoint_sha=digest, sha256=digest, bytes=path.stat().st_size,
            actor_snapshot_id=actor_snapshot_id, version=self.version, head_version=self.version,
            completed_outer=self.completed_outer, feature_spec_sha=self.feature_spec_sha,
            replay_identity_sha256=self.replay_identity_sha256, replay_groups=len(self.replay_labels),
            update_count=self.update_count, head_adam_steps=self._adam_steps(),
            head_epochs_per_outer=self.config.head_epochs_per_outer, bootstrap_head_epochs=self.config.bootstrap_head_epochs,
            loss_mode=HEAD_KIND, loss_identity=self.config.loss_identity,
            initialization_provenance=self.initialization_provenance)

    @classmethod
    def load(cls, path, expected_actor_snapshot_id=None, expected_feature_spec_sha=None,
             expected_checkpoint_sha=None, *, feature_cache=None, feature_provider=None, device='cpu'):
        if expected_checkpoint_sha is not None and _sha_file(path) != expected_checkpoint_sha:
            raise ValueError('QPROBE_CHECKPOINT_SHA_MISMATCH')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (expected_actor_snapshot_id is not None and state.get('actor_snapshot_id') != expected_actor_snapshot_id):
            raise ValueError('QPROBE_ACTOR_BINDING_MISMATCH')
        if expected_feature_spec_sha is not None and state.get('feature_spec_sha') != expected_feature_spec_sha:
            raise ValueError('QPROBE_FEATURE_SPEC_MISMATCH')
        _required_text(state.get('actor_snapshot_id'), 'actor_snapshot_id'); _sha(state.get('feature_spec_sha'), 'feature_spec_sha')
        instance = cls(state['feature_dim'], state['config'], state['feature_spec_sha'],
                       feature_cache=feature_cache, feature_provider=feature_provider, device=device)
        instance.load_state_dict(state)
        return instance
