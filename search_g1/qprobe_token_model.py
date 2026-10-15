"""Fresh token-wise binary question head; training-set warmup only.

Every real question token enters MLP -> one self-attention layer -> masked
global right padding -> flatten -> two outputs. There is no token pooling,
truncation, count-class output, actor update, or independent validation split.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TokenBinaryQProbeConfig:
    max_tokens: int
    feature_dim: int = 2048
    hidden_size: int = 128
    attention_heads: int = 4
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 3e-4
    auxiliary_weight: float = .3
    target_statistic: str = 'sample_std'
    top_k: int = 512
    seed: int = 1729

    def __post_init__(self):
        for key in ('max_tokens', 'feature_dim', 'epochs', 'batch_size', 'top_k'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError('TOKEN_QPROBE_INVALID_CONFIG: ' + key)
        if self.hidden_size != 128 or self.attention_heads != 4:
            raise ValueError('TOKEN_QPROBE_REQUIRES_128_HIDDEN_AND_4_HEADS')
        if (type(self.seed) is not int or not 0 <= self.seed < 2**63 or
            not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or
            not math.isfinite(self.auxiliary_weight) or self.auxiliary_weight < 0):
            raise ValueError('TOKEN_QPROBE_INVALID_OPTIMIZER_CONFIG')
        if self.target_statistic not in ('sample_std', 'sample_variance'):
            raise ValueError('TOKEN_QPROBE_UNKNOWN_REGRESSION_TARGET')

    @property
    def statistic_bound(self):
        return math.sqrt(.3) if self.target_statistic == 'sample_std' else .3

    @property
    def loss_identity(self):
        scale = 'sqrt(0.3)' if self.target_statistic == 'sample_std' else '0.3'
        return (f'BCEWithLogits(K in 1..4)+{self.auxiliary_weight:g}*'
                f'((predicted_{self.target_statistic}-{self.target_statistic}(K))/{scale})^2')


def _integer_vector(values, name, size=None):
    value = torch.as_tensor(values)
    if value.ndim != 1 or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('TOKEN_QPROBE_REQUIRES_INTEGER_VECTOR: ' + name)
    if size is not None and len(value) != size:
        raise ValueError('TOKEN_QPROBE_VECTOR_LENGTH_MISMATCH: ' + name)
    return value.detach().long()


class TokenBinaryQuestionProbe(nn.Module):
    def __init__(self, config: TokenBinaryQProbeConfig):
        super().__init__()
        self.config = config
        self.token_mlp = nn.Sequential(nn.Linear(config.feature_dim, 128), nn.GELU(), nn.Linear(128, 128))
        self.attention = nn.MultiheadAttention(128, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.output = nn.Linear(config.max_tokens * 128, 2)

    def forward(self, features, lengths):
        if (not isinstance(features, torch.Tensor) or features.ndim != 3 or
            features.shape[-1] != self.config.feature_dim or not len(features)):
            raise ValueError('TOKEN_QPROBE_FEATURE_SHAPE')
        if features.shape[1] > self.config.max_tokens:
            raise ValueError('TOKEN_QPROBE_LENGTH_EXCEEDS_GLOBAL_BMAX')
        lengths = _integer_vector(lengths, 'lengths', len(features)).to(features.device)
        if bool(((lengths < 1) | (lengths > features.shape[1]) | (lengths > self.config.max_tokens)).any()):
            raise ValueError('TOKEN_QPROBE_INVALID_OR_OVERSIZE_TOKEN_LENGTH')
        # Remove only the batch-wide trailing padding before attention. Every
        # real token remains; the output still has the fixed global Bmax shape.
        active_tokens = int(lengths.max())
        features = features[:, :active_tokens].to(dtype=self.output.weight.dtype)
        mask = torch.arange(active_tokens, device=features.device)[None, :] < lengths[:, None]
        # Even arbitrary/NaN padding cannot contribute through MLP biases or queries.
        features = features.masked_fill(~mask[:, :, None], 0.)
        if not torch.isfinite(features).all():
            raise ValueError('TOKEN_QPROBE_NONFINITE_REAL_TOKEN')
        hidden = self.token_mlp(features).masked_fill(~mask[:, :, None], 0.)
        attended, _ = self.attention(hidden, hidden, hidden, key_padding_mask=~mask, need_weights=False)
        hidden = self.norm(hidden + attended).masked_fill(~mask[:, :, None], 0.)
        hidden = F.pad(hidden, (0, 0, 0, self.config.max_tokens - active_tokens))
        raw = self.output(hidden.flatten(start_dim=1))
        if not torch.isfinite(raw).all():
            raise ValueError('TOKEN_QPROBE_NONFINITE_OUTPUT')
        return dict(raw_outputs=raw, binary_logit=raw[:, 0], mixed_probability=raw[:, 0].sigmoid(),
                    predicted_statistic=raw[:, 1].sigmoid() * self.config.statistic_bound)


def token_binary_loss(prediction, counts, config):
    counts = _integer_vector(counts, 'counts', len(prediction['binary_logit'])).to(prediction['binary_logit'].device)
    if bool(((counts < 0) | (counts > 5)).any()):
        raise ValueError('TOKEN_QPROBE_K_OUTSIDE_ZERO_TO_FIVE')
    variance = counts.float() * (5-counts.float()) / 20
    target = variance.sqrt() if config.target_statistic == 'sample_std' else variance
    truth = ((counts > 0) & (counts < 5)).float()
    bce = F.binary_cross_entropy_with_logits(prediction['binary_logit'], truth, reduction='none')
    raw_mse = (prediction['predicted_statistic'] - target).square()
    normalized_mse = ((prediction['predicted_statistic'] - target) / config.statistic_bound).square()
    auxiliary = config.auxiliary_weight * normalized_mse
    loss = (bce + auxiliary).mean()
    if not torch.isfinite(loss):
        raise ValueError('TOKEN_QPROBE_NONFINITE_LOSS')
    return loss, dict(binary_bce=bce.mean(), regression_mse_raw=raw_mse.mean(),
                      regression_mse_normalized=normalized_mse.mean(), regression_contribution=auxiliary.mean())


class TokenBinaryWarmup:
    """One fresh head/Adam fit, with a private shuffle RNG and scalar histories."""
    def __init__(self, config, device='cpu'):
        self.config = config if isinstance(config, TokenBinaryQProbeConfig) else TokenBinaryQProbeConfig(**config)
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.model = TokenBinaryQuestionProbe(self.config)
        self.model.to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.rng = torch.Generator(device='cpu').manual_seed(self.config.seed + 1)
        self.update_count = 0
        self.fitted = False

    def _adam_steps(self):
        steps = [int(state['step'].item()) for state in self.optimizer.state.values() if 'step' in state]
        if steps and len(set(steps)) != 1:
            raise ValueError('TOKEN_QPROBE_INCONSISTENT_ADAM_STEPS')
        return steps[0] if steps else 0

    def state_dict(self):
        return copy.deepcopy(dict(schema_version=1, config=asdict(self.config), model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(), private_rng=self.rng.get_state(),
            update_count=self.update_count, fitted=self.fitted, loss_identity=self.config.loss_identity))

    def _batch(self, source, index, lengths, ids):
        actual_lengths = lengths[index]
        local_max = int(actual_lengths.max())
        if hasattr(source, 'get_batch'):
            batch = source.get_batch([ids[i] for i in index.tolist()], device=self.device)
            features = batch['features']
            if 'attention_mask' in batch:
                expected = torch.arange(features.shape[1], device=features.device)[None, :] < actual_lengths.to(features.device)[:, None]
                if not torch.equal(batch['attention_mask'].bool(), expected):
                    raise ValueError('TOKEN_QPROBE_CACHE_IS_NOT_EXACT_RIGHT_PADDING')
        elif isinstance(source, torch.Tensor):
            if source.ndim != 3 or source.shape[1] > self.config.max_tokens:
                raise ValueError('TOKEN_QPROBE_FEATURE_SHAPE_OR_GLOBAL_BMAX')
            features = source[:, :local_max].index_select(0, index.to(source.device))
        else:
            rows = [torch.as_tensor(source[i]) for i in index.tolist()]
            if any(row.ndim != 2 or row.shape != (int(length), self.config.feature_dim)
                   for row, length in zip(rows, actual_lengths)):
                raise ValueError('TOKEN_QPROBE_UNPADDED_DATASET_ROW_SHAPE')
            features = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        if features.shape[1] > self.config.max_tokens or features.shape[1] < local_max:
            raise ValueError('TOKEN_QPROBE_FEATURE_SHAPE_OR_GLOBAL_BMAX')
        return features[:, :local_max].detach().to(device=self.device, dtype=torch.float32), actual_lengths.to(self.device)

    def _summary(self, source, lengths, counts, ids, epoch):
        prior_mode = self.model.training
        probabilities, predicted_statistic = [], []
        sums = dict(total_loss=0., binary_bce=0., regression_mse_raw=0., regression_mse_normalized=0., regression_contribution=0.)
        try:
            self.model.eval()
            with torch.no_grad():
                for index in torch.arange(len(ids)).split(self.config.batch_size):
                    features, local_lengths = self._batch(source, index, lengths, ids)
                    prediction = self.model(features, local_lengths)
                    loss, parts = token_binary_loss(prediction, counts[index], self.config)
                    for key, value in dict(total_loss=loss, **parts).items():
                        sums[key] += float(value) * len(index)
                    probabilities.append(prediction['mixed_probability'].detach().cpu())
                    predicted_statistic.append(prediction['predicted_statistic'].detach().cpu())
        finally:
            self.model.train(prior_mode)
        probability = torch.cat(probabilities); statistic = torch.cat(predicted_statistic)
        truth = (counts > 0) & (counts < 5); positive = probability >= .5
        tp, tn, fp, fn = [int(value.sum()) for value in
                         (truth & positive, ~truth & ~positive, ~truth & positive, truth & ~positive)]
        ratio = lambda n, d: n/d if d else 0.
        recall, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
        recalls = ([recall] if tp+fn else []) + ([specificity] if tn+fp else [])
        variance = counts.float() * (5-counts.float()) / 20
        std = variance.sqrt()
        predicted_std = statistic if self.config.target_statistic == 'sample_std' else statistic.sqrt()
        predicted_variance = statistic.square() if self.config.target_statistic == 'sample_std' else statistic
        # Same epoch predictions supply TopK; neither regression nor row order breaks a tie.
        order = sorted(range(len(ids)), key=lambda i: (-float(probability[i]), hashlib.sha256(ids[i].encode()).hexdigest()))
        selected = torch.tensor(order[:min(self.config.top_k, len(ids))], dtype=torch.long)
        summary = dict(epoch=epoch, training_groups=len(ids), dataset_scope='all_warmup_training_groups_not_independent_holdout',
            **{key: value/len(ids) for key, value in sums.items()},
            target_statistic=self.config.target_statistic, regression_bound=self.config.statistic_bound,
            auxiliary_weight=self.config.auxiliary_weight, loss_identity=self.config.loss_identity,
            actual_head_adam_updates=self._adam_steps(), batch_size=self.config.batch_size,
            learning_rate=float(self.optimizer.param_groups[0]['lr']),
            binary_confusion=dict(true_positive=tp,true_negative=tn,false_positive=fp,false_negative=fn),
            threshold=.5, accuracy=ratio(tp+tn,len(ids)), balanced_accuracy=sum(recalls)/len(recalls),
            balanced_accuracy_definition='mean recall over classes present in training data',
            precision=ratio(tp,tp+fp), recall=recall, specificity=specificity, f1=ratio(2*tp,2*tp+fp+fn),
            true_mixed_fraction=float(truth.float().mean()), predicted_mixed_fraction=float(positive.float().mean()),
            threshold_tie_count=int((probability == .5).sum()), p_mixed_mean=float(probability.mean()),
            p_mixed_min=float(probability.min()), p_mixed_max=float(probability.max()),
            sample_std_mae=float((predicted_std-std).abs().mean()),
            reported_std_origin='direct_bounded_sample_std_output' if self.config.target_statistic == 'sample_std' else 'sqrt_of_bounded_sample_variance_output',
            predicted_sample_variance_mean=float(predicted_variance.mean()),
            top_k=dict(requested_k=self.config.top_k, selected_count=len(selected), selection_metric='binary_p_mixed',
                tie_break='sha256(question_id)', mixed_precision=float(truth[selected].float().mean()),
                true_sample_std_mean=float(std[selected].mean()), true_sample_variance_mean=float(variance[selected].mean()),
                predicted_mixed_probability_mean=float(probability[selected].mean()),
                selected_question_ids_sha256=hashlib.sha256(json.dumps([ids[i] for i in selected.tolist()],
                    separators=(',', ':'),ensure_ascii=False).encode()).hexdigest()))
        summary[self.config.target_statistic + '_mse_raw'] = summary['regression_mse_raw']
        summary[self.config.target_statistic + '_mse_normalized'] = summary['regression_mse_normalized']
        return summary

    def fit(self, token_features, lengths, counts, ids, *, progress_callback=None):
        if self.fitted or self.update_count or self.optimizer.state:
            raise ValueError('TOKEN_QPROBE_WARMUP_REQUIRES_FRESH_HEAD_AND_ADAM')
        ids = list(ids); size = len(ids)
        if not size or any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != size:
            raise ValueError('TOKEN_QPROBE_REQUIRES_UNIQUE_QUESTION_IDS')
        lengths = _integer_vector(lengths, 'lengths', size).cpu()
        counts = _integer_vector(counts, 'counts', size).cpu()
        if bool(((lengths < 1) | (lengths > self.config.max_tokens)).any()):
            raise ValueError('TOKEN_QPROBE_LENGTH_EXCEEDS_GLOBAL_BMAX')
        if bool(((counts < 0) | (counts > 5)).any()):
            raise ValueError('TOKEN_QPROBE_K_OUTSIDE_ZERO_TO_FIVE')
        if not hasattr(token_features, 'get_batch') and len(token_features) != size:
            raise ValueError('TOKEN_QPROBE_FEATURE_COUNT_MISMATCH')
        prior_mode = self.model.training
        epoch_history, minibatch_history = [], []
        def publish(stage, row):
            if progress_callback is not None:
                progress_callback(dict(stage=stage, **copy.deepcopy(row)))
        try:
            self.model.train()
            epoch_history.append(self._summary(token_features,lengths,counts,ids,0))
            publish('epoch_training_summary', epoch_history[-1])
            for epoch in range(1, self.config.epochs+1):
                order = torch.randperm(size, generator=self.rng)
                for batch_index, index in enumerate(order.split(self.config.batch_size)):
                    features, local_lengths = self._batch(token_features,index,lengths,ids)
                    prediction = self.model(features,local_lengths)
                    loss, parts = token_binary_loss(prediction,counts[index],self.config)
                    row = dict(epoch=epoch,minibatch_index=batch_index,batch_size=len(index),
                        head_update_ordinal=self.update_count+1,head_adam_steps_before=self._adam_steps(),
                        learning_rate=float(self.optimizer.param_groups[0]['lr']),pre_step_loss=float(loss.detach()),
                        **{key:float(value.detach()) for key,value in parts.items()})
                    self.optimizer.zero_grad(set_to_none=True);loss.backward()
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()):
                        raise ValueError('TOKEN_QPROBE_NONFINITE_GRADIENT')
                    self.optimizer.step();self.update_count+=1
                    if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
                        raise ValueError('TOKEN_QPROBE_NONFINITE_PARAMETERS')
                    row['head_adam_steps_after']=self._adam_steps()
                    minibatch_history.append(row);publish('head_optimizer_update',row)
                epoch_history.append(self._summary(token_features,lengths,counts,ids,epoch))
                publish('epoch_training_summary',epoch_history[-1])
            self.fitted=True
            return dict(schema_version=1,role='token_binary_question_head_training_warmup',
                config=asdict(self.config),training_groups=size,fit_epochs=self.config.epochs,
                actual_head_adam_updates=self._adam_steps(),
                expected_head_adam_updates=self.config.epochs*math.ceil(size/self.config.batch_size),
                epoch_history=epoch_history,minibatch_history=minibatch_history,
                training_summary=epoch_history[-1],dataset_scope='training_only_not_independent_holdout',
                actor_updated=False,reference_updated=False,formal_training_started=False)
        finally:
            self.model.train(prior_mode)
