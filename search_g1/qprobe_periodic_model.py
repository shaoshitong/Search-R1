"""Continuous native last-block probe; refits never read student parameters.

Only the *existing* probe state can construct this learner. The changing
student supplies detached question features through a separate provider.
"""
from dataclasses import asdict, fields, replace
import copy
import hashlib
import io

import torch

from search_g1.qprobe_qwen_block_model import (
    QwenBlockQProbeConfig, QwenBlockWarmup, QwenLastBlockQuestionProbe,
    native_implementation_binding,
)

HEAD_KIND = 'qwen_native_periodic_lastblock_binary_v1'
REFIT_OUTERS = tuple(range(0, 200, 25))
SAVE_OUTERS = tuple(range(25, 201, 25))


def core_config(values):
    values = dict(values)
    aliases = dict(head_batch_size='batch_size', head_learning_rate='learning_rate',
                   std_loss_weight='auxiliary_weight', head_seed='seed',
                   bootstrap_head_epochs='epochs', feature_dim='hidden_size')
    for old, new in aliases.items():
        if old in values:
            values[new] = values[old]
    names = {f.name for f in fields(QwenBlockQProbeConfig)}
    return QwenBlockQProbeConfig(**{k: v for k, v in values.items() if k in names})


def restore_engine(state, *, device='cpu', epochs=20):
    """Allocate native modules, then restore ALL probe state; no official copy."""
    if (state.get('role') != 'qwen_last_block_question_head' or
            state.get('schema_version') != 1):
        raise ValueError('PERIODIC_REQUIRES_EXISTING_NATIVE_PROBE')
    config = replace(QwenBlockQProbeConfig(**state['config']), epochs=epochs)
    if state['native_implementation'] != native_implementation_binding(config):
        raise ValueError('PERIODIC_NATIVE_IMPLEMENTATION_CHANGED')
    engine = QwenBlockWarmup.__new__(QwenBlockWarmup)
    engine.config, engine.device = config, torch.device(device)
    # Constructor randomness is isolated and overwritten, including readout.
    # No student state_dict, copy_official_block, or fresh learning state exists.
    with torch.random.fork_rng(devices=[]):
        engine.model = QwenLastBlockQuestionProbe(config)
    engine.model.load_state_dict(state['model'], strict=True)
    engine.model.to(device=engine.device, dtype=torch.float32)
    engine.optimizer = torch.optim.Adam(engine.model.parameters(), lr=config.learning_rate)
    engine.optimizer.load_state_dict(state['optimizer'])
    engine.rng = torch.Generator(device='cpu')
    engine.rng.set_state(state['private_rng'])
    engine.update_count, engine.fitted = state['update_count'], state['fitted']
    engine._initial_block_proof = copy.deepcopy(state['initial_block_proof'])
    engine.initial_readout_proof = copy.deepcopy(state['initial_readout_proof'])
    engine.native_implementation = copy.deepcopy(state['native_implementation'])
    if engine._adam_steps() != engine.update_count or not engine.optimizer.state:
        raise ValueError('PERIODIC_PROBE_ADAM_STATE_MISSING_OR_DIVERGENT')
    return engine


def fit_phase(engine, provider, lengths, counts, ids, phase, checkpoint):
    """Fit this pool only; save actual Adam/RNG every epoch and on exceptions.

    The saved active permutation permits exact continuation after a caught
    mid-epoch error. An external in-flight ledger must refuse unseen Adam steps
    after a SIGKILL, instead of silently calling an older epoch a lossless resume.
    """
    from search_g1.qprobe_token_model import token_binary_loss
    ids = list(ids)
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    counts = torch.as_tensor(counts, dtype=torch.long)
    if (len(ids) != len(set(ids)) or len(counts) != len(ids) or
            len(lengths) != len(ids) or bool(((counts < 0) | (counts > 5)).any())):
        raise ValueError('PERIODIC_REQUIRES_UNIQUE_COMPLETE_G5_GROUPS')
    batches = (len(ids) + engine.config.batch_size - 1) // engine.config.batch_size
    expected = phase['start_steps'] + phase['epoch'] * batches + phase['batch']
    if engine.update_count != expected or engine._adam_steps() != expected:
        raise ValueError('PERIODIC_PHASE_ADAM_COUNTER_MISMATCH')
    previous = engine.model.training
    try:
        engine.model.train()
        while phase['epoch'] < engine.config.epochs:
            if phase.get('order') is None:
                if phase['batch']:
                    raise ValueError('PERIODIC_MISSING_ACTIVE_PERMUTATION')
                phase['order'] = torch.randperm(len(ids), generator=engine.rng)
                checkpoint('permutation', phase)
            order = phase['order']
            if sorted(order.tolist()) != list(range(len(ids))):
                raise ValueError('PERIODIC_INVALID_SAVED_PERMUTATION')
            for batch_number, index in enumerate(order.split(engine.config.batch_size)):
                if batch_number < phase['batch']:
                    continue
                checkpoint('before_step', phase)
                engine.optimizer.zero_grad(set_to_none=True)
                total = 0.
                for micro in index.split(engine.config.micro_batch_size):
                    features, sizes = engine._batch(provider, micro, lengths, ids)
                    prediction = engine.model(features, sizes)
                    loss, _ = token_binary_loss(prediction, counts[micro], engine.config)
                    fraction = len(micro) / len(index)
                    (loss * fraction).backward()
                    total += float(loss.detach()) * fraction
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in engine.model.parameters()):
                    raise ValueError('PERIODIC_NONFINITE_GRADIENT')
                engine.optimizer.step()
                engine.update_count += 1
                phase['batch'] += 1
                if engine._adam_steps() != engine.update_count or any(
                        not torch.isfinite(p).all() for p in engine.model.parameters()):
                    raise ValueError('PERIODIC_NONFINITE_OR_DIVERGENT_ADAM')
                phase['minibatches'].append(dict(epoch=phase['epoch'] + 1,
                    batch=batch_number, loss=total, adam_steps=engine.update_count))
                checkpoint('after_step', phase)
            phase['epoch'] += 1
            phase['batch'], phase['order'] = 0, None
            phase['summaries'].append(engine._summary(provider, lengths, counts, ids, phase['epoch']))
            checkpoint('epoch', phase)
        engine.fitted = True
        checkpoint('complete', phase)
    except BaseException:
        # Save the true live state, never roll model/Adam back to the phase start.
        checkpoint('failure', phase)
        raise
    finally:
        engine.model.train(previous)
    return dict(fit_epochs=engine.config.epochs, groups=len(ids),
                head_adam_steps_before=phase['start_steps'],
                head_adam_steps_after=engine._adam_steps(),
                head_optimizer_updates=engine._adam_steps() - phase['start_steps'],
                training_summary=phase['summaries'][-1], replay_groups=0)


class PeriodicQwenProbe:
    def __init__(self, engine, feature_spec_sha, *, completed_outer=0,
                 actor_snapshot_id=None, phases=None):
        self._engine, self.config = engine, engine.config
        self.model, self.optimizer, self.rng = engine.model, engine.optimizer, engine.rng
        self.device, self.feature_dim = engine.device, engine.config.feature_dim
        self.feature_spec_sha, self.completed_outer = feature_spec_sha, completed_outer
        self.actor_snapshot_id, self.phases = actor_snapshot_id, list(phases or [])
        self.feature_provider, self._version_cache = None, None

    def attach_feature_provider(self, provider):
        self.feature_provider = provider

    def _adam_steps(self):
        return self._engine._adam_steps()

    @property
    def version(self):
        if self._version_cache is None:
            h = hashlib.sha256()
            for name, value in self.model.state_dict().items():
                h.update(name.encode())
                h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            h.update(str(self._adam_steps()).encode())
            self._version_cache = h.hexdigest()
        return self._version_cache

    def predict(self, qids):
        if self.feature_provider is None:
            raise ValueError('PERIODIC_MISSING_CURRENT_STUDENT_PROVIDER')
        batch = self.feature_provider.get_batch(qids, device=self.device)
        prior = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model(batch['features'], batch['lengths'])
            return dict(mixed_probability=result['mixed_probability'].cpu(),
                        predicted_sample_std=result['predicted_statistic'].cpu())
        finally:
            self.model.train(prior)

    def state_dict(self):
        return dict(schema_version=1, head_kind=HEAD_KIND, head=self._engine.state_dict(),
                    feature_spec_sha=self.feature_spec_sha, completed_outer=self.completed_outer,
                    actor_snapshot_id=self.actor_snapshot_id, phases=copy.deepcopy(self.phases))

    def save(self, path, *, actor_snapshot_id):
        from search_g1.training_state import atomic_save
        self.actor_snapshot_id = actor_snapshot_id
        return atomic_save(path, self.state_dict())

    @classmethod
    def load(cls, path, *, expected_actor_snapshot_id=None, expected_feature_spec_sha=None,
             expected_checkpoint_sha=None, feature_provider=None, device='cpu'):
        from search_g1.qprobe_runtime import _sha
        if expected_checkpoint_sha is not None and _sha(path) != expected_checkpoint_sha:
            raise ValueError('PERIODIC_HEAD_SHA_CHANGED')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('head_kind') != HEAD_KIND or saved.get('schema_version') != 1:
            raise ValueError('PERIODIC_HEAD_KIND_CHANGED')
        if (expected_actor_snapshot_id is not None and
                saved['actor_snapshot_id'] != expected_actor_snapshot_id):
            raise ValueError('PERIODIC_HEAD_ACTOR_MISMATCH')
        if (expected_feature_spec_sha is not None and
                saved['feature_spec_sha'] != expected_feature_spec_sha):
            raise ValueError('PERIODIC_HEAD_FEATURE_SCHEMA_CHANGED')
        result = cls(restore_engine(saved['head'], device=device), saved['feature_spec_sha'],
                     completed_outer=saved['completed_outer'], actor_snapshot_id=saved['actor_snapshot_id'],
                     phases=saved['phases'])
        result.attach_feature_provider(feature_provider)
        return result
