"""Opt-in, detached GRPO/GiGPO runtime evidence. No collectives or model gathers."""
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

import torch


def enabled():
    return bool(os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR'))


def runtime_env_vars():
    if not enabled():
        return {}
    names = ('AUDIT_DIR', 'RUN_ID', 'COMMIT', 'CONFIG_SHA256', 'DATA_IDENTITY')
    result = {f'SEARCH_R1_BASELINE_{name}': os.environ[f'SEARCH_R1_BASELINE_{name}']
              for name in names if f'SEARCH_R1_BASELINE_{name}' in os.environ}
    for name in ('RBA_H100_SHARED_EXECUTION_ID','CPATH','TRITON_LIBCUDA_PATH','LIBRARY_PATH','SLURM_JOB_ID', 'SEARCH_R1_MODEL_IDENTITY', 'SEARCH_R1_BASELINE_OUTPUT',
                 'TREE_COMMON_DATA', 'TREE_INSTRUCT_MODEL', 'SEARCH_R1_EXECUTION_BACKEND',
                 'SEARCH_R1_TREE_RANK_AUDIT_DIR','SEARCH_R1_TREE_AUDIT_DIR','SEARCH_R1_TRAJECTORY_DIR',
                 'RECURSIVE_A_CONTINUATION_BINDING','RECURSIVE_A_CONTINUATION_SHA256','RECURSIVE_A_H100_FRESH_PROFILE','RECURSIVE_A_SPARSE_DETAILS','RECURSIVE_A_OMIT_FULL_BETA','RECURSIVE_A_GUARD_ROUND_PROGRESS','TREE_LINEAR_PREFIX_SCAN','TREE_LIGHTWEIGHT_CANDIDATE_REWARD'):
        if name in os.environ:
            result[name] = os.environ[name]
    resource = 'SEARCH_R1_BASELINE_TEACHER_CPU_THREADS'
    if resource in os.environ:
        result[resource] = str(teacher_cpu_threads())
    return result


def config_boundary(config, split_manifest):
    """Verify the actual post-holdout entry against the reviewed resolved config."""
    if not enabled():
        return
    from omegaconf import OmegaConf
    text = OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(config, resolve=True)))
    actual = hashlib.sha256(text.encode()).hexdigest()
    identity = split_manifest['content_identity']['sha256']
    if (actual != os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256')
            or identity != os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY')):
        raise ValueError('Actual baseline entry config or split differs from reviewed audit identity')
    root = Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'entry-config.yaml').open('x') as stream:
        stream.write(text)
    emit('config_boundary', dict(actual_config_sha256=actual, actual_data_identity=identity,
        entry_total_training_steps=int(config.trainer.total_training_steps),
        config_scope='resolved Hydra + common holdout before trainer optim.total_training_steps injection'), 'entry')


def driver_outer(batch, outer_step, method):
    if not enabled() or method not in ('grpo', 'gigpo'):
        return
    width = batch.batch['responses'].shape[1]
    mask = batch.batch['loss_mask'][:, -width:]
    tensors = {key: stats(batch.batch[key], mask) for key in
        ('old_log_probs', 'advantages', 'policy_weights', 'token_level_rewards', 'returns')
        if key in batch.batch.keys()}
    emit('driver_outer', dict(outer_step=outer_step, method=method, rows=len(batch),
        questions=len(set(batch.non_tensor_batch['uid'])), tensors=tensors,
        action_tokens=int(mask.sum()), observation_tokens=int((
            batch.batch['attention_mask'][:, -width:] * (1 - batch.batch['info_mask'][:, -width:])).sum())))


def teacher_cpu_threads():
    """Optional driver-only thread budget; actor and retrieval budgets are separate."""
    value = os.environ.get('SEARCH_R1_BASELINE_TEACHER_CPU_THREADS')
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 16:
        raise ValueError('TOP-D teacher CPU threads must be an integer in 1..16')
    return int(value)


def configure_teacher_cpu_threads():
    count = teacher_cpu_threads()
    if count is not None:
        # Called only in the TOP-D driver, before loading its frozen CPU teacher.
        torch.set_num_threads(count)
    return torch.get_num_threads()


def emit(kind, record, rank='driver'):
    directory = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record = dict(record, kind=kind, rank=rank, pid=os.getpid(),
                  run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID', root.name),
                  commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
                  config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
                  data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'))
    # Rank files have one writer; driver records never contain reward_model/gold.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    with (root / f'{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(clean(record), allow_nan=False, sort_keys=True) + '\n')


def tensor_hash(tensor):
    value = tensor.detach()
    flat = value.view(-1) if value.is_contiguous() else None
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    # <=8 MiB CPU chunks, including FP32 teacher tensors; no full model copy.
    chunk = max(1, (8 * 1024 * 1024) // tensor.element_size())
    for start in range(0, value.numel(), chunk):
        if flat is not None:
            part = flat[start:start + chunk]
        else:
            # Avoid reshape copying a whole noncontiguous teacher parameter.
            indices = torch.arange(start, min(start + chunk, value.numel()), device=value.device)
            part = value[torch.unravel_index(indices, value.shape)]
        raw = part.to('cpu').contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def stats(tensor, mask=None):
    value = tensor.detach()
    if mask is not None:
        value = value[mask.bool()]
    finite = torch.isfinite(value)
    good = value[finite]
    return dict(count=value.numel(), finite=int(finite.sum()),
                nonzero=int((good != 0).sum()),
                minimum=float(good.min()) if good.numel() else None,
                maximum=float(good.max()) if good.numel() else None)


def fixed_hashes(batch):
    return {key: tensor_hash(batch[key]) for key in ('old_log_probs', 'advantages', 'policy_weights', 'step_ids', 'loss_mask') if key in batch.keys()}


def shard_sample(model, maximum=8192):
    """Sample evenly spaced local coordinates, never summon full FSDP parameters."""
    params = [(name, param.detach()) for name, param in model.named_parameters() if param.numel()]
    total = sum(param.numel() for _, param in params)
    count = min(maximum, total)
    coordinates = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long().tolist() if count else []
    values, layout = [], []
    offset = 0
    for name, param in params:
        indices = [i - offset for i in coordinates if offset <= i < offset + param.numel()]
        if indices:
            index = torch.tensor(indices, device=param.device)
            values.append(param.reshape(-1).index_select(0, index).float().cpu())
            layout.append(dict(name=name, local_shape=list(param.shape), indices=indices))
        offset += param.numel()
    return layout, torch.cat(values) if values else torch.empty(0)


class ActorAudit:
    def __init__(self, model, batch, meta, epochs):
        self.model, self.batch = model, batch
        self.step = int(meta['baseline_outer_step'])
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.before = fixed_hashes(batch)
        self.layout, self.values = shard_sample(model)
        self.rows, self.epochs = len(batch), epochs
        self.grads, self.epoch_hashes, self.micro_stats = [], [], []

    def epoch(self):
        self.epoch_hashes.append(fixed_hashes(self.batch))

    def micro(self, logp, old, mask, loss, step_ids=None, advantages=None, loss_parts=None, ref_logp=None):
        valid = mask.bool()
        self.micro_stats.append(dict(current_log_probs=stats(logp, valid),
                                     ratios=stats(torch.exp(logp.detach()[valid] - old.detach()[valid])),
                                     policy_loss=stats(loss)))
        if step_ids is not None:
            ratios=[]
            deltas=[]
            for row in range(len(logp)):
                for action in torch.unique(step_ids[row][valid[row]]):
                    chosen=valid[row] & (step_ids[row] == action)
                    delta=(logp.detach()[row][chosen]-old.detach()[row][chosen]).sum()
                    deltas.append(delta)
                    ratios.append(delta.exp())
            self.micro_stats[-1]['joint_action_ratios']=stats(torch.stack(ratios) if ratios else logp.new_empty(0))
        if not enabled():
            return
        record=dict(outer_step=self.step, micro_index=len(self.micro_stats),
                    optimizer_steps_completed=len(self.grads), statistics=self.micro_stats[-1])
        if step_ids is not None:
            record['joint_action_logratios']=stats(torch.stack(deltas) if deltas else logp.new_empty(0))
        if advantages is not None:
            record['advantages']=stats(advantages,valid)
            record['advantage_sign_counts']={name:int((test & valid).sum()) for name,test in
                [('positive',advantages>0),('negative',advantages<0),('zero',advantages==0)]}
        if ref_logp is not None:
            record['reference_minus_current']=stats(ref_logp.detach()-logp.detach(),valid)
        record['loss_parts']={name:stats(value.detach()) for name,value in (loss_parts or {}).items()}
        directory=Path(os.environ['SEARCH_R1_BASELINE_AUDIT_DIR'])/'numeric'
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'micro-{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')

    def gradient(self, norm):
        value = float(norm.detach())
        self.grads.append(value if math.isfinite(value) else None)

    def finish(self, updates, skips):
        layout, values = shard_sample(self.model)
        compatible = layout == self.layout
        change = stats(values - self.values) if compatible else None
        after = fixed_hashes(self.batch)
        mask = self.batch.get('loss_mask', None)
        emit('actor_outer', dict(outer_step=self.step, rows=self.rows, epochs=self.epochs,
             optimizer_steps=updates, skipped_empty_minibatches=skips,
             micro_stats=self.micro_stats, grad_norm_scope='existing_optimizer_norm_FSDP_global',
             grad_norms=self.grads, grad_finite=sum(v is not None for v in self.grads),
             grad_nonzero=sum(v is not None and v != 0 for v in self.grads),
             fixed_before=self.before, fixed_epochs=self.epoch_hashes, fixed_after=after,
             fixed_unchanged=self.before == after and all(v == self.before for v in self.epoch_hashes),
             shard_layout=layout, shard_layout_unchanged=compatible, shard_delta=change,
             shard_sample_before_sha256=tensor_hash(self.values),
             shard_sample_after_sha256=tensor_hash(values),
             shard_sample_before=stats(self.values), shard_sample_after=stats(values),
             mask=stats(mask) if mask is not None else None,
             mask_binary=bool(((mask == 0) | (mask == 1)).all()) if mask is not None else None,
             tensors={key: stats(self.batch[key], mask) for key in self.before}), self.rank)
