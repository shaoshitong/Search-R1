"""Inherited E9 group-variance probe on the fixed official Qwen encoder.

The actor supplies only sampled tokens.  Its changing hidden states never enter
this module.  Features stay in device memory and are discarded after backward.
"""
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import inspect
import json
import random
from pathlib import Path

import torch
from torch import nn

from .probe_model.packed_variance import PackedVarianceHead, PackedVarianceProbe

OFFICIAL_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
WARMUP_SHA256 = '5f90fb17bee993353f47a7d687bdf2ea235dcd3d2eba287465567f4c286b13d7'


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state(device):
    import numpy as np
    result = dict(python=random.getstate(), numpy=np.random.get_state(),
                  torch=torch.get_rng_state())
    if torch.device(device).type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(value, device):
    import numpy as np
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value.get('torch', value.get('torch_cpu')).cpu())
    if torch.device(device).type == 'cuda':
        state = value.get('cuda', value.get('torch_cuda'))
        if state is not None:
            torch.cuda.set_rng_state(state.cpu(), device)


@contextmanager
def probe_rng(adapter):
    """Probe shuffle/initialization must not advance the actor's RNG streams."""
    previous = rng_state(adapter.device)
    try:
        restore_rng(adapter.rng, adapter.device)
        yield
    finally:
        adapter.rng = rng_state(adapter.device)
        restore_rng(previous, adapter.device)


def pack_group(group, separator, prefix, pad_token=151643):
    """Warmup packing: question once, five own policy prefixes, no EM access."""
    prompt = group.get('prompt_token_ids', group.get('initial_prompt_token_ids'))
    trajectories = group['policy_token_ids']
    if not prompt or len(trajectories) != 5:
        raise ValueError('Probe input requires one nonempty prompt and five own trajectories')
    ids = list(prompt)
    mask, types = [True] * len(ids), [0] * len(ids)
    for trajectory in trajectories:
        policy = list(trajectory[:prefix])
        ids += list(separator) + policy + [pad_token] * (prefix - len(policy))
        mask += [True] * len(separator) + [True] * len(policy) + [False] * (prefix - len(policy))
        types += [2] * len(separator) + [1] * prefix
    positions, used = [], 0
    for valid in mask:
        positions.append(used if valid else 0)
        used += int(valid)
    return dict(input_ids=ids, attention_mask=mask, position_ids=positions,
                token_types=types, valid_positions=used)


def target4variance(strict_em):
    if len(strict_em) != 5 or any(value not in (0, 1) for value in strict_em):
        raise ValueError('Exactly five binary strict-EM labels are required')
    mean = sum(strict_em) / 5
    return 4 * mean * (1 - mean)


class GroupProbeAdapter:
    def __init__(self, encoder, probe, ema, optimizer, separator, device, state, *, rank=0):
        self.encoder, self.probe, self.ema = encoder, probe, ema
        self.optimizer, self.separator = optimizer, list(separator)
        self.device = torch.device(device)
        self.rank = rank
        self.binding = copy.deepcopy(state['binding'])
        self.warmup_cursor = state.get('warmup_cursor', state['cursor'])
        self.online_updates = state.get('online_updates', 0)
        self.EMA_updates = state.get('EMA_updates', self.warmup_cursor)
        self.completed_update_outer = state.get('completed_update_outer', 0)
        self.pending_update = copy.deepcopy(state.get('pending_update'))
        self.last_update = copy.deepcopy(state.get('last_update'))
        self.current_prefix = state.get('current_prefix', 80)
        self.source_checkpoint = copy.deepcopy(state.get('source_checkpoint'))
        self.source_RNG_by_rank = cpu_tree(state.get('source_RNG_by_rank', state['RNG_by_rank']))
        self.RNG_expansion = copy.deepcopy(state.get('RNG_expansion', {}))
        sources = state['RNG_by_rank']
        world = (torch.distributed.get_world_size() if torch.distributed.is_initialized()
                 else max(len(sources), rank + 1))
        for new_rank in range(len(sources), world):
            self.RNG_expansion[str(new_rank)] = dict(source_rank=new_rank % len(sources),
                draws_advanced=(new_rank // len(sources)) * 4096)
        self.rng = cpu_tree(sources[rank % len(sources)])
        if rank >= len(sources):
            # Preserve the two original streams. New ranks get reproducible
            # advanced branches, explicitly recorded rather than called exact
            # continuation of a nonexistent old rank.
            with probe_rng(self):
                skip = (rank // len(sources)) * 4096
                import numpy as np
                for _ in range(skip):
                    random.random()
                np.random.random(skip)
                torch.rand(skip)
                if self.device.type == 'cuda':
                    torch.rand(skip, device=self.device)
            self.RNG_expansion[str(rank)] = dict(source_rank=rank % len(sources),
                                                  draws_advanced=skip)

    @classmethod
    def from_warmup(cls, model_dir, checkpoint_path, *, checkpoint_sha256=WARMUP_SHA256,
                    device='cuda', rank=0, encoder_attn_implementation='flash_attention_2',
                    attention_installer=None):
        """Load only official layers0..14, not an extra full 28-layer GPU model."""
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_sha256 is not None and file_sha256(checkpoint_path) != checkpoint_sha256:
            raise ValueError('Inherited probe checkpoint SHA256 differs')
        old = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        if (config.model_type, config.hidden_size, config.num_hidden_layers) != ('qwen2', 3584, 28):
            raise ValueError('Official Qwen2.5-7B architecture differs')
        if old['binding'].get('objective') != 'mse' or old['binding'].get('layer') != 15:
            raise ValueError('Expected inherited L15 packed variance-MSE probe')
        config._attn_implementation = encoder_attn_implementation
        encoder_config = copy.deepcopy(config)
        encoder_config.num_hidden_layers = 15
        encoder = AutoModel.from_pretrained(model_dir, config=encoder_config,
            local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
            attn_implementation=encoder_attn_implementation)
        encoder.norm = nn.Identity()
        encoder.requires_grad_(False).eval()
        block = Qwen2DecoderLayer(config, layer_idx=15).float()
        head = PackedVarianceHead(max_positions=old['raw']['group_head.positions.weight'].shape[0])
        rotary = copy.deepcopy(encoder.rotary_emb).float()
        probe = PackedVarianceProbe(block, rotary,
            'cache_position' in inspect.signature(block.forward).parameters, head)
        ema = copy.deepcopy(probe).eval().requires_grad_(False)
        probe.load_state_dict(old['raw'], strict=True)
        ema.load_state_dict(old['EMA'], strict=True)
        optimizer = torch.optim.Adam(probe.parameters(), lr=5e-6)
        optimizer.load_state_dict(old['Adam'])
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        separator = tokenizer.encode('\n### trajectory\n', add_special_tokens=False)
        old.setdefault('source_checkpoint', dict(path=str(checkpoint_path), sha256=checkpoint_sha256))
        adapter = cls(encoder, probe, ema, optimizer, separator, device, old, rank=rank)
        if attention_installer is not None:
            attention_installer(encoder, probe.block, ema.block)
        return adapter

    def activate(self, *, training=False):
        self.encoder.to(self.device)
        self.probe.to(self.device)
        if training:
            self.ema.to(self.device)
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and key != 'step':
                        state[key] = value.to(self.device)

    def offload(self):
        self.encoder.to('cpu')
        self.probe.to('cpu')
        self.ema.to('cpu')
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def ensure_position_capacity(self, required):
        """Extend only if needed; retain old raw/EMA coordinates and Adam step."""
        old = self.probe.group_head.positions
        if required <= old.num_embeddings:
            return
        capacity = ((required + 255) // 256) * 256
        new = nn.Embedding(capacity, old.embedding_dim, device=old.weight.device,
                           dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:old.num_embeddings].copy_(old.weight)
        previous = old.weight
        for group in self.optimizer.param_groups:
            group['params'] = [new.weight if p is previous else p for p in group['params']]
        state = self.optimizer.state.pop(previous, {})
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == previous.shape:
                expanded = value.new_zeros(new.weight.shape)
                expanded[:old.num_embeddings].copy_(value)
                state[key] = expanded
        self.optimizer.state[new.weight] = state
        ema_old = self.ema.group_head.positions
        ema_new = nn.Embedding(capacity, old.embedding_dim, device=ema_old.weight.device,
                              dtype=ema_old.weight.dtype)
        with torch.no_grad():
            ema_new.weight.copy_(new.weight.to(ema_new.weight.device))
            ema_new.weight[:old.num_embeddings].copy_(ema_old.weight)
        ema_new.requires_grad_(False)
        self.probe.group_head.positions = new
        self.ema.group_head.positions = ema_new

    def capture(self, groups, prefix):
        packed = [pack_group(g, self.separator, prefix) for g in groups]
        self.ensure_position_capacity(max(p['valid_positions'] for p in packed))
        width = max(len(p['input_ids']) for p in packed)
        ids = torch.full((len(groups), width), 151643, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        positions, types = torch.zeros_like(ids), torch.zeros_like(ids)
        for i, row in enumerate(packed):
            n = len(row['input_ids'])
            ids[i, :n] = torch.tensor(row['input_ids'], device=self.device)
            mask[i, :n] = torch.tensor(row['attention_mask'], device=self.device)
            positions[i, :n] = torch.tensor(row['position_ids'], device=self.device)
            types[i, :n] = torch.tensor(row['token_types'], device=self.device)
        with torch.no_grad():
            features = self.encoder(input_ids=ids, attention_mask=mask,
                position_ids=positions, use_cache=False, return_dict=True).last_hidden_state.detach()
        return dict(features=features, attention_mask=mask,
                    position_ids=positions, token_types=types)

    def autocast(self):
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if self.device.type == 'cuda' else nullcontext())

    def score_groups(self, groups, *, prefix=80, batch_size=8, state='raw'):
        if state not in ('raw', 'EMA'):
            raise ValueError('Probe inference state is raw or EMA')
        self.activate(training=state == 'EMA')
        model = self.probe if state == 'raw' else self.ema
        model.eval()
        scores = []
        try:
            with torch.no_grad(), self.autocast():
                for start in range(0, len(groups), batch_size):
                    scores.extend(torch.sigmoid(model(self.capture(
                        groups[start:start + batch_size], prefix))).cpu().tolist())
        finally:
            self.offload()
        return scores

    def state_dict(self, rng_by_rank=None):
        rngs = [self.rng] if rng_by_rank is None else rng_by_rank
        return cpu_tree(dict(format='DYNRPO7B_GROUP_VARIANCE_PROBE_V1',
            raw=self.probe.state_dict(), EMA=self.ema.state_dict(),
            Adam=self.optimizer.state_dict(), RNG_by_rank=rngs,
            warmup_cursor=self.warmup_cursor,
            cursor=self.warmup_cursor + self.online_updates,
            actual_Adam_updates=self.warmup_cursor + self.online_updates,
            EMA_updates=self.EMA_updates, online_updates=self.online_updates,
            completed_update_outer=self.completed_update_outer,
            pending_update=self.pending_update, current_prefix=self.current_prefix,
            last_update=self.last_update,
            source_checkpoint=self.source_checkpoint, binding=self.binding,
            source_RNG_by_rank=self.source_RNG_by_rank, RNG_expansion=self.RNG_expansion))

    def load_state_dict(self, saved):
        self.offload()
        previous_rng = rng_state(self.device)
        try:
            self.ensure_position_capacity(saved['raw']['group_head.positions.weight'].shape[0])
        finally:
            restore_rng(previous_rng, self.device)
        self.probe.load_state_dict(saved['raw'], strict=True)
        self.ema.load_state_dict(saved['EMA'], strict=True)
        self.optimizer.load_state_dict(saved['Adam'])
        self.rng = cpu_tree(saved['RNG_by_rank'][self.rank])
        self.warmup_cursor = saved['warmup_cursor']
        self.online_updates, self.EMA_updates = saved['online_updates'], saved['EMA_updates']
        self.completed_update_outer = saved['completed_update_outer']
        self.pending_update = copy.deepcopy(saved.get('pending_update'))
        self.last_update = copy.deepcopy(saved.get('last_update'))
        self.current_prefix = saved['current_prefix']
        self.RNG_expansion = copy.deepcopy(saved.get('RNG_expansion', {}))
