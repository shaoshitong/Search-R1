"""Three full-copy Qwen decoder blocks over frozen question token features.

This module is the sequence probe core, not a replacement for the old pooled
cache/online fitting protocol. Integration must explicitly bind sequence data.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import torch
from torch import nn
from search_g1.qprobe_model import binary_predictions


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_full_copy(source, copies):
    original = source.state_dict()
    owned = set()
    for block in copies:
        state = block.state_dict()
        if state.keys() != original.keys():
            raise ValueError('QPROBE_QWEN_COPY_KEYS_DIFFER')
        for name, value in state.items():
            expected = original[name]
            if value.dtype != expected.dtype or value.shape != expected.shape or not torch.equal(value, expected):
                raise ValueError('QPROBE_QWEN_COPY_TENSOR_DIFFERS:' + name)
            if value.numel():
                pointer = (value.device, value.untyped_storage().data_ptr())
                source_pointer = (expected.device, expected.untyped_storage().data_ptr())
                if pointer == source_pointer or pointer in owned:
                    raise ValueError('QPROBE_QWEN_COPY_SHARED_STORAGE:' + name)
                owned.add(pointer)
    return {'full_state_sha256': state_digest(source),
            'tensor_count_per_block': len(original), 'independent_copies': len(copies)}


class QwenBlockBinaryProbe(nn.Module):
    """Official final-layer features -> three copied blocks -> binary mixed/variance projections."""
    def __init__(self, last_decoder_layer, final_norm, qwen_config):
        super().__init__()
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        if not isinstance(last_decoder_layer, Qwen2DecoderLayer):
            raise TypeError('QPROBE_REQUIRES_ACTUAL_QWEN2_DECODER_LAYER')
        if getattr(qwen_config, 'model_type', None) != 'qwen2':
            raise ValueError('QPROBE_REQUIRES_QWEN2_CONFIG')
        self.feature_dim = qwen_config.hidden_size
        # deepcopy retains all parameters, buffers and exact attention/MLP class.
        self.blocks = nn.ModuleList([copy.deepcopy(last_decoder_layer) for _ in range(3)])
        self.initialization = verify_full_copy(last_decoder_layer, self.blocks)
        self.final_norm = copy.deepcopy(final_norm)
        dtype = next(last_decoder_layer.parameters()).dtype
        device = next(last_decoder_layer.parameters()).device
        self.rotary = Qwen2RotaryEmbedding(config=copy.deepcopy(qwen_config), device=device)
        self.head = nn.Linear(self.feature_dim, 2, bias=True, device=device, dtype=dtype)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        for parameter in self.parameters(): parameter.requires_grad_(True)
        self.config = copy.deepcopy(qwen_config)

    def forward(self, frozen_hidden, attention_mask, position_ids):
        if frozen_hidden.ndim != 3 or frozen_hidden.shape[-1] != self.feature_dim:
            raise ValueError('QPROBE_REQUIRES_FULL_TOKEN_FEATURES')
        if attention_mask.shape != frozen_hidden.shape[:2] or position_ids.shape != attention_mask.shape:
            raise ValueError('QPROBE_SEQUENCE_MASK_POSITION_SHAPE')
        if position_ids.dtype != torch.long or bool((position_ids < 0).any()):
            raise ValueError('QPROBE_INVALID_ORIGINAL_POSITIONS')
        if not torch.isfinite(frozen_hidden).all() or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('QPROBE_INVALID_SEQUENCE_FEATURES')
        if bool((attention_mask.sum(-1) == 0).any()):
            raise ValueError('QPROBE_EMPTY_QUESTION_SEQUENCE')
        parameter = next(self.blocks.parameters())
        hidden = frozen_hidden.detach().to(device=parameter.device, dtype=parameter.dtype)
        valid = attention_mask.to(device=parameter.device, dtype=torch.bool)
        positions = position_ids.to(parameter.device)
        length = hidden.shape[1]
        causal = torch.arange(length, device=parameter.device)
        allowed = (causal[:, None] >= causal[None, :])[None, None] & valid[:, None, None, :]
        mask = torch.zeros(allowed.shape, dtype=hidden.dtype, device=hidden.device)
        mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        rotary = self.rotary(hidden, positions)
        for block in self.blocks:
            block_mask = valid if type(block.self_attn).__name__ == 'Qwen2FlashAttention2' else mask
            kwargs = dict(attention_mask=block_mask, position_ids=positions, use_cache=False, output_attentions=False)
            if 'position_embeddings' in inspect.signature(block.forward).parameters:
                kwargs['position_embeddings'] = rotary
            hidden = block(hidden, **kwargs)[0]
        hidden = self.final_norm(hidden)
        last = causal.expand(valid.shape).masked_fill(~valid, -1).max(-1).values
        logits = self.head(hidden[torch.arange(len(hidden), device=hidden.device), last]).float()
        return binary_predictions(logits[:,0],logits[:,1])


def load_official_initialization(path, *, expected_file_sha256, expected_encoder_files_sha256):
    """Construct all three blocks from verified frozen encoder source tensors."""
    from pathlib import Path
    from search_g1.qprobe_features import OFFICIAL_REVISION, sha_file
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
    if sha_file(path) != expected_file_sha256:raise ValueError('QPROBE_INITIALIZATION_FILE_SHA')
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    initializer=(state.get('schema_version'),state.get('architecture'))
    allowed={'schema_version','architecture','official_revision','encoder_files_sha256','qwen_config',
        'attention_implementation','source_layer_index','source_layer_state','source_layer_sha256',
        'source_norm_state','source_norm_sha256'}
    if set(state)!=allowed or initializer not in ((2,'qwen2_decoder3_count6'),(3,'official_qwen2_last_layer_source_v1')):
        raise ValueError('QPROBE_INITIALIZATION_ARCHITECTURE')
    if state.get('official_revision')!=OFFICIAL_REVISION or state.get('encoder_files_sha256')!=expected_encoder_files_sha256:raise ValueError('QPROBE_INITIALIZATION_OFFICIAL_SOURCE')
    config=Qwen2Config.from_dict(copy.deepcopy(state['qwen_config']))
    config._attn_implementation=state['attention_implementation']
    if state['source_layer_index']!=config.num_hidden_layers-1:raise ValueError('QPROBE_INITIALIZATION_NOT_FINAL_LAYER')
    dtype=next(iter(state['source_layer_state'].values())).dtype
    # State is the complete actual layer captured during official feature
    # preparation. Random constructor values are all overwritten strictly.
    with torch.random.fork_rng(devices=[]):
        source=Qwen2DecoderLayer(config,layer_idx=state['source_layer_index']).to(dtype=dtype)
        norm=Qwen2RMSNorm(config.hidden_size,eps=config.rms_norm_eps).to(dtype=dtype)
        source.load_state_dict(state['source_layer_state'],strict=True)
        norm.load_state_dict(state['source_norm_state'],strict=True)
        if state_digest(source)!=state['source_layer_sha256'] or state_digest(norm)!=state['source_norm_sha256']:raise ValueError('QPROBE_INITIALIZATION_SOURCE_TENSOR_SHA')
        source.requires_grad_(False);norm.requires_grad_(False)
        result=QwenBlockBinaryProbe(source,norm,config)
    result.initialization.update(fresh_binary_head=True,initializer_schema=state['schema_version'],
        initializer_architecture=state['architecture'],probe_architecture='qwen2_decoder3_binary_mix_v1',
        official_revision=state['official_revision'],
        encoder_files_sha256=state['encoder_files_sha256'],source_layer_index=state['source_layer_index'],
        initialization_file_sha256=expected_file_sha256)
    return result
