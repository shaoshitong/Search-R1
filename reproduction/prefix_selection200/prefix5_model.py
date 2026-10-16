"""The official native Qwen block35 and exact question+policy5 flattened head.

This changes the reference's retained-token count, not its transformer design.
Initialization imports an explicitly qualified complete question5 warmup state;
its optimizer, EMA clock, and sampler are never reused by online actor training.
"""
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from .contract import QMAX, PREFIX_TOKENS, WIDTH, DIM
from reproduction.prefix20_online.model import validate_prefix_positions

OFFICIAL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
WARMUP_RUN = '8af926a49630428097b94027864da68b'
EPOCH3_SHA = '4fccd80f9c0761e0b613f8fbf45f0b562c08ec9d007fa868aa746bc49b4f4437'
PARAMETER_SHAPES = {
    'block.self_attn.q_proj.weight': (2048, 2048),
    'block.self_attn.q_proj.bias': (2048,),
    'block.self_attn.k_proj.weight': (256, 2048),
    'block.self_attn.k_proj.bias': (256,),
    'block.self_attn.v_proj.weight': (256, 2048),
    'block.self_attn.v_proj.bias': (256,),
    'block.self_attn.o_proj.weight': (2048, 2048),
    'block.mlp.gate_proj.weight': (11008, 2048),
    'block.mlp.up_proj.weight': (11008, 2048),
    'block.mlp.down_proj.weight': (2048, 11008),
    'block.input_layernorm.weight': (2048,),
    'block.post_attention_layernorm.weight': (2048,),
    'output.weight': (1, WIDTH * DIM),
    'output.bias': (1,),
}
BUFFER_SHAPES = {'rotary.inv_freq': (64,)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def validate_model_state(state):
    if not isinstance(state, dict) or set(state) != set(PARAMETER_SHAPES) or len(state) != 14:
        raise ValueError('Require exactly the fourteen native block35/Linear learned tensors')
    for name, shape in PARAMETER_SHAPES.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Warmup learned tensor shape/FP32/finite CPU identity differs: '+name)


def validate_native_buffers(state):
    if not isinstance(state, dict) or set(state) != set(BUFFER_SHAPES):
        raise ValueError('Require all native nonpersistent RoPE buffers explicitly')
    for name, shape in BUFFER_SHAPES.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Warmup native buffer shape/FP32/finite CPU identity differs: '+name)


def warmup_model_weights(source, *, epoch, model_kind='ema', source_run=WARMUP_RUN):
    """Strict complete-state extraction; a historical hash is never a model."""
    if type(epoch) is not int or not 1 <= epoch <= 200 or model_kind not in ('ema', 'raw'):
        raise ValueError('Require an explicit complete warmup epoch and ema/raw weight kind')
    binding, counts = source.get('binding', {}), source.get('counters', {})
    head = binding.get('head_copy', {})
    if (source.get('role') != 'fresh_question5_joint200_ema099'
            or binding.get('role') != 'fresh_question5_joint200_ema099'
            or binding.get('run_id') != source_run
            or source.get('epoch') != epoch or source.get('completed_epoch') is not True
            or source.get('next_update_in_epoch') != 0 or counts.get('complete_epochs') != epoch
            or binding.get('official_revision') != OFFICIAL_REVISION
            or binding.get('common_manifest_sha256') != '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
            or binding.get('question_max_tokens') != QMAX or binding.get('prefix_tokens') != PREFIX_TOKENS
            or head.get('question_max_tokens') != QMAX or head.get('policy_tokens') != PREFIX_TOKENS
            or head.get('hidden_size') != DIM or head.get('feature_layer') != 27
            or head.get('feature_boundary') != 'post_block'
            or head.get('source_revision') != OFFICIAL_REVISION or head.get('source_layer') != 35
            or head.get('layout') != 'question_right_pad_then_fixed_five_policy_tail'
            or head.get('master_dtype') != 'float32'
            or binding.get('trajectory_bce_weight') != 1.0 or binding.get('mixed_bce_weight') != 1.0
            or binding.get('ema_decay') != .99 or binding.get('actor_updates') != 0
            or source.get('actor_updates') != 0):
        raise ValueError('Complete question5 warmup run/epoch/architecture/science identity differs')
    steps = counts.get('optimizer_steps')
    if (type(steps) is not int or steps < 1 or counts.get('ema_updates') != steps
            or source.get('actual_adam_steps_per_rank') != [steps]*4
            or source.get('ema', {}).get('updates') != steps
            or source.get('ema', {}).get('decay') != .99
            or counts.get('global_groups_seen') != 445440*epoch
            or counts.get('global_trajectories_seen') != 2227200*epoch):
        raise ValueError('Complete warmup four-rank Adam/EMA clock differs')
    metrics = source.get('epoch_metrics')
    if (not isinstance(metrics, dict) or metrics.get('epoch') != epoch
            or metrics.get('ema_updates') != steps
            or metrics.get('same_train_test_weight_kind') != 'ema0.99'
            or metrics.get('train_ema', {}).get('groups') != 445440
            or metrics.get('train_ema', {}).get('trajectories') != 2227200
            or metrics.get('test_ema', {}).get('groups') != 512
            or metrics.get('test_ema', {}).get('trajectories') != 2560
            or source.get('epoch_metrics_sha256') != hashlib.sha256(_canonical(metrics)).hexdigest()):
        raise ValueError('Complete warmup checkpoint lacks its bound same-epoch EMA metrics')
    model = source['ema'].get('model') if model_kind == 'ema' else source.get('raw_model')
    buffers = source['ema'].get('nonpersistent_buffers') if model_kind == 'ema' else source.get('raw_nonpersistent_buffers')
    validate_model_state(model)
    validate_native_buffers(buffers)
    return model, buffers, dict(source_warmup_run=source_run, source_epoch=epoch,
        source_model_kind=model_kind, source_actual_adam_updates=steps,
        source_ema_updates=source['ema']['updates'], source_binding_sha256=hashlib.sha256(_canonical(binding)).hexdigest(),
        learned_tensor_count=14, nonpersistent_buffer_names=sorted(buffers),
        warmup_optimizer_imported=False, warmup_rng_imported=False, warmup_model_weights_only=True)


def restore_native_buffers(model, saved):
    persistent = model.state_dict()
    targets = {name: value for name, value in model.named_buffers() if name not in persistent}
    if set(saved) != set(targets):
        raise ValueError('Native model buffer identities differ from the saved source')
    with torch.no_grad():
        for name, target in targets.items():
            value = saved[name]
            if target.shape != value.shape or target.dtype != value.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Native model buffer shape/dtype/finite identity differs: '+name)
            target.copy_(value.to(target.device))


def nonpersistent_buffers(model):
    persistent = model.state_dict()
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers() if name not in persistent}


def validate_online_layout(batch, max_positions):
    features, mask, positions = batch['features'], batch['attention_mask'], batch['position_ids']
    lengths, question, policy = batch['question_lengths'], batch['question_mask'], batch['policy_mask']
    if (features.ndim != 3 or features.shape[1:] != (WIDTH, DIM) or mask.shape != features.shape[:2]
            or positions.shape != mask.shape or positions.dtype != torch.long or mask.dtype != torch.bool
            or question.shape != mask.shape or policy.shape != mask.shape
            or question.dtype != torch.bool or policy.dtype != torch.bool
            or lengths.shape != (features.shape[0],) or lengths.dtype != torch.long
            or len(batch['policy_segments']) != features.shape[0]):
        raise ValueError('Question+5 head requires the exact complete native feature layout')
    slots = torch.arange(WIDTH, device=mask.device).unsqueeze(0)
    if (bool(((lengths < 1) | (lengths > QMAX)).any()) or not torch.equal(question, slots < lengths[:, None])
            or bool(policy[:, :QMAX].any()) or not torch.equal(mask, question | policy)
            or not torch.equal(positions[question], slots.expand_as(mask)[question])):
        raise ValueError('Question slots, masks, or actual RoPE positions differ')
    actual_lengths = policy[:, QMAX:].sum(-1)
    if 'policy_lengths' in batch and not torch.equal(actual_lengths, batch['policy_lengths']):
        raise ValueError('Policy mask count differs from its authentic prefix length')
    for i, bounds in enumerate(batch['policy_segments']):
        if bool(policy[i, QMAX:].any()):
            validate_prefix_positions(policy[i:i+1, QMAX:], positions[i:i+1, QMAX:], [bounds], max_positions)
        elif bounds:
            raise ValueError('Actual zero-token EOS cannot have synthetic policy segments')


class OnlineQuestionProbe(nn.Module):
    """Native official FA2 block35, FP32 masters, and the learned prefix5 Linear."""
    def __init__(self, model_dir, question_max_tokens=QMAX):
        super().__init__()
        from safetensors import safe_open
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        root = Path(model_dir)
        config = Qwen2Config.from_pretrained(str(root), local_files_only=True)
        if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
                config.num_key_value_heads, config.intermediate_size) != (2048, 36, 16, 2, 11008):
            raise ValueError('Require the official Qwen2.5-3B native architecture')
        if question_max_tokens != QMAX or WIDTH != QMAX + PREFIX_TOKENS or PREFIX_TOKENS != 5:
            raise ValueError('The source question5 model requires Q273 plus five policy slots')
        config._attn_implementation = 'flash_attention_2'
        self.question_max_tokens, self.policy_tokens = QMAX, PREFIX_TOKENS
        self.max_tokens, self.hidden_size = WIDTH, DIM
        self.max_positions = int(config.max_position_embeddings)
        self.block = Qwen2DecoderLayer(config, layer_idx=35).float()
        self.rotary = Qwen2RotaryEmbedding(config=config).float()
        self.output = nn.Linear(WIDTH * DIM, 1, dtype=torch.float32)
        prefix = 'model.layers.35.'
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        keys = {key: filename for key, filename in index.items() if key.startswith(prefix)}
        state = {}
        for filename in sorted(set(keys.values())):
            with safe_open(str(root/filename), framework='pt', device='cpu') as handle:
                for key, owner in keys.items():
                    if owner == filename:
                        state[key[len(prefix):]] = handle.get_tensor(key).float()
        self.block.load_state_dict(state, strict=True)
        if not all(torch.equal(value, state[name]) for name, value in self.block.state_dict().items()):
            raise RuntimeError('Native official block35 copy differs')
        if 'FlashAttention2' not in type(self.block.self_attn).__name__:
            raise RuntimeError('The reference native FlashAttention2 backend was not constructed')
        self._cache_position = 'cache_position' in inspect.signature(self.block.forward).parameters

    def forward(self, batch):
        validate_online_layout(batch, self.max_positions)
        features, mask, positions = batch['features'].detach(), batch['attention_mask'], batch['position_ids']
        kwargs = dict(attention_mask=mask, position_ids=positions,
                      position_embeddings=self.rotary(features, positions), use_cache=False, output_attentions=False)
        if self._cache_position:
            kwargs['cache_position'] = torch.arange(WIDTH, device=features.device)
        hidden = self.block(features, **kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0).flatten(1)
        return self.output(hidden).squeeze(-1).float()
