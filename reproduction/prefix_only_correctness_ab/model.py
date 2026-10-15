"""One official Qwen block attends only contextual prefix features, at original RoPE positions."""
import inspect
import json
from pathlib import Path

import torch
from torch import nn


def expected_policy_positions(segments, max_positions=32768):
    """Validate retained policy segments without assuming cross-turn position monotonicity."""
    expected = []
    previous_turn = -1
    for segment in segments:
        offset = segment['policy_start']
        length = segment['policy_length']
        start = segment['position_start']
        turn = segment['turn_index']
        if any(type(value) is not int for value in (offset, length, start, turn)):
            raise ValueError('Segment coordinates must be actual integer metadata')
        if offset != len(expected) or length < 1 or start < 1 or turn <= previous_turn:
            raise ValueError('Policy segments must cover the retained sequence in actual turn order')
        if start + length > max_positions:
            raise ValueError('Policy segment exceeds the official RoPE position range')
        expected.extend(range(start, start + length))
        previous_turn = turn
    if not 1 <= len(expected) <= 80:
        raise ValueError('Require one to eighty actual retained policy tokens')
    return expected


def validate_prefix_positions(mask, positions, policy_segments, max_positions=32768):
    if positions.dtype != torch.long or positions.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError('Invalid policy mask/position metadata')
    if len(policy_segments) != mask.shape[0]:
        raise ValueError('One segment identity record is required per trajectory')
    rows = [expected_policy_positions(segments, max_positions) for segments in policy_segments]
    width = mask.shape[1]
    if any(len(row) > width for row in rows):
        raise ValueError('Policy segment length exceeds feature sequence width')
    expected_positions = torch.tensor([row + [0] * (width - len(row)) for row in rows],
                                     dtype=torch.long, device=positions.device)
    expected_mask = torch.tensor([[True] * len(row) + [False] * (width - len(row)) for row in rows],
                                dtype=torch.bool, device=mask.device)
    if not torch.equal(mask, expected_mask) or not torch.equal(positions[mask], expected_positions[mask]):
        raise ValueError('Head tokens must match original per-turn positions and retained segment boundaries')


class PrefixCorrectnessProbe(nn.Module):
    def __init__(self, model_dir, max_tokens):
        super().__init__()
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        from safetensors import safe_open
        root = Path(model_dir)
        config = Qwen2Config.from_pretrained(str(root), local_files_only=True)
        if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
                config.num_key_value_heads, config.intermediate_size) != (2048, 36, 16, 2, 11008):
            raise ValueError('Expected official Qwen2.5-3B architecture')
        config._attn_implementation = 'flash_attention_2'
        self.max_tokens = int(max_tokens)
        if self.max_tokens != 80:
            raise ValueError('All four prefix-only heads must have identical width80')
        self.max_positions = int(config.max_position_embeddings)
        self.block = Qwen2DecoderLayer(config, layer_idx=35)
        self.rotary = Qwen2RotaryEmbedding(config=config)
        self.output = nn.Linear(self.max_tokens * config.hidden_size, 1)
        index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        prefix = 'model.layers.35.'
        keys = {k: v for k, v in index.items() if k.startswith(prefix)}
        if not keys:
            raise ValueError('Official last-block weights missing')
        state = {}
        for filename in sorted(set(keys.values())):
            with safe_open(str(root / filename), framework='pt', device='cpu') as handle:
                for key, owner in keys.items():
                    if owner == filename:
                        state[key[len(prefix):]] = handle.get_tensor(key).float()
        self.block.load_state_dict(state, strict=True)
        if not all(torch.equal(v, state[k]) for k, v in self.block.state_dict().items()):
            raise RuntimeError('Official block copy differs')
        if 'FlashAttention2' not in type(self.block.self_attn).__name__:
            raise RuntimeError('FlashAttention2 backend was not constructed')
        self.copy_metadata = dict(source_layer=35, tensors=len(state),
                                  parameters=sum(v.numel() for v in state.values()),
                                  attention_class=type(self.block.self_attn).__name__,
                                  shared_max_tokens=self.max_tokens, head_input='prefix_positions_only',
                                  question_hidden_passed_to_head=False, master_dtype='float32')
        self._cache_position = 'cache_position' in inspect.signature(self.block.forward).parameters

    def forward(self, batch):
        features = batch['features'].detach()
        mask = batch['attention_mask'].bool()
        positions = batch['position_ids'].long()
        if features.ndim != 3 or features.shape[:2] != mask.shape or positions.shape != mask.shape:
            raise ValueError('Feature/mask/position shape mismatch')
        if features.shape[1] > self.max_tokens or not bool(mask[:, 0].all()):
            raise ValueError('Invalid sequence length or empty sequence')
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError('Expected right padding')
        validate_prefix_positions(mask, positions, batch['policy_segments'], self.max_positions)
        kwargs = dict(attention_mask=mask, position_ids=positions,
                      position_embeddings=self.rotary(features, positions),
                      use_cache=False, output_attentions=False)
        if self._cache_position:
            kwargs['cache_position'] = torch.arange(features.shape[1], device=features.device)
        hidden = self.block(features, **kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0)
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, self.max_tokens - hidden.shape[1]))
        return self.output(hidden.flatten(1)).squeeze(-1).float()
