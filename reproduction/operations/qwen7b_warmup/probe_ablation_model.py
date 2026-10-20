"""Independent native-Qwen7B block/readout ablations; no existing probe edits.

All layer numbers are ZERO BASED.  A probe block21/24/27 requires the *post*
block20/23/26 residual respectively.  Existing layer26 feature banks therefore
support block27 only; callers must provide verified capture metadata per batch.

Every variant first executes its native causal block on the complete question
+ Prefix20 input.  ``prefix_only`` changes only pooling after that block; it
does not remove question context from attention.  Readout heads are3584->1,
never flattened293*3584 heads.  No constructor resumes or replaces an existing
probe checkpoint; the caller owns independent initialization/recovery lineage.
"""
from dataclasses import asdict, dataclass
import inspect
import json
import math
from pathlib import Path

import torch
from torch import nn


MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
QUESTION_MAX_TOKENS = 273
PREFIX_TOKENS = 20
HIDDEN_SIZE = 3584
CAPTURE_PAIRS = {21:20, 24:23, 27:26}
CAPTURE_KIND = 'post_block_residual'
READOUTS = ('last_valid_token', 'mean', 'learned_attention')
SCOPES = ('prefix_only', 'question_prefix')


@dataclass(frozen=True)
class ProbeAblationSpec:
    """This explicit contract belongs in the experiment/bank manifest."""
    block_layer: int
    capture_layer: int
    readout: str = 'mean'
    scope: str = 'question_prefix'
    question_max_tokens: int = QUESTION_MAX_TOKENS
    prefix_tokens: int = PREFIX_TOKENS
    hidden_size: int = HIDDEN_SIZE
    capture_kind: str = CAPTURE_KIND
    layer_index_basis: str = 'zero_based'
    model_revision: str = REVISION

    def __post_init__(self):
        if type(self.block_layer) is not int or self.block_layer not in CAPTURE_PAIRS:
            raise ValueError('Native probe block must be zero-based21,24,or27')
        if type(self.capture_layer) is not int or self.capture_layer != CAPTURE_PAIRS[self.block_layer]:
            raise ValueError('Probe block requires matching previous post-block residual:20->21,23->24,26->27')
        if self.readout not in READOUTS or self.scope not in SCOPES:
            raise ValueError('Unsupported token readout or readout scope')
        if (self.question_max_tokens, self.prefix_tokens, self.hidden_size) != (273, 20, 3584):
            raise ValueError('This native7B contract is QMAX273,Prefix20,hidden3584')
        if self.capture_kind != CAPTURE_KIND or self.layer_index_basis != 'zero_based':
            raise ValueError('Only explicit zero-based post-block residual captures are accepted')
        if self.model_revision != REVISION:
            raise ValueError('The ablation is bound to the pinned official7B snapshot')

    @property
    def width(self):
        return self.question_max_tokens+self.prefix_tokens

    def contract(self):
        return asdict(self)


class TokenReadout(nn.Module):
    """Padding-safe token pooling plus one small hidden->1 binary-logit head.

    This pure module does not detach its input: gradients must reach the probe
    block.  The native wrapper detaches the *captured student features* before
    executing that block.  Learned attention starts with a zero key, giving
    exactly uniform pooling initially without consuming extra random numbers;
    matching an experiment seed thus also matches mean/attention head starts.

    ``pool`` returns pooled representations and token weights on their current
    device for testing/remote diagnostics.  Nothing is serialized or copied to
    a local file by this module.  Empty eligible sets have zero representations
    and zero token weights, so their only output is the classifier bias.
    """
    def __init__(self, hidden_size=HIDDEN_SIZE, readout='mean', scope='question_prefix',
                 question_max_tokens=QUESTION_MAX_TOKENS, prefix_tokens=PREFIX_TOKENS):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError('Positive hidden width required')
        if readout not in READOUTS or scope not in SCOPES:
            raise ValueError('Unsupported readout or scope')
        if type(question_max_tokens) is not int or question_max_tokens < 1:
            raise ValueError('Positive question token capacity required')
        if type(prefix_tokens) is not int or prefix_tokens < 1:
            raise ValueError('Positive prefix token capacity required')
        self.hidden_size, self.readout, self.scope = hidden_size, readout, scope
        self.qmax, self.prefix = question_max_tokens, prefix_tokens
        self.width = self.qmax+self.prefix
        if readout == 'learned_attention':
            self.attention_key = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        else:
            self.register_parameter('attention_key', None)
        self.output = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def eligible_mask(self, attention_mask):
        if attention_mask.ndim != 2 or attention_mask.shape[1] != self.width:
            raise ValueError('Complete question+prefix attention mask required')
        if attention_mask.dtype != torch.bool:
            raise TypeError('The true token mask must be boolean')
        if self.scope == 'question_prefix':
            return attention_mask
        eligible = attention_mask.clone()
        eligible[:, :self.qmax] = False
        return eligible

    def pool(self, hidden, attention_mask):
        if hidden.ndim != 3 or hidden.shape[1:] != (self.width, self.hidden_size):
            raise ValueError('Readout expects complete hidden[B,QMAX+Prefix20,H]')
        if hidden.shape[0] != attention_mask.shape[0] or hidden.device != attention_mask.device:
            raise ValueError('Readout hidden and token mask must share batch/device')
        if not hidden.is_floating_point():
            raise TypeError('Floating point hidden representations required')
        eligible = self.eligible_mask(attention_mask)
        # Mask before score/multiply: NaN padding times zero would still be NaN.
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        safe = hidden.to(dtype=dtype).masked_fill(~eligible.unsqueeze(-1), 0)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError('Nonfinite eligible readout representation')
        nonempty = eligible.any(dim=1)
        weights = torch.zeros(eligible.shape, dtype=dtype, device=hidden.device)
        if self.readout == 'last_valid_token':
            indices = torch.arange(self.width, device=hidden.device).expand_as(eligible)
            last = indices.masked_fill(~eligible, -1).amax(dim=1)
            weights.scatter_(1, last.clamp_min(0).unsqueeze(1), nonempty.to(dtype).unsqueeze(1))
        elif self.readout == 'mean':
            weights = eligible.to(dtype)/eligible.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            # Explicit FP32/FP64 scores remain stable under an outer BF16 autocast.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                scores = torch.matmul(safe, self.attention_key.to(dtype=dtype))/math.sqrt(self.hidden_size)
                scores = scores.masked_fill(~eligible, -torch.inf)
                scores = torch.where(nonempty.unsqueeze(1), scores, torch.zeros_like(scores))
                weights = scores.softmax(dim=1).masked_fill(~eligible, 0)
        pooled = (safe*weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

    def forward(self, hidden, attention_mask, return_weights=False):
        pooled, weights = self.pool(hidden, attention_mask)
        logits = self.output(pooled.to(dtype=self.output.weight.dtype)).squeeze(-1)
        if return_weights:
            return dict(logits=logits, token_weights=weights)
        return logits


def validate_capture_batch(batch, spec):
    """Reject wrong-layer/native-position or mask geometry before block use."""
    if 'capture_layer' not in batch or 'capture_kind' not in batch:
        raise ValueError('Verified capture_layer and capture_kind metadata are required; do not infer from a filename')
    if batch['capture_kind'] != spec.capture_kind:
        raise ValueError('A post-block residual capture is required')
    layer = batch['capture_layer']
    if torch.is_tensor(layer):
        if layer.dtype not in (torch.int32, torch.int64) or layer.numel() == 0:
            raise ValueError('Integer capture layer metadata required')
        if not bool((layer == spec.capture_layer).all()):
            raise ValueError('Feature capture does not match the selected probe block')
    elif type(layer) is not int or layer != spec.capture_layer:
        raise ValueError('Feature capture does not match the selected probe block')
    features, mask, positions = batch['features'], batch['attention_mask'], batch['position_ids']
    if features.ndim != 3 or features.shape[1:] != (spec.width, spec.hidden_size):
        raise ValueError('Native feature geometry does not match the ablation contract')
    if features.shape[0] < 1:
        raise ValueError('A nonempty native batch is required')
    if torch.is_tensor(layer) and layer.numel() not in (1, features.shape[0]):
        raise ValueError('Capture metadata must be one shared layer or one layer per trajectory')
    if mask.shape != features.shape[:2] or positions.shape != mask.shape:
        raise ValueError('Native feature/mask/position alignment is required')
    if mask.dtype != torch.bool or positions.dtype != torch.long:
        raise TypeError('True boolean mask and native int64 position IDs required')
    if not features.is_floating_point():
        raise TypeError('Floating point native features required')
    if not features.device == mask.device == positions.device:
        raise ValueError('Native batch tensors must share a device')
    if not bool(mask[:, 0].all()):
        raise ValueError('Every native sequence must retain its initial question context')
    if not bool(torch.isfinite(features.masked_fill(~mask.unsqueeze(-1), 0)).all()):
        raise ValueError('Nonfinite real native feature')
    if bool((positions.masked_select(mask) < 0).any()):
        raise ValueError('Native real-token positions must be nonnegative')
    # Permit the existing QMAX padding gap while rejecting order inversions.
    for row in range(len(mask)):
        own = positions[row].masked_select(mask[row])
        if len(own) > 1 and not bool((own[1:] > own[:-1]).all()):
            raise ValueError('Question+prefix native positions must remain in causal order')


class NativeBlockProbe(nn.Module):
    """Fresh independent official native block, RoPE and small token readout.

    ``model_dir`` is an already checksum-verified remote official snapshot.
    The inventory revision is checked here; full model-file SHA verification
    remains the asset-preparation gate, avoiding rehashing multi-GB shards per
    candidate construction.  Every selected tensor is strictly loaded and
    compared against the supplied official shard before this object is usable.
    """
    def __init__(self, model_dir, spec):
        super().__init__()
        if not isinstance(spec, ProbeAblationSpec):
            raise TypeError('An explicit ProbeAblationSpec is required')
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        from safetensors import safe_open

        self.spec = spec
        root = Path(model_dir)
        inventory = json.loads((root/'official-inventory.json').read_text())
        if inventory['sha'] != spec.model_revision:
            raise ValueError('Model inventory is not the pinned official7B revision')
        if 'id' in inventory and inventory['id'] != MODEL:
            raise ValueError('Unexpected official model identity')
        config = Qwen2Config.from_pretrained(str(root), local_files_only=True)
        if (config.num_hidden_layers, config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.intermediate_size) != (28, 3584, 28, 4, 18944):
            raise ValueError('Only the official Qwen2.5-7B architecture is accepted')
        config._attn_implementation = 'flash_attention_2'
        self.block = Qwen2DecoderLayer(config, layer_idx=spec.block_layer).float()
        self.rotary = Qwen2RotaryEmbedding(config=config).float()
        if 'FlashAttention2' not in type(self.block.self_attn).__name__:
            raise RuntimeError('This ablation must retain the available native FA2 runtime')
        if not self.block.self_attn.is_causal:
            raise RuntimeError('A native causal probe block is required')
        self.cache_position = 'cache_position' in inspect.signature(self.block.forward).parameters
        prefix = 'model.layers.%d.' % spec.block_layer
        mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        keys = {key:filename for key, filename in mapping.items() if key.startswith(prefix)}
        if not keys:
            raise ValueError('Selected native block is absent from the official snapshot index')
        state = {}
        for filename in sorted(set(keys.values())):
            if Path(filename).name != filename:
                raise ValueError('Shard index must reference a local snapshot filename')
            with safe_open(str(root/filename), framework='pt', device='cpu') as handle:
                for key, owner in keys.items():
                    if owner == filename:
                        state[key[len(prefix):]] = handle.get_tensor(key).float()
        self.block.load_state_dict(state, strict=True)
        if not all(torch.equal(value, state[name]) for name, value in self.block.state_dict().items()):
            raise RuntimeError('Native block differs from its supplied official shard tensors')
        self.readout = TokenReadout(spec.hidden_size, spec.readout, spec.scope,
                                    spec.question_max_tokens, spec.prefix_tokens)
        self.initialization_proof = dict(model=MODEL, revision=spec.model_revision,
            zero_based_block_layer=spec.block_layer, zero_based_post_block_capture_layer=spec.capture_layer,
            official_tensor_load_exact=True, source_shards=sorted(set(keys.values())),
            spec=spec.contract(), classifier_shape=list(self.readout.output.weight.shape),
            attention_key_initialization='all_zero_uniform' if spec.readout == 'learned_attention' else None)

    def forward(self, batch, return_weights=False):
        validate_capture_batch(batch, self.spec)
        mask = batch['attention_mask'].contiguous()
        positions = batch['position_ids'].contiguous()
        # The ONLY detach boundary: frozen student features cannot get gradients.
        features = batch['features'].detach().masked_fill(~mask.unsqueeze(-1), 0).contiguous()
        kwargs = dict(attention_mask=mask, position_ids=positions,
                      position_embeddings=self.rotary(features, positions),
                      use_cache=False, output_attentions=False)
        if self.cache_position:
            kwargs['cache_position'] = torch.arange(self.spec.width, device=features.device)
        # Both scopes see exactly the same full causal question+prefix context.
        hidden = self.block(features, **kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        result = self.readout(hidden, mask, return_weights=return_weights)
        if return_weights:
            result['logits'] = result['logits'].float()
            return result
        return result.float()
