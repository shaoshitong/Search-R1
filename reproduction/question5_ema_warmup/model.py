"""Official Qwen block35 over detached question + authentic policy5 features.

Architecture/layout derives from prefix20_question_replay/model.py. This fresh
experiment has no checkpoint migration and never imports a trained old head.
"""

import inspect
import json
from pathlib import Path

import torch
from torch import nn

POLICY_TOKENS = 5
HIDDEN_SIZE = 2048
OFFICIAL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def expected_policy_positions(segments, max_positions):
    expected = []
    previous_turn = -1
    for segment in segments:
        offset, length, start, turn = (segment[name] for name in
                                      ("policy_start", "policy_length", "position_start", "turn_index"))
        if any(type(value) is not int for value in (offset, length, start, turn)):
            raise ValueError("Policy segment coordinates must be real integer metadata")
        if offset != len(expected) or length < 1 or start < 1 or turn <= previous_turn:
            raise ValueError("Segments must cover the original policy tokens in turn order")
        if start + length > max_positions:
            raise ValueError("Policy segment exceeds the original Qwen position limit")
        expected.extend(range(start, start + length))
        previous_turn = turn
    if not 1 <= len(expected) <= POLICY_TOKENS:
        raise ValueError("Require 1..5 actual policy tokens; no fabricated padding tokens")
    return expected


def validate_layout(batch, question_max_tokens, max_positions):
    """Validate fixed storage slots independently of original per-turn RoPE positions."""
    features = batch["features"]
    mask, positions = batch["attention_mask"], batch["position_ids"]
    question_mask, policy_mask = batch["question_mask"], batch["policy_mask"]
    lengths = batch["question_lengths"]
    width = question_max_tokens + POLICY_TOKENS
    if (features.ndim != 3 or features.shape[0] < 1
            or features.shape[1:] != (width, HIDDEN_SIZE)
            or mask.shape != features.shape[:2] or mask.dtype != torch.bool
            or positions.shape != mask.shape or positions.dtype != torch.long):
        raise ValueError("Require complete [N,Qmax+5,2048] features and exact bool/long masks/positions")
    if lengths.shape != (features.shape[0],) or lengths.dtype != torch.long:
        raise ValueError("Every trajectory needs its actual initial question length")
    if any(t.shape != mask.shape or t.dtype != torch.bool for t in (question_mask, policy_mask)):
        raise ValueError("Question and policy masks must cover the fixed storage layout")
    if bool(((lengths < 1) | (lengths > question_max_tokens)).any()):
        raise ValueError("Question length exceeds scanned Qmax; truncation is prohibited")
    slots = torch.arange(width, device=mask.device).unsqueeze(0)
    expected_question = slots < lengths.unsqueeze(1)
    if not torch.equal(question_mask, expected_question):
        raise ValueError("All initial question tokens must occupy the right-padded question slots")
    if bool(policy_mask[:, :question_max_tokens].any()) or not torch.equal(mask, question_mask | policy_mask):
        raise ValueError("Question/policy slot masks overlap or do not match the effective mask")
    expected_question_positions = slots.expand(features.shape[0], -1)
    if not torch.equal(positions[question_mask], expected_question_positions[question_mask]):
        raise ValueError("Question positions must preserve their original 0..qlen-1 indices")
    segments = batch["policy_segments"]
    if len(segments) != features.shape[0]:
        raise ValueError("One authentic policy segment list is required per trajectory")
    rows = [expected_policy_positions(row, max_positions) for row in segments]
    tail_mask = torch.tensor([[True] * len(row) + [False] * (POLICY_TOKENS - len(row))
                              for row in rows], device=mask.device, dtype=torch.bool)
    tail_positions = torch.tensor([row + [0] * (POLICY_TOKENS - len(row)) for row in rows],
                                  device=positions.device, dtype=torch.long)
    if (not torch.equal(policy_mask[:, question_max_tokens:], tail_mask)
            or not torch.equal(positions[:, question_max_tokens:][tail_mask], tail_positions[tail_mask])):
        raise ValueError("Policy positions differ from the same trajectory's original retained segments")
    if "policy_lengths" in batch:
        actual_lengths = torch.tensor([len(row) for row in rows], device=lengths.device, dtype=torch.long)
        if not torch.equal(batch["policy_lengths"], actual_lengths):
            raise ValueError("Policy lengths disagree with authentic segment coverage")


def masked_flatten(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Invalid hidden/mask shape for masked flatten")
    return hidden.masked_fill(~mask.unsqueeze(-1), 0).flatten(1)


def causal_padding_mask(mask, dtype):
    """Native SDPA/eager additive mask; causal order follows retained token order.

RoPE positions may reset across retrieval turns, so they do not define causal
ordering. Storage order does. Question padding holes are never visible as keys.
"""
    n, width = mask.shape
    slots = torch.arange(width, device=mask.device)
    allowed = (slots.unsqueeze(1) >= slots.unsqueeze(0))[None, None, :, :] & mask[:, None, None, :]
    result = torch.zeros((n, 1, width, width), device=mask.device, dtype=dtype)
    return result.masked_fill(~allowed, torch.finfo(dtype).min)


class Question5CorrectnessProbe(nn.Module):
    """A fresh full official last block and fresh Linear; frozen inputs are detached."""
    def __init__(self, model_dir, question_max_tokens, *, attention_implementation="sdpa"):
        super().__init__()
        from safetensors import safe_open
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

        root = Path(model_dir)
        config = Qwen2Config.from_pretrained(str(root), local_files_only=True)
        expected = (2048, 36, 16, 2, 11008)
        actual = (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
                  config.num_key_value_heads, config.intermediate_size)
        if actual != expected:
            raise ValueError("Require official Qwen2.5-3B 2048/36/16Q/2KV/11008 architecture")
        if (type(question_max_tokens) is not int or question_max_tokens < 1
                or question_max_tokens >= config.max_position_embeddings):
            raise ValueError("Qmax must come from the real full-dataset initial-question scan")
        if attention_implementation not in ("sdpa", "eager", "flash_attention_2"):
            raise ValueError("Use a supported native Qwen attention backend")
        config._attn_implementation = attention_implementation
        self.question_max_tokens = question_max_tokens
        self.policy_tokens = POLICY_TOKENS
        self.hidden_size = HIDDEN_SIZE
        self.max_tokens = question_max_tokens + POLICY_TOKENS
        self.max_positions = int(config.max_position_embeddings)
        self.attention_implementation = attention_implementation
        self.block = Qwen2DecoderLayer(config, layer_idx=35).float()
        self.rotary = Qwen2RotaryEmbedding(config=config)
        self.output = nn.Linear(self.max_tokens * HIDDEN_SIZE, 1, dtype=torch.float32)
        prefix = "model.layers.35."
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())["weight_map"]
            keys = {name: filename for name, filename in index.items() if name.startswith(prefix)}
        else:
            with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as handle:
                keys = {name: "model.safetensors" for name in handle.keys() if name.startswith(prefix)}
        if not keys:
            raise ValueError("Official block35 source tensors are missing")
        state = {}
        for filename in sorted(set(keys.values())):
            with safe_open(str(root / filename), framework="pt", device="cpu") as handle:
                for key, owner in keys.items():
                    if owner == filename:
                        state[key[len(prefix):]] = handle.get_tensor(key).float()
        self.block.load_state_dict(state, strict=True)
        if not all(torch.equal(value, state[name]) for name, value in self.block.state_dict().items()):
            raise RuntimeError("Copied native block35 differs from the official source tensors")
        self.copy_metadata = dict(source_revision=OFFICIAL_REVISION, source_layer=35,
                                  source_tensor_count=len(state),
                                  source_parameter_count=sum(value.numel() for value in state.values()),
                                  attention_class=type(self.block.self_attn).__name__,
                                  attention_implementation=attention_implementation,
                                  question_max_tokens=question_max_tokens, policy_tokens=POLICY_TOKENS,
                                  hidden_size=HIDDEN_SIZE, head_input="question_and_policy5",
                                  feature_layer=27, feature_boundary="post_block",
                                  layout="question_right_pad_then_fixed_five_policy_tail",
                                  initial_block="exact_official_layer35_copy", initial_linear="fresh_nnLinear",
                                  old_probe_imported=False, master_dtype="float32")
        self._cache_position = "cache_position" in inspect.signature(self.block.forward).parameters

    def forward(self, batch):
        if batch is None:
            # A final global batch can contain fewer groups than ranks. An empty
            # rank still participates in the same DDP reduction without reading
            # or repeating a trajectory; every trainable parameter gets zero grad.
            return sum(parameter.reshape(-1)[0] * 0 for parameter in self.parameters()
                       if parameter.requires_grad)
        validate_layout(batch, self.question_max_tokens, self.max_positions)
        features = batch["features"].detach()
        mask, positions = batch["attention_mask"], batch["position_ids"]
        if self.attention_implementation == "flash_attention_2":
            attention_mask = mask
        else:
            attention_mask = causal_padding_mask(mask, features.dtype)
        kwargs = dict(attention_mask=attention_mask, position_ids=positions,
                      position_embeddings=self.rotary(features, positions),
                      use_cache=False, output_attentions=False)
        if self._cache_position:
            kwargs["cache_position"] = torch.arange(features.shape[1], device=features.device)
        hidden = self.block(features, **kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.output(masked_flatten(hidden, mask)).squeeze(-1).float()
