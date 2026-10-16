"""Batch-local native frozen Qwen layer27 features for authentic question+20.

The complete official 36-layer AutoModel is loaded unchanged. Its layer27 hook
captures post-block hidden states and immediately aborts the remaining forward,
so only decoder layers0..27 execute. No labels enter the encoder. Feature tensors
are detached ordinary BF16 tensors, safe for the trainable copied-block backward.
"""

from .data import ARRAY_NAMES, FEATURE_DIM, FEATURE_ROLE, POLICY_TOKENS, REVISION
from .data import expected_policy_positions, initial_question_segment, retained_twenty, sha


class _LayerCaptured(Exception):
    pass


def encode_rows(rows, array_provider, question_max_tokens, capture, *, device='cpu',
                micro_batch_size=32, pad_token_id=151643, token_budget=None):
    """Assemble one requested batch around a supplied causal layer27 capture.

    ``array_provider(source_id)`` returns the three immutable packed token mmaps.
    ``capture(ids, mask, positions)`` returns post-block layer27 hidden states.
    This pure assembly function is also usable with a CPU fake capture fixture.
    """
    import numpy as np
    import torch
    if not rows or type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError('A nonempty batch and positive encoder micro batch are required')
    if type(question_max_tokens) is not int or question_max_tokens < 1:
        raise ValueError('A scanned, fixed global question slot width is required')
    if token_budget is not None and (type(token_budget) is not int or token_budget < 1):
        raise ValueError('Encoder token budget must be a positive integer')
    if len({row['trajectory_id'] for row in rows}) != len(rows):
        raise ValueError('A requested feature batch contains duplicate trajectory events')
    device = torch.device(device)
    width = question_max_tokens + POLICY_TOKENS
    features = torch.zeros((len(rows), width, FEATURE_DIM), dtype=torch.bfloat16, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    question_mask, policy_mask = torch.zeros_like(mask), torch.zeros_like(mask)
    positions = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    question_lengths, policy_lengths, bounds, work = [], [], [], []
    for index, original in enumerate(rows):
        question = initial_question_segment(original)
        segments, policy_bounds, plen = retained_twenty(original)
        qlen = question['question_length']
        if qlen > question_max_tokens:
            raise ValueError('Actual initial request exceeds fixed Qmax; truncation is forbidden')
        arrays = array_provider(original['source_id'])
        contexts, policy_ids, policy_positions = (arrays[name] for name in ARRAY_NAMES)
        offset = original['policy_offset']
        if np.asarray(policy_positions[offset:offset + plen]).tolist() != expected_policy_positions(policy_bounds):
            raise ValueError('Policy features must preserve their original native positions')
        mask[index, :qlen] = question_mask[index, :qlen] = True
        mask[index, question_max_tokens:question_max_tokens + plen] = policy_mask[index, question_max_tokens:question_max_tokens + plen] = True
        positions[index, :qlen] = torch.arange(qlen, device=device)
        positions[index, question_max_tokens:question_max_tokens + plen] = torch.as_tensor(
            np.array(policy_positions[offset:offset + plen], copy=True), dtype=torch.long, device=device)
        question_lengths.append(qlen)
        policy_lengths.append(plen)
        bounds.append(policy_bounds)
        for segment in segments:
            lo, req, take = segment['context_offset'], segment['request_length'], segment['policy_length']
            start = offset + segment['policy_start']
            if not np.array_equal(contexts[lo + req:lo + req + take], policy_ids[start:start + take]):
                raise ValueError('Same-trajectory policy token IDs differ from the authentic context')
            work.append(dict(segment, batch_row=index, source_id=original['source_id'], question_length=qlen))
    # Length bucketing changes execution order only. Each segment is copied into
    # its original trajectory's slots, with actual positions and padding masks.
    work.sort(key=lambda item: item['context_length'])
    cursor = 0
    with torch.no_grad():
        while cursor < len(work):
            size = min(micro_batch_size, len(work) - cursor)
            if token_budget is not None:
                while size > 1 and size * work[cursor + size - 1]['context_length'] > token_budget:
                    size -= 1
                if work[cursor]['context_length'] > token_budget:
                    raise ValueError('One actual causal context exceeds the configured encoder token budget')
            selected = work[cursor:cursor + size]
            context_width = max(segment['context_length'] for segment in selected)
            ids = torch.full((size, context_width), int(pad_token_id), dtype=torch.long, device=device)
            valid, rope = torch.zeros_like(ids), torch.zeros_like(ids)
            for slot, segment in enumerate(selected):
                contexts = array_provider(segment['source_id'])[ARRAY_NAMES[0]]
                lo, length = segment['context_offset'], segment['context_length']
                tokens = np.asarray(contexts[lo:lo + length])
                if len(tokens) != length:
                    raise ValueError('A causal request payload is incomplete')
                ids[slot, :length] = torch.as_tensor(np.array(tokens, copy=True), dtype=torch.long, device=device)
                valid[slot, :length] = 1
                rope[slot, :length] = torch.arange(length, device=device)
            hidden = capture(ids, valid, rope).detach()
            if hidden.shape != (size, context_width, FEATURE_DIM):
                raise ValueError('Native layer27 post-block representation shape differs')
            for slot, segment in enumerate(selected):
                index, request, take = segment['batch_row'], segment['request_length'], segment['policy_length']
                target = question_max_tokens + segment['policy_start']
                features[index, target:target + take] = hidden[slot, request:request + take].to(torch.bfloat16)
                if segment['turn_index'] == 0:
                    qlen = segment['question_length']
                    if qlen != request:
                        raise ValueError('Actual turn0 request and question layout differ')
                    features[index, :qlen] = hidden[slot, :qlen].to(torch.bfloat16)
            cursor += size
    return dict(features=features.detach(), attention_mask=mask, position_ids=positions,
                question_mask=question_mask, policy_mask=policy_mask,
                question_lengths=torch.tensor(question_lengths, dtype=torch.long, device=device),
                policy_lengths=torch.tensor(policy_lengths, dtype=torch.long, device=device),
                policy_segments=bounds, feature_role=FEATURE_ROLE,
                labels=torch.tensor([row['strict_em'] for row in rows], dtype=torch.float32, device=device),
                trajectory_ids=[row['trajectory_id'] for row in rows],
                qids=[str(row['qid']) for row in rows], event_ids=[row['event_id'] for row in rows])


class FrozenQuestion20Encoder:
    """Reader adapter exposing get_batch(flat_event_trajectory_ids, device=...)."""
    def __init__(self, dataset, model_dir, *, question_max_tokens=None, micro_batch_size=32,
                 device='cuda:0', attention_implementation='sdpa', token_budget=None,
                 verify_model_files=True):
        import torch
        from pathlib import Path
        from transformers import AutoModel
        self.dataset = dataset
        self.device = torch.device(device)
        self.question_max_tokens = dataset.question_max_tokens if question_max_tokens is None else question_max_tokens
        if self.question_max_tokens != dataset.question_max_tokens:
            raise ValueError('Encoder must use the composite manifest scanned Qmax')
        self.max_tokens = self.question_max_tokens + POLICY_TOKENS
        self.feature_dim = FEATURE_DIM
        self.micro_batch_size, self.token_budget = micro_batch_size, token_budget
        self.path, self.manifest_sha, self.spec = dataset.path, dataset.manifest_sha, dict(dataset.spec)
        self.index, self.ids_by_split, self.ids, self.groups = dataset.index, dataset.ids_by_split, dataset.ids, dataset.groups
        self.model_dir = Path(model_dir).resolve(strict=True)
        if verify_model_files:
            for name, expected in dataset.spec['encoder_files_sha256'].items():
                expected = expected['sha256'] if isinstance(expected, dict) else expected
                if sha(self.model_dir / name) != expected:
                    raise ValueError('Actual official frozen encoder file SHA differs: ' + name)
        self.model = AutoModel.from_pretrained(str(self.model_dir), torch_dtype=torch.bfloat16,
            attn_implementation=attention_implementation, local_files_only=True,
            trust_remote_code=False, low_cpu_mem_usage=True)
        self.model.requires_grad_(False).eval().to(self.device)
        config = self.model.config
        if (len(self.model.layers), config.hidden_size, config.num_attention_heads,
                config.num_key_value_heads, config.intermediate_size) != (36, 2048, 16, 2, 11008):
            raise ValueError('Require complete native official Qwen2.5-3B architecture')
        self.pad_token_id = int(config.pad_token_id if config.pad_token_id is not None else 151643)
        self._captured = []
        self._hook = self.model.layers[27].register_forward_hook(self._capture_hook)
        self.spec.update(encoder_revision=REVISION, encoder_attention_implementation=attention_implementation,
                         feature_storage='batch_memory_only', executed_decoder_indices=list(range(28)),
                         encoder_optimizer_updates=0, labels_passed_to_encoder=False)

    def __len__(self):
        return len(self.dataset)

    def _capture_hook(self, _module, _inputs, value):
        self._captured.append(value[0] if isinstance(value, tuple) else value)
        raise _LayerCaptured()

    def _capture(self, ids, attention_mask, position_ids):
        import torch
        self._captured.clear()
        with torch.no_grad():
            try:
                self.model(input_ids=ids, attention_mask=attention_mask, position_ids=position_ids,
                           use_cache=False, output_hidden_states=False, return_dict=True)
            except _LayerCaptured:
                pass
        if len(self._captured) != 1:
            raise RuntimeError('Native layer27 post-block capture is missing')
        return self._captured.pop().detach()

    def encode_rows(self, rows, array_provider=None):
        return encode_rows(rows, array_provider or self.dataset.source_arrays, self.question_max_tokens, self._capture,
                           device=self.device, micro_batch_size=self.micro_batch_size,
                           pad_token_id=self.pad_token_id, token_budget=self.token_budget)

    def get_batch(self, trajectory_ids, device=None):
        import torch
        if device is not None and torch.device(device) != self.device:
            raise ValueError('Frozen encoder and copied trainable block must share the rank device')
        return self.encode_rows(self.dataset.get_rows(trajectory_ids))

    def close(self):
        self._hook.remove()


class EncodedPrefix20Reader:
    """Share one frozen encoder across train/test composite reader views."""
    def __init__(self, dataset, encoder):
        if dataset.question_max_tokens != encoder.question_max_tokens:
            raise ValueError('Reader and encoder must share the scanned question slot width')
        self.dataset, self.encoder = dataset, encoder
        self.groups, self.index = dataset.groups, dataset.index
        self.ids, self.ids_by_split = dataset.ids, dataset.ids_by_split
        self.spec, self.manifest_sha = dataset.spec, dataset.manifest_sha
        self.question_max_tokens, self.max_tokens = dataset.question_max_tokens, dataset.max_tokens

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, trajectory_ids, device=None):
        import torch
        if device is not None and torch.device(device) != self.encoder.device:
            raise ValueError('Reader and frozen encoder must use the same rank device')
        return self.encoder.encode_rows(self.dataset.get_rows(trajectory_ids),
                                        array_provider=self.dataset.source_arrays)
