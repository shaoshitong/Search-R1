"""Batch-local frozen layer27 encoding of authentic question+policy20 contexts.

Only token scans are read from disk. Every call encodes its requested trajectory
IDs, returns ordinary detached BF16 tensors for the trainable head, and writes no
feature tensors. The trainer may retain consumed batches in temporary CPU RAM.
"""
from reproduction.prefix20_question_replay.features import (
    FEATURE_DIM, FEATURE_ROLE, POLICY_TOKENS, REVISION, _source_scan,
    initial_question_segment, retained_twenty,
)
from reproduction.prefix20_online.model import expected_policy_positions
from reproduction.prefix_only_correctness_ab.features import _LayerCaptured, remote, sha


def normalized_row(original, question_max_tokens):
    """Select only the authentic initial request and first20 kept action tokens."""
    question = initial_question_segment(original)
    segments, bounds, length = retained_twenty(original)
    qlen = question['question_length']
    if not 1 <= qlen <= question_max_tokens:
        raise ValueError('Actual initial request exceeds the fixed question slot width')
    return dict(original, question_length=qlen, question_context_offset=question['context_offset'],
                length=length, policy_length=length, prefix_length=length,
                segments=segments, segment_bounds=bounds)


def encode_rows(rows, contexts, policy_ids, policy_positions, question_max_tokens,
                capture, device='cpu', micro_batch_size=32):
    """Small CPU-testable batch assembly around a supplied causal capture call.

    capture(input_ids, attention_mask, position_ids) returns layer27 post-block
    hidden states. Per-segment length bucketing changes only execution order;
    every output remains at its own trajectory and original RoPE position.
    """
    import numpy as np
    import torch
    if not rows or micro_batch_size < 1:
        raise ValueError('A nonempty requested batch and positive encoder micro batch are required')
    qmax = int(question_max_tokens)
    normalized = [normalized_row(row, qmax) for row in rows]
    if len({row['trajectory_id'] for row in normalized}) != len(normalized):
        raise ValueError('A batch contains duplicate requested trajectory identities')
    device = torch.device(device)
    width = qmax + POLICY_TOKENS
    # These are ordinary tensors, allocated outside inference_mode, so a trainable
    # decoder can save them for backward without inference-tensor restrictions.
    features = torch.zeros((len(rows), width, FEATURE_DIM), dtype=torch.bfloat16, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    question_mask, policy_mask = torch.zeros_like(mask), torch.zeros_like(mask)
    positions = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    work = []
    for i, row in enumerate(normalized):
        qlen, plen, offset = row['question_length'], row['policy_length'], row['policy_offset']
        expected = expected_policy_positions(row['segment_bounds'])
        actual = np.asarray(policy_positions[offset:offset + plen])
        if actual.tolist() != expected:
            raise ValueError('Actual policy positions differ from their own retained turn boundaries')
        mask[i, :qlen] = question_mask[i, :qlen] = True
        mask[i, qmax:qmax + plen] = policy_mask[i, qmax:qmax + plen] = True
        positions[i, :qlen] = torch.arange(qlen, device=device)
        positions[i, qmax:qmax + plen] = torch.as_tensor(actual.copy(), dtype=torch.long, device=device)
        for segment in row['segments']:
            lo, req, take = segment['context_offset'], segment['request_length'], segment['policy_length']
            poff = offset + segment['policy_start']
            if not np.array_equal(contexts[lo + req:lo + req + take], policy_ids[poff:poff + take]):
                raise ValueError('Own policy tokens differ from their exact causal request context')
            work.append(dict(segment, batch_row=i, question_length=qlen))
    work.sort(key=lambda item: item['context_length'])
    with torch.no_grad():
        for start in range(0, len(work), micro_batch_size):
            selected = work[start:start + micro_batch_size]
            context_width = max(item['context_length'] for item in selected)
            # Padding tokens are masked; no synthetic token is ever stored as an
            # effective question or policy feature. Original input IDs are copied.
            tokens = torch.zeros((len(selected), context_width), dtype=torch.long, device=device)
            valid = torch.zeros_like(tokens)
            rope = torch.zeros_like(tokens)
            for slot, segment in enumerate(selected):
                lo, n = segment['context_offset'], segment['context_length']
                actual = np.asarray(contexts[lo:lo + n])
                if len(actual) != n:
                    raise ValueError('Actual causal context is incomplete')
                tokens[slot, :n] = torch.as_tensor(actual.copy(), dtype=torch.long, device=device)
                valid[slot, :n] = 1
                rope[slot, :n] = torch.arange(n, device=device)
            hidden = capture(tokens, valid, rope).detach()
            if hidden.shape != (len(selected), context_width, FEATURE_DIM):
                raise ValueError('Layer27 post-block representation shape differs')
            for slot, segment in enumerate(selected):
                i, req, plen = segment['batch_row'], segment['request_length'], segment['policy_length']
                policy_lo = qmax + segment['policy_start']
                features[i, policy_lo:policy_lo + plen] = hidden[slot, req:req + plen].to(torch.bfloat16)
                if segment['turn_index'] == 0:
                    qlen = segment['question_length']
                    if qlen != req:
                        raise ValueError('Initial question length and actual turn0 request differ')
                    features[i, :qlen] = hidden[slot, :qlen].to(torch.bfloat16)
    return dict(features=features.detach(), attention_mask=mask, position_ids=positions,
                question_mask=question_mask, policy_mask=policy_mask,
                question_lengths=torch.tensor([r['question_length'] for r in normalized], dtype=torch.long, device=device),
                policy_lengths=torch.tensor([r['policy_length'] for r in normalized], dtype=torch.long, device=device),
                policy_segments=[r['segment_bounds'] for r in normalized], feature_role=FEATURE_ROLE,
                labels=torch.tensor([r['strict_em'] for r in normalized], dtype=torch.float32, device=device),
                trajectory_ids=[r['trajectory_id'] for r in normalized], qids=[str(r['qid']) for r in normalized])


class InlineEncoder:
    """One frozen encoder on the same device as the trainable probe head."""
    def __init__(self, inputs_dir, model_dir, question_max_tokens=273,
                 micro_batch_size=32, device='cuda:0', attention_implementation='sdpa'):
        import torch
        from transformers import AutoModel
        self.device = torch.device(device)
        self.question_max_tokens = int(question_max_tokens)
        self.max_tokens, self.feature_dim = self.question_max_tokens + POLICY_TOKENS, FEATURE_DIM
        self.micro_batch_size = int(micro_batch_size)
        self.model_dir = remote(model_dir)
        self.set_inputs(inputs_dir)
        self.model = AutoModel.from_pretrained(str(self.model_dir), torch_dtype=torch.bfloat16,
            attn_implementation=attention_implementation, local_files_only=True,
            trust_remote_code=False, low_cpu_mem_usage=True)
        self.model.requires_grad_(False).eval().to(self.device)
        if len(self.model.layers) != 36 or self.model.config.hidden_size != FEATURE_DIM:
            raise ValueError('Wrong frozen official Qwen2.5-3B encoder shape')
        self._captured = []
        self._hook = self.model.layers[27].register_forward_hook(self._capture_hook)

    def set_inputs(self, inputs_dir):
        """Swap the current token scan without reloading frozen model weights."""
        import numpy as np
        source, proof, rows, arrays = _source_scan(inputs_dir)
        mapping = proof['model_files_sha256']
        if hasattr(self, 'spec') and self.spec['encoder_files_sha256'] != mapping:
            raise ValueError('A new cached batch cannot change the official frozen encoder')
        self.path, self.manifest = source, proof
        self.manifest_sha = sha(source / 'scan-complete.json')
        self.spec = dict(encoder_revision=REVISION, encoder_files_sha256=mapping,
                         question_max_tokens=self.question_max_tokens,
                         head_max_tokens=self.max_tokens, feature_layer=27, boundary='post_block',
                         question_hidden_stored=True, prefix_length=20, encoder_frozen=True,
                         feature_role=FEATURE_ROLE, feature_storage='batch_memory_only')
        self.index, self.ids_by_split = {}, {'train': [], 'test': []}
        for original in rows:
            row = normalized_row(original, self.question_max_tokens)
            tid = row['trajectory_id']
            if tid in self.index or type(row['strict_em']) is not int or row['strict_em'] not in (0, 1):
                raise ValueError('Duplicate cached trajectory identity or invalid own EM label')
            self.index[tid] = row
            self.ids_by_split[row['split']].append(tid)
        self.ids = list(self.index)
        self.contexts = np.load(arrays / 'packed_context_ids.npy', allow_pickle=False, mmap_mode='r')
        self.input_ids = np.load(arrays / 'packed_policy_ids.npy', allow_pickle=False, mmap_mode='r')
        self.position_ids = np.load(arrays / 'packed_policy_positions.npy', allow_pickle=False, mmap_mode='r')

    def _capture_hook(self, _module, _inputs, value):
        self._captured.append(value[0] if isinstance(value, tuple) else value)
        raise _LayerCaptured()

    def _capture(self, input_ids, attention_mask, position_ids):
        import torch
        self._captured.clear()
        with torch.no_grad():
            try:
                self.model(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, use_cache=False,
                           output_hidden_states=False, return_dict=True)
            except _LayerCaptured:
                pass
        if len(self._captured) != 1:
            raise RuntimeError('Actual layer27 post-block capture is missing')
        return self._captured.pop().detach()

    def get_batch(self, ids, device=None):
        import torch
        if device is not None and torch.device(device) != self.device:
            raise ValueError('Inline encoder and trainable head must use the same rank device')
        return encode_rows([self.index[tid] for tid in ids], self.contexts, self.input_ids,
                           self.position_ids, self.question_max_tokens, self._capture,
                           device=self.device, micro_batch_size=self.micro_batch_size)

    def close(self):
        self._hook.remove()
