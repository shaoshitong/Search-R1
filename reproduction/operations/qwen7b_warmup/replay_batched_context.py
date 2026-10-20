"""Isolated length-bucketed HF causal replay; no IO, scheduler, or model loader.

Only the frozen prefill is batched. Saved full causal token contexts and their
RoPE positions remain unchanged, then the qualified legacy replay_features
function gathers the initial question and own retained Prefix20 slots. Native
vLLM equivalence and scalar-versus-batched bit equality are NOT claimed.
Actual feature tensors must stay in the caller's remote GPU/shared-store scope.
"""
from itertools import islice
import time
from . import replay_probe_layers as r


def checked_contexts(contexts, positions=None):
    contexts = [list(row) for row in contexts]
    if any(not row or any(type(token) is not int or token < 0 for token in row) for row in contexts):
        raise ValueError('Every actual causal context needs nonempty integer token IDs')
    positions = [list(range(len(row))) for row in contexts] if positions is None else [list(row) for row in positions]
    if len(positions) != len(contexts) or any(row != list(range(len(context))) or
            any(type(p) is not int for p in row) for row, context in zip(positions, contexts)):
        raise ValueError('Each saved full causal context must keep actual positions0..length-1')
    return contexts, positions


def plan_context_batches(contexts, positions=None, batch_size=16, max_context_tokens=16384):
    """Stable length buckets bounded by B*maxLength, including padding.

    A single context longer than the declared batch token budget is preserved
    whole and run alone. It is never truncated or split into another context.
    """
    contexts, _ = checked_contexts(contexts, positions)
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError('Explicit batch size1..16 required')
    if type(max_context_tokens) is not int or max_context_tokens < 1:
        raise ValueError('Positive explicit padded full-context token budget required')
    order = sorted(range(len(contexts)), key=lambda i:(len(contexts[i]), i))
    result, current = [], []
    for index in order:
        length = len(contexts[index])
        if current and (len(current) >= batch_size or (len(current)+1)*length > max_context_tokens):
            result.append(current); current = []
        current.append(index)
        if length > max_context_tokens:
            if len(current) != 1:raise AssertionError('Oversized contexts must be isolated')
            result.append(current); current = []
    if current:result.append(current)
    assert sorted(i for batch in result for i in batch) == list(range(len(contexts)))
    return result


class BatchedContextCapture:
    """Frozen AutoModel capture adapter with bounded remote-RAM replay windows."""
    def __init__(self, model, batch_size=16, max_context_tokens=16384, device=None,
                 pad_token_id=None, expected_hidden_size=r.DIM):
        import torch
        plan_context_batches([], batch_size=batch_size, max_context_tokens=max_context_tokens)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Caller must supply an eval(), requires_grad_(False) frozen actor')
        config = model.config
        if config.num_hidden_layers != 28 or config.hidden_size != expected_hidden_size:
            raise ValueError('Exactly28 native decoder blocks and matching declared hidden size required')
        if device is None:
            first = next(iter(model.parameters()), None)
            device = first.device if first is not None else torch.device('cpu')
        self.device, self.model = torch.device(device), model
        self.batch_size, self.max_context_tokens = batch_size, max_context_tokens
        self.hidden_size = expected_hidden_size
        self.pad_token_id = (getattr(config, 'pad_token_id', None) or 0) if pad_token_id is None else pad_token_id
        if type(self.pad_token_id) is not int or self.pad_token_id < 0:
            raise ValueError('Nonnegative explicit padding token ID required')
        self.total_model_calls = self.total_contexts = self.total_valid_tokens = self.total_padded_tokens = 0
        self.total_seconds = 0.; self.last_capture = None

    def capture_many(self, contexts, positions=None):
        """Return per-context {20,23,26: detached CPU BF16[actualLength,D]}.

        Returned order matches input order even though inference uses length
        buckets. Padding has an explicit zero attention mask and is discarded.
        """
        import torch
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Frozen actor contract changed before prefill')
        contexts, positions = checked_contexts(contexts, positions)
        batches = plan_context_batches(contexts, positions, self.batch_size, self.max_context_tokens)
        result = [None]*len(contexts); started = time.monotonic(); padded = 0
        with torch.inference_mode():
            for batch in batches:
                width = max(len(contexts[i]) for i in batch); padded += len(batch)*width
                ids = torch.full((len(batch), width), self.pad_token_id, dtype=torch.long, device=self.device)
                mask = torch.zeros_like(ids); rope = torch.zeros_like(ids)
                for local, index in enumerate(batch):
                    n = len(contexts[index]); ids[local, :n] = torch.tensor(contexts[index], dtype=torch.long, device=self.device)
                    mask[local, :n] = 1; rope[local, :n] = torch.tensor(positions[index], dtype=torch.long, device=self.device)
                output = self.model(input_ids=ids, attention_mask=mask, position_ids=rope,
                    use_cache=False, output_hidden_states=True, return_dict=True)
                states = output.hidden_states
                if len(states) != 29:raise ValueError('HF residual boundary must be hidden_states[layer+1] of29 entries')
                for index in batch:result[index] = {}
                for layer in r.LAYERS:
                    value = states[layer+1]
                    if value.shape != (len(batch), width, self.hidden_size) or value.requires_grad:
                        raise ValueError('Captured native block tensor geometry/gradient state differs')
                    # Three device copies per batch, rather than one copy per
                    # layer per row. Clone cropped CPU slices so callers cannot
                    # retain an entire padded batch through a small tensor view.
                    cpu = value.detach().to('cpu', dtype=torch.bfloat16, copy=True)
                    for local, index in enumerate(batch):
                        own = cpu[local, :len(contexts[index])].clone()
                        if not bool(torch.isfinite(own).all()):raise ValueError('Nonfinite real causal hidden state')
                        result[index][layer] = own
                    del cpu
                del output, states, value, ids, mask, rope
        elapsed = time.monotonic()-started
        self.total_model_calls += len(batches); self.total_contexts += len(contexts)
        self.total_valid_tokens += sum(map(len, contexts)); self.total_padded_tokens += padded; self.total_seconds += elapsed
        self.last_capture = dict(contexts=len(contexts), model_calls=len(batches), valid_tokens=sum(map(len, contexts)),
            padded_tokens=padded, batch_sizes=[len(b) for b in batches], batch_padded_widths=[max(len(contexts[i]) for i in b) for b in batches],
            oversized_single_contexts=sum(len(c)>self.max_context_tokens for c in contexts), elapsed_seconds=elapsed,
            capture_layers=list(r.LAYERS), hf_hidden_state_indices=[i+1 for i in r.LAYERS],
            actor_optimizer_updates=0, contexts_truncated=0, native_bitexact_claim=False, scalar_batch_bitexact_claim=False)
        return result

    def replay_many(self, normalized_rows):
        """Gather a bounded list with the unchanged legacy mask/position logic."""
        rows = list(islice(iter(normalized_rows), 65))
        if len(rows)>64:raise ValueError('Replay window exceeds64 trajectories; use iter_replay to bound RAM')
        contexts = [list(segment['context_tokens']) for row in rows for segment in row['segments']]
        captured = self.capture_many(contexts); offset = 0; result = []
        for row in rows:
            count = len(row['segments']); position = [0]
            def saved_capture(context, positions):
                index = offset+position[0]
                if position[0] >= count or list(context) != contexts[index] or list(positions) != list(range(len(context))):
                    raise ValueError('Legacy gather requested another context/order/position')
                position[0] += 1; return captured[index]
            result.append(r.replay_features(row, saved_capture, hidden_size=self.hidden_size))
            if position[0] != count:raise ValueError('Not every saved causal segment was gathered')
            offset += count
        if offset != len(captured):raise ValueError('Unexpected extra captured causal context')
        return result

    def iter_replay(self, normalized_rows, window_trajectories=16):
        """Yield (unchanged normalized row, features) without buffering a bank."""
        if type(window_trajectories) is not int or not 1 <= window_trajectories <= 64:
            raise ValueError('Explicit replay window1..64 trajectories required')
        iterator = iter(normalized_rows)
        while True:
            rows = list(islice(iterator, window_trajectories))
            if not rows:return
            values = self.replay_many(rows)
            for row, features in zip(rows, values):yield row, features
            del values, rows

    def stats(self):
        return dict(model_calls=self.total_model_calls, contexts=self.total_contexts,
            valid_tokens=self.total_valid_tokens, padded_tokens=self.total_padded_tokens,
            elapsed_seconds=self.total_seconds, actor_optimizer_updates=0,
            native_bitexact_claim=False, scalar_batch_bitexact_claim=False)


def compare_replay_features(scalar, batched):
    """Small numeric-only report; no tensors/tokens/labels in returned output."""
    import torch
    scalar, batched = list(scalar), list(batched)
    if len(scalar) != len(batched) or not scalar:raise ValueError('Same nonempty ordered replay rows required')
    metrics = {layer:{kind:r.Comparison() for kind in ('all_retained','initial_question','policy_prefix20')} for layer in r.LAYERS}
    for first, second in zip(scalar, batched):
        for layer in r.LAYERS:
            a, b = first[layer], second[layer]
            if a['question_length']!=b['question_length'] or a['policy_length']!=b['policy_length'] or not torch.equal(a['attention_mask'],b['attention_mask']) or not torch.equal(a['position_ids'],b['position_ids']):
                raise ValueError('Scalar/batched retained masks, positions or lengths differ')
            qlen, plen = a['question_length'], a['policy_length']
            metrics[layer]['all_retained'].add(b['features'][b['attention_mask']],a['features'][a['attention_mask']])
            metrics[layer]['initial_question'].add(b['features'][:qlen],a['features'][:qlen])
            metrics[layer]['policy_prefix20'].add(b['features'][r.QMAX:r.QMAX+plen],a['features'][r.QMAX:r.QMAX+plen])
    return dict(trajectories=len(scalar), layers={str(layer):{kind:obj.result() for kind,obj in values.items()} for layer,values in metrics.items()},
        masks_positions_lengths_exact=True,native_bitexact_claim=False,scalar_batch_bitexact_claim=False,
        numeric_acceptance_pending=True,raw_tensors_returned=False)
