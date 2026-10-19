"""Pinned reference native residual capture, parameterized for official 7B.

Only representation dimensions/depth differ; authentic request/token alignment,
post-block residual sum and hidden-flush checks are retained from the reference.
"""
from pathlib import Path
import types
import os
DIM = 3584
PREFIX = int(os.environ.get('SEARCH_G1_B_PREFIX_TOKENS','20'))
QMAX = int(os.environ['QWEN7B_QUESTION_MAX_TOKENS'])
WIDTH = QMAX + PREFIX
FEATURE_LAYER = 26

def query_mapping(metadata, query_lens, tokens, positions):
    """Actual prefill->decode ordered query rows; n=1, TP1, no graph padding."""
    result, offset, lens = [], 0, iter(query_lens)
    for group in metadata:
        if len(group.seq_data) != 1:
            raise ValueError('Capture requires independently sampled n1 requests')
        n = next(lens)
        if type(n) is not int or n < 1:
            raise ValueError('Native query length must be positive')
        result.append(dict(request_id=str(group.request_id), start=offset, end=offset+n,
                           tokens=tokens[offset:offset+n], positions=positions[offset:offset+n]))
        offset += n
    if list(lens) or offset != len(tokens) or offset != len(positions):
        raise ValueError('Native request/query/token alignment differs')
    return result

class ActorHiddenCapture:
    """Per-worker CPU buffers; GPU tensors live only for one native model call."""
    def __init__(self, engine, actor_version):
        import torch
        self.torch = torch
        self.actor_version = actor_version
        self.requests, self.trajectories = {}, {}
        self.calls, self.captured_rows, self.elapsed_seconds = 0, 0, 0.0
        self.query_rows, self.native_hidden = None, None
        self.runner = engine.model_executor.driver_worker.model_runner
        runner = self.runner
        if not runner.model_config.enforce_eager or runner.cache_config.enable_prefix_caching:
            raise ValueError('Actual question capture requires eager execution/no skipped prefill hidden')
        model = runner.model.model
        if len(model.layers) != 28 or model.config.hidden_size != DIM:
            raise ValueError('Require official28-layer Qwen2.5-7B')
        self.hook = model.layers[FEATURE_LAYER].register_forward_hook(self._layer_output)
        self.original_prepare = runner.prepare_model_input
        self.original_execute = runner.execute_model
        capture = self

        def prepare(_runner, metadata, *args, **kwargs):
            value = capture.original_prepare(metadata, *args, **kwargs)
            tokens = value.input_tokens.detach().cpu().tolist()
            positions = value.input_positions.detach().cpu().tolist()
            capture.query_rows = query_mapping(metadata, value.query_lens, tokens, positions)
            return value

        def execute(_runner, *args, **kwargs):
            import time
            start = time.monotonic()
            capture.native_hidden = None
            value = capture.original_execute(*args, **kwargs)
            capture._consume_native()
            capture.elapsed_seconds += time.monotonic()-start
            return value

        runner.prepare_model_input = types.MethodType(prepare, runner)
        runner.execute_model = types.MethodType(execute, runner)

    def register(self, state, request):
        """Bind positions before add_request; only actual causal inputs qualify."""
        torch = self.torch
        tid, qlen = state['trajectory_id'], len(state['prompt'])
        if state['actor_version'] != self.actor_version or not 1 <= qlen <= QMAX:
            raise ValueError('Actor version/question width differs from capture')
        if tid not in self.trajectories:
            self.trajectories[tid] = dict(features=torch.zeros(WIDTH, DIM, dtype=torch.bfloat16),
                filled=torch.zeros(WIDTH, dtype=torch.bool), positions=torch.zeros(WIDTH, dtype=torch.long),
                token_ids=[None]*WIDTH, question_length=qlen)
        action = state['action']
        self.requests[request['request_id']] = dict(trajectory_id=tid,
            request_tokens=list(request['actual_request_token_ids']),
            logical_request_length=len(action['actual_request_token_ids']),
            retained_before_turn=sum(state['info_mask']), turn_index=state['ordinal'])

    def _layer_output(self, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Unexpected native Qwen decoder residual contract')
        hidden, residual = output
        self.native_hidden = (hidden + residual).detach()

    def _consume_native(self):
        torch = self.torch
        if self.native_hidden is None or self.query_rows is None:
            raise ValueError('Actual sampling forward did not capture layer27')
        hidden = self.native_hidden
        if hidden.ndim != 2 or hidden.shape[1] != DIM:
            raise ValueError('Actual actor hidden shape differs')
        wanted, targets = [], []
        for query in self.query_rows:
            request = self.requests.get(query['request_id'])
            if request is None:
                raise ValueError('Native forward contains an unbound request')
            data = self.trajectories[request['trajectory_id']]
            for j, (token, pos) in enumerate(zip(query['tokens'], query['positions'])):
                slot = None
                if request['turn_index'] == 0 and 0 <= pos < data['question_length']:
                    slot = pos
                elif pos >= request['logical_request_length']:
                    policy = request['retained_before_turn'] + pos-request['logical_request_length']
                    if 0 <= policy < PREFIX:
                        slot = QMAX+policy
                if slot is not None:
                    # All positions inside the original request are token bound;
                    # subsequent decode tokens are bound to the saved prefix at finish.
                    if pos < len(request['request_tokens']) and token != request['request_tokens'][pos]:
                        raise ValueError('Prepared actor input differs from registered real token IDs')
                    wanted.append(query['start']+j)
                    targets.append((data, slot, token, pos))
        if wanted:
            indices = torch.tensor(wanted, dtype=torch.long, device=hidden.device)
            values = hidden.index_select(0, indices).to(device='cpu', dtype=torch.bfloat16)
            for value, (data, slot, token, pos) in zip(values, targets):
                data['features'][slot].copy_(value)
                data['filled'][slot] = True
                data['positions'][slot] = pos
                data['token_ids'][slot] = token
            self.captured_rows += len(wanted)
        self.calls += 1
        self.native_hidden, self.query_rows = None, None

    def finish(self, prefix):
        """Only retained5/short-EOS tokens; no sampled lookahead can enter head."""
        torch = self.torch
        tid = prefix['trajectory_id']; data = self.trajectories.pop(tid)
        qlen, plen = data['question_length'], prefix['retained_policy_length']
        if not 0 <= plen <= PREFIX:
            raise ValueError('Native prefix includes tokens beyond the retained first5')
        slots = list(range(qlen)) + list(range(QMAX,QMAX+plen))
        tokens = prefix['initial_prompt_token_ids'] + prefix['policy_token_ids']
        positions = list(range(qlen)) + prefix['policy_position_ids']
        if prefix['actor_version'] != self.actor_version:
            raise ValueError('Captured hidden belongs to another actor version')
        if (not bool(data['filled'][slots].all()) or [data['token_ids'][j] for j in slots] != tokens
                or data['positions'][slots].tolist() != positions):
            raise ValueError('Missing/off-by-one actor hidden or changed retained prefix')
        mask = torch.zeros(WIDTH, dtype=torch.bool); mask[slots] = True
        data['features'][~mask] = 0; data['positions'][~mask] = 0
        return dict(features=data['features'], attention_mask=mask, position_ids=data['positions'],
                    question_length=qlen, policy_length=plen)

    def forget_request(self, request_id):
        self.requests.pop(request_id, None)

    def close(self):
        self.hook.remove()
        self.runner.prepare_model_input = self.original_prepare
        self.runner.execute_model = self.original_execute
