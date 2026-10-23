"""Small CPU numerical checks; no model weights or tensor files are created."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from reproduction.dynrpo7b.group_probe_adapter import (
    GroupProbeAdapter, pack_group, rng_state, target4variance)
from reproduction.dynrpo7b.prefix_schedule import DEFAULT_SCHEDULE
from reproduction.dynrpo7b.probe_update import ProbeUpdater


class _TinyProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(1, 1)
        self.group_head = nn.Module()
        self.group_head.positions = nn.Embedding(2048, 1)

    def forward(self, batch):
        return self.block(batch['features']).squeeze(-1)


def _tiny_adapter():
    torch.manual_seed(5)
    probe = _TinyProbe()
    ema = copy.deepcopy(probe).requires_grad_(False)
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-6)
    probe({'features': torch.ones(1, 1)}).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    for state in optimizer.state.values():
        state['step'].fill_(11889)
    adapter = GroupProbeAdapter(nn.Identity(), probe, ema, optimizer, [7], 'cpu',
        dict(binding={}, cursor=11889, EMA_updates=11889,
             RNG_by_rank=[rng_state('cpu')]))
    adapter.capture = types.MethodType(lambda self, groups, prefix: {
        'features': torch.tensor([[float(int(g['qid']) % 7) / 7] for g in groups])}, adapter)
    return adapter


class ProbeCPUContractTest(unittest.TestCase):
    def test_original_warmup_pack_golden_and_no_em_leak(self):
        group = dict(qid='q', prompt_token_ids=[11, 12, 13],
            policy_token_ids=[[21] * 80, [22] * 8, [23] * 15, [24] * 3, [25] * 79],
            strict_em=[0, 1, 0, 1, 0])
        packed = pack_group(group, [30, 31], 80)
        packed.pop('valid_positions')
        # Golden obtained directly from the accepted warmup's pack_group.
        digest = hashlib.sha256(json.dumps(packed, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, '9a79a3ceb47d88d766cc9fb1873bf064a86d01bd5d8c83590a30fee5171ed20c')
        changed = dict(group, strict_em=[1, 1, 1, 1, 1])
        self.assertEqual(pack_group(group, [30, 31], 80), pack_group(changed, [30, 31], 80))

    def test_variance_targets_and_100_boundary(self):
        for count, target in enumerate([0, .64, .96, .96, .64, 0]):
            self.assertAlmostEqual(target4variance([1] * count + [0] * (5 - count)), target)
        s = DEFAULT_SCHEDULE
        self.assertEqual([s.selection_prefix(x) for x in [1, 100, 101, 200]], [80, 80, 160, 160])
        self.assertEqual([s.update_prefix(x) for x in [10, 90, 100, 190]], [80, 80, 160, 160])
        self.assertFalse(s.update_due(200))

    def test_three_epochs_and_resume_keep_adam_and_raw_ema(self):
        groups = [dict(qid=str(i), prompt_token_ids=[1, 2],
            policy_token_ids=[[j + 3] for j in range(5)],
            strict_em=[1] * (i % 6) + [0] * (5 - i % 6)) for i in range(2560)]
        snapshots = []

        def save_in_memory(adapter, path):
            snapshots.append(copy.deepcopy(adapter.state_dict()))
            return dict(path=str(path), sha256='test-memory-only')

        with tempfile.TemporaryDirectory() as tmp, \
                patch('reproduction.dynrpo7b.probe_update.save_probe_checkpoint', save_in_memory), \
                patch('reproduction.dynrpo7b.probe_update.atomic_json'), \
                contextlib.redirect_stdout(io.StringIO()):
            first = _tiny_adapter()
            actor_rng_before = rng_state('cpu')
            ProbeUpdater(first).fit(groups, completed_actor_outer=100, rollout_actor_outer=99,
                actor_checkpoint_ref=dict(path='actor100'), output_dir=tmp)
            self.assertTrue(torch.equal(actor_rng_before['torch'], rng_state('cpu')['torch']))
            resumed = _tiny_adapter()
            resumed.load_state_dict(copy.deepcopy(snapshots[0]))
            ProbeUpdater(resumed).fit(groups, completed_actor_outer=100, rollout_actor_outer=99,
                actor_checkpoint_ref=dict(path='actor100'), output_dir=tmp)
        self.assertEqual(first.online_updates, 60)
        self.assertEqual(resumed.online_updates, 60)
        self.assertEqual(resumed.current_prefix, 160)
        self.assertEqual(resumed.completed_update_outer, 100)
        self.assertIsNone(resumed.pending_update)
        for original, restored in [(first.probe, resumed.probe), (first.ema, resumed.ema)]:
            for key, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[key]), key)
        for state in resumed.optimizer.state.values():
            self.assertEqual(int(state['step']), 11949)

        # A fresh actor-checkpoint restore still discovers and loads a later
        # complete probe epoch before doing any additional optimizer updates.
        epoch1 = snapshots[0]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('reproduction.dynrpo7b.probe_update.save_probe_checkpoint', save_in_memory), \
                patch('reproduction.dynrpo7b.probe_update.atomic_json'), \
                patch('reproduction.dynrpo7b.probe_update._restore_epoch',
                      side_effect=lambda adapter, ref: adapter.load_state_dict(copy.deepcopy(epoch1))), \
                contextlib.redirect_stdout(io.StringIO()):
            (Path(tmp) / 'probe-epoch-01.pt.json').write_text(json.dumps(dict(
                path='memory-epoch1', sha256='test-memory-only',
                pending_update=epoch1['pending_update'])))
            recovered = _tiny_adapter()
            result = ProbeUpdater(recovered).fit(groups, completed_actor_outer=100,
                rollout_actor_outer=99, actor_checkpoint_ref=dict(path='actor100'), output_dir=tmp)
            self.assertEqual([v['epoch'] for v in result['epochs']], [1, 2, 3])
            self.assertEqual(recovered.online_updates, 60)
            for key, tensor in first.probe.state_dict().items():
                self.assertTrue(torch.equal(tensor, recovered.probe.state_dict()[key]), key)

    def test_position_extension_keeps_existing_adam_coordinates(self):
        adapter = _tiny_adapter()
        old = adapter.probe.group_head.positions.weight
        before = old.detach().clone()
        adapter.optimizer.state[old] = dict(step=torch.tensor(11889.),
            exp_avg=torch.ones_like(old), exp_avg_sq=torch.full_like(old, 2.))
        adapter.ensure_position_capacity(2200)
        new = adapter.probe.group_head.positions.weight
        state = adapter.optimizer.state[new]
        self.assertTrue(torch.equal(new[:2048], before))
        self.assertTrue(torch.equal(state['exp_avg'][:2048], torch.ones_like(old)))
        self.assertEqual(float(state['step']), 11889.)
        self.assertEqual(int(state['exp_avg'][2048:].count_nonzero()), 0)
        self.assertTrue(any(p is new for group in adapter.optimizer.param_groups for p in group['params']))

    def test_driver_rpc_payload_and_rank_score_merge(self):
        from reproduction.dynrpo7b.runtime import ProbeRuntime

        class FakeWorkerGroup:
            def __init__(self):
                self.calls = []

            def dynrpo7b_probe_operation(self, operation, payload):
                self.calls.append((operation, payload))
                if operation == 'init':
                    self.asserted_init = payload['model_dir']
                    return [dict(rank=r, initialized=True) for r in range(4)]
                if operation == 'predict':
                    return [dict(rank=r, predictions=[dict(qid=g['qid'], score=int(g['qid']) / 10)
                        for g in payload['groups'][r::4]]) for r in range(4)]
                raise AssertionError(operation)

        with tempfile.TemporaryDirectory() as tmp:
            workers = FakeWorkerGroup()
            runtime = ProbeRuntime(dict(output_dir=tmp, model_dir='official',
                checkpoint_path='E9', checkpoint_sha256='source-sha'), tmp, workers)
            self.assertEqual(runtime.score_groups([dict(qid=str(i)) for i in range(8)], prefix=80),
                             [i / 10 for i in range(8)])
            self.assertEqual([op for op, _ in workers.calls], ['init', 'predict'])


if __name__ == '__main__':
    unittest.main()
