"""CPU gradient/parity checks for the I2 objective and actual I4 monitor calls.

No model download, GPU, rollout, baseline regeneration or admission is used.
The PPO callable is the real repository function, loaded without optional Ray/
TensorDict runtime imports; the frozen-monitor statistics are the original code.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from .actor import (ActorBudget, LOSS_REDUCTION, OriginalPolicyMonitor,
                    original_pair_policy_objective, pair_token_masses,
                    frozen_policy_monitor)


def _actual_function(path, name, globals):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), globals)
    return globals[name]


ROOT = Path(__file__).resolve().parents[2]
MASKED_MEAN = _actual_function(ROOT / 'verl/utils/torch_functional.py', 'masked_mean', dict(torch=torch))
PPO = _actual_function(ROOT / 'verl/trainer/ppo/core_algos.py', 'compute_policy_loss',
                       dict(torch=torch, verl_F=SimpleNamespace(masked_mean=MASKED_MEAN)))


def _rows(n=16, *, empty=False):
    rows = []
    for i in range(n):
        # Alternating authored policy spans and actual external observations.
        mask = [1] * (i % 7 + 1) + [0] * 3 + [1] * (i % 5 + 1)
        if empty: mask = [0] * len(mask)
        rows.append(dict(trajectory_id='tid-%d' % i, initial_prompt_token_ids=[11, 12],
            response_token_ids=list(range(21, 21 + len(mask))), response_info_mask=mask,
            response_log_probs=[-.1 if m else 0. for m in mask]))
    return rows


class OriginalObjectiveTests(unittest.TestCase):
    def test_true_ppo_loss_and_gradients_match_legacy_micro2_at_every_physical_size(self):
        torch.manual_seed(17)
        shape = (16, 11)
        old = torch.randn(shape, dtype=torch.float64)
        current = (old + torch.randn(shape, dtype=torch.float64) * .7).requires_grad_()
        advantages = torch.randn((16, 1), dtype=torch.float64).expand(shape)
        mask = torch.zeros(shape, dtype=torch.int64)
        for i in range(16): mask[i, :1 + i % 11] = 1
        # One zero-token EOS pair plus masked tool observations inside a pair.
        mask[4:6] = 0; mask[8, 1:4] = 0
        masses = [int(mask[i:i + 2].sum()) for i in range(0, 16, 2)]
        reference = current.sum() * 0.
        for i in range(0, 16, 2):
            if masses[i // 2]:
                reference = reference + PPO(old[i:i + 2], current[i:i + 2], advantages[i:i + 2],
                    mask[i:i + 2], cliprange=.2, dual_clip=3.)[0] / 8
        expected_gradient = torch.autograd.grad(reference, current)[0]
        for physical in (16, 8, 4, 2, 1):
            with self.subTest(physical_micro=physical):
                actual_current = current.detach().clone().requires_grad_()
                actual = actual_current.sum() * 0.
                for start in range(0, 16, physical):
                    stop = start + physical
                    loss, _clip, _kl, details = original_pair_policy_objective(
                        old[start:stop], actual_current[start:stop], advantages[start:stop], mask[start:stop],
                        pair_masses=masses, row_offset=start, loss_fn=PPO)
                    self.assertTrue(all(d['original_pair_token_mass'] == masses[d['pair_index']] for d in details))
                    actual = actual + loss
                gradient = torch.autograd.grad(actual, actual_current)[0]
                torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(gradient, expected_gradient, rtol=1e-13, atol=1e-13)
                self.assertTrue(torch.equal(gradient[~mask.bool()], torch.zeros_like(gradient[~mask.bool()])))
        # These lengths demonstrably distinguish the repaired objective from the defect.
        wrong = PPO(old, current, advantages, mask, cliprange=.2, dual_clip=3.)[0]
        self.assertGreater(abs(float((wrong - reference).detach())), 1e-4)

    def test_pair_identity_and_policy_mass_include_all_turns_exclude_observations(self):
        rows = _rows(); ids = list(reversed(range(16)))
        masses = pair_token_masses(rows, ids)
        self.assertEqual(masses, [sum(sum(rows[i]['response_info_mask']) for i in ids[j:j + 2])
                                  for j in range(0, 16, 2)])
        with self.assertRaises(ValueError): pair_token_masses(rows, [0] * 16)
        self.assertIn('original_two_row', LOSS_REDUCTION)

    def test_all_empty_pair_zero_with_grad_and_unchanged_eight_pair_denominator(self):
        current = torch.full((16, 1), 3., requires_grad=True)
        mask = torch.zeros_like(current, dtype=torch.int64)
        loss, clip, kl, details = original_pair_policy_objective(current.detach(), current,
            current.detach(), mask, pair_masses=[0] * 8, row_offset=0, loss_fn=PPO)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.); self.assertEqual(float(clip), 0.); self.assertEqual(float(kl), 0.)
        self.assertEqual(len(details), 8)
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))


class MonitorTests(unittest.TestCase):
    def make_monitor(self, directory, *, empty=False):
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): module.weight.fill_(.1)
        forwards = []
        def forward(batch, **kwargs):
            forwards.append(tuple(batch['responses'].shape))
            return None, module.weight[0, 0].expand_as(batch['old_log_probs'])
        actor = SimpleNamespace(actor_module=module, _forward_micro_batch=forward)
        monitor = OriginalPolicyMonitor(actor, _rows(empty=empty), [list(range(16))], [-1.] * 16,
            SimpleNamespace(pad_token_id=0), budget=ActorBudget(), device='cpu',
            run_id='fixture', outer=1, rank=0, actor_version='official0', ledger=Path(directory) / 'ledger.jsonl')
        return monitor, module, forwards

    def test_real_before_after_and_adjacent_cache_reuse_preserve_model_training_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor, module, calls = self.make_monitor(tmp)
            before = monitor.observe('before_minibatch', 0, 0)
            with torch.no_grad(): module.weight.add_(.1)
            after = monitor.observe('after_minibatch', 0, 1)
            reused = monitor.observe('before_minibatch', 1, 1)
            self.assertTrue(module.training)
            self.assertEqual(len(calls), 2)
            self.assertFalse(before['forward_reused']); self.assertFalse(after['forward_reused'])
            self.assertTrue(reused['forward_reused'])
            self.assertEqual(reused['sampled_old_policy_kl'], after['sampled_old_policy_kl'])
            # A real parameter mutation invalidates the after cache.
            monitor.observe('after_minibatch', 1, 2)
            with torch.no_grad(): module.weight.add_(.01)
            self.assertFalse(monitor.observe('before_minibatch', 2, 2)['forward_reused'])
            recorded = [json.loads(s) for s in monitor.ledger.read_text().splitlines()]
            self.assertEqual([r['stage'] for r in recorded[:3]],
                             ['before_minibatch', 'after_minibatch', 'before_minibatch'])
            self.assertEqual(len(recorded), 5)
            self.assertEqual(recorded[2]['forward_source']['actual_adam_steps'], 1)

    def test_finite_kl_tail_breaches_warn_and_do_not_skip_or_change_dual_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor, module, calls = self.make_monitor(tmp)
            with torch.no_grad(): module.weight.fill_(2.)
            report = monitor.observe('before_minibatch', 0, 0)
            self.assertTrue(report['allowed'])
            self.assertIn('old_policy_kl', report['diagnostic_warnings'])
            self.assertIn('negative_advantage_tail', report['diagnostic_warnings'])
            self.assertEqual(report['contract']['dual_clip'], 3.)
            self.assertEqual(report['contract']['finite_thresholds_diagnostic_only'], True)
            self.assertEqual(len(calls), 1)

    def test_actual_nonfinite_effective_input_rejects_and_retains_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor, module, _calls = self.make_monitor(tmp)
            with torch.no_grad(): module.weight.fill_(float('nan'))
            with self.assertRaisesRegex(FloatingPointError, 'Nonfinite actual old-policy monitor'):
                monitor.observe('before_minibatch', 0, 0)
            report = json.loads(monitor.ledger.read_text())
            self.assertFalse(report['allowed'])
            self.assertGreater(report['invalid_count'], 0)
            self.assertIsNone(report['sampled_old_policy_kl'])

    def test_legitimate_all_eos_has_same_four_row_forward_schedule_and_explicit_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor, module, calls = self.make_monitor(tmp, empty=True)
            report = monitor.observe('before_minibatch', 0, 0)
            self.assertTrue(report['allowed']); self.assertFalse(report['observation_available'])
            self.assertIsNone(report['sampled_old_policy_kl'])
            self.assertEqual(len(report['monitor_identity']['trajectory_ids']), 4)
            self.assertEqual(calls[0][0], 4)

    def test_monitor_pair_weights_do_not_depend_on_physical_oom_micro(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor, _module, calls = self.make_monitor(tmp)
            expected = monitor.observe('before_minibatch', 0, 0)
            for micro in (64, 32, 16, 8):
                monitor.budget = ActorBudget(micro)
                result = monitor.observe('after_minibatch', 0, 1)
                self.assertAlmostEqual(result['sampled_old_policy_kl'], expected['sampled_old_policy_kl'], places=12)
                self.assertEqual(result['negative_tail_weight_fraction'], expected['negative_tail_weight_fraction'])


if __name__ == '__main__': unittest.main()
