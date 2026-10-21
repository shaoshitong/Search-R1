"""Tiny CPU actor mechanics; these tests do not qualify 7B CUDA/FSDP numerics.

The local test runtime has torch/numpy but not the unrelated verl/tensordict
stack. Only for these fixtures, load the exact existing masked_mean and
compute_policy_loss function ASTs instead of importing package-level extras.
Production helper imports the real core_algos module normally.
"""
import ast
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from search_g1 import sparse_policy_qualification as module


def actual_policy_loss():
    root = Path(__file__).resolve().parents[1]
    namespace = dict(torch=torch)
    def function(path, name, globals_):
        tree = ast.parse((root/path).read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), globals_)
        return globals_[name]
    masked_mean = function(Path('verl/utils/torch_functional.py'), 'masked_mean', namespace)
    namespace = dict(torch=torch, verl_F=SimpleNamespace(masked_mean=masked_mean))
    return function(Path('verl/trainer/ppo/core_algos.py'), 'compute_policy_loss', namespace)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.15)
        self.head = torch.nn.Linear(5, 11).double()
        with torch.no_grad():
            self.head.weight.mul_(.03); self.head.bias.mul_(.03)


class TinyActor:
    def __init__(self, *, sparse_offset=0.0, fail_call=None):
        self.actor_module = TinyPolicy()
        self.actor_module.eval()
        # Preserve an intentionally mixed initial mode, too.
        self.actor_module.dropout.train(True)
        self.config = SimpleNamespace(ppo_micro_batch_size=8, ppo_mini_batch_size=32,
            ppo_logical_micro_batch_size=4, entropy_coeff=0., kl_loss_coef=0.,
            state_masking=True, clip_ratio=.2,
            policy_safety=SimpleNamespace(enabled=True, dual_clip=3.))
        self._selected_policy_logits = True
        self._actual_precision_logged = False
        self.last_forward_precision = dict(original='must survive qualification')
        self.calls = []; self.noise = []
        self.sparse_offset = sparse_offset; self.fail_call = fail_call
        self.optimizer_steps = self.scheduler_steps = 0

    def _forward_micro_batch(self, micro, temperature, compute_entropy):
        self.calls.append((self._selected_policy_logits, temperature, compute_entropy))
        noise = random.random() + float(np.random.random()) + float(torch.rand(()))
        self.noise.append(noise)
        self._actual_precision_logged = self._selected_projection_logged = True
        self.last_forward_precision = dict(temporary=True)
        if len(self.calls) == self.fail_call:
            raise RuntimeError('Injected independent diagnostic worker failure')
        hidden = self.actor_module.dropout(micro['features']) + noise * .001
        labels = micro['responses']; mask = micro['loss_mask'].bool()
        if self._selected_policy_logits:
            logits = self.actor_module.head(hidden[mask]) / temperature
            selected = logits.log_softmax(-1).gather(-1, labels[mask].unsqueeze(-1)).squeeze(-1)
            log_prob = selected.new_zeros(labels.shape).masked_scatter(mask, selected)
            log_prob = log_prob + self.sparse_offset * mask
        else:
            logits = self.actor_module.head(hidden) / temperature
            log_prob = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return None, log_prob


class SparsePolicyQualificationTests(unittest.TestCase):
    def setUp(self):
        random.seed(3021); np.random.seed(3021); torch.manual_seed(3021)
        self.micro = dict(features=torch.randn(8, 6, 5, dtype=torch.float64),
            responses=torch.randint(11, (8, 6)),
            loss_mask=torch.tensor([[0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0]] * 4),
            old_log_probs=torch.full((8, 6), -np.log(11), dtype=torch.float64),
            advantages=torch.linspace(-.7, .8, 48, dtype=torch.float64).reshape(8, 6))
        self.loss = actual_policy_loss()

    def run_qualification(self, actor):
        with patch.object(module, '_policy_loss_function', return_value=self.loss):
            return module.qualify_sparse_policy(actor, self.micro, temperature=1., expected_world_size=1)

    def state(self, actor):
        return dict(rng=module._rng(torch.device('cpu')),
            parameters={name:p.detach().clone() for name,p in actor.actor_module.named_parameters()},
            modes=[child.training for child in actor.actor_module.modules()],
            precision=copy.deepcopy(actor.last_forward_precision),
            micro={name:value.clone() for name,value in self.micro.items()})

    def assert_clean(self, actor, before):
        self.assertTrue(actor._selected_policy_logits)
        self.assertFalse(actor._actual_precision_logged)
        self.assertFalse(hasattr(actor, '_selected_projection_logged'))
        self.assertEqual(actor.last_forward_precision, before['precision'])
        self.assertEqual([child.training for child in actor.actor_module.modules()], before['modes'])
        after_rng = module._rng(torch.device('cpu'))
        self.assertEqual(after_rng['python'], before['rng']['python'])
        self.assertTrue(np.array_equal(after_rng['numpy'][1], before['rng']['numpy'][1]))
        self.assertEqual(after_rng['numpy'][2:], before['rng']['numpy'][2:])
        self.assertTrue(torch.equal(after_rng['torch_cpu'], before['rng']['torch_cpu']))
        for name, parameter in actor.actor_module.named_parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(parameter, before['parameters'][name]))
        for name,value in self.micro.items():
            self.assertTrue(torch.equal(value,before['micro'][name]))
        self.assertEqual((actor.optimizer_steps, actor.scheduler_steps), (0, 0))

    def test_same_rng_dense_repeats_sparse_loss_and_all_gradients_match(self):
        actor = TinyActor(); before = self.state(actor)
        result = self.run_qualification(actor)
        self.assertTrue(result['accepted'])
        self.assertEqual([row[0] for row in actor.calls], [False, False, False, True])
        self.assertEqual(len(set(actor.noise)), 1)
        self.assertTrue(all(not row[2] for row in actor.calls))
        expected = sum(parameter.numel() for parameter in actor.actor_module.parameters())
        for comparison in result['comparisons'].values():
            self.assertTrue(comparison['accepted'])
            self.assertEqual(comparison['gradients']['global_compared_gradient_elements'], expected)
            self.assertEqual(comparison['log_probabilities']['global_authored_tokens'], int(self.micro['loss_mask'].sum()))
        json.dumps(result, allow_nan=False)
        self.assert_clean(actor, before)

    def test_fixed_thresholds_reject_sparse_difference_and_leave_actor_clean(self):
        actor = TinyActor(sparse_offset=.005); before = self.state(actor)
        result = self.run_qualification(actor)
        self.assertFalse(result['accepted'])
        self.assertIn('log_probability_max_abs', result['comparisons']['D1_S1']['failed_checks'])
        self.assertEqual(result['thresholds'], dict(module.THRESHOLDS))
        self.assertTrue(all(row['accepted'] for row in result['dense_repeat_noise'].values()))
        self.assert_clean(actor, before)

    def test_worker_failure_propagates_without_retry_and_restores_flags_rng_and_grads(self):
        actor = TinyActor(fail_call=2); before = self.state(actor)
        with self.assertRaisesRegex(RuntimeError, 'independent diagnostic worker failure'):
            self.run_qualification(actor)
        self.assertEqual(len(actor.calls), 2)
        self.assert_clean(actor, before)

    def test_zero_gradient_cannot_masquerade_as_a_passing_cosine_comparison(self):
        self.micro['advantages'].zero_()
        actor = TinyActor(); before = self.state(actor)
        result = self.run_qualification(actor)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['comparisons']['D1_S1']['gradients']['cosine'])
        self.assertIn('gradient_relative_l2', result['comparisons']['D1_S1']['failed_checks'])
        json.dumps(result, allow_nan=False)
        self.assert_clean(actor, before)

    def test_pending_gradient_and_default_wrong_world_are_rejected(self):
        actor = TinyActor()
        next(actor.actor_module.parameters()).grad = torch.ones_like(next(actor.actor_module.parameters()))
        with self.assertRaisesRegex(ValueError, 'pending gradients'):
            self.run_qualification(actor)
        with self.assertRaisesRegex(ValueError, 'actual rank count'):
            module.qualify_sparse_policy(TinyActor(), self.micro, temperature=1.)


if __name__ == '__main__': unittest.main()
