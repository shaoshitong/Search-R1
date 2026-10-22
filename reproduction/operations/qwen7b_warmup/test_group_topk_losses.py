"""Check the metric bounds independently of the differentiable implementation."""
import importlib.util
import itertools
import unittest
from reproduction.operations.qwen7b_warmup.group_topk_losses import (
    reference_regret, reference_topk_ramp, reference_pairwise_hinge_sum,
    precision_at_k_ramp, squared_epsilon_regression,
)


class FinitePoolBounds(unittest.TestCase):
    def test_exhaustive_small_pools_including_ties_and_too_few_positives(self):
        cases = 0
        for n in range(1, 6):
            for labels in itertools.product((0, 1), repeat=n):
                m = sum(labels)
                for scores in itertools.product((-1.0, 0.0, 1.0), repeat=n):
                    pair = reference_pairwise_hinge_sum(scores, labels)
                    for k in range(1, n+1):
                        regret = reference_regret(scores, labels, k)
                        ramp = reference_topk_ramp(scores, labels, k)
                        self.assertGreaterEqual(ramp+1e-12, regret/k)
                        self.assertGreaterEqual(pair+1e-12, regret*(regret+abs(m-k)))
                        shifted = reference_topk_ramp([s+2.5 for s in scores], labels, k)
                        self.assertAlmostEqual(ramp, shifted, places=12)
                        cases += 1
        self.assertEqual(cases, 44790)

    def test_lower_variance_mse_can_reduce_real_mixed_precision(self):
        target = (.16, .24, 0.0)
        labels = (1, 1, 0)
        a, b = (.16, 0.0, 0.0), (.07, .08, .09)
        mse = lambda x: sum((p-y)**2 for p, y in zip(x, target))/3
        self.assertLess(mse(b), mse(a))
        self.assertEqual(reference_regret(a, labels, 1), 0)
        self.assertEqual(reference_regret(b, labels, 1), 1)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Run tensor checks in existing remote CPU runtime')
class ActualTensorLoss(unittest.TestCase):
    def test_value_matches_reference_and_pushes_wrong_top_negative_down(self):
        import torch
        s = torch.tensor([.1, -.3, .4, -.2], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([1, 1, 0, 0])
        loss = precision_at_k_ramp(s, y, 1)
        self.assertAlmostEqual(loss.item(), reference_topk_ramp(s.detach().tolist(), y.tolist(), 1))
        loss.backward()
        self.assertLess(s.grad[0].item(), 0)
        self.assertGreater(s.grad[2].item(), 0)
        self.assertAlmostEqual(s.grad.sum().item(), 0)

    def test_margin_satisfaction_has_exact_zero_gradient(self):
        import torch
        for values, labels, k in [([2., 1.5, -.5, -2.], [1, 1, 0, 0], 1),
                                   ([2., 1.5, -.5, -2.], [1, 1, 0, 0], 3),
                                   ([0., 1., 2.], [0, 0, 0], 2),
                                   ([0., 1., 2.], [1, 1, 1], 2)]:
            s = torch.tensor(values, requires_grad=True)
            loss = precision_at_k_ramp(s, torch.tensor(labels), k)
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(torch.equal(s.grad, torch.zeros_like(s)))

    def test_dead_zone_regression_stops_inside_band(self):
        import torch
        p = torch.tensor([.11, .13], requires_grad=True)
        loss = squared_epsilon_regression(p, torch.tensor([.1, .14]), .02)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(p.grad, torch.zeros_like(p)))


if __name__ == '__main__':
    unittest.main()
