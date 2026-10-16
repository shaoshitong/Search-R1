"""Focused CPU mathematics tests, with no model or remote data."""
import math
import unittest

import torch

from .loss import joint_loss


class JointG5LossTests(unittest.TestCase):
    def test_half_probabilities_hand_value_group_stationary_trajectory_learns(self):
        z = torch.zeros((1, 5), dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float64)
        result = joint_loss(z, labels)
        self.assertAlmostEqual(result['mixed_probability'].item(), 15 / 16, places=14)
        self.assertAlmostEqual(result['group_bce'].item(), -math.log(15 / 16), places=14)
        self.assertAlmostEqual(result['trajectory_bce'].item(), math.log(2), places=14)
        group_gradient = torch.autograd.grad(result['group_bce'], z, retain_graph=True)[0]
        trajectory_gradient = torch.autograd.grad(result['trajectory_bce'], z)[0]
        self.assertTrue(torch.allclose(group_gradient, torch.zeros_like(z), atol=1e-14, rtol=0))
        self.assertTrue(torch.allclose(trajectory_gradient,
            torch.tensor([[-.1, .1, .1, .1, .1]], dtype=torch.float64), atol=1e-14, rtol=0))

    def test_mixed_label_includes_k1_and_excludes_only_k0_k5(self):
        labels = torch.tensor([[1] * k + [0] * (5 - k) for k in range(6)], dtype=torch.float64)
        result = joint_loss(torch.zeros_like(labels), labels)
        self.assertEqual(result['mixed_label'].tolist(), [0, 1, 1, 1, 1, 0])
        expected = (2 * math.log(16) - 4 * math.log(15 / 16)) / 6
        self.assertAlmostEqual(result['group_bce'].item(), expected, places=14)

    def test_extreme_logits_have_finite_loss_and_gradient_without_clamping(self):
        z = torch.tensor([[10000.] * 5, [-10000.] * 5,
                          [10000., -10000., 10000., -10000., 10000.],
                          [-10000., 10000., -10000., 10000., -10000.]], requires_grad=True)
        labels = torch.tensor([[1., 0., 0., 0., 0.], [0., 0., 0., 0., 1.],
                               [0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]])
        result = joint_loss(z, labels)
        for key in ('loss', 'trajectory_bce', 'group_bce', 'mixed_probability'):
            self.assertTrue(torch.isfinite(result[key]).all())
        result['loss'].backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0)

    def test_permutation_and_independent_mean_weighting(self):
        z = torch.tensor([[-1., .2, .7, -.4, 1.1], [.4, -.7, .6, .1, -.2]], dtype=torch.float64)
        labels = torch.tensor([[1., 0., 1., 0., 0.], [0., 1., 0., 0., 0.]], dtype=torch.float64)
        result = joint_loss(z, labels, group_weight=.7)
        permutation = [3, 0, 4, 1, 2]
        shuffled = joint_loss(z[:, permutation], labels[:, permutation], group_weight=.7)
        for key in ('loss', 'trajectory_bce', 'group_bce', 'mixed_probability', 'mixed_label'):
            self.assertTrue(torch.allclose(result[key], shuffled[key], atol=1e-14, rtol=0))
        self.assertTrue(torch.equal(result['loss'], result['total_loss']))
        self.assertTrue(torch.equal(result['loss'], result['trajectory_bce'] + .7 * result['group_bce']))
        pieces = [joint_loss(z[i:i + 1], labels[i:i + 1], .7) for i in range(2)]
        for key in ('loss', 'trajectory_bce', 'group_bce'):
            self.assertTrue(torch.allclose(result[key], torch.stack([p[key] for p in pieces]).mean(), atol=1e-14, rtol=0))

    def test_group_graph_propagates_exact_product_gradient(self):
        z = torch.tensor([[-1., .2, .7, -.4, 1.1]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1., 0., 1., 0., 0.]], dtype=torch.float64)
        result = joint_loss(z, labels)
        actual = torch.autograd.grad(result['group_bce'], z)[0]
        p = z.detach().sigmoid()
        a, b = p.prod(dim=1, keepdim=True), (1 - p).prod(dim=1, keepdim=True)
        q = 1 - a - b
        dq = p * b - (1 - p) * a
        expected = (q - 1) / (q * (1 - q)) * dq
        self.assertGreater(actual.abs().sum().item(), 0)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-14, rtol=1e-12))
        self.assertTrue(torch.allclose(result['mixed_probability'], q.squeeze(1), atol=1e-14, rtol=0))


if __name__ == '__main__':
    unittest.main()
