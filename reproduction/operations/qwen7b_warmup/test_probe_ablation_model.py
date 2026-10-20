"""Synthetic CPU contracts only; no local model weights or remote operations."""
import unittest

import torch
from torch import nn

from .probe_ablation_model import (CAPTURE_KIND, CAPTURE_PAIRS, NativeBlockProbe,
                                   ProbeAblationSpec, TokenReadout, validate_capture_batch)


def readout(kind, scope='prefix_only'):
    result = TokenReadout(3, kind, scope, question_max_tokens=2, prefix_tokens=3).double()
    with torch.no_grad():
        result.output.weight.copy_(torch.tensor([[1., 0., 0.]], dtype=torch.float64))
        result.output.bias.fill_(.7)
    return result


class TokenReadoutTests(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.tensor([[[1., 2., 3.], [3., 4., 5.], [7., 8., 9.],
                                     [11., 12., 13.], [1000., 1000., 1000.]],
                                    [[2., 4., 6.], [4., 6., 8.], [99., 99., 99.],
                                     [99., 99., 99.], [99., 99., 99.]]], dtype=torch.float64)
        self.mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

    def test_explicit_zero_based_layer_contract(self):
        for block, capture in CAPTURE_PAIRS.items():
            self.assertEqual(ProbeAblationSpec(block, capture).capture_layer, capture)
        for block, capture in ((21, 21), (24, 26), (27, 20), (26, 25)):
            with self.assertRaises(ValueError):
                ProbeAblationSpec(block, capture)

    def test_last_and_mean_use_only_real_tokens_in_selected_scope(self):
        last, weights = readout('last_valid_token').pool(self.hidden, self.mask)
        torch.testing.assert_close(last[0], self.hidden[0, 3])
        self.assertTrue(torch.equal(last[1], torch.zeros(3, dtype=torch.float64)))
        self.assertTrue(torch.equal(weights[0], torch.tensor([0., 0., 0., 1., 0.], dtype=torch.float64)))
        mean, _ = readout('mean').pool(self.hidden, self.mask)
        torch.testing.assert_close(mean[0], self.hidden[0, 2:4].mean(0))
        full, _ = readout('mean', 'question_prefix').pool(self.hidden, self.mask)
        torch.testing.assert_close(full[0], self.hidden[0, :4].mean(0))
        torch.testing.assert_close(full[1], self.hidden[1, :2].mean(0))
        full_last, _ = readout('last_valid_token', 'question_prefix').pool(self.hidden, self.mask)
        torch.testing.assert_close(full_last[1], self.hidden[1, 1])

    def test_attention_padding_weights_empty_prefix_and_bias_are_safe(self):
        module = readout('learned_attention')
        with torch.no_grad():
            module.attention_key.copy_(torch.tensor([1., -.2, .3], dtype=torch.float64))
        hidden = self.hidden.clone()
        hidden[~self.mask] = float('nan')
        pooled, weights = module.pool(hidden, self.mask)
        scores = (self.hidden[0, 2:4]@module.attention_key)/3**.5
        torch.testing.assert_close(weights[0, 2:4], scores.softmax(0))
        self.assertTrue(torch.equal(weights.masked_select(~module.eligible_mask(self.mask)),
                                    torch.zeros_like(weights.masked_select(~module.eligible_mask(self.mask)))))
        self.assertAlmostEqual(float(weights[0].sum().detach()), 1.)
        self.assertEqual(float(weights[1].sum().detach()), 0.)
        self.assertTrue(torch.equal(pooled[1], torch.zeros_like(pooled[1])))
        logits = module(hidden, self.mask)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertAlmostEqual(float(logits[1].detach()), .7)
        logits[1].backward()
        self.assertTrue(torch.equal(module.attention_key.grad, torch.zeros_like(module.attention_key)))
        self.assertTrue(torch.equal(module.output.weight.grad, torch.zeros_like(module.output.weight)))
        self.assertEqual(float(module.output.bias.grad), 1.)

    def test_attention_initialization_matches_mean_and_small_head_shape(self):
        torch.manual_seed(1729)
        mean = TokenReadout(3, 'mean', 'prefix_only', 2, 3)
        torch.manual_seed(1729)
        attention = TokenReadout(3, 'learned_attention', 'prefix_only', 2, 3)
        self.assertTrue(torch.equal(mean.output.weight, attention.output.weight))
        self.assertTrue(torch.equal(mean.output.bias, attention.output.bias))
        a, aw = mean.pool(self.hidden, self.mask)
        b, bw = attention.pool(self.hidden, self.mask)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(aw, bw)
        self.assertEqual(tuple(attention.output.weight.shape), (1, 3))
        self.assertEqual(sum(p.numel() for p in attention.parameters()), 7)

    def test_attention_key_learns_and_all_padding_is_zero_for_every_readout(self):
        module = readout('learned_attention')
        module(self.hidden[:1], self.mask[:1]).sum().backward()
        self.assertGreater(float(module.attention_key.grad.abs().sum()), 0.)
        padded = torch.full_like(self.hidden, float('nan'))
        empty = torch.zeros_like(self.mask)
        for kind in ('last_valid_token', 'mean', 'learned_attention'):
            result = readout(kind, 'question_prefix')
            pooled, weights = result.pool(padded, empty)
            self.assertTrue(torch.equal(pooled, torch.zeros_like(pooled)))
            self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
            torch.testing.assert_close(result(padded, empty), torch.full((2,), .7, dtype=torch.float64))

    def test_mean_attention_permutation_invariance_and_last_order_sensitivity(self):
        permutation = torch.tensor([0, 1, 3, 2, 4])
        for kind in ('mean', 'learned_attention'):
            module = readout(kind)
            if module.attention_key is not None:
                with torch.no_grad():
                    module.attention_key.copy_(torch.tensor([.3, .7, -.2], dtype=torch.float64))
            original, _ = module.pool(self.hidden, self.mask)
            shuffled, _ = module.pool(self.hidden[:, permutation], self.mask[:, permutation])
            torch.testing.assert_close(original, shuffled)
        module = readout('last_valid_token')
        original, _ = module.pool(self.hidden, self.mask)
        shuffled, _ = module.pool(self.hidden[:, permutation], self.mask[:, permutation])
        self.assertFalse(torch.equal(original[0], shuffled[0]))


class FakeCausalBlock(nn.Module):
    """A differentiable native-context stand-in; never real GPU qualification."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.seen_mask = None

    def forward(self, hidden, attention_mask, **kwargs):
        self.seen_mask = attention_mask.detach().clone()
        return (hidden+(hidden*attention_mask.unsqueeze(-1)).cumsum(1)*self.scale,)


class FakeRotary(nn.Module):
    def forward(self, features, positions):
        return None


def synthetic_native_probe():
    # Avoid constructors that read official weights.  Only the real forward and
    # capture contract are exercised with tiny synthetic values of native shape.
    result = NativeBlockProbe.__new__(NativeBlockProbe)
    nn.Module.__init__(result)
    result.spec = ProbeAblationSpec(27, 26, 'mean', 'prefix_only')
    result.block, result.rotary = FakeCausalBlock(), FakeRotary()
    result.cache_position = True
    result.readout = TokenReadout(3584, 'mean', 'prefix_only', 273, 20)
    with torch.no_grad():
        result.readout.output.weight.zero_()
        result.readout.output.weight[0, 0] = 1.
        result.readout.output.bias.zero_()
    return result


def native_batch():
    features = torch.zeros(1, 293, 3584)
    features[0, 0, 0], features[0, 273, 0] = 3., 2.
    features.requires_grad_(True)
    mask = torch.zeros(1, 293, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 273] = True
    positions = torch.zeros(1, 293, dtype=torch.long)
    positions[:, 273] = 1
    return dict(features=features, attention_mask=mask, position_ids=positions,
                capture_layer=26, capture_kind=CAPTURE_KIND)


class NativeForwardContractTests(unittest.TestCase):
    def test_prefix_pooling_keeps_question_context_and_detaches_student_only(self):
        probe, batch = synthetic_native_probe(), native_batch()
        result = probe(batch)
        self.assertEqual(float(result.detach()), 7.)  # Prefix2 + causal(question3+prefix2).
        self.assertTrue(torch.equal(probe.block.seen_mask, batch['attention_mask']))
        self.assertTrue(bool(probe.block.seen_mask[0, 0]))
        result.sum().backward()
        self.assertIsNone(batch['features'].grad)
        self.assertIsNotNone(probe.block.scale.grad)
        self.assertIsNotNone(probe.readout.output.weight.grad)

    def test_empty_prefix_still_runs_question_context_and_returns_only_bias(self):
        probe, batch = synthetic_native_probe(), native_batch()
        batch['attention_mask'][:, 273:] = False
        with torch.no_grad():
            probe.readout.output.bias.fill_(.3)
        result = probe(batch, return_weights=True)
        self.assertTrue(bool(probe.block.seen_mask[0, 0]))
        self.assertTrue(torch.equal(result['token_weights'], torch.zeros_like(result['token_weights'])))
        self.assertAlmostEqual(float(result['logits'].detach()), .3, places=6)

    def test_wrong_capture_or_noncausal_positions_are_rejected(self):
        spec, batch = ProbeAblationSpec(24, 23), native_batch()
        with self.assertRaises(ValueError):
            validate_capture_batch(batch, spec)
        spec = ProbeAblationSpec(27, 26)
        missing = dict(batch)
        missing.pop('capture_layer')
        with self.assertRaises(ValueError):
            validate_capture_batch(missing, spec)
        invalid = dict(batch, position_ids=torch.zeros_like(batch['position_ids']))
        with self.assertRaises(ValueError):
            validate_capture_batch(invalid, spec)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
