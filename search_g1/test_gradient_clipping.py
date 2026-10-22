"""Focused CPU checks for optional measurements around the real clip math."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import torch

from search_g1 import gradient_clipping as clipping


class GradientClippingMeasurementTests(unittest.TestCase):
    @staticmethod
    def module(dtype, values):
        module = torch.nn.Linear(len(values), 1, bias=False, dtype=dtype)
        module.weight.grad = torch.tensor([values], dtype=dtype)
        return module

    def test_disabled_measurements_keep_exact_fp64_clip_and_dtype_rounding(self):
        for dtype, values in (
            (torch.float32, [1e20, -1e20, 3e19, 0.]),
            (torch.bfloat16, [1e38, -1e38, 3e37, 0.]),
            (torch.float16, [60000., -60000., 17000., 0.]),
            (torch.float32, [.01, -.02, .03, 0.]),
        ):
            for chunk_size in (1, 3, 1048576):
                with self.subTest(dtype=dtype, chunk_size=chunk_size):
                    legacy = self.module(dtype, values)
                    fast = self.module(dtype, values)
                    # A previous real measurement must not survive a skipped pass.
                    fast._last_clipped_global_l2 = torch.tensor(123.)
                    fast._last_clipped_global_l2_measured = True
                    original = fast.weight.grad.clone()
                    with contextlib.redirect_stdout(io.StringIO()):
                        expected_norm = clipping.clip_global_l2(legacy, 1., chunk_size=chunk_size)
                    with patch.object(clipping, 'require_finite', side_effect=AssertionError('finite scan')), \
                         patch.object(torch, 'maximum', side_effect=AssertionError('max diagnostic')), \
                         patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar sync')), \
                         patch.object(clipping, '_sum_squares', wraps=clipping._sum_squares) as passes:
                        norm = clipping.clip_global_l2(fast, 1., chunk_size=chunk_size,
                            diagnostic=False, check_finite=False, measure_post_norm=False)
                    self.assertTrue(torch.equal(norm, expected_norm))
                    self.assertTrue(torch.equal(fast.weight.grad, legacy.weight.grad))
                    coefficient = torch.clamp(1. / (norm + 1e-6), max=1.)
                    expected = (original.to(torch.float64) * coefficient).to(dtype)
                    self.assertTrue(torch.equal(fast.weight.grad, expected))
                    self.assertEqual(passes.call_count, 2)  # Pre-clip shards + replicas only.
                    self.assertTrue(all(call.kwargs == {'measure_max_abs': False}
                                        for call in passes.call_args_list))
                    self.assertIsNone(fast._last_clipped_global_l2)
                    self.assertIs(fast._last_clipped_global_l2_measured, False)
                    self.assertIs(legacy._last_clipped_global_l2_measured, True)
                    self.assertIsInstance(legacy._last_clipped_global_l2, torch.Tensor)

    def test_only_preclip_sum_collective_remains_when_measurements_disabled(self):
        collective_calls = []

        def reduce(tensor, op=None, group=None):
            collective_calls.append('MAX' if op == clipping.dist.ReduceOp.MAX else 'SUM')

        for fast, expected_calls in ((False, ['SUM', 'MAX', 'SUM']), (True, ['SUM'])):
            module = self.module(torch.float32, [3., 4.])
            collective_calls.clear()
            with patch.object(clipping, '_partition', return_value=([module.weight], [], 'group', None)), \
                 patch.object(clipping.dist, 'is_initialized', return_value=True), \
                 patch.object(clipping.dist, 'all_reduce', side_effect=reduce), \
                 patch.object(clipping, 'require_finite'):
                norm = clipping.clip_global_l2(module, 1., diagnostic=False,
                    check_finite=not fast, measure_post_norm=not fast)
            self.assertEqual(collective_calls, expected_calls)
            self.assertTrue(torch.equal(norm, torch.tensor(5., dtype=torch.float64)))
            self.assertTrue(torch.equal(module.weight.grad,
                (torch.tensor([[3., 4.]], dtype=torch.float64) / (5. + 1e-6)).float()))

    def test_legacy_defaults_keep_max_diagnostic_and_measured_post_norm(self):
        module = self.module(torch.float32, [3., -4.])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
             patch.object(clipping, '_sum_squares', wraps=clipping._sum_squares) as passes:
            norm = clipping.clip_global_l2(module, 1., context={'outer': 1})
        prefix = 'SEARCH_G1_GRADIENT_NORM_DIAGNOSTIC='
        line = output.getvalue().strip()
        self.assertTrue(line.startswith(prefix))
        diagnostic = json.loads(line[len(prefix):])
        self.assertEqual(diagnostic['global_l2_fp64'], 5.)
        self.assertEqual(diagnostic['max_abs'], 4.)
        self.assertEqual(diagnostic['context'], {'outer': 1})
        self.assertEqual(passes.call_count, 4)
        self.assertTrue(torch.equal(norm, torch.tensor(5., dtype=torch.float64)))
        measured = torch.linalg.vector_norm(module.weight.grad.double())
        self.assertTrue(torch.equal(module._last_clipped_global_l2, measured))
        self.assertIs(module._last_clipped_global_l2_measured, True)


if __name__ == '__main__':
    unittest.main()
