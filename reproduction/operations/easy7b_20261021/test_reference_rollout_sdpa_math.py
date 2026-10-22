import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from reproduction.operations.easy7b_20261021.reference_rollout_sdpa_math import (
    continuation_logits, difference, fp32_math_attention, greedy_divergences,
)


class ReferenceTests(unittest.TestCase):
    def test_suffix_uses_previous_position_and_full_vocabulary_normalization(self):
        logits = torch.tensor([[[1., 2., 3., 4., 5.], [2., 4., 1., 3., 0.],
                                [0., 1., 2., 3., 4.], [9., 1., 0., 0., 0.]]])
        rows, selected = continuation_logits(logits, [1, 2, 3, 4], 2)
        expected = torch.log_softmax(logits[0, 1:3].float(), -1)
        self.assertTrue(torch.equal(rows, expected))
        self.assertTrue(torch.equal(selected, torch.stack([expected[0, 3], expected[1, 4]])))

    def test_reference_passes_explicit_fp32_qkv_and_restores_bf16_output_and_original_function(self):
        original = functional.scaled_dot_product_attention
        observed = []
        q = torch.randn(1, 2, 4, 8).bfloat16()
        def spy(query, key, value, *args, **kwargs):
            observed.append((query.dtype, key.dtype, value.dtype))
            return original(query, key, value, *args, **kwargs)
        expected = original(q.float(), q.float(), q.float(), is_causal=True).to(q.dtype)
        with patch('torch.nn.functional.scaled_dot_product_attention', spy):
            with fp32_math_attention() as audit:
                output = functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            self.assertIs(functional.scaled_dot_product_attention, spy)
        self.assertIs(functional.scaled_dot_product_attention, original)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(observed, [(torch.float32,) * 3])
        self.assertEqual(audit['output_dtypes'], {'torch.float32': 1})

    def test_errors_compare_same_token_not_unmatched_greedy_sequences(self):
        value = difference([[-1., -2.], [-3.]], [[-1.1, -1.8], [-3.]])
        self.assertEqual(value['tokens'], 3)
        self.assertAlmostEqual(value['mean_abs'], .1)
        self.assertAlmostEqual(value['max_abs'], .2)

    def test_only_first_greedy_divergence_is_explained_on_common_teacher_prefix(self):
        backend = {'xformers': {'greedy_check': [{'token_ids': [3, 4]}]},
                   'fa2': {'greedy_check': [{'token_ids': [3, 4]}]},
                   'fa3': {'greedy_check': [{'token_ids': [3, 5]}]}}
        details = [[{}, dict(top1_id=4, top2_id=5, top1_minus_top2=.001,
            target_logprob=-.7, candidate_logprobs={'fa3': -.701})]]
        report = greedy_divergences(backend, [[1, 2, 3, 4]], [2], details)
        row = report['xformers_vs_fa3']['first_divergences'][0]
        self.assertEqual(row['first_divergent_continuation_index'], 1)
        self.assertTrue(row['same_prefix_as_teacher_before_divergence'])
        self.assertEqual(row['reference_prefers'], 'xformers')
        self.assertAlmostEqual(row['reference_xformers_minus_candidate_logprob'], .001)


if __name__ == '__main__':
    unittest.main()
