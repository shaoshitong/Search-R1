import types
import unittest
import torch

from search_g1.vllm_flash3 import Flash3Functions


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.record = []
        def invoke(*args, **kwargs):
            self.record.append((args, kwargs))
            return args[0], 'lse'
        self.f = Flash3Functions(types.SimpleNamespace(
            module=types.SimpleNamespace(flash_attn_varlen_func=invoke,
                                         flash_attn_with_kvcache=invoke), provenance={}))
        self.q = torch.randn(5, 28, 8)
        self.k = torch.randn(5, 4, 8)
        self.cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    def test_prefill_keeps_compact_kv_objects_scale_and_offsets(self):
        out = self.f.varlen(self.q, self.k, self.k, self.cu, self.cu, 3, 3,
                            softmax_scale=.25, causal=True)
        args, kw = self.record[-1]
        self.assertIs(out, self.q)
        self.assertIs(args[1], self.k)
        self.assertIs(args[3], self.cu)
        self.assertEqual(kw['softmax_scale'], .25)
        self.assertTrue(kw['causal'])
        self.assertEqual(self.f.calls['prefill'], 1)

    def test_paged_prefix_uses_original_page_table_and_actual_sequence_lengths(self):
        table = torch.tensor([[3], [1]], dtype=torch.int32)
        self.f.varlen(self.q, self.k, self.k, self.cu, self.cu, 3, 3,
                      block_table=table, causal=True)
        args, kw = self.record[-1]
        self.assertIs(kw['page_table'], table)
        self.assertTrue(torch.equal(kw['cache_seqlens'], torch.tensor([2, 3], dtype=torch.int32)))
        self.assertIs(kw['cu_seqlens_q'], self.cu)
        self.assertEqual(self.f.calls['paged_prefill'], 1)

    def test_decode_preserves_cache_and_does_not_append_kv_twice(self):
        q = self.q[:2].unsqueeze(1)
        table = torch.tensor([[3], [1]], dtype=torch.int32)
        lengths = torch.tensor([13, 27], dtype=torch.int32)
        self.f.kvcache(q, self.k, self.k, block_table=table,
                      cache_seqlens=lengths, causal=True)
        args, kw = self.record[-1]
        self.assertIs(args[1], self.k)
        self.assertIs(kw['cache_seqlens'], lengths)
        self.assertNotIn('k', kw)
        self.assertEqual(self.f.calls['decode'], 1)

    def test_unsupported_semantics_never_fall_back(self):
        with self.assertRaises(ValueError):
            self.f.varlen(self.q, self.k, self.k, self.cu, self.cu, 3, 3,
                          alibi_slopes=torch.ones(28))
        self.assertFalse(self.record)


if __name__ == '__main__':
    unittest.main()
