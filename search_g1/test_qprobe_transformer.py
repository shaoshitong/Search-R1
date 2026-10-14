import unittest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from search_g1.qprobe_transformer import QwenBlockCountProbe, verify_full_copy


class TransformerProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            attention_dropout=0.)
        self.config._attn_implementation = 'eager'
        self.source = Qwen2DecoderLayer(self.config, layer_idx=1)
        self.source.requires_grad_(False)
        self.probe = QwenBlockCountProbe(self.source, Qwen2RMSNorm(32, eps=self.config.rms_norm_eps), self.config)
        with torch.no_grad(): self.probe.head.weight.normal_(std=.02)

    def test_each_of_three_layers_is_full_equal_independent_copy(self):
        receipt = verify_full_copy(self.source, self.probe.blocks)
        self.assertEqual(receipt['independent_copies'], 3)
        for block in self.probe.blocks:
            self.assertTrue(all(p.requires_grad for p in block.parameters()))
        with torch.no_grad(): next(self.probe.blocks[0].parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'COPY_TENSOR_DIFFERS'):
            verify_full_copy(self.source, self.probe.blocks)
        self.assertTrue(torch.equal(next(self.probe.blocks[1].parameters()), next(self.source.parameters())))

    def test_frozen_features_and_source_have_no_grad_but_all_blocks_train(self):
        hidden = torch.randn(2, 5, 32, requires_grad=True)
        out = self.probe(hidden, torch.ones(2, 5), torch.arange(5).expand(2, 5))
        out['logits'].square().mean().backward()
        self.assertIsNone(hidden.grad)
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))
        for block in self.probe.blocks:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()))
            self.assertTrue(any(bool((p.grad != 0).any()) for p in block.parameters()))

    def test_question_isolation_and_padding_invariance(self):
        a = torch.randn(1, 4, 32)
        one = self.probe(a, torch.ones(1,4), torch.arange(4).expand(1,4))['logits']
        batch = torch.randn(2, 7, 32); batch[0,:4] = a[0]
        mask = torch.ones(2,7); mask[0,4:] = 0
        two = self.probe(batch, mask, torch.arange(7).expand(2,7))['logits']
        torch.testing.assert_close(one[0], two[0], atol=2e-6, rtol=2e-5)
        batch[1].mul_(100);batch[0,4:].mul_(100)
        changed = self.probe(batch, mask, torch.arange(7).expand(2,7))['logits']
        torch.testing.assert_close(two[0], changed[0], atol=2e-6, rtol=2e-5)

    def test_reject_pooled_cache_and_empty_question(self):
        with self.assertRaisesRegex(ValueError, 'FULL_TOKEN_FEATURES'):
            self.probe(torch.randn(2,32),torch.ones(2),torch.zeros(2,dtype=torch.long))
        with self.assertRaisesRegex(ValueError, 'EMPTY_QUESTION'):
            self.probe(torch.randn(2,4,32),torch.zeros(2,4),torch.arange(4).expand(2,4))

if __name__ == '__main__':unittest.main()
