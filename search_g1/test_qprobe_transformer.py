import unittest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from search_g1.qprobe_transformer import QwenBlockBinaryProbe, verify_full_copy


class TransformerProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            attention_dropout=0.)
        self.config._attn_implementation = 'eager'
        self.source = Qwen2DecoderLayer(self.config, layer_idx=1)
        self.source.requires_grad_(False)
        self.probe = QwenBlockBinaryProbe(self.source, Qwen2RMSNorm(32, eps=self.config.rms_norm_eps), self.config)
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
        out['mix_logit'].square().mean().backward()
        self.assertIsNone(hidden.grad)
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))
        for block in self.probe.blocks:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()))
            self.assertTrue(any(bool((p.grad != 0).any()) for p in block.parameters()))

    def test_question_isolation_and_padding_invariance(self):
        a = torch.randn(1, 4, 32)
        one = self.probe(a, torch.ones(1,4), torch.arange(4).expand(1,4))['mix_logit']
        batch = torch.randn(2, 7, 32); batch[0,:4] = a[0]
        mask = torch.ones(2,7); mask[0,4:] = 0
        two = self.probe(batch, mask, torch.arange(7).expand(2,7))['mix_logit']
        torch.testing.assert_close(one[0], two[0], atol=2e-6, rtol=2e-5)
        batch[1].mul_(100);batch[0,4:].mul_(100)
        changed = self.probe(batch, mask, torch.arange(7).expand(2,7))['mix_logit']
        torch.testing.assert_close(two[0], changed[0], atol=2e-6, rtol=2e-5)

    def test_initializer_reloads_actual_source_tensors_and_rejects_wrong_source(self):
        import tempfile
        from pathlib import Path
        from search_g1.qprobe_features import OFFICIAL_REVISION, sha_file
        from search_g1.qprobe_transformer import state_digest, load_official_initialization
        norm=Qwen2RMSNorm(32,eps=self.config.rms_norm_eps)
        state=dict(schema_version=3,architecture='official_qwen2_last_layer_source_v1',official_revision=OFFICIAL_REVISION,
            encoder_files_sha256='a'*64,qwen_config=self.config.to_dict(),attention_implementation='eager',
            source_layer_index=1,source_layer_state=self.source.state_dict(),source_layer_sha256=state_digest(self.source),
            source_norm_state=norm.state_dict(),source_norm_sha256=state_digest(norm))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'initialization.pt';torch.save(state,p)
            probe=load_official_initialization(p,expected_file_sha256=sha_file(p),expected_encoder_files_sha256='a'*64)
            self.assertEqual(probe.initialization['full_state_sha256'],state_digest(self.source))
            self.assertEqual(verify_full_copy(self.source,probe.blocks)['independent_copies'],3)
            self.assertTrue(probe.initialization['fresh_binary_head'])
            self.assertTrue(torch.equal(probe.head.weight,torch.zeros_like(probe.head.weight)))
            # Old metadata is allowed only for authenticated official-only source
            # tensors; never an old trained six-class head or optimizer.
            legacy=dict(state,schema_version=2,architecture='qwen2_decoder3_count6')
            torch.save(legacy,p)
            fresh=load_official_initialization(p,expected_file_sha256=sha_file(p),expected_encoder_files_sha256='a'*64)
            self.assertEqual(fresh.head.out_features,2)
            self.assertTrue(fresh.initialization['fresh_binary_head'])
            for key in ('model','optimizer','head_state','learned_state'):
                torch.save(dict(legacy,**{key:{}}),p)
                with self.assertRaisesRegex(ValueError,'ARCHITECTURE'):
                    load_official_initialization(p,expected_file_sha256=sha_file(p),expected_encoder_files_sha256='a'*64)
            torch.save(state,p)
            with self.assertRaisesRegex(ValueError,'OFFICIAL_SOURCE'):
                load_official_initialization(p,expected_file_sha256=sha_file(p),expected_encoder_files_sha256='b'*64)
            with self.assertRaisesRegex(ValueError,'FILE_SHA'):
                load_official_initialization(p,expected_file_sha256='f'*64,expected_encoder_files_sha256='a'*64)

    def test_reject_pooled_cache_and_empty_question(self):
        with self.assertRaisesRegex(ValueError, 'FULL_TOKEN_FEATURES'):
            self.probe(torch.randn(2,32),torch.ones(2),torch.zeros(2,dtype=torch.long))
        with self.assertRaisesRegex(ValueError, 'EMPTY_QUESTION'):
            self.probe(torch.randn(2,4,32),torch.zeros(2,4),torch.arange(4).expand(2,4))

if __name__ == '__main__':unittest.main()
