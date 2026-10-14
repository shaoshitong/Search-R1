import json
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from search_g1.qprobe_features import OFFICIAL_ENCODER, OFFICIAL_REVISION, COMMON_MANIFEST_SHA, QUESTION_EXTRACTION_RULE
from search_g1.qprobe_sequence import SequenceFeatures, SequenceCacheWriter, SequenceCache, extract_question_sequences


class SequenceCacheTests(unittest.TestCase):
    def spec(self):return dict(encoder_id=OFFICIAL_ENCODER,encoder_revision=OFFICIAL_REVISION,
        encoder_files_sha256='a'*64,tokenizer_sha256='b'*64,prompt_template_sha256='c'*64,
        feature_layer=1,pooling='all_nonpadding_prompt_tokens',feature_dtype='float16',feature_dim=8,
        max_prompt_tokens=32,question_extraction=QUESTION_EXTRACTION_RULE,train_manifest_sha256=COMMON_MANIFEST_SHA,
        split='train',encoder_compute_dtype='float32',representation='frozen_token_sequence_v2')
    def features(self):return SequenceFeatures([torch.randn(3,8),torch.randn(5,8)],
        [torch.tensor([2,3,4]),torch.arange(5)])
    def test_ragged_roundtrip_original_positions_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache';w=SequenceCacheWriter(p,self.spec(),2);f=self.features();w.append(['a','b'],['qa','qb'],f);w.finish()
            cache=SequenceCache(p);self.assertEqual(cache.verify_all()['total_tokens'],8)
            restored=cache.get_features(['b','a'],['qb','qa']);self.assertEqual([len(h) for h in restored.hidden],[5,3])
            torch.testing.assert_close(restored.hidden[1],f.hidden[0].half());self.assertTrue(torch.equal(restored.positions[1],torch.tensor([2,3,4])))
            h,mask,pos=restored.padded('cpu',torch.float32);self.assertEqual(h.shape,(2,5,8));self.assertEqual(mask[1].tolist(),[True,True,True,False,False])
            with self.assertRaisesRegex(ValueError,'QUESTION_TEXT'):cache.get_features(['a'],['changed'])
    def test_tampered_token_file_rejected_per_question_and_full_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache';w=SequenceCacheWriter(p,self.spec(),2);w.append(['a','b'],['qa','qb'],self.features());w.finish()
            with (p/'hidden.bin').open('r+b') as f:f.write(b'\xff\xff')
            cache=SequenceCache(p)
            with self.assertRaisesRegex(ValueError,'CORRUPT_SEQUENCE'):cache.get_features(['a'],['qa'])
            with self.assertRaisesRegex(ValueError,'FULL_SHA'):cache.verify_all()
    def test_incomplete_cache_cannot_publish_and_pooled_spec_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache';w=SequenceCacheWriter(p,self.spec(),2);w.append(['a'],['qa'],self.features()[[0]])
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):w.finish()
            w.abort();self.assertFalse(p.exists());self.assertTrue(w.partial.exists())
            spec=self.spec();spec['pooling']='last_nonpadding_prompt_token'
            with self.assertRaisesRegex(ValueError,'SPEC_REQUIRED'):SequenceCacheWriter(p,spec,2)
    def test_sequence_precompute_entry_publishes_verifiable_cache_and_initializer(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import sys,types
        import search_g1.qprobe_prepare as prepare
        from search_g1.qprobe_transformer import load_official_initialization
        from search_g1.qprobe_features import sha_file
        config=Qwen2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,vocab_size=30)
        config._attn_implementation='eager';model=Qwen2Model(config)
        class Tokenizer:
            pad_token_id=0
            def __call__(self,questions,**kw):return dict(input_ids=torch.tensor([[1,2,3]]*len(questions)),attention_mask=torch.ones(len(questions),3,dtype=torch.long))
        dependency=types.ModuleType('reproduction.common8.prepare_run');dependency.verify_model=lambda *args:{'tokenizer.json':'a'*64,'model.safetensors':'b'*64}
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'config.json').write_text(json.dumps(config.to_dict()));(base/'official.json').write_text('{}')
            args=SimpleNamespace(data=d,prompt_key='prompt',model=d,official_manifest=str(base/'official.json'),
                feature_layer=None,max_prompt_tokens=16,device='cpu',cache_dir=str(base/'cache'),batch_size=2,sequence_probe=True)
            with patch.dict(sys.modules,{'reproduction.common8.prepare_run':dependency}), \
                    patch.object(prepare,'train_questions',return_value=([('qa','a'),('qb','b')],{'products':{'train':{'sha256':'c'*64}}})), \
                    patch('transformers.AutoModel.from_pretrained',return_value=model), \
                    patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                proof=prepare.precompute(args)
            self.assertEqual(args.feature_layer,1);self.assertEqual(proof['count'],2)
            cache=SequenceCache(base/'cache');cache.verify_all();self.assertEqual(cache.manifest['total_tokens'],6)
            artifact=base/'cache/initialization.pt'
            probe=load_official_initialization(artifact,expected_file_sha256=sha_file(artifact),expected_encoder_files_sha256=cache.spec['encoder_files_sha256'])
            self.assertEqual(probe.initialization['independent_copies'],3)

    def test_actual_qwen_last_layer_extraction_preserves_frozen_full_sequence(self):
        config=Qwen2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,vocab_size=30,attention_dropout=0.)
        config._attn_implementation='eager';model=Qwen2Model(config)
        class Tokenizer:
            def __call__(self,*a,**kw):return dict(input_ids=torch.tensor([[0,0,4,5],[6,7,8,9]]),attention_mask=torch.tensor([[0,0,1,1],[1,1,1,1]]))
        f=extract_question_sequences(model,Tokenizer(),['qa','qb'],feature_layer=1,max_prompt_tokens=4)
        self.assertEqual([len(h) for h in f.hidden],[2,4]);self.assertTrue(torch.equal(f.positions[0],torch.tensor([2,3])))
        self.assertTrue(all(not h.requires_grad for h in f.hidden));self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        with self.assertRaisesRegex(ValueError,'FINAL_DECODER'):extract_question_sequences(model,Tokenizer(),['qa'],feature_layer=0)

if __name__=='__main__':unittest.main()
