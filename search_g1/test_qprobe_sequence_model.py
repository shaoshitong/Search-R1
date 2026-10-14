import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm
from search_g1.qprobe_model import OnlineQProbe
from search_g1.qprobe_sequence import SequenceFeatures
from search_g1.qprobe_sequence_model import OnlineSequenceQProbe
from search_g1.qprobe_transformer import QwenBlockCountProbe
from search_g1.qprobe_features import OFFICIAL_REVISION


class OnlineSequenceTests(unittest.TestCase):
    def model(self):
        config=Qwen2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.)
        config._attn_implementation='eager'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);source=Qwen2DecoderLayer(config,layer_idx=1)
            model=QwenBlockCountProbe(source,Qwen2RMSNorm(8,eps=config.rms_norm_eps),config)
        model.initialization.update(official_revision=OFFICIAL_REVISION,encoder_files_sha256='a'*64,source_layer_index=1,initialization_file_sha256='b'*64)
        return model
    def probe(self,micro=2):return OnlineSequenceQProbe(self.model(),dict(head_batch_size=4,head_epochs_per_outer=2,seed=17,head_learning_rate=.01),'c'*64,train_micro=micro,predict_micro=2)
    def features(self):
        rng=torch.Generator().manual_seed(21)
        return SequenceFeatures([torch.randn(n,8,generator=rng) for n in [2,3,4,5]],[torch.arange(n) for n in [2,3,4,5]])
    def labels(self):return [dict(qid=str(i),question_hash='d'*64,selection_source='top',complete=True,strict_em=[int(j<k) for j in range(5)]) for i,k in enumerate([0,1,3,5])]
    def update(self,p,outer=1):return p.update_completed_outer(outer,self.labels(),self.features(),policy_snapshot_id='official0',sampling_contract_sha='sampling',strict_scoring_contract_sha='scoring',retriever_version='retriever',run_id='fresh-run',attempt_id='attempt'+str(outer))
    def test_micro_accumulation_preserves_effective_batch_and_original_rng(self):
        a,b=self.probe(1),self.probe(4);rng=torch.get_rng_state().clone()
        x=self.update(a);y=self.update(b);self.assertEqual(x['head_optimizer_updates'],2);self.assertEqual(y['head_optimizer_updates'],2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in a.model.state_dict().items():torch.testing.assert_close(value,b.model.state_dict()[key],atol=3e-6,rtol=3e-5)
    def test_complete_checkpoint_roundtrip_and_next_update_equivalence(self):
        a=self.probe();self.update(a)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'probe.pt';receipt=a.save(p,actor_snapshot_id='actor1')
            b=OnlineQProbe.load(p,expected_actor_snapshot_id='actor1',expected_feature_spec_sha='c'*64,expected_checkpoint_sha=receipt['sha256'])
            self.assertIsInstance(b,OnlineSequenceQProbe);self.assertEqual(a.version,b.version);self.assertTrue(torch.equal(a.rng.get_state(),b.rng.get_state()))
            self.assertEqual(a.replay_labels,b.replay_labels);self.assertEqual([len(h) for h in b.replay_features.hidden],[2,3,4,5])
            for h,j in zip(a.replay_features.hidden,b.replay_features.hidden):torch.testing.assert_close(h,j,rtol=0,atol=0)
            self.update(a,2);self.update(b,2);self.assertEqual(a.version,b.version)
            for i,j in zip(a.optimizer.state.values(),b.optimizer.state.values()):
                for key in i:torch.testing.assert_close(i[key],j[key],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'ACTOR_BINDING'):OnlineQProbe.load(p,expected_actor_snapshot_id='wrong')
    def test_failed_fit_restores_blocks_head_optimizer_rng_and_replay(self):
        a=self.probe();self.update(a);before=a.state_dict();version=a.version
        def broken(outer):
            with torch.no_grad():next(a.model.parameters()).add_(1)
            torch.randperm(10,generator=a.rng);raise RuntimeError('injected_fit_failure')
        with patch.object(a,'_fit',side_effect=broken):
            with self.assertRaisesRegex(RuntimeError,'injected'):self.update(a,2)
        self.assertEqual(a.version,version);self.assertTrue(torch.equal(a.rng.get_state(),before['rng']));self.assertEqual(a.replay_labels,before['replay_labels']);self.assertEqual(a.completed_outer,1)
        for key,value in before['model'].items():torch.testing.assert_close(value,a.model.state_dict()[key],rtol=0,atol=0)
    def test_runtime_scores_sequences_and_restores_paired_head(self):
        from types import SimpleNamespace
        from search_g1.qprobe_runtime import QProbeRuntime
        from search_g1.qprobe_selection import PredictQuestion
        class Cache:
            spec_sha='c'*64;feature_dim=8
            spec={'representation':'frozen_token_sequence_v2'}
            manifest={'files':{'initialization.pt':{'sha256':'b'*64}}}
            def get_features(cache,qids,questions):return self.features()[:len(qids)]
        cfg=dict(probe_architecture='qwen2_decoder3_count6',head_batch_size=4,head_epochs_per_outer=2,
            seed=17,head_learning_rate=.01,sequence_train_micro=2,sequence_predict_micro=2,sequence_device='cpu')
        with tempfile.TemporaryDirectory() as d:
            runtime=QProbeRuntime(cfg,d,feature_cache=Cache(),head=self.probe())
            features,pred,version=runtime.score([PredictQuestion(0,'a','qa'),PredictQuestion(1,'b','qb')])
            self.assertIsInstance(features,SequenceFeatures);self.assertEqual(pred['probabilities'].shape,(2,6))
            self.update(runtime.head);runtime.completed_outer=1
            path=Path(d)/'paired.pt';runtime.save(path,'actor1')
            other=QProbeRuntime(cfg,d,feature_cache=Cache(),head=self.probe());other.restore(path,'actor1')
            self.assertEqual(other.head.version,runtime.head.version)
            with self.assertRaisesRegex(ValueError,'ARCHITECTURE_CONFIG'):
                QProbeRuntime(dict(cfg,probe_architecture='count_mlp'),d,feature_cache=Cache(),head=self.probe())

    def test_incompatible_source_or_physical_micro_cannot_restore(self):
        a=self.probe();self.update(a);state=a.state_dict();state['initialization']['source_layer_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'SOURCE_CHANGED'):a.load_state_dict(state)
        state=a.state_dict();state['sequence_train_micro']=1
        with self.assertRaisesRegex(ValueError,'MICRO_CHANGED'):a.load_state_dict(state)

if __name__=='__main__':unittest.main()
