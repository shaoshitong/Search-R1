"""Actual tiny native-Qwen CPU adoption/replay/transaction/checkpoint fixture."""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from search_g1.qprobe_qwen_block_model import QwenBlockWarmup
from search_g1.qprobe_qwen_block_online import OnlineQwenBlockQProbe, QwenBlockOnlineConfig
from search_g1.qprobe_token_model import token_binary_loss
from search_g1.test_qprobe_token_online import TinyTokenProvider, labels, advance


class NativeProvider(TinyTokenProvider):
    feature_dim = 8

    def get_batch(self,qids,device='cpu'):
        batch = super().get_batch(qids,device=device)
        batch['features'] = torch.nn.functional.pad(batch['features'],(0,5))
        return batch


@unittest.skipUnless(importlib.util.find_spec('transformers') is not None,'actual native transformers required')
class QwenOnlineTests(unittest.TestCase):
    def test_adoption_excludes_teacher_raw_recent8_and_exact_transactions(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            config = QwenBlockOnlineConfig(max_tokens=2,hidden_size=8,num_attention_heads=2,
                num_key_value_heads=1,intermediate_size=16,micro_batch_size=256)
            mapped = QwenBlockOnlineConfig.from_mapping(dict(asdict(config),seed=20261013,head_seed=1729))
            self.assertEqual(mapped,config)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator().manual_seed(43).get_state())
                native = Qwen2DecoderLayer(config.native_config(),layer_idx=35)
            official = {'model.layers.35.'+key:value.detach().to(torch.bfloat16).clone()
                        for key,value in native.state_dict().items()}
            provider = NativeProvider(); qids = provider.ids
            counts = torch.arange(5120)%6; lengths = [provider.index[qid]['length'] for qid in qids]
            global_rng = torch.get_rng_state().clone()
            warmup = QwenBlockWarmup(config.core_config(),official)
            metrics = warmup.fit(provider,lengths,counts,qids)
            self.assertEqual(metrics['actual_head_adam_updates'],200)
            corpus = labels(provider,qids,0,bootstrap=True)
            with tempfile.TemporaryDirectory() as directory:
                raw_path = Path(directory)/'native-warmup.pt'
                wrapper = dict(token_feature_spec_sha256=provider.spec_sha,head=warmup.state_dict(),
                    label_producer_provenance=dict(protocol=dict(sampling_contract_sha='s'*64,
                        strict_scoring_contract_sha='c'*64,retriever_version='fixed-retriever')))
                torch.save(wrapper,raw_path)
                digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                kwargs = dict(expected_checkpoint_sha=digest,expected_feature_spec_sha=provider.spec_sha,
                    provenance=dict(actor_snapshot_id='actor:0',warmup_ready_sha256='b'*64),config=config,feature_cache=provider)
                head = OnlineQwenBlockQProbe.adopt_warmup(raw_path,corpus,qids,**kwargs)
                self.assertEqual((head.completed_outer,head.update_count,head._adam_steps()),(0,0,200))
                self.assertEqual((head.replay_labels,head.replay_features,head.protocol),([],[],None))
                provenance = head.initialization_provenance
                self.assertTrue(provenance['teacher_replay_excluded'])
                self.assertEqual(provenance['bootstrap_groups'],5120)
                self.assertEqual(provenance['formal_replay_groups_at_adoption'],0)
                self.assertTrue(head.verify_native_provenance()['accepted'])
                for key,value in warmup.model.state_dict().items():
                    torch.testing.assert_close(head.model.state_dict()[key],value,rtol=0,atol=0)
                torch.testing.assert_close(head.rng.get_state(),warmup.rng.get_state(),rtol=0,atol=0)
                with mock.patch.object(head.model,'state_dict',wraps=head.model.state_dict) as reads:
                    initial_version = head.version
                    for _ in range(4):
                        head.predict(qids[:3]); self.assertEqual(head.version,initial_version)
                    self.assertEqual(reads.call_count,1)  # No full77m hash on each prediction.
                before_rng = head.rng.get_state().clone()
                permutation = torch.randperm(512,generator=torch.Generator().set_state(before_rng))[:256]
                selected = qids[:512]
                batch,sizes = warmup._batch(provider,permutation,torch.tensor(lengths[:512]),selected)
                with torch.no_grad():
                    unweighted,_ = token_binary_loss(warmup.model(batch,sizes),counts[:512][permutation],config)
                first = advance(head,provider,1)
                self.assertEqual((first['fit_epochs'],first['head_optimizer_updates']),(2,4))
                self.assertEqual(first['replay_groups'],512)  # Not5632 teacher+formal groups.
                self.assertEqual((first['head_adam_steps_before'],first['head_adam_steps_after']),(200,204))
                self.assertAlmostEqual(first['minibatch_history'][0]['pre_step_weighted_loss'],float(2*unweighted),places=6)
                self.assertNotEqual(head.version,initial_version)
                self.assertTrue(all(row['source_outer'] == 1 for row in head.replay_labels))
                self.assertTrue(all(row['selection_source'] != 'bootstrap_all_candidate' for row in head.replay_labels))
                for outer in range(2,10):
                    result = advance(head,provider,outer)
                    expected_outers = set(range(max(1,outer-7),outer+1))
                    self.assertEqual({row['source_outer'] for row in head.replay_labels},expected_outers)
                    self.assertEqual(result['replay_groups'],512*len(expected_outers))
                    self.assertEqual(result['head_optimizer_updates'],4*len(expected_outers))
                    for row in result['minibatch_history']:
                        self.assertAlmostEqual(row['pre_step_weighted_loss'],row['loss_parts']['binary_bce']+
                                               .3*row['loss_parts']['regression_mse_normalized'],places=6)
                self.assertEqual(head._adam_steps(),376)
                self.assertEqual({row['source_outer'] for row in head.replay_labels},set(range(2,10)))
                checkpoint = Path(directory)/'formal.pt'
                receipt = head.save(checkpoint,actor_snapshot_id='actor:9')
                restored = OnlineQwenBlockQProbe.load(checkpoint,expected_actor_snapshot_id='actor:9',
                    expected_feature_spec_sha=provider.spec_sha,expected_checkpoint_sha=receipt['sha256'])
                self.assertEqual(restored.version,head.version)
                self.assertEqual(restored.replay_identity_sha256,head.replay_identity_sha256)
                self.assertTrue(restored.verify_native_provenance()['teacher_replay_excluded'])
                with self.assertRaisesRegex(ValueError,'NOT_ATTACHED'): restored.predict(qids[:2])
                restored.attach_feature_provider(provider)
                for key,value in head.predict(qids[:3]).items():
                    torch.testing.assert_close(restored.predict(qids[:3])[key],value,rtol=0,atol=0)
                self.assertEqual(receipt['head_kind'],'qwen_native_lastblock_binary_v1')
                self.assertEqual(receipt['replay_groups'],4096)
                self.assertEqual(receipt['head_version'],head.version)
                wrong = labels(provider,qids[:512],10); wrong[0]['question_hash'] = 'f'*64
                before = head.state_dict()
                with self.assertRaisesRegex(ValueError,'QUESTION_HASH'):
                    head.update_completed_outer(10,wrong,qids[:512],policy_snapshot_id='actor-before-10',
                        sampling_contract_sha='s'*64,strict_scoring_contract_sha='c'*64,
                        retriever_version='fixed-retriever',run_id='formal',attempt_id='attempt-10')
                self.assertEqual(head._adam_steps(),376)
                provider.calls,provider.fail_after = 0,1
                with self.assertRaisesRegex(RuntimeError,'after actual Adam'): advance(head,provider,10)
                self.assertEqual((head._adam_steps(),head._engine.update_count),(377,377))
                self.assertTrue(head.state_dict()['fit_failed'])
                self.assertEqual(head.completed_outer,9)
                with self.assertRaisesRegex(ValueError,'FAILED_FIT'): advance(head,provider,10)
                provider.fail_after = None
                head.load_state_dict(before)
                self.assertEqual((head._adam_steps(),head.version),(376,restored.version))
                self.assertEqual(advance(head,provider,10)['minibatch_history'],advance(restored,provider,10)['minibatch_history'])
                self.assertEqual(head.version,restored.version)
                tampered = copy.deepcopy(before)
                tampered['initialization_provenance']['teacher_replay_excluded'] = False
                with self.assertRaisesRegex(ValueError,'TOKEN_QPROBE_INVALID_CHECKPOINT_COUNTERS'): restored.load_state_dict(tampered)
                bad_wrapper = copy.deepcopy(wrapper); bad_wrapper['head']['fitted'] = False
                bad_path = Path(directory)/'partial.pt'; torch.save(bad_wrapper,bad_path)
                bad_kwargs = dict(kwargs,expected_checkpoint_sha=hashlib.sha256(bad_path.read_bytes()).hexdigest())
                with self.assertRaisesRegex(ValueError,'UNQUALIFIED_NATIVE_WARMUP'):
                    OnlineQwenBlockQProbe.adopt_warmup(bad_path,corpus,qids,**bad_kwargs)
                self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
