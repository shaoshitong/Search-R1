"""Mathematical, leakage, online chronology and exact recovery invariants."""
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from search_g1.qprobe_model import BinaryMixVarianceProbe, OnlineQProbe, QProbeConfig, binary_mix_variance_loss, binary_predictions


def labels(outer, n=6, count=None):
    return [dict(qid=f'q{i}', question_hash='a'*64, strict_em=[1]*(i%6 if count is None else count)+
                 [0]*(5-(i%6 if count is None else count)), complete=True, selection_source='explore')
            for i in range(n)]


def advance(probe, outer, features, values=None, **overrides):
    metadata = dict(policy_snapshot_id=f'actor-before-{outer}', sampling_contract_sha='s'*64,
                    strict_scoring_contract_sha='c'*64, retriever_version='retriever-fixed',
                    run_id='test-run', attempt_id=f'attempt-{outer}')
    metadata.update(overrides)
    return probe.update_completed_outer(outer, labels(outer, len(features)) if values is None else values,
                                        features, **metadata)


class QProbeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_32_binary_groups_match_unbiased_sample_variance(self):
        for em in itertools.product([0,1],repeat=5):
            values=torch.tensor(em,dtype=torch.float32);k=sum(em)
            self.assertAlmostEqual(k*(5-k)/20,values.var(correction=1).item(),places=6)
            self.assertEqual(int(1<=k<=4),int(0<values.sum()<5))
        self.assertAlmostEqual(max(k*(5-k)/20 for k in range(6)),.3)

    def test_two_output_binary_head_has_no_fake_count_or_accuracy_distribution(self):
        prediction=BinaryMixVarianceProbe(4)(torch.ones(2,4))
        self.assertEqual(set(prediction),{'mix_logit','variance_logit','mixed_probability','expected_variance'})
        self.assertEqual(prediction['mixed_probability'].shape,(2,))
        self.assertTrue(torch.equal(prediction['mixed_probability'],torch.full((2,),.5)))
        torch.testing.assert_close(prediction['expected_variance'],torch.full((2,),.15))

    def test_binary_rank_is_independent_of_variance(self):
        prediction=binary_predictions(torch.tensor([-2.,2.]),torch.tensor([20.,-20.]))
        self.assertLess(prediction['mixed_probability'][0],prediction['mixed_probability'][1])
        self.assertGreater(prediction['expected_variance'][0],prediction['expected_variance'][1])

    def test_cold_head_ties_every_question_and_preserves_global_rng(self):
        torch.manual_seed(123);before=torch.get_rng_state().clone();probe=OnlineQProbe(8)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        pred=probe.predict(torch.arange(32).float().reshape(4,8))
        self.assertTrue(torch.equal(pred['mixed_probability'],torch.full((4,),.5)))
        self.assertFalse(any(v.requires_grad for v in pred.values()))

    def test_constructor_seeds_only_cpu_without_touching_actor_cuda_rng(self):
        torch.random.default_generator.manual_seed(123)
        before = torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all', side_effect=AssertionError('actor CUDA RNG must remain untouched')) as cuda_seed:
            OnlineQProbe(8)
        cuda_seed.assert_not_called()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_real_labels_reduce_binary_nll(self):
        probe=OnlineQProbe(4,dict(head_learning_rate=.03,head_epochs_per_outer=6,head_batch_size=32))
        x=torch.tensor([[1.,2.,3.,4.]]*32)
        before=-probe.predict(x)['mixed_probability'].log().mean().item()
        for outer in range(1,4):advance(probe,outer,x,labels(outer,32,count=2))
        after=-probe.predict(x)['mixed_probability'].log().mean().item()
        self.assertLess(after,before*.6)

    def test_age_weighted_binary_aux_objective_matches_manual_replay(self):
        prediction=binary_predictions(torch.tensor([-.3,.8],requires_grad=True),torch.tensor([.1,-.4],requires_grad=True))
        counts=torch.tensor([0,2]);weights=torch.tensor([1.,.25])
        loss,parts=binary_mix_variance_loss(prediction,counts,QProbeConfig(),weights)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(prediction['mix_logit'],torch.tensor([0.,1.]),reduction='none')
        variance=counts.float()*(5-counts.float())/20
        per=bce+.1*((prediction['expected_variance']-variance)/.3).square()
        torch.testing.assert_close(loss,(per*weights).sum()/weights.sum())
        self.assertEqual(set(parts),{'mixed_bce','variance_mse_normalized'})
        loss.backward();self.assertTrue(torch.isfinite(prediction['mix_logit'].grad).all())

    def test_complement_targets_are_symmetric_and_old_loss_rejected(self):
        prediction=binary_predictions(torch.tensor([.2]),torch.tensor([-.3]))
        for k in range(6):
            left,_=binary_mix_variance_loss(prediction,torch.tensor([k]),QProbeConfig())
            right,_=binary_mix_variance_loss(prediction,torch.tensor([5-k]),QProbeConfig())
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        with self.assertRaises(ValueError):QProbeConfig(loss_version='std_class_ce_v1')
        with self.assertRaises(ValueError):QProbeConfig.from_mapping({'count_aux_loss_weight':.1})
        with self.assertRaises(ValueError):QProbeConfig(variance_loss_weight=.2)

    def test_updates_do_not_perturb_actor_rng_and_evict_old_labels(self):
        probe = OnlineQProbe(4, dict(replay_window_outer=2, head_epochs_per_outer=1))
        x = torch.arange(24).float().reshape(6,4)
        torch.manual_seed(991); rng = torch.get_rng_state().clone()
        for outer in range(1,4): advance(probe, outer, x)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual({x['source_outer'] for x in probe.replay_labels}, {2,3})
        self.assertEqual(len(probe.replay_labels), 12)
        self.assertEqual(probe.replay_labels[-1]['policy_snapshot_id'], 'actor-before-3')

    def test_no_partial_missing_or_duplicate_group_is_admitted(self):
        probe = OnlineQProbe(4); x = torch.ones(6,4)
        incomplete = labels(1); incomplete[0]['complete'] = False
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE'): advance(probe,1,x,incomplete)
        wrong = labels(1); wrong[0]['strict_em'] = [0,0,0,0]
        with self.assertRaisesRegex(ValueError, 'FIVE_STRICT'): advance(probe,1,x,wrong)
        duplicates = labels(1); duplicates[1]['qid'] = duplicates[0]['qid']
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'): advance(probe,1,x,duplicates)
        advance(probe,1,x)
        with self.assertRaisesRegex(ValueError, 'ALREADY_UPDATED'): advance(probe,1,x)
        with self.assertRaisesRegex(ValueError, 'PROTOCOL_CHANGE'):
            advance(probe,2,x,sampling_contract_sha='different')
        self.assertEqual(probe.update_count,1)

    def test_failure_rolls_back_head_optimizer_labels_and_rng(self):
        probe = OnlineQProbe(4); x = torch.ones(6,4); advance(probe,1,x)
        before = probe.state_dict(); version = probe.version
        original_fit = probe._fit
        def failed_fit(outer):
            original_fit(outer); raise RuntimeError('simulated publication failure')
        with mock.patch.object(probe, '_fit', side_effect=failed_fit):
            with self.assertRaises(RuntimeError): advance(probe,2,x)
        self.assertEqual(probe.version,version)
        self.assertEqual(probe.replay_labels,before['replay_labels'])
        self.assertTrue(torch.equal(probe.rng.get_state(),before['rng']))
        advance(probe,2,x)
        self.assertEqual(probe.completed_outer,2)

    def test_snapshot_recovery_reproduces_next_update_exactly(self):
        probe = OnlineQProbe(4,feature_spec_sha='f'*64)
        x = torch.arange(24).float().reshape(6,4); advance(probe,1,x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'outer-1.pt'
            receipt = probe.save(path,actor_snapshot_id='actor-after-1')
            clone = OnlineQProbe.load(path,expected_actor_snapshot_id='actor-after-1',
                                     expected_feature_spec_sha='f'*64,expected_checkpoint_sha=receipt['checkpoint_sha'])
            self.assertEqual(probe.version,clone.version)
            advance(probe,2,x); advance(clone,2,x)
            self.assertEqual(probe.version,clone.version)
            self.assertTrue(torch.equal(probe.predict(x)['mixed_probability'],clone.predict(x)['mixed_probability']))
            with self.assertRaisesRegex(ValueError,'ACTOR_BINDING'): OnlineQProbe.load(path,'wrong-actor')
            with self.assertRaisesRegex(ValueError,'FEATURE_SPEC'): OnlineQProbe.load(path,expected_feature_spec_sha='wrong')
            with self.assertRaises(FileExistsError): probe.save(path,actor_snapshot_id='actor-after-2')

    def test_bootstrap_retains_real_policy_age_and_rejects_future(self):
        probe = OnlineQProbe(4, dict(bootstrap_max_groups_per_outer=4))
        history=[]
        for outer in (3,4):
            for item in labels(outer):
                item.update(source_outer=outer,selection_source='bootstrap_all_candidate',run_id='old',
                            attempt_id=f'attempt-{outer}',policy_snapshot_id=f'actor-before-{outer}',
                            sampling_contract_sha='s',strict_scoring_contract_sha='c',retriever_version='r')
                history.append(item)
        probe.bootstrap(history,torch.ones(12,4),actor_outer=4,actor_snapshot_id='actor-after-4')
        self.assertEqual(len(probe.replay_labels),8)
        self.assertEqual(probe.completed_outer,4); self.assertEqual(probe.update_count,0)
        self.assertEqual({x['policy_snapshot_id'] for x in probe.replay_labels}, {'actor-before-3','actor-before-4'})
        future=dict(history[0],source_outer=5)
        with self.assertRaisesRegex(ValueError,'FUTURE'):
            OnlineQProbe(4).bootstrap([future],torch.ones(1,4),actor_outer=4,actor_snapshot_id='actor-after-4')

    def test_old_six_class_checkpoints_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt'
            for version in (1,2):
                torch.save({'schema_version':version,'probe_architecture':'qwen2_decoder3_count6'},path)
                with self.assertRaisesRegex(ValueError,'SCHEMA'):
                    OnlineQProbe.load(path)

    def test_numeric_and_config_guards(self):
        with self.assertRaises(ValueError): QProbeConfig(replay_half_life_outer=0)
        self.assertEqual(QProbeConfig.from_mapping({'head_hidden_size':32}).hidden_size,32)
        with self.assertRaises(ValueError): QProbeConfig.from_mapping({'head_hidden_size':32,'hidden_size':64})
        with self.assertRaises(ValueError): OnlineQProbe(4).predict(torch.full((1,4),float('nan')))
        with self.assertRaises(ValueError): OnlineQProbe(4).predict(torch.ones(4))


if __name__ == '__main__': unittest.main()
