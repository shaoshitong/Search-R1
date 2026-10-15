"""Mathematical, leakage, online chronology and exact recovery invariants."""
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from search_g1.qprobe_model import CountMomentProbe, OnlineQProbe, QProbeConfig, count_moment_loss


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

    def test_all_32_binary_groups_match_sample_std(self):
        probe = CountMomentProbe(4)
        for em in itertools.product([0,1], repeat=5):
            values = torch.tensor(em, dtype=torch.float32)
            count = sum(em)
            self.assertAlmostEqual(probe.means[count].item(), values.mean().item())
            self.assertAlmostEqual(probe.stds[count].item(), values.std(correction=1).item())
        self.assertAlmostEqual(probe.stds.max().item(), math.sqrt(.3), places=6)

    def test_same_expected_accuracy_can_have_opposite_diversity(self):
        model = CountMomentProbe(4)
        logits = torch.full((2,6), -100.0)
        logits[0,[0,5]] = 0; logits[1,[2,3]] = 0
        prediction = model.moments(logits)
        self.assertTrue(torch.allclose(prediction['mean'], torch.tensor([.5,.5])))
        self.assertLess(prediction['expected_std'][0].item(), 1e-30)
        self.assertAlmostEqual(prediction['expected_std'][1].item(), math.sqrt(.3), places=6)
        self.assertTrue(torch.allclose(prediction['mixed_probability'], torch.tensor([0.,1.]), atol=1e-6))

    def test_expected_variance_is_not_expected_std_squared(self):
        model = CountMomentProbe(3)
        logits = torch.full((1,6), -100.); logits[0,[0,2]] = 0.
        pred = model.moments(logits)
        self.assertGreater(pred['expected_variance'].item(), pred['expected_std'].square().item())

    def test_cold_head_ties_every_question_and_preserves_global_rng(self):
        torch.manual_seed(123); before = torch.get_rng_state().clone()
        probe = OnlineQProbe(8)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        pred = probe.predict(torch.arange(32).float().reshape(4,8))
        self.assertTrue(torch.allclose(pred['probabilities'], torch.full((4,6), 1/6)))
        self.assertEqual(len(set(pred['expected_std'].tolist())), 1)
        self.assertFalse(any(value.requires_grad for value in pred.values()))

    def test_real_labels_reduce_prequential_count_nll(self):
        probe = OnlineQProbe(4, dict(head_learning_rate=.03, head_epochs_per_outer=6, head_batch_size=32))
        x = torch.tensor([[1.,2.,3.,4.]]*32)
        before = -probe.predict(x)['probabilities'][:,2].log().mean().item()
        for outer in range(1,4): advance(probe, outer, x, labels(outer,32,count=2))
        after = -probe.predict(x)['probabilities'][:,2].log().mean().item()
        self.assertLess(after, before*.6)
        self.assertEqual(probe.update_count, 3)

    def test_age_weighted_objective_matches_manual_complete_replay(self):
        model = CountMomentProbe(3); p = model(torch.ones(2,3))
        weights = torch.tensor([1.,.25]); counts = torch.tensor([0,2])
        loss, pieces = count_moment_loss(p, counts, QProbeConfig(), weights)
        ce = -p['probabilities'][torch.arange(2),counts].log()
        std = (counts.float()*(5-counts.float())/20).sqrt()
        q = torch.stack([p['probabilities'][:, pair].sum(-1) for pair in ((0,5),(1,4),(2,3))], -1)
        cls = torch.minimum(counts, 5-counts)
        per = -q[torch.arange(2), cls].log() + .1*ce + ((p['expected_std']-std)/math.sqrt(.3)).square()
        self.assertAlmostEqual(loss.item(), ((per*weights).sum()/weights.sum()).item(), places=6)
        self.assertTrue(torch.isfinite(pieces['count_ce']))

    def test_std_main_loss_is_symmetric_and_ignores_mean(self):
        model = CountMomentProbe(3)
        logits = torch.tensor([[1., -2., 3., .5, -1., 2.]], requires_grad=True)
        pred = model.moments(logits)
        config = QProbeConfig(count_aux_loss_weight=0)
        left, _ = count_moment_loss(pred, torch.tensor([1]), config)
        pred['mean'] = torch.tensor([999.])
        right, _ = count_moment_loss(pred, torch.tensor([4]), config)
        self.assertTrue(torch.equal(left, right))
        left.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError): QProbeConfig(mean_loss_weight=.1)

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
            self.assertTrue(torch.equal(probe.predict(x)['probabilities'],clone.predict(x)['probabilities']))
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

    def test_numeric_and_config_guards(self):
        with self.assertRaises(ValueError): QProbeConfig(replay_half_life_outer=0)
        self.assertEqual(QProbeConfig.from_mapping({'head_hidden_size':32}).hidden_size,32)
        with self.assertRaises(ValueError): QProbeConfig.from_mapping({'head_hidden_size':32,'hidden_size':64})
        with self.assertRaises(ValueError): OnlineQProbe(4).predict(torch.full((1,4),float('nan')))
        with self.assertRaises(ValueError): OnlineQProbe(4).predict(torch.ones(4))


if __name__ == '__main__': unittest.main()
