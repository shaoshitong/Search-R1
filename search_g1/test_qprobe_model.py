"""Mathematical, leakage, online chronology and exact recovery invariants."""
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from search_g1.qprobe_model import (CountMomentProbe, OnlineQProbe, QProbeConfig,
                                   count_moment_loss, count_moment_group_loss)


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


def sigma3_config(**overrides):
    config = dict(loss_mode='sigma3_std_aux6_v1', mean_loss_weight=0.0,
                  std_loss_weight=3.0, count_aux_loss_weight=0.1)
    config.update(overrides)
    return QProbeConfig(**config)


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
        per = ce + .1*(p['mean']-counts.float()/5).square() + .1*((p['expected_std']-std)/math.sqrt(.3)).square()
        self.assertAlmostEqual(loss.item(), ((per*weights).sum()/weights.sum()).item(), places=6)
        self.assertTrue(torch.isfinite(pieces['count_ce']))

    def test_sigma3_formula_mapping_and_gradients(self):
        model = CountMomentProbe(3)
        logits = torch.tensor([[2.,-1.,.4,.8,-.2,1.], [-.3,1.8,.2,-.9,.4,1.],
                               [1.,.6,-.4,2.,.1,-1.], [.2,-.4,1.3,.7,-.8,1.],
                               [1.4,.3,-.6,.5,1.7,-.2], [-.7,.1,.5,-1.,.4,2.]],
                              requires_grad=True)
        counts = torch.arange(6)
        weights = torch.tensor([1.,.25,2.,.5,1.5,.75])
        pred = model.moments(logits)
        p = logits.softmax(-1)
        q = torch.stack((p[:,0]+p[:,5], p[:,1]+p[:,4], p[:,2]+p[:,3]), dim=-1)
        target = torch.tensor([0,1,2,2,1,0])
        ce3 = -q[torch.arange(6), target].log()
        ce6 = -p[torch.arange(6), counts].log()
        expected_std = math.sqrt(.2)*q[:,1] + math.sqrt(.3)*q[:,2]
        torch.testing.assert_close(pred['expected_std'], expected_std)
        target_std = (counts.float()*(5-counts.float())/20).sqrt()
        std_mse = ((expected_std-target_std)/math.sqrt(.3)).square()
        manual = ((ce3+3*std_mse+.1*ce6)*weights).sum()/weights.sum()
        loss, pieces = count_moment_loss(pred, counts, sigma3_config(), weights)
        torch.testing.assert_close(loss, manual)
        torch.testing.assert_close(pieces['sigma3_ce'], (ce3*weights).sum()/weights.sum())
        torch.testing.assert_close(pieces['normalized_sample_std_mse'], pieces['std_mse'])
        torch.testing.assert_close(loss, pieces['sigma3_ce_contribution']+
                                   pieces['std_mse_contribution']+pieces['count_ce_contribution'])
        actual_grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        manual_grad, = torch.autograd.grad(manual, logits)
        torch.testing.assert_close(actual_grad, manual_grad)
        self.assertNotIn('mean_mse', pieces)

    def test_sigma3_extreme_logits_have_finite_loss_and_gradients(self):
        logits = torch.tensor([[1000.,-1000.,-1000.,-1000.,-1000.,1000.],
                               [-1000.,-1000.,1000.,1000.,-1000.,-1000.]], requires_grad=True)
        pred = CountMomentProbe(3).moments(logits)
        counts = torch.tensor([2,0])
        self.assertEqual(pred['probabilities'][0,2].item(), 0)
        loss, pieces = count_moment_loss(pred, counts, sigma3_config())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(pieces['sigma3_ce'].item(), 1900)
        gradient, = torch.autograd.grad(loss, logits)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_sigma3_fit_matches_independent_global_recency_adam(self):
        config = sigma3_config(head_learning_rate=.01, head_epochs_per_outer=1,
                               head_batch_size=1, replay_half_life_outer=1.)
        actual = OnlineQProbe(3, config)
        manual = OnlineQProbe(3, config)
        x = torch.tensor([[1.,2.,-1.], [-1.,.5,2.]])
        counts = torch.tensor([0,2])
        actual.replay_labels = [dict(correct_count=0, source_outer=1),
                                dict(correct_count=2, source_outer=3)]
        actual.replay_features = x.clone()
        weights = torch.tensor([.25,1.]); weights *= len(weights)/weights.sum()
        order = torch.randperm(len(counts), generator=manual.rng)
        for index in order.split(1):
            pred = manual.model(x[index])
            logits = pred['logits']
            paired = torch.stack([torch.logsumexp(logits[:, [k,5-k]], -1)
                                  for k in range(3)], -1)
            ce3 = torch.nn.functional.cross_entropy(paired, torch.minimum(counts[index],5-counts[index]),
                                                    reduction='none')
            ce6 = torch.nn.functional.cross_entropy(logits, counts[index], reduction='none')
            target_std = (counts[index].float()*(5-counts[index].float())/20).sqrt()
            std_mse = ((pred['expected_std']-target_std)/math.sqrt(.3)).square()
            objective = ((ce3+3*std_mse+.1*ce6)*weights[index]).mean()
            manual.optimizer.zero_grad(set_to_none=True)
            objective.backward(); manual.optimizer.step()
        report = actual._fit(3)
        for key, value in actual.model.state_dict().items():
            torch.testing.assert_close(value, manual.model.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(actual.rng.get_state(), manual.rng.get_state()))
        expected, _ = count_moment_loss(actual.predict(x), counts, config, weights)
        self.assertAlmostEqual(report['replay_weighted_loss'], expected.item(), places=6)
        self.assertEqual(report['head_optimizer_updates'], 2)
        self.assertEqual(report['loss_mode'], config.loss_mode)
        self.assertEqual(report['loss_identity'], config.loss_identity)

    def test_sigma3_bootstrap_and_online_use_shared_loss_and_progress(self):
        import search_g1.qprobe_model as module
        config = sigma3_config(head_learning_rate=.03, head_epochs_per_outer=1, head_batch_size=32)
        probe = OnlineQProbe(3, config)
        x = torch.tensor([[1.,2.,3.]]*4)
        history = labels(1,4,count=2)
        for item in history:
            item.update(source_outer=1,selection_source='bootstrap_all_candidate',run_id='old',
                        attempt_id='old-1',policy_snapshot_id='actor-before-1',sampling_contract_sha='s'*64,
                        strict_scoring_contract_sha='c'*64,retriever_version='retriever-fixed')
        before, _ = count_moment_loss(probe.predict(x), torch.full((4,),2), config)
        with mock.patch.object(module, 'count_moment_group_loss', wraps=count_moment_group_loss) as shared:
            bootstrap = probe.bootstrap(history,x,actor_outer=1,actor_snapshot_id='actor-after-1')
            self.assertEqual(shared.call_count, 2)  # One Adam batch and one report.
            online = advance(probe,2,x,labels(2,4,count=2))
            self.assertEqual(shared.call_count, 4)
        self.assertLess(bootstrap['replay_weighted_loss'], before.item())
        self.assertLess(online['replay_weighted_loss'], bootstrap['replay_weighted_loss'])
        for report in (bootstrap, online):
            self.assertEqual(report['loss_identity'], config.loss_identity)
            self.assertAlmostEqual(report['replay_weighted_loss'], report['sigma3_ce_contribution']+
                                   report['std_mse_contribution']+report['count_ce_contribution'], places=6)

    def test_loss_checkpoint_legacy_defaults_and_sigma3_binding(self):
        x = torch.ones(2,3)
        legacy = OnlineQProbe(3,feature_spec_sha='f'*64)
        advance(legacy,1,x)
        old_state = legacy.state_dict()
        old_state['config'].pop('loss_mode'); old_state['config'].pop('count_aux_loss_weight')
        old_state.pop('loss_identity'); old_state['actor_snapshot_id'] = 'actor-after-1'
        restored = OnlineQProbe(3,feature_spec_sha='f'*64)
        restored.load_state_dict(old_state)
        self.assertEqual(restored.config.loss_mode, 'count6_moments_v1')
        self.assertEqual(restored.version, legacy.version)
        fresh = OnlineQProbe(3,sigma3_config(),feature_spec_sha='f'*64)
        with self.assertRaisesRegex(ValueError, 'NAMESPACE_MISMATCH'):
            fresh.load_state_dict(old_state)
        advance(fresh,1,x)
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory)/'old.pt'; torch.save(old_state, old_path)
            self.assertEqual(OnlineQProbe.load(old_path).config.loss_mode, 'count6_moments_v1')
            path = Path(directory)/'sigma3.pt'
            receipt = fresh.save(path,actor_snapshot_id='actor-after-1')
            reloaded = OnlineQProbe.load(path)
            self.assertEqual(reloaded.config, fresh.config)
            self.assertEqual(receipt['loss_identity'], fresh.config.loss_identity)
            advance(fresh,2,x); advance(reloaded,2,x)
            self.assertEqual(reloaded.version, fresh.version)
        forged = fresh.state_dict(); forged['loss_identity'] = 'CE6'
        with self.assertRaisesRegex(ValueError, 'LOSS_IDENTITY_MISMATCH'):
            fresh.load_state_dict(forged)

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
        with self.assertRaises(ValueError): QProbeConfig(loss_mode='unknown')
        with self.assertRaisesRegex(ValueError,'mean_loss_weight=0'):
            QProbeConfig(loss_mode='sigma3_std_aux6_v1')
        with self.assertRaises(ValueError): sigma3_config(count_aux_loss_weight=float('nan'))
        self.assertEqual(QProbeConfig.from_mapping({'head_hidden_size':32}).hidden_size,32)
        with self.assertRaises(ValueError): QProbeConfig.from_mapping({'head_hidden_size':32,'hidden_size':64})
        with self.assertRaises(ValueError): OnlineQProbe(4).predict(torch.full((1,4),float('nan')))
        with self.assertRaises(ValueError): OnlineQProbe(4).predict(torch.ones(4))


if __name__ == '__main__': unittest.main()
