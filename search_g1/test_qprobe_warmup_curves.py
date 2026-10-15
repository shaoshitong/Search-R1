"""Remote CPU fixtures for warmup budget/curves and binary diagnostics."""
import copy
import json
import math
import unittest
import torch
from search_g1 import test_qprobe_sequence_model as fixtures
from search_g1.qprobe_training_metrics import binary_mixed_training_metrics


class BinaryMixedMetricsTests(unittest.TestCase):
    def test_binary_confusion_brier_bce_and_distinct_rankings(self):
        counts=torch.tensor([0,1,4,5])
        pred=dict(mixed_probability=torch.tensor([.9,.8,.2,.1],dtype=torch.float64),
                  expected_std=torch.tensor([0.,.1,.4,.3],dtype=torch.float64))
        m=binary_mixed_training_metrics(pred,counts,top_k=2)
        self.assertEqual([m['binary_tp'],m['binary_fp'],m['binary_tn'],m['binary_fn']],[1,1,1,1])
        for key in ('binary_accuracy','binary_precision','binary_recall','binary_f1'):
            self.assertEqual(m[key],.5)
        self.assertAlmostEqual(m['binary_brier'],(.81+.04+.64+.01)/4)
        self.assertAlmostEqual(m['binary_bce'],-math.log(.1*.8*.2*.9)/4)
        self.assertEqual(m['p_mix_top_mixed_hits'],1)
        self.assertEqual(m['sigma_top_mixed_hits'],1)
        self.assertEqual(m['random_mixed_hits_expected'],1.)
        self.assertEqual(m['metric_scope'],'same_training_questions_in_sample')
        json.dumps(m,allow_nan=False)

    def test_top448_stable_ties_and_no_invented_heldout(self):
        counts=torch.tensor([1]*448+[0]*64)
        pred=dict(mixed_probability=torch.full((512,),.5),expected_std=torch.zeros(512))
        m=binary_mixed_training_metrics(pred,counts)
        self.assertEqual(m['top_effective'],448)
        self.assertEqual(m['top_mixed_hits'],448)
        self.assertEqual(m['sigma_top_mixed_hits'],448)
        self.assertEqual(m['random_mixed_rate_expected'],448/512)
        self.assertEqual(m['binary_majority_accuracy_baseline'],448/512)
        self.assertEqual(m['p_mix_mean'],.5);self.assertEqual(m['p_mix_std'],0.)
        self.assertEqual(m['binary_tp'],448);self.assertEqual(m['binary_fp'],64)
        self.assertEqual(m['ranking_tie_policy'],'stable_training_input_order')

    def test_zero_denominators_and_invalid_probabilities(self):
        pred=dict(mixed_probability=torch.zeros(4),expected_std=torch.zeros(4))
        m=binary_mixed_training_metrics(pred,torch.zeros(4,dtype=torch.long))
        self.assertEqual(m['binary_precision'],0.);self.assertEqual(m['binary_recall'],0.)
        self.assertEqual(m['binary_f1'],0.);self.assertEqual(m['top_effective'],4)
        bad=dict(pred,mixed_probability=torch.tensor([float('nan')]*4))
        with self.assertRaisesRegex(ValueError,'PREDICTIONS_INVALID'):
            binary_mixed_training_metrics(bad,torch.zeros(4,dtype=torch.long))


class WarmupCurveTests(unittest.TestCase):
    def fixture(self):return fixtures.OnlineSequenceTests()
    def bootstrap(self,p,**kwargs):
        f=self.fixture();labels=[]
        for x in f.labels():
            labels.append(dict(x,selection_source='bootstrap_all_candidate',source_outer=0,
                run_id='warmup',attempt_id='teacher0',policy_snapshot_id='official0',
                sampling_contract_sha='sampling',strict_scoring_contract_sha='scoring',retriever_version='retriever'))
        return p.bootstrap(labels,f.features(),actor_outer=0,actor_snapshot_id='warmup:0',**kwargs)

    def test_ten_epochs_emit_steps_epochs_real_adam_and_keep_online_two(self):
        p=self.fixture().probe();records=[];global_rng=torch.get_rng_state().clone()
        result=self.bootstrap(p,warmup_epochs=10,curve_callback=records.append)
        steps=[x for x in records if x['event']=='optimizer_step']
        epochs=[x for x in records if x['event']=='epoch']
        self.assertEqual(result['head_optimizer_updates'],10)
        self.assertEqual(result['fit_epochs'],10);self.assertEqual(p.config.head_epochs_per_outer,2)
        self.assertEqual(len(steps),10);self.assertEqual(len(epochs),11)
        self.assertEqual([x['epoch'] for x in epochs],list(range(11)))
        self.assertEqual(epochs[0]['optimizer_update'],0)
        self.assertEqual(epochs[0]['adam_step_min'],0);self.assertEqual(epochs[0]['adam_step_max'],0)
        self.assertEqual(epochs[0]['binary_majority_accuracy_baseline'],.5)
        self.assertEqual(epochs[0]['p_mix_min'],epochs[0]['p_mix_max'])
        self.assertEqual(epochs[0]['p_mix_std'],0.)
        self.assertEqual([x['adam_step_before_min'] for x in steps],list(range(10)))
        self.assertEqual([x['adam_step_after_max'] for x in steps],list(range(1,11)))
        self.assertEqual(epochs[-1]['training_groups'],4)
        self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
        json.dumps(records,allow_nan=False)
        online=p._fit(0)
        self.assertEqual(online['head_optimizer_updates'],2)
        self.assertEqual(p._adam_step_range(),(12,12))
        self.assertFalse(hasattr(p,'_bootstrap_fit_options'))

    def test_curve_evaluation_does_not_change_fit_or_private_shuffle_rng(self):
        a=self.fixture().probe();b=self.fixture().probe();records=[]
        x=self.bootstrap(a,warmup_epochs=3)
        y=self.bootstrap(b,warmup_epochs=3,curve_callback=records.append)
        self.assertEqual(x['replay_weighted_loss'],y['replay_weighted_loss'])
        self.assertTrue(torch.equal(a.rng.get_state(),b.rng.get_state()))
        for key,value in a.model.state_dict().items():
            torch.testing.assert_close(value,b.model.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(a.config.loss_version,'std_class_ce_v1')
        self.assertEqual(a.config.std_loss_weight,1.);self.assertEqual(a.config.count_aux_loss_weight,.1)

    def test_invalid_override_and_sink_failure_are_not_silently_completed(self):
        p=self.fixture().probe()
        with self.assertRaisesRegex(ValueError,'EPOCHS_INVALID'):
            self.bootstrap(p,warmup_epochs=True)
        self.assertFalse(p.replay_labels)
        def fail(record):raise OSError('durable_sink_failed')
        with self.assertRaisesRegex(OSError,'durable_sink_failed'):
            self.bootstrap(p,warmup_epochs=10,curve_callback=fail)
        self.assertIsNone(p.actor_snapshot_id)
        self.assertFalse(hasattr(p,'_bootstrap_fit_options'))


if __name__=='__main__':unittest.main()
