"""Remote CPU tests for binary selection/prequential/RPC/checkpoint bindings.

Tensor/numerical tests are intentionally executed only on the remote environment.
"""
import ast
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from search_g1.qprobe_selection import PredictQuestion,binary_predictions,select_questions,prequential_metrics
from search_g1.qprobe_training_metrics import binary_mixed_training_metrics
from search_g1.qprobe_runtime import validate_config
from search_g1.qprobe_sequence_rpc import RemoteSequenceQProbe
from reproduction.h1008_qprobe.first_five_admission import selected


class BinarySelectionTests(unittest.TestCase):
    def questions(self,n):return [PredictQuestion(i,str(i),'Question '+str(i)) for i in range(n)]
    def test_focus_uses_probability_not_auxiliary_variance(self):
        questions=self.questions(6)
        prediction=dict(mixed_probability=np.array([.1,.2,.3,.4,.5,.6]),expected_variance=np.array([.3,.25,.2,.15,.1,0.]))
        result=select_questions(questions,prediction,selected_questions=2,exploration_questions=0,rng=np.random.default_rng(1))
        self.assertEqual([r.ordinal for r in result],[4,5])
        self.assertTrue(all(r.selection_source=='predicted_mixed' for r in result))
    def test_uniform64_first_then_top448_remaining_disjoint(self):
        questions=self.questions(5120)
        prediction=dict(mixed_probability=np.linspace(0,1,5120),expected_variance=np.zeros(5120))
        expected_random=set(np.random.default_rng(41).permutation(5120)[:64])
        expected_top=set(sorted(set(range(5120))-expected_random,reverse=True)[:448])
        result=select_questions(questions,prediction,rng=np.random.default_rng(41))
        self.assertEqual({r.ordinal for r in result if r.selection_source=='random_exploration'},expected_random)
        self.assertEqual({r.ordinal for r in result if r.selection_source=='predicted_mixed'},expected_top)
        self.assertEqual(len(result),512);self.assertEqual(len({r.ordinal for r in result}),512)
        self.assertEqual([r.ordinal for r in result],sorted(expected_random|expected_top))
    def test_ties_reproducible_private_rng(self):
        q=self.questions(30);p=dict(mixed_probability=np.full(30,.5),expected_variance=np.zeros(30))
        first=select_questions(q,p,selected_questions=8,exploration_questions=2,rng=np.random.default_rng(9),tie_seed=22)
        second=select_questions(q,p,selected_questions=8,exploration_questions=2,rng=np.random.default_rng(9),tie_seed=22)
        self.assertEqual(first,second)
    def test_old_count_probabilities_rejected(self):
        with self.assertRaises(ValueError):binary_predictions(dict(probabilities=np.full((4,6),1/6)))
        with self.assertRaises(ValueError):binary_predictions(dict(mixed_probability=np.ones(4),expected_variance=np.zeros(4),mean=np.zeros(4)))
    def test_binary_support_and_shapes_fail_closed(self):
        for p,v in (([1.1],[0]),([.5],[.31]),([float('nan')],[0]),([.5,.3],[0])):
            with self.assertRaises(ValueError):binary_predictions(dict(mixed_probability=p,expected_variance=v))
    def test_prequential_strict_group_targets_and_top_random(self):
        labels=[[0,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,1,1,1,1]]
        p=dict(mixed_probability=np.array([.1,.8,.9,.2]),expected_variance=np.array([0,.2,.2,0]))
        m=prequential_metrics(p,labels,['random_exploration','predicted_mixed','predicted_mixed','random_exploration'])
        self.assertEqual(m['all']['actual_mixed_fraction'],.5)
        self.assertEqual(m['predicted_mixed']['actual_mixed_fraction'],1.)
        self.assertEqual(m['random_exploration']['actual_mixed_fraction'],0.)
        self.assertAlmostEqual(m['all']['variance_normalized_mse'],0.)
        self.assertNotIn('count_nll',m['all']);self.assertNotIn('mean_mae',m['all'])
    def test_training_metrics_use_direct_binary_and_variance(self):
        counts=torch.tensor([0,1,2,3,4,5])
        p=dict(mixed_probability=torch.tensor([.01,.9,.8,.8,.9,.01]),expected_variance=counts.float()*(5-counts.float())/20)
        m=binary_mixed_training_metrics(p,counts,top_k=4)
        self.assertEqual(m['binary_accuracy'],1.)
        self.assertEqual(m['top_mixed_hits'],4)
        self.assertEqual(m['metric_scope'],'same_training_questions_in_sample')
        self.assertAlmostEqual(m['variance_normalized_mse'],0.)
    def test_saturated_fp32_variance_endpoint_accepted_but_true_overbound_rejected(self):
        counts=torch.tensor([2])
        p=dict(mixed_probability=torch.tensor([.99]),expected_variance=.3*torch.sigmoid(torch.tensor([20.])))
        result=binary_mixed_training_metrics(p,counts,top_k=1)
        self.assertLess(result['variance_mse'],1e-12)
        with self.assertRaises(ValueError):
            binary_mixed_training_metrics(dict(p,expected_variance=torch.tensor([.31])),counts)

    def test_new_admission_and_reject_old_runtime_architecture(self):
        run=dict(execution_site='h1008',startup_admission_mode='actual_first_five',training_world_size=8,
                 question_probe=dict(probe_architecture='qwen2_decoder3_binary_mix_v1'))
        self.assertTrue(selected(run))
        with self.assertRaises(ValueError):validate_config(dict(probe_architecture='qwen2_decoder3_count6'))
    def test_rpc_rejects_old_head_schema(self):
        proxy=object.__new__(RemoteSequenceQProbe)
        with self.assertRaises(ValueError):proxy._accept(dict(probe_architecture='qwen2_decoder3_count6',head_schema_version=2))
    def test_prequential_diagnostic_failure_does_not_block_actual_probe_fit(self):
        from search_g1.test_qprobe_runtime import runtime,stage_pending
        with tempfile.TemporaryDirectory() as directory:
            value=runtime(directory)
            with patch('search_g1.qprobe_runtime.prequential_metrics',side_effect=RuntimeError('diagnostic_only')):
                stage_pending(value)
            self.assertFalse(value.pending['prequential']['diagnostic_complete'])
            before=value.head.update_count
            event=value.complete_outer(1,'run:1')
            self.assertEqual(value.completed_outer,1)
            self.assertGreater(value.head.update_count,before)
            self.assertEqual(event['prequential']['diagnostic_error']['stage'],'prequential_metrics')

    def test_nonserializable_diagnostic_does_not_rollback_valid_update(self):
        from search_g1.test_qprobe_runtime import runtime,stage_pending
        with tempfile.TemporaryDirectory() as directory:
            value=runtime(directory)
            with patch('search_g1.qprobe_runtime.prequential_metrics',return_value={'diagnostic_nan':float('nan')}):
                stage_pending(value)
            event=value.complete_outer(1,'run:1')
            self.assertEqual(value.completed_outer,1)
            self.assertFalse(event['prequential']['diagnostic_complete'])

    def test_bad_label_alignment_is_still_fatal_before_diagnostics(self):
        from search_g1.test_qprobe_runtime import runtime,stage_pending
        with tempfile.TemporaryDirectory() as directory:
            value=runtime(directory);stage_pending(value)
            labels=value.pending['labels'];value.pending['labels']=None
            wrong=[[0]*5 for _ in range(512)]
            with self.assertRaisesRegex(ValueError,'alignment'):
                value.observe_labels(labels,wrong)

    def test_runtime_selection_uses_frozen_direct_predictions(self):
        source=Path(__file__).with_name('qprobe_runtime.py').read_text()
        self.assertIn('select_questions(questions, predictions,',source)
        self.assertIn('ranking="mixed_probability"',source)
        self.assertNotIn('predictions["probabilities"]',source)
        self.assertNotIn('predicted_mean=',source)
        self.assertNotIn('predicted_std=',source)

class BinaryHydraCompositionTests(unittest.TestCase):
    def test_actual_yaml_and_manifest_arguments_resolve_without_legacy_loss_keys(self):
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        from reproduction.common8.qprobe_configuration import arguments
        from search_g1.qprobe_model import QProbeConfig
        from search_g1.qprobe_runtime import validate_config
        repo=Path(__file__).resolve().parents[1]
        settings=dict(probe_architecture='qwen2_decoder3_binary_mix_v1',cache_dir='/mnt/cache',
            loss_version='binary_mix_variance_v1',variance_loss_weight=.1,
            teacher_warmup_epochs=10,bootstrap_max_groups_per_outer=5120)
        overrides=arguments({'question_probe':settings},'/mnt/binary-test')
        with initialize_config_dir(config_dir=str(repo/'verl/trainer/config'),version_base=None):
            config=compose(config_name='ppo_trainer',overrides=overrides)
        resolved=OmegaConf.to_container(config.question_probe,resolve=True)
        self.assertEqual(QProbeConfig.from_mapping(resolved).variance_loss_weight,.1)
        validate_config(resolved)
        self.assertEqual(resolved['teacher_warmup_epochs'],10)
        self.assertEqual(resolved['bootstrap_max_groups_per_outer'],5120)
        self.assertFalse({'mean_loss_weight','std_loss_weight','count_aux_loss_weight'}&set(resolved))

if __name__=='__main__':unittest.main()
