"""Real Hydra composition and initial-calibration routing for the 7B adapter."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reproduction.common8 import configuration
from reproduction.common8.formal_contract import assert_formal_run, formal_mathematics
from reproduction.common8.model_identity import SEVEN, identity
from reproduction.common8.run_pipeline import Pipeline
from reproduction.common8 import test_configuration as config_fixture
from reproduction.easy2560_joint4.control import BASE, MODEL, launch_command
from reproduction.easy2560_joint4.model7b import expected
from reproduction.easy2560_joint4.prepare_training import selection_settings
from reproduction.easy2560_joint4.test_common_contracts import easy_run


class SevenTests(unittest.TestCase):
    def fixture(self, directory):
        base, run = config_fixture.ConfigurationTests().fixture(directory)
        run.update(easy_run(8), model_revision=SEVEN[1], initial_model=MODEL, reference_model=MODEL)
        run.update(selection_settings('/remote/predecessor', 8))
        run['vllm_gpu_memory_utilization'] = .1
        run['budget'].update(model_id=SEVEN[0], model_revision=SEVEN[1], world_size=4)
        run['budget'].update(formal_mathematics(run))
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        return base, run

    @unittest.skipUnless(importlib.util.find_spec('hydra'), 'Hydra composition runs in existing HPC runtime')
    def test_actual_resolved_math_and_model_binding(self):
        with tempfile.TemporaryDirectory() as td:
            base, run = self.fixture(td)
            _, config, args = configuration.resolved(base)
            self.assertEqual(config.actor_rollout_ref.model.path, MODEL)
            self.assertEqual(config.actor_rollout_ref.rollout.gpu_memory_utilization, .1)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_micro_batch_size, 32)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_logical_micro_batch_size, 4)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size, 128)
            self.assertEqual(config.data.selection_candidate_questions, 2560)
            self.assertEqual(config.data.selection_selected_questions, 512)
            self.assertEqual(config.trainer.n_gpus_per_node, 4)
            self.assertFalse(config.algorithm.all_correct_virtual_zero)
            self.assertTrue(config.trainer.rolling_checkpoint)
            self.assertEqual(config.trainer.save_freq, 1)
            self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 1e-6)
            self.assertEqual(config.actor_rollout_ref.actor.optim.lr_warmup_steps, 57)
            self.assertNotIn('initial_reward_probe_source_base', run)
            self.assertEqual(2560 // config.actor_rollout_ref.actor.ppo_mini_batch_size, 20)
            self.assertEqual(len(args), len({a.split('=',1)[0] for a in args}))

    def test_seven_identity_cannot_be_non_easy_or_three_revision(self):
        with tempfile.TemporaryDirectory() as td:
            _, run = self.fixture(td)
            for mutation in [dict(easy2560_selection=False), dict(model_revision='wrong'),
                             dict(reference_model='/remote/3b')]:
                with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                    assert_formal_run(dict(run, **mutation))

    def test_round_zero_uses_actual_seven_refit_not_three_head_adoption(self):
        with tempfile.TemporaryDirectory() as td:
            base, run = self.fixture(td)
            pipeline = Mock()
            pipeline.run = run; pipeline.base = base; pipeline.python = '/remote/env/python'
            pipeline.probe_gpus = [3]
            with patch('reproduction.easy2560_joint4.adopt_reward_zero.adopt',
                       side_effect=AssertionError('3B head reuse is forbidden')):
                Pipeline.initial_probe_calibration(pipeline)
            command = pipeline.start.call_args.args[1]
            self.assertEqual(command, ['/remote/env/python','-m','reproduction.common8.refit',
                                      '--step','0','--model',MODEL])
            pipeline.wait.assert_called_once()

    def test_official_source_and_default_native_four_launch(self):
        manifest = expected()
        self.assertEqual((manifest['repo'],manifest['revision']), SEVEN)
        self.assertEqual(len(manifest['files']), 11)
        self.assertEqual(sum(r['bytes'] for r in manifest['files']), 15242788168)
        self.assertEqual(sum(r['file'].endswith('.safetensors') for r in manifest['files']), 4)
        args = launch_command(BASE, 'a'*40)
        self.assertIn('--gres=gpu:4', args)
        self.assertEqual(args[-2:], ['8',''])


if __name__ == '__main__': unittest.main()
