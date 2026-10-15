"""The resolved launch preserves 512/G5 and only changes the selection mechanism."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from reproduction.common8 import configuration
from reproduction.common8.qprobe_configuration import arguments
from reproduction.common8 import test_configuration as baseline_tests


class QProbeConfigurationTests(unittest.TestCase):
    def fixture(self, root, exploration):
        base, run = baseline_tests.ConfigurationTests().fixture(root)
        run["question_probe"] = dict(cache_dir=str(base/"question-features"),
                                    exploration_questions=exploration)
        run["retriever_version"] = "immutable-index-fixture"
        (base/"logs/run-manifest.json").write_text(json.dumps(run))
        return base, run

    def test_actual_hydra_all_exploration_values_preserve_actor_recipe(self):
        for exploration in (0, 64, 128, 256):
            with self.subTest(exploration=exploration), tempfile.TemporaryDirectory() as root:
                base, run = self.fixture(root, exploration)
                _, config, overrides = configuration.resolved(base)
                self.assertEqual(config.data.selection_mode, "question_probe")
                self.assertFalse(config.data.mixed_first_selection)
                self.assertEqual(config.question_probe.exploration_questions, exploration)
                self.assertEqual(config.question_probe.candidate_questions, 5120)
                self.assertEqual(config.data.train_batch_size*config.actor_rollout_ref.rollout.n_agent, 2560)
                self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size, 128)
                self.assertEqual(config.actor_rollout_ref.actor.ppo_epochs, 1)
                self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 1e-6)
                self.assertEqual(config.actor_rollout_ref.actor.policy_safety.dual_clip, 3.)
                self.assertEqual(len(overrides), len({x.split("=", 1)[0] for x in overrides}))

    def test_unbound_defaults_leave_legacy_recipe_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            base, _ = baseline_tests.ConfigurationTests().fixture(root)
            _, config, _ = configuration.resolved(base)
            self.assertEqual(config.data.selection_mode, "legacy")
            self.assertTrue(config.data.mixed_first_selection)

    def test_sigma3_arguments_bind_all_loss_fields_without_changing_actor_controls(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=self.fixture(root,0)
            loss=dict(loss_mode='sigma3_std_aux6_v1',mean_loss_weight=0.0,
                      std_loss_weight=3.0,count_aux_loss_weight=0.1)
            run['question_probe'].update(loss)
            overrides=arguments(run,base)
            parsed={key:json.loads(raw) for key,raw in (row.split('=',1) for row in overrides[1:])}
            for name,value in loss.items():self.assertEqual(parsed['question_probe.'+name],value)
            self.assertEqual(overrides[0],'data.selection_mode=question_probe')
            self.assertTrue(all(row.startswith('question_probe.') for row in overrides[1:]))
            for name in loss:
                changed=copy.deepcopy(run);changed['question_probe'].pop(name)
                if name=='loss_mode':continue  # A legacy manifest may omit its historical default.
                with self.subTest(missing=name),self.assertRaisesRegex(ValueError,'explicitly bind'):
                    arguments(changed,base)

    def test_sigma3_real_hydra_resolves_explicit_loss_fields(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=self.fixture(root,0)
            loss=dict(loss_mode='sigma3_std_aux6_v1',mean_loss_weight=0.0,std_loss_weight=3.0,count_aux_loss_weight=0.1)
            run['question_probe'].update(loss);(base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,config,_=configuration.resolved(base)
            for name,value in loss.items():self.assertEqual(config.question_probe[name],value)

    def test_invalid_budgets_or_relative_paths_rejected_before_launch(self):
        for changed in ({"exploration_questions": True}, {"exploration_questions": -1},
                        {"exploration_questions": 513}, {"selected_questions": 128},
                        {"responses_per_question": 4}, {"cache_dir": "relative"},
                        {"unknown": 1}, {"loss_mode":"unknown"},
                        {"loss_mode":"sigma3_std_aux6_v1"}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as root:
                base, run = self.fixture(root, 64)
                run["question_probe"].update(changed)
                with self.assertRaises(ValueError):
                    arguments(run, base)


if __name__ == "__main__":
    unittest.main()
