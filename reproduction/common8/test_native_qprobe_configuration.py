"""Native manifest and real Hydra composition; no model or GPU execution."""
import copy
from pathlib import Path
import tempfile
import unittest

from reproduction.common8.qprobe_configuration import arguments


def native_settings(base):
    return dict(head_kind='qwen_native_lastblock_binary_v1', loss_mode='token_binary_std_v1',
        ranking='binary_mixed_probability', candidate_questions=5120, selected_questions=512,
        responses_per_question=5, exploration_questions=0, feature_dim=2048, hidden_size=2048,
        head_hidden_size=2048, num_attention_heads=16, num_key_value_heads=2, intermediate_size=11008,
        rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=32768, source_layer=35,
        num_hidden_layers=36, attention_implementation='sdpa', micro_batch_size=32,
        mean_loss_weight=0.0, count_aux_loss_weight=0.0, std_loss_weight=0.3, target_statistic='sample_std',
        bootstrap_head_epochs=10, bootstrap_max_groups_per_outer=5120, head_batch_size=256,
        head_epochs_per_outer=2, replay_window_outer=8, teacher_replay_excluded=True,
        head_learning_rate=0.0003, seed=20261013, head_seed=1729, max_tokens=244, head_device='cuda:0',
        cache_dir=str(base/'question-features'), initial_head_checkpoint=str(base/'initial.pt'),
        initial_head_checkpoint_sha256='a'*64, initial_head_ready=str(base/'initial.json'),
        initial_head_ready_sha256='b'*64)


class NativeHeadManifestTests(unittest.TestCase):
    def test_effective_native_recipe_and_paired_restore_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);settings=native_settings(base)
            result=arguments({'question_probe':settings},base)
            self.assertIn('question_probe.head_hidden_size=2048',result)
            self.assertIn('question_probe.head_epochs_per_outer=2',result)
            self.assertIn('question_probe.replay_window_outer=8',result)
            self.assertEqual(len(result),len({row.split('=',1)[0] for row in result}))
            settings['snapshot_path']=str(base/'paired-state.pt')
            for key in ('initial_head_checkpoint','initial_head_checkpoint_sha256',
                        'initial_head_ready','initial_head_ready_sha256'):
                settings[key]=None
            self.assertTrue(arguments({'question_probe':settings},base))

    def test_old_architecture_window_loss_and_unbound_initialization_are_refused(self):
        invalid=[dict(head_hidden_size=128),dict(num_attention_heads=4),dict(num_key_value_heads=16),
                 dict(intermediate_size=2048),dict(bootstrap_head_epochs=20),dict(std_loss_weight=3.0),
                 dict(count_aux_loss_weight=0.1),dict(replay_window_outer=16),dict(exploration_questions=64),
                 dict(teacher_replay_excluded=False),dict(teacher_replay_excluded=1),dict(max_tokens=245),
                 dict(initial_head_checkpoint=None),dict(head_device='cuda:4')]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            for changed in invalid:
                settings=copy.deepcopy(native_settings(base));settings.update(changed)
                with self.subTest(changed=changed),self.assertRaises(ValueError):
                    arguments({'question_probe':settings},base)

    def test_actual_hydra_resolves_native_recipe_without_legacy_hidden_alias(self):
        import json
        from reproduction.common8 import configuration
        from reproduction.common8.test_configuration import ConfigurationTests
        from search_g1.qprobe_qwen_block_online import QwenBlockOnlineConfig
        with tempfile.TemporaryDirectory() as directory:
            base,run=ConfigurationTests().fixture(directory)
            run['question_probe']=native_settings(base)
            run['retriever_version']='immutable-native-config-cpu-fixture'
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,config,overrides=configuration.resolved(base)
            self.assertEqual(config.question_probe.head_hidden_size,2048)
            self.assertEqual(config.question_probe.head_device,'cuda:0')
            online=QwenBlockOnlineConfig.from_mapping(dict(config.question_probe))
            self.assertEqual(online.hidden_size,2048)
            self.assertEqual(online.seed,1729)
            self.assertEqual(online.epochs,10)
            self.assertEqual(online.head_epochs_per_outer,2)
            self.assertEqual(online.replay_window_outer,8)
            self.assertEqual(config.data.train_batch_size*config.actor_rollout_ref.rollout.n_agent,2560)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size,128)
            self.assertEqual(config.actor_rollout_ref.actor.optim.lr_warmup_steps,57)
            self.assertEqual(len(overrides),len({row.split('=',1)[0] for row in overrides}))


if __name__=='__main__':
    unittest.main()
