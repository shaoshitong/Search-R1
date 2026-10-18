"""CPU launch plumbing: explicit receipt, once-only claim, pre-bind Easy scope."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import control, prepare_training
from reproduction.common8 import prepare_run


class RestoreEntryTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('hydra'), 'Use existing HPC environment for real Hydra composition')
    def test_rolling_checkpoint_cadence_preserves_four_gpu_scientific_recipe(self):
        from reproduction.common8 import configuration, test_configuration
        from reproduction.common8.formal_contract import formal_mathematics, SKIP_INITIAL_EVALUATION_POLICY
        from reproduction.common8.precision_contract import NATIVE4_BF16_PROFILE, SAMPLING_JOURNAL_POLICY
        with tempfile.TemporaryDirectory() as td:
            base, run = test_configuration.ConfigurationTests().fixture(td)
            run.update(prepare_training.selection_settings(base/'old', 8))
            run.update(execution_site='hpc3', execution_backend='slurm', training_world_size=4,
                precision_profile=NATIVE4_BF16_PROFILE, vllm_gpu_memory_utilization=.1,
                behavior_probability_policy=SAMPLING_JOURNAL_POLICY,
                pretraining_baseline_policy=SKIP_INITIAL_EVALUATION_POLICY)
            run['budget']['world_size'] = 4
            run['budget'].update(formal_mathematics(run))
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            _, config, args = configuration.resolved(base)
            self.assertEqual(len({a.split('=',1)[0] for a in args}), len(args))
            self.assertEqual((config.trainer.save_freq, config.trainer.test_freq), (1,50))
            self.assertTrue(config.trainer.rolling_checkpoint)
            self.assertEqual(config.trainer.n_gpus_per_node, 4)
            self.assertEqual(config.trainer.total_training_steps, 201)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size, 128)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_micro_batch_size, 32)
            self.assertEqual(config.actor_rollout_ref.actor.ppo_logical_micro_batch_size, 4)
            self.assertEqual(config.data.selection_candidate_questions, 2560)
            self.assertFalse(config.algorithm.all_correct_virtual_zero)

    def test_explicit_fourth_argument_and_once_claim_pin_the_receipt_sha(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve(); receipt = base / 'saved-state.json'
            receipt.write_text('{"status":"CPU_VALIDATED_SOURCE"}\n')
            with patch.object(control, 'own_base', side_effect=Path), patch.object(
                    control, 'bootstrap'), patch.object(control.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 0, '731001\n', '')) as submit:
                result = control.submit(base, 'a' * 40, 16, '/old/run', receipt)
                command = submit.call_args.args[0]
                self.assertEqual(command[-4:], [str(base), 'a' * 40, '16', str(receipt)])
                claim = json.loads((base / 'preparation/training-submit-claim.json').read_text())
                self.assertEqual(claim['restore_receipt'], str(receipt))
                self.assertEqual(claim['restore_receipt_sha256'], control.sha(receipt))
                self.assertEqual(result['restore_receipt_sha256'], claim['restore_receipt_sha256'])
                with self.assertRaisesRegex(ValueError, 'SUBMISSION_ALREADY_ATTEMPTED'):
                    control.submit(base, 'a' * 40, 16, '/old/run', receipt)
                self.assertEqual(submit.call_count, 1)

    def test_shell_passes_restore_as_cli_argument_and_fresh_has_no_restore(self):
        script = Path(prepare_training.__file__).with_name('train.sh')
        subprocess.run(['bash', '-n', str(script)], check=True)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for restored in (False, True):
                base = root / ('restored' if restored else 'fresh')
                (base / 'repo').mkdir(parents=True); (base / 'env/bin').mkdir(parents=True)
                log = base / 'calls.jsonl'
                python = base / 'env/bin/python'
                python.write_text('#!/usr/bin/env python3\nimport json,sys\n'
                    'with open(' + repr(str(log)) + ', "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
                python.chmod(0o755)
                receipt = str(root / 'source.json') if restored else ''
                subprocess.run(['bash', str(script), str(base), 'a' * 40, '16', receipt], check=True)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(calls), 2)
                self.assertIn('reproduction.easy2560_joint4.prepare_training', calls[0])
                self.assertIn('reproduction.easy2560_joint4.native_ray_resources', calls[1])
                if restored:
                    self.assertEqual(calls[0][-2:], ['--restore-receipt', receipt])
                else:
                    self.assertNotIn('--restore-receipt', calls[0])

    def test_predecessor_easy_manifest_reuses_actual_eligible_pool_without_a_head(self):
        with tempfile.TemporaryDirectory() as td:
            old = Path(td); (old / 'logs').mkdir()
            pool = '/data/user/jxu957/experiments/original/assets/prefix20/eligible-question-pool.json'
            (old / 'logs/run-manifest.json').write_text(json.dumps({'selection_eligible_pool_path': pool}))
            settings = prepare_training.selection_settings(old, 16, restoring=True)
            self.assertEqual(settings['selection_eligible_pool_path'], pool)
            self.assertNotIn('initial_reward_probe_source_base', settings)
            self.assertNotIn('question_probe', settings)
            self.assertIs(settings['all_correct_virtual_zero'], False)

    def training_fixture(self, root):
        base = root / 'new'; old = root / 'old'
        (base / 'preparation').mkdir(parents=True); (base / 'logs').mkdir()
        (old / 'logs').mkdir(parents=True)
        (old / 'logs/run-manifest.json').write_text(json.dumps({'selection_eligible_pool_path': '/actual/pool.json'}))
        receipt = root / 'source-receipt.json'; receipt.write_text('{"actual":"receipt"}\n')
        (base / 'preparation/environment-reuse.json').write_text(json.dumps({'predecessor': str(old)}))
        (base / 'preparation/official-model-manifest.json').write_text('{}')
        claim = dict(source_commit='a' * 40, actor_micro_batch_per_rank=16, **control.restore_binding(receipt))
        (base / 'preparation/training-submit-claim.json').write_text(json.dumps(claim))
        return base, receipt

    def test_training_prepare_passes_receipt_and_all_settings_directly_to_common(self):
        with tempfile.TemporaryDirectory() as td:
            base, receipt = self.training_fixture(Path(td).resolve())
            def prepare_actor(*args, **kwargs):
                self.assertEqual(kwargs['restore_receipt'], str(receipt))
                settings = kwargs['easy2560_settings']
                self.assertEqual(settings['pretraining_baseline_policy'], 'reuse_historical')
                self.assertEqual(settings['selection_eligible_pool_path'], '/actual/pool.json')
                self.assertIn('infra_rollout', settings); self.assertIn('retriever_version', settings)
                self.assertNotIn('initial_reward_probe_source_base', settings)
                return dict(run_id='b' * 32, full_state_restore={'completed_outer': 5})
            config = SimpleNamespace(data=SimpleNamespace(selection_mode='mixed_first'),
                                     algorithm=SimpleNamespace(all_correct_virtual_zero=False))
            with patch.object(prepare_training, 'own_base', side_effect=Path), patch.dict(
                    os.environ, SEARCH_G1_TRAINING_WORLD_SIZE='4', SLURM_JOB_ID='731001'), patch(
                    'reproduction.common8.prepare_run.prepare', side_effect=prepare_actor) as common, patch(
                    'reproduction.common8.configuration.resolved', return_value=(None, config, None)):
                proof = prepare_training.prepare(base, 'a' * 40, 16, receipt)
                self.assertEqual(common.call_count, 1)
                self.assertEqual(proof['restored_completed_outer'], 5)
                self.assertFalse(proof['actual_cuda_restore_accepted'])

    def test_changed_receipt_rejected_before_model_preparation(self):
        with tempfile.TemporaryDirectory() as td:
            base, receipt = self.training_fixture(Path(td).resolve()); receipt.write_text('changed')
            with patch.object(prepare_training, 'own_base', side_effect=Path), patch.dict(
                    os.environ, SEARCH_G1_TRAINING_WORLD_SIZE='4', SLURM_JOB_ID='731001'), patch(
                    'reproduction.common8.prepare_run.prepare') as common:
                with self.assertRaisesRegex(ValueError, 'SUBMITTED_RESTORE_RECEIPT_DIFFERS'):
                    prepare_training.prepare(base, 'a' * 40, 16, receipt)
                common.assert_not_called()

    def test_common_restore_method_is_not_fresh_and_scope_is_complete_before_bind(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(); base = root / 'new'; origin = root / 'old'
            repo = base / 'repo'; (repo / 'reproduction/common8').mkdir(parents=True)
            (base / 'data').mkdir(); (base / 'logs').mkdir(); (origin / 'data').mkdir(parents=True)
            source_repo = Path(prepare_run.__file__).parent
            for name in ('recipe.json', 'budget.json'):
                shutil.copy2(source_repo / name, repo / 'reproduction/common8' / name)
            audit = {'calibration_question_ids': ['q1', 'q2']}
            for name in control.DATA_NAMES: (base / 'data' / name).write_text(name)
            (origin / 'data/calibration-300.parquet').write_text('actual saved calibration')
            (origin / 'data/audit.json').write_text(json.dumps(audit))
            marker = repo / 'actual.py'; marker.write_text('pass\n')
            settings = prepare_training.selection_settings(origin, 16, restoring=True)
            settings.update(infra_rollout={'actual': 'rollout'}, retriever_version='retriever-sha',
                            pretraining_baseline_policy='reuse_historical')
            receipt = root / 'receipt.json'; receipt.write_text('{}')
            from search_g1.easy2560_restore import STATUS
            def bind(run, path):
                self.assertEqual(path, str(receipt))
                self.assertEqual(run['training_world_size'], 4)
                self.assertEqual(run['actor_micro_batch_per_rank'], 16)
                self.assertEqual(run['actor_logical_micro_batch_per_rank'], 4)
                self.assertEqual(run['selection_eligible_pool_path'], settings['selection_eligible_pool_path'])
                self.assertEqual(run['infra_rollout'], settings['infra_rollout'])
                self.assertEqual(run['retriever_version'], 'retriever-sha')
                method = json.loads((base / 'preparation/formal-inline-method.json').read_text())
                self.assertFalse(method['fresh_states'])
                run['run_id'] = 'c' * 32
                run['full_state_restore'] = dict(kind='easy2560_native4', completed_outer=5)
            with patch.dict(os.environ, SEARCH_G1_TRAINING_WORLD_SIZE='4',
                    SEARCH_G1_EXECUTION_PROFILE='native4_bf16_varlen_fp32_master'), patch.object(
                    prepare_run, 'verify', return_value={'manifest_sha256': '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'}), patch.object(
                    prepare_run, 'verify_model', return_value={'config.json': 'd' * 64}), patch.object(
                    prepare_run, 'derive_calibration', side_effect=AssertionError('restore cannot derive fresh data')), patch(
                    'reproduction.common8.source_files.runtime_files', return_value=[marker]), patch(
                    'search_g1.full_state_restore.source', return_value=(receipt, {'status': STATUS, 'world_size': 4}, root / 'cp', origin,
                                                                         {'actor_micro_batch_per_rank': 16})), patch(
                    'search_g1.full_state_restore.bind', side_effect=bind), patch(
                    'reproduction.common8.topology.validate_base', return_value=base), patch(
                    'reproduction.common8.shared_slurm.bind'), patch('reproduction.common8.native_ray.prepare'):
                run = prepare_run.prepare(base, root / 'official', root / 'official.json', 'a' * 40,
                    restore_receipt=str(receipt), execution_site='hpc3', actor_micro_batch_per_rank=16,
                    actor_logical_micro_batch_per_rank=4, easy2560_selection=True, easy2560_settings=settings)
                self.assertEqual(run['run_id'], 'c' * 32)
                self.assertEqual(run['full_state_restore']['completed_outer'], 5)
                self.assertEqual((base / 'data/calibration-300.parquet').read_text(), 'actual saved calibration')


if __name__ == '__main__': unittest.main()
