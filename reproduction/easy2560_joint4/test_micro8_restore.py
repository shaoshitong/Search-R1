"""CPU regression for physical16 CP5 -> physical8 recovery, no tensor reload."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.common8 import prepare_run
from reproduction.easy2560_joint4 import control, prepare_training
from search_g1 import easy2560_restore as saved, full_state_restore
from search_g1.test_easy2560_restore import Fixture, write


class Micro8RestoreTests(unittest.TestCase):
    def fixture(self, directory):
        fixture = Fixture(directory)
        fixture.run['budget'].update(ppo_micro_batch_size=64, per_rank_actor_micro_batch=16,
            ppo_mini_batch_size=128, per_rank_actor_logical_micro_batch=4,
            objective_reduction='original four-row micro masked_mean')
        write(fixture.base/'logs/run-manifest.json', fixture.run)
        fixture.create()
        return fixture

    def child(self, fixture, physical=8):
        child = copy.deepcopy(fixture.run)
        child.update(run_id='new-allocation', source_commit='9'*40, actor_micro_batch_per_rank=physical)
        child['budget'].update(ppo_micro_batch_size=physical*4, per_rank_actor_micro_batch=physical)
        return child

    def test_real_CP5_receipt_binds_eight_without_rewriting_saved_state(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(td); child = self.child(fixture)
            original = fixture.output.read_bytes()
            saved.bind(child, fixture.output)
            binding = child['full_state_restore']
            transition = binding['physical_micro_transition']
            self.assertEqual(child['actor_micro_batch_per_rank'], 8)
            self.assertEqual(transition['source_accumulation_steps'], 2)
            self.assertEqual(transition['target_accumulation_steps'], 4)
            self.assertEqual(transition['global_mini_batch'], 128)
            self.assertEqual(transition['logical_micro_per_rank'], 4)
            self.assertFalse(transition['bitwise_continuation_claimed'])
            self.assertEqual(binding['completed_outer'], 5)
            self.assertEqual(binding['actual_optimizer_writes_per_rank'], 100)
            self.assertEqual(binding['candidate_source_cursor'], 12800)
            self.assertEqual(child['initial_probe_version'], fixture.version)
            self.assertEqual(child['run_id'], fixture.run_id)
            self.assertEqual(fixture.output.read_bytes(), original)
            self.assertFalse(binding['post_callback_rng_captured'])

    def test_no_upward_physical_change_or_scientific_budget_drift(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(td)
            for physical in (32, 2, True):
                with self.subTest(physical=physical), self.assertRaises(ValueError):
                    saved.bind(self.child(fixture, physical), fixture.output)
            for key, value in (('learning_rate', 2e-6), ('ppo_mini_batch_size', 64),
                               ('per_rank_actor_logical_micro_batch', 8), ('world_size', 8),
                               ('ppo_micro_batch_size', 64), ('per_rank_actor_micro_batch', 16)):
                child = self.child(fixture); child['budget'][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    saved.bind(child, fixture.output)
            child = self.child(fixture); child['input_sha256']['train.parquet'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'SCIENTIFIC_INPUT'):
                saved.bind(child, fixture.output)

    def test_saved_original_scope_remains_physical16_only(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = Fixture(td); fixture.run['actor_micro_batch_per_rank'] = 8
            write(fixture.base/'logs/run-manifest.json', fixture.run)
            with self.assertRaisesRegex(ValueError, 'EXACT_NATIVE4_EASY_SCOPE'):
                fixture.create()

    def test_training_constructor_accepts_global_micro32_with_mini128(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(td)
            config = SimpleNamespace(trainer=dict(full_state_restore_receipt=str(fixture.output)))
            config.trainer = type('Trainer', (dict,), {'__getattr__':dict.__getitem__})(
                full_state_restore_receipt=str(fixture.output), total_training_steps=201, acceptance_step=5)
            actor = SimpleNamespace(ppo_micro_batch_size=32, ppo_mini_batch_size=128,
                                    ppo_logical_micro_batch_size=4)
            config.actor_rollout_ref = SimpleNamespace(actor=actor,
                model=SimpleNamespace(actor_checkpoint_path=str(fixture.cp)))
            config.data = SimpleNamespace(selection_mode='mixed_first')
            config.algorithm = SimpleNamespace(all_correct_virtual_zero=False)
            self.assertEqual(full_state_restore.from_config(config)['step'], 5)
            for key, value in (('ppo_micro_batch_size',128), ('ppo_micro_batch_size',31),
                               ('ppo_mini_batch_size',64), ('ppo_logical_micro_batch_size',8)):
                previous = getattr(actor, key); setattr(actor, key, value)
                with self.subTest(key=key), self.assertRaises((ValueError, AssertionError)):
                    full_state_restore.from_config(config)
                setattr(actor, key, previous)

    def test_common_preparation_reaches_bind_with_source16_target8(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve(); base = root/'new'; origin = root/'old'; repo = base/'repo'
            (repo/'reproduction/common8').mkdir(parents=True)
            (base/'data').mkdir(); (base/'logs').mkdir(); (origin/'data').mkdir(parents=True)
            for name in ('recipe.json','budget.json'):
                shutil.copy2(Path(prepare_run.__file__).parent/name, repo/'reproduction/common8'/name)
            for name in control.DATA_NAMES: (base/'data'/name).write_text(name)
            (origin/'data/calibration-300.parquet').write_text('saved calibration')
            write(origin/'data/audit.json', dict(calibration_question_ids=['q1']))
            marker = repo/'actual.py'; marker.write_text('pass\n')
            receipt = root/'receipt.json'; receipt.write_text('{}')
            settings = prepare_training.selection_settings(origin, 8, restoring=True)
            def bind(run, path):
                self.assertEqual(run['actor_micro_batch_per_rank'], 8)
                self.assertEqual(run['budget']['ppo_micro_batch_size'], 32)
                self.assertEqual(run['budget']['ppo_mini_batch_size'], 128)
                self.assertEqual(run['budget']['per_rank_actor_logical_micro_batch'], 4)
                run['full_state_restore'] = dict(kind=saved.KIND, completed_outer=5)
            with patch.dict(os.environ, SEARCH_G1_TRAINING_WORLD_SIZE='4',
                    SEARCH_G1_EXECUTION_PROFILE='native4_bf16_varlen_fp32_master'), patch.object(
                    prepare_run,'verify', return_value={'manifest_sha256':'0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'}), patch.object(
                    prepare_run,'verify_model',return_value={'config.json':'d'*64}), patch(
                    'reproduction.common8.source_files.runtime_files',return_value=[marker]), patch(
                    'search_g1.full_state_restore.source',return_value=(receipt,{'status':saved.STATUS,'world_size':4},root/'cp',origin,
                                                                         {'actor_micro_batch_per_rank':16})), patch(
                    'search_g1.full_state_restore.bind',side_effect=bind) as bound, patch(
                    'reproduction.common8.topology.validate_base',return_value=base), patch(
                    'reproduction.common8.shared_slurm.bind'), patch('reproduction.common8.native_ray.prepare'):
                run = prepare_run.prepare(base,root/'official',root/'official.json','a'*40,
                    restore_receipt=str(receipt),execution_site='hpc3',actor_micro_batch_per_rank=8,
                    actor_logical_micro_batch_per_rank=4,easy2560_selection=True,easy2560_settings=settings)
            bound.assert_called_once()
            self.assertEqual(run['full_state_restore']['completed_outer'],5)


if __name__ == '__main__': unittest.main()
