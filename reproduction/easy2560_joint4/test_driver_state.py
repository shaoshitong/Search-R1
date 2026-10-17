"""CPU driver/eligible-stream/RNG round trips; no model or GPU restore claims."""
import ast
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reproduction.common8 import behavior_runtime
from search_g1.training_state import (capture_easy2560_driver, restore_easy2560_driver,
                                      driver_rng, compact_stream_state)
from search_g1.full_state_restore import equal_tree


ROOT = Path(__file__).resolve().parents[2]


def run_fixture():
    return dict(profile='common8', easy2560_selection=True, execution_site='hpc3',
        execution_backend='slurm', training_world_size=4, data_selection_mode='mixed_first',
        actor_micro_batch_per_rank=32, actor_logical_micro_batch_per_rank=4,
        run_id='easy-cpu-fixture', source_commit='a'*40,
        selection_eligible_pool_path='/remote/eligible-question-pool.json',
        completed_updates=0, source_sha256={})


class Stream:
    def __init__(self, state): self.saved = copy.deepcopy(state)
    def state(self): return copy.deepcopy(self.saved)


def trainer_fixture(step=50):
    state = dict(candidate_order=[3, 1, 0, 2], candidate_cursor=4, epoch=2,
        consumed_source_rows=128002, pending_source=[dict(tokens=torch.tensor([3, 4]))]*3,
        pending_offset=1, prefix20_pool_binding=dict(eligible_pool_sha256='eligible-sha'))
    return NS(global_steps=step, mixed_stream=Stream(state),
        inline_attempt_progress=NS(completed_outer=step, cumulative_writes=step*20-3,
            attempt_index=step+2, aborted_attempts=2, full_budget_outer=step-1),
        reward_fn=NS(expected_probe_version='easy-cpu-fixture:0:sha', no_self=True),
        val_reward_fn=NS(expected_probe_version=None), train_dataloader=object())


class DriverTests(unittest.TestCase):
    def setUp(self): self.original = driver_rng()
    def tearDown(self):
        torch.set_rng_state(self.original['torch_cpu'])
        np.random.set_state(self.original['numpy']); random.setstate(self.original['python'])
    def live(self, **changes):
        return dict(dict(run_id='easy-cpu-fixture', version='easy-cpu-fixture:0:sha', round=0), **changes)

    def test_capture_does_not_draw_rng_and_retains_partial_pending_without_head(self):
        trainer = trainer_fixture(); before = driver_rng()
        saved = capture_easy2560_driver(trainer, run_fixture(), live_probe=self.live())
        equal_tree(driver_rng(), before); equal_tree(saved['rng'], before)
        equal_tree(saved['candidate_stream'], compact_stream_state(trainer.mixed_stream.state()))
        self.assertTrue(saved['reward_refresh_pending']); self.assertIsNone(saved['question_probe'])
        trainer.mixed_stream.saved['pending_source'][0]['tokens'][0] = 99
        self.assertEqual(int(saved['candidate_stream']['pending_source'][0]['tokens'][0]), 3)

    def test_actual_default_restore_uses_bound_eligible_pool_and_exact_rng_future(self):
        trainer = trainer_fixture(); run = run_fixture()
        saved = capture_easy2560_driver(trainer, run, live_probe=self.live())
        expected = (torch.rand(9), np.random.rand(9), random.random())
        torch.rand(7); np.random.rand(8); random.random()
        calls = []
        class RestoredEligible(Stream):
            def __init__(self, loader, path, *, restored_state):
                calls.append((loader, path)); super().__init__(restored_state)
        target = trainer_fixture(); target.reward_fn.expected_probe_version = None
        with patch.dict(sys.modules, {'search_g1.prefix20_data': NS(Prefix20CandidateStream=RestoredEligible)}):
            proof = restore_easy2560_driver(target, saved, run,
                source_run_id=run['run_id'], live_probe=self.live())
        self.assertEqual(calls, [(target.train_dataloader, run['selection_eligible_pool_path'])])
        equal_tree(target.mixed_stream.state(), saved['candidate_stream'])
        equal_tree(target._restored_attempt_progress, saved['attempt_progress'])
        self.assertTrue(target._restored_reward_refresh_pending)
        self.assertTrue(proof['actual_driver_rng_readback_equal'])
        self.assertTrue(proof['actual_candidate_stream_readback_equal'])
        self.assertTrue(torch.equal(expected[0], torch.rand(9)))
        self.assertTrue(np.array_equal(expected[1], np.random.rand(9)))
        self.assertEqual(expected[2], random.random())
        self.assertEqual(target.val_reward_fn.expected_probe_version, saved['probe_version'])

    def test_restore_rejects_head_foreign_scope_or_live_version_before_rng_mutation(self):
        run = run_fixture(); saved = capture_easy2560_driver(trainer_fixture(), run, live_probe=self.live())
        for kind in ('head', 'logical', 'mode', 'version', 'round'):
            record = copy.deepcopy(saved); actual_run = copy.deepcopy(run); live = self.live()
            if kind == 'head': record['question_probe'] = {'path': 'forbidden'}
            if kind == 'logical': actual_run['actor_logical_micro_batch_per_rank'] = 32
            if kind == 'mode': actual_run['data_selection_mode'] = 'prefix20_probe'
            if kind == 'version': live['version'] = 'wrong'
            if kind == 'round': live['round'] = 50
            before = driver_rng()
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                restore_easy2560_driver(trainer_fixture(), record, actual_run,
                    source_run_id=run['run_id'], live_probe=live,
                    stream_factory=lambda loader, restored_state: Stream(restored_state))
            equal_tree(driver_rng(), before)

    def test_final200_restore_has_no_pending_refit_and_preserves_actual_write_count(self):
        trainer = trainer_fixture(200); run = run_fixture()
        trainer.reward_fn.expected_probe_version = 'easy-cpu-fixture:150:sha'
        live = self.live(version=trainer.reward_fn.expected_probe_version, round=150)
        saved = capture_easy2560_driver(trainer, run, live_probe=live)
        self.assertFalse(saved['reward_refresh_pending'])
        proof = restore_easy2560_driver(trainer, saved, run,
            source_run_id=run['run_id'], live_probe=live,
            stream_factory=lambda loader, restored_state: Stream(restored_state))
        self.assertEqual(proof['attempt_progress']['cumulative_writes'], 3997)
        self.assertFalse(proof['reward_refresh_pending'])

    def test_actual_save_checkpoint_persists_easy_driver_without_question_head(self):
        tree = ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_save_checkpoint')
        namespace = dict(os=os)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     'actual_save_checkpoint.py', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); (base/'logs').mkdir(); (base/'artifacts/live').mkdir(parents=True)
            run = run_fixture()
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            (base/'artifacts/live/manifest.json').write_text(json.dumps(self.live()))
            trainer = trainer_fixture()
            trainer.config = NS(trainer=NS(default_local_dir=str(base/'checkpoints'), default_hdfs_dir=None))
            trainer.actor_rollout_wg = NS(
                save_checkpoint=lambda local, remote: Path(local).mkdir(parents=True),
                capture_mixed_state=lambda directory, checkpoint: [{'rank': rank} for rank in range(4)])
            trainer.use_critic = False
            with patch.dict(os.environ, SEARCH_G1_BASE=folder): namespace['_save_checkpoint'](trainer)
            directory = base/'checkpoints/actor/global_step_50/training_state'
            saved = torch.load(directory/'driver.pt', map_location='cpu', weights_only=False)
            manifest = json.loads((directory/'manifest.json').read_text())
            self.assertIsNone(saved['question_probe']); self.assertIsNone(manifest['question_probe'])
            self.assertEqual(saved['selection_mode'], 'mixed_first')
            self.assertEqual(saved['attempt_progress']['cumulative_writes'], 997)
            equal_tree(saved['candidate_stream'], trainer.mixed_stream.state())

    def test_behavior_native_scope_accepts_easy_and_rejects_wrong_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); (base/'logs').mkdir(); run = run_fixture()
            with patch.dict(os.environ, SEARCH_G1_BASE=folder), \
                    patch.object(behavior_runtime, 'validate_base'), \
                    patch.object(behavior_runtime, 'execution_job_id', return_value='cpu-only'):
                for logical in (4, 32):
                    run['actor_logical_micro_batch_per_rank'] = logical
                    (base/'logs/run-manifest.json').write_text(json.dumps(run))
                    if logical == 4:
                        self.assertEqual(behavior_runtime._native_run()[1], run)
                    else:
                        with self.assertRaises(AssertionError): behavior_runtime._native_run()


if __name__ == '__main__': unittest.main()
