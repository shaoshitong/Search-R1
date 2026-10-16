"""Tiny CPU contracts for four-rank planning and fail-closed source extraction."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from .contract import WORLD, QMAX, WIDTH, DIM
from . import prefix5_model as model_code
from .probe_plan import complete_pool_plan, online_plan, rank_question_slice, ddp_mean_scale


def tiny_complete_source():
    shapes = {name: (1,) for name in model_code.PARAMETER_SHAPES}
    raw = {name: torch.tensor([1.]) for name in shapes}
    ema = {name: torch.tensor([2.]) for name in shapes}
    buffers = {'rotary.inv_freq': torch.ones(64)}
    metrics = dict(epoch=3, ema_updates=26100, same_train_test_weight_kind='ema0.99',
                   train_ema=dict(groups=445440, trajectories=2227200),
                   test_ema=dict(groups=512, trajectories=2560))
    source = dict(role='fresh_question5_joint200_ema099', epoch=3, completed_epoch=True,
        next_update_in_epoch=0, actor_updates=0, raw_model=raw, raw_nonpersistent_buffers=buffers,
        actual_adam_steps_per_rank=[26100]*4,
        ema=dict(model=ema, nonpersistent_buffers=copy.deepcopy(buffers), decay=.99, updates=26100),
        counters=dict(complete_epochs=3, optimizer_steps=26100, ema_updates=26100,
                      global_groups_seen=445440*3, global_trajectories_seen=2227200*3),
        binding=dict(role='fresh_question5_joint200_ema099', run_id=model_code.WARMUP_RUN,
            official_revision=model_code.OFFICIAL_REVISION,
            common_manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
            question_max_tokens=273, prefix_tokens=5, trajectory_bce_weight=1., mixed_bce_weight=1.,
            ema_decay=.99, actor_updates=0,
            head_copy=dict(question_max_tokens=273, policy_tokens=5, hidden_size=2048,
                           feature_layer=27, feature_boundary='post_block',
                           source_revision=model_code.OFFICIAL_REVISION, source_layer=35,
                           layout='question_right_pad_then_fixed_five_policy_tail', master_dtype='float32')),
        epoch_metrics=metrics,
        epoch_metrics_sha256=hashlib.sha256(model_code._canonical(metrics)).hexdigest())
    return source, shapes


class Prefix5Contracts(unittest.TestCase):
    def test_actual_transformers446_dual_native_rope_buffers_are_preserved(self):
        source, shapes = tiny_complete_source()
        for state in (source['raw_nonpersistent_buffers'], source['ema']['nonpersistent_buffers']):
            state['block.self_attn.rotary_emb.inv_freq'] = torch.ones(64)
        with patch.dict(model_code.PARAMETER_SHAPES, shapes, clear=True):
            _, buffers, proof = model_code.warmup_model_weights(source, epoch=3)
        self.assertEqual(set(buffers), {'rotary.inv_freq', 'block.self_attn.rotary_emb.inv_freq'})
        self.assertEqual(proof['nonpersistent_buffer_names'], sorted(buffers))
        with self.assertRaises(ValueError):
            model_code.validate_native_buffers({'rotary.inv_freq': torch.ones(64), 'invented': torch.ones(64)})

    def test_complete_pool_covers_all_own_g5_and_matches_true_global_ddp_mean(self):
        plan = complete_pool_plan(1729, 20)
        self.assertEqual(len(plan), 50)
        self.assertEqual(sorted(i for row in plan for i in row['question_indices']), list(range(2560)))
        self.assertEqual([sum(len(rank_question_slice(row, rank)) for row in plan)
                          for rank in range(WORLD)], [640]*4)
        for row in plan:
            parts = [rank_question_slice(row, rank) for rank in range(WORLD)]
            self.assertTrue(all(len(part) in (12, 13) for part in parts))
            # Distinct per-question contributions detect incorrect equal-rank weighting.
            ddp_gradient = sum(sum(i*i for i in part)/len(part)
                *ddp_mean_scale(len(part),row['global_questions']) for part in parts)/WORLD
            global_gradient = sum(i*i for i in row['question_indices'])/row['global_questions']
            self.assertAlmostEqual(ddp_gradient, global_gradient, places=7)
        self.assertEqual(len(online_plan(1729, 1)), 10)
        with self.assertRaises(ValueError):
            complete_pool_plan(1729, 19)
        with self.assertRaises(ValueError):
            rank_question_slice(plan[0], 4)

    def test_source_kind_preserves_ema_or_raw_and_never_imports_optimizer(self):
        source, shapes = tiny_complete_source()
        with patch.dict(model_code.PARAMETER_SHAPES, shapes, clear=True):
            ema, buffers, proof = model_code.warmup_model_weights(source, epoch=3)
            raw, _, raw_proof = model_code.warmup_model_weights(source, epoch=3, model_kind='raw')
        self.assertEqual(float(ema['output.weight'][0]), 2.)
        self.assertEqual(float(raw['output.weight'][0]), 1.)
        self.assertEqual(set(buffers), {'rotary.inv_freq'})
        self.assertFalse(proof['warmup_optimizer_imported'])
        self.assertFalse(raw_proof['warmup_rng_imported'])
        self.assertEqual(proof['source_model_kind'], 'ema')

    def test_source_rejects_wrong_epoch_partial_state_missing_buffer_and_extra_tensor(self):
        source, shapes = tiny_complete_source()
        with patch.dict(model_code.PARAMETER_SHAPES, shapes, clear=True):
            with self.assertRaises(ValueError):
                model_code.warmup_model_weights(source, epoch=4)
            for change in ('partial', 'buffer', 'extra', 'ema_clock', 'layout', 'metrics'):
                broken = copy.deepcopy(source)
                if change == 'partial':
                    broken['completed_epoch'] = False
                elif change == 'buffer':
                    broken['ema']['nonpersistent_buffers'] = {}
                elif change == 'extra':
                    broken['ema']['model']['invented.weight'] = torch.ones(1)
                elif change == 'ema_clock':
                    broken['ema']['updates'] -= 1
                elif change == 'layout':
                    broken['binding']['prefix_tokens'] = 20
                elif change == 'metrics':
                    broken['epoch_metrics']['epoch'] = 4
                with self.subTest(change=change), self.assertRaises(ValueError):
                    model_code.warmup_model_weights(broken, epoch=3)

    def test_missing_actual_cp3_fails_before_any_initialization_claim(self):
        from .probe import initialize
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(state_dir=str(root/'online'), source_checkpoint=str(root/'missing-cp3.pt'),
                source_epoch=3, source_model_kind='ema', source_sha256=model_code.EPOCH3_SHA,
                source_run=model_code.WARMUP_RUN, run_id='new-run')
            with patch('reproduction.prefix_selection200.probe.remote', side_effect=Path):
                with self.assertRaises(FileNotFoundError):
                    initialize(args)
            self.assertFalse((root/'online').exists())

    def test_authentic_zero_eos_and_short_prefix_layout(self):
        features = torch.zeros(1, WIDTH, DIM)
        question = torch.zeros(1, WIDTH, dtype=torch.bool)
        question[:, :2] = True
        policy = torch.zeros_like(question)
        positions = torch.zeros(1, WIDTH, dtype=torch.long)
        positions[0, 1] = 1
        batch = dict(features=features, attention_mask=question|policy, question_mask=question,
            policy_mask=policy, position_ids=positions, question_lengths=torch.tensor([2]),
            policy_lengths=torch.tensor([0]), policy_segments=[[]])
        model_code.validate_online_layout(batch, 32768)
        policy[:, QMAX] = True
        positions[:, QMAX] = 2
        batch.update(attention_mask=question|policy, policy_lengths=torch.tensor([1]),
                     policy_segments=[[dict(policy_start=0, policy_length=1, position_start=2, turn_index=0)]])
        model_code.validate_online_layout(batch, 32768)
        batch['policy_lengths'] = torch.tensor([6])
        with self.assertRaises(ValueError):
            model_code.validate_online_layout(batch, 32768)


if __name__ == '__main__':
    unittest.main()
