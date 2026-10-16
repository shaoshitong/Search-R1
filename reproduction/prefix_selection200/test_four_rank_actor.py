"""CPU-only equivalence checks for physical4/reference virtual8 actor math."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from .actor import (ActorBudget, OBJECTIVE, WORLD, _state_binding, learning_rate,
                    micro_loss_weight, oom_retry_plan, rank_plan, remote,
                    restore_metadata, virtual_minibatches)
from .resume import boundary
from . import storage


class FourRankActorTests(unittest.TestCase):
    def test_exact_natural_twenty_update_virtual8_assignment(self):
        plan = rank_plan(1729, 3)
        self.assertEqual(len(plan), 4)
        self.assertEqual({len(steps) for steps in plan}, {20})
        self.assertEqual({len(ids) for steps in plan for ids in steps}, {32})
        self.assertEqual(sorted(i for steps in plan for ids in steps for i in ids), list(range(2560)))
        order = list(range(2560)); random.Random(1732).shuffle(order)
        for step in range(20):
            virtual = [block for rank in range(4) for block in virtual_minibatches(plan[rank][step])]
            self.assertEqual(virtual, [order[step * 128 + v * 16:step * 128 + (v + 1) * 16]
                                      for v in range(8)])
        self.assertEqual(learning_rate(0), 0.)
        self.assertEqual(learning_rate(57), 1e-6)

    def test_micro_never_combines_virtual16_token_denominators(self):
        for requested, local, accumulation, effective in ((4, 1, 32, 4), (16, 4, 8, 16),
                                                         (64, 16, 2, 64), (128, 16, 2, 64)):
            budget = ActorBudget(requested)
            self.assertEqual((budget.local_micro, budget.accumulation, budget.effective_global_micro),
                             (local, accumulation, effective))
            self.assertEqual(budget.global_micro, requested)
            for block in virtual_minibatches(list(range(32))):
                chunks = [block[start:start + budget.local_micro]
                          for start in range(0, len(block), budget.local_micro)]
                self.assertEqual(sum(map(len, chunks)), 16)
                self.assertTrue(all(len(chunk) == local for chunk in chunks))
        for invalid in (3, 24, 256):
            with self.assertRaises(ValueError): ActorBudget(invalid)

    def test_unequal_token_lengths_preserve_reference_objective_and_gradient(self):
        import torch
        # Deliberately unequal token masses; a local32 token mean would differ.
        lengths = torch.tensor([1 + (i * 7) % 9 for i in range(128)])
        mask = (torch.arange(9)[None, :] < lengths[:, None]).to(torch.float64)
        mask[16:32] = 0.  # An empty virtual rank is zero, not omitted from1/8.
        features = torch.arange(128 * 9, dtype=torch.float64).reshape(128, 9) / 113.
        parameter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)

        def loss(ids):
            values = (features[ids] * parameter + .3).square()
            mass = mask[ids].sum()
            return (values * mask[ids]).sum() / mass if mass else values.sum() * 0.

        reference = sum(loss(list(range(v * 16, (v + 1) * 16))) for v in range(8)) / 8
        reference.backward(); expected_grad = parameter.grad.clone()
        for requested in (4, 16, 64, 128):
            parameter.grad = None; budget = ActorBudget(requested); physical = []
            for rank in range(WORLD):
                reduced = parameter * 0.
                for block in virtual_minibatches(list(range(rank * 32, (rank + 1) * 32))):
                    mass = int(mask[block].sum())
                    for start in range(0, 16, budget.local_micro):
                        ids = block[start:start + budget.local_micro]
                        weight = micro_loss_weight(int(mask[ids].sum()), mass) / 2
                        reduced = reduced + loss(ids) * weight
                physical.append(reduced)
            actual = sum(physical) / WORLD
            actual.backward()
            torch.testing.assert_close(actual.detach(), reference.detach(), rtol=1e-14, atol=1e-14)
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-14, atol=1e-14)
        naive = sum(loss(list(range(rank * 32, (rank + 1) * 32))) for rank in range(4)) / 4
        self.assertGreater(abs(float(naive.detach() - reference.detach())), 1e-4)

    def test_four_rank_restore_and_partial_oom_loss_are_explicit(self):
        config = dict(nonmixed_advantage_mode='zero_anchor_all_correct_only',actor_hidden_native=True,run_id='new4', official_model_dir='/data/models/official', grounding_probe_version='fixed0',
                      resume_joint=dict(completed_outer=3))
        actor = dict(run_id='new4', completed_outer=3, binding=_state_binding(config),
                     actual_adam_steps_per_rank=[60] * 4, scheduler_completed_outer=3)
        probe = dict(run_id='new4', completed_outer=0, actual_adam_steps_per_rank=[0] * 4)
        pool = dict(next_outer=4, consumed_candidates=7680, cycle=0, cursor=7680, order=list(range(169028)))
        joint = dict(run_id='new4', outer=3, actor=actor, probe=probe)
        self.assertEqual(boundary(config, joint, actor, probe, pool), 3)
        self.assertEqual(actor['binding']['loss_reduction'], OBJECTIVE)
        changed = copy.deepcopy(actor); changed['actual_adam_steps_per_rank'] *= 2
        with self.assertRaises(ValueError): restore_metadata(config, changed, 4)
        retry = oom_retry_plan(outer=4, failed_micro=16, latest_complete_outer=3,
            rollout_actor_version='v3', latest_actor_version='v3', partial_writes=[7] * 4)
        self.assertEqual(retry['actor_global_micro_batch_size'], 8)
        self.assertEqual(retry['lost_unsaved_updates_on_reload'], [7] * 4)
        self.assertFalse(retry['automatic_rollback_performed'])
        self.assertFalse(retry['stitch_partial_writes'])
        self.assertTrue(retry['partial_updates_occurred'])

    def test_a800_remote_namespace_retains_owned_storage_guard(self):
        self.assertEqual(remote('/data/shaoshitong/checkpoint'), Path('/data/shaoshitong/checkpoint'))
        with self.assertRaises(ValueError): remote('/Users/shaoshitong/checkpoint')
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve(); base = parent / 'search-g1-prefix-selection200-a800-prefix5-fixture'
            (base / 'preparation').mkdir(parents=True)
            (base / 'preparation/launch-config.json').write_text(json.dumps(dict(base=str(base), run_id='owned',
                outers=200, world_size=4, probe_updates_per_outer=0, actor_hidden_native=True,
                probe_update_period=10, probe_updates_per_period=50,probe_update_at_final=False)))
            with patch.object(storage, 'A800_REMOTE_PARENT', parent):
                self.assertEqual(storage.validate_base(base)[0], base)
                with self.assertRaises(ValueError): storage.validate_base(parent)


if __name__ == '__main__': unittest.main()
