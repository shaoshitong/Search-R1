"""CPU-only contracts for the human-approved half-pool, ten-outer probe recipe."""
import unittest

from .contract import (CANDIDATE_QUESTIONS, SELECTED_QUESTIONS, FINAL_OUTER,
                       PROBE_PERIOD, PROBE_UPDATES, PROBE_LR, G, WORLD)
from .periodic import is_probe_update, probe_boundary
from .probe_plan import complete_pool_plan, rank_question_slice, ddp_mean_scale
from .pool import initial_state, reserve_indices
from .rollout import question_count
from .probe import LR, summarize_loss_curve


class Period10HalfPoolContracts(unittest.TestCase):
    def test_schedule_has_nineteen_updates_and_never_updates_outer200(self):
        updates = [outer for outer in range(1, FINAL_OUTER+1) if is_probe_update(outer)]
        self.assertEqual(updates, list(range(10, 200, 10)))
        self.assertEqual((CANDIDATE_QUESTIONS, SELECTED_QUESTIONS, PROBE_PERIOD, PROBE_UPDATES),
                         (2560, 512, 10, 50))
        self.assertEqual(LR, PROBE_LR)
        self.assertEqual(LR, 5e-6)
        self.assertEqual(probe_boundary(9), dict(last_update_outer=0, actual_adam_per_rank=0))
        self.assertEqual(probe_boundary(10), dict(last_update_outer=10, actual_adam_per_rank=50))
        self.assertEqual(probe_boundary(199), dict(last_update_outer=190, actual_adam_per_rank=950))
        self.assertEqual(probe_boundary(200), probe_boundary(199))
        for outer in (0, 1, 9, 11, 199, 200, 201):
            with self.subTest(outer=outer), self.assertRaises(ValueError):
                complete_pool_plan(1729, outer)

    def test_one_complete_2560_g5_pass_keeps_global_weighting(self):
        plan = complete_pool_plan(1729, 10)
        self.assertEqual(len(plan), 50)
        self.assertEqual(sum(row['global_questions'] == 51 for row in plan), 40)
        self.assertEqual(sum(row['global_questions'] == 52 for row in plan), 10)
        self.assertEqual(sorted(i for row in plan for i in row['question_indices']), list(range(2560)))
        self.assertEqual([sum(len(rank_question_slice(row, rank)) for row in plan)
                          for rank in range(WORLD)], [640]*4)
        for row in plan:
            parts = [rank_question_slice(row, rank) for rank in range(WORLD)]
            self.assertTrue(all(len(part) in (12, 13) for part in parts))
            global_mean = sum(i*i for i in row['question_indices'])/row['global_questions']
            ddp_mean = sum(sum(i*i for i in part)/len(part)*ddp_mean_scale(len(part), row['global_questions'])
                           for part in parts)/WORLD
            self.assertAlmostEqual(ddp_mean, global_mean, places=7)
        self.assertEqual(sum(row['global_questions']*G for row in plan), 12800)

    def test_half_pool_reservation_and_rollout_counts_are_question_counts(self):
        state = initial_state(168510)
        first, _, state = reserve_indices(state, 1)
        second, _, state = reserve_indices(state, 2)
        self.assertEqual(len(first), 2560)
        self.assertEqual(len(second), 2560)
        self.assertFalse(set(first)&set(second))
        self.assertEqual(state['consumed_candidates'], 5120)
        self.assertEqual(question_count({}, 'prefix'), 2560)
        self.assertEqual(question_count({}, 'continue'), 512)
        self.assertEqual(question_count(dict(rollout_continuation_questions=2560), 'continue'), 2560)
        self.assertEqual(question_count(dict(rollout_candidate_questions=512), 'prefix'), 512)
        with self.assertRaises(ValueError):
            question_count(dict(rollout_candidate_questions=5120), 'prefix')

    def test_loss_curve_is_exactly_fifty_global_adam_observations(self):
        rows = [dict(step=step, trajectory_bce=1., group_bce=.5, total_loss=1.5)
                for step in range(1, 51)]
        self.assertEqual(summarize_loss_curve(rows)['actual_adam_updates_per_rank'], 50)
        with self.assertRaises(ValueError):
            summarize_loss_curve(rows[:-1])


if __name__ == '__main__':
    unittest.main()
