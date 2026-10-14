"""CPU tests for moment coherence, fair exploration, and pre-fit measurement."""
import copy
import math
import unittest

import numpy as np

from search_g1.qprobe_selection import (
    COUNT_STDS, PredictQuestion, QuestionPool, count_predictions, prequential_metrics,
    select_questions, validate_budget,
)


def questions(n):
    return [PredictQuestion(i, str(i), "question " + str(i)) for i in range(n)]


class QuestionProbeSelectionTests(unittest.TestCase):
    def test_zero_exploration_selects_top512_and_ignores_exploration_rng(self):
        candidates = questions(600)
        probabilities = np.zeros((600, 6))
        probabilities[:, 0] = 1
        probabilities[87:] = np.array([0, 0, .5, .5, 0, 0])
        first = select_questions(candidates, probabilities, exploration_questions=0,
                                 rng=np.random.default_rng(1), tie_seed="top512")
        other = select_questions(candidates, probabilities, exploration_questions=0,
                                 rng=np.random.default_rng(999), tie_seed="top512")
        self.assertEqual(first, other)
        self.assertEqual(len(first), 512)
        self.assertTrue(all(row.ordinal >= 87 and row.selection_source == "predicted_diversity"
                            for row in first))
        self.assertEqual([row.ordinal for row in first], sorted(row.ordinal for row in first))

    def test_mean_half_does_not_imply_within_group_diversity(self):
        prediction = count_predictions([[.5, 0, 0, 0, 0, .5], [0, 0, .5, .5, 0, 0]])
        np.testing.assert_allclose(prediction["mean"], [.5, .5])
        self.assertEqual(prediction["expected_std"][0], 0)
        self.assertAlmostEqual(prediction["expected_std"][1], math.sqrt(.3))
        self.assertAlmostEqual(COUNT_STDS[1], math.sqrt(.2))
        result = select_questions(questions(2), prediction["probabilities"], selected_questions=1,
                                  exploration_questions=0, rng=np.random.default_rng(42))
        self.assertEqual([x.ordinal for x in result], [1])

    def test_every_binary_vector_matches_ddof_one_and_distribution_moments(self):
        import itertools
        for vector in itertools.product((0, 1), repeat=5):
            count = sum(vector)
            p = np.eye(6)[count:count+1]
            output = count_predictions(p)
            self.assertAlmostEqual(output["mean"][0], np.mean(vector))
            self.assertAlmostEqual(output["expected_std"][0], np.std(vector, ddof=1))
            self.assertAlmostEqual(output["expected_variance"][0], np.var(vector, ddof=1))

    def test_5120_acquisition_all_ablation_budgets_and_original_order(self):
        candidates = questions(5120)
        p = np.zeros((5120, 6)); p[:, 0] = 1
        p[-1024:] = np.array([0, 0, .5, .5, 0, 0])
        state = np.random.default_rng(20261013).bit_generator.state
        exploration_sets = []
        for explore in (0, 64, 128, 256):
            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(state)
            selected = select_questions(candidates, p, exploration_questions=explore, rng=rng, tie_seed=5)
            random_rows = {x.ordinal for x in selected if x.selection_source == "random_exploration"}
            focus = [x for x in selected if x.selection_source == "predicted_diversity"]
            self.assertEqual(len(selected), 512)
            self.assertEqual(len(random_rows), explore)
            self.assertEqual(len(focus), 512-explore)
            self.assertEqual(len({x.qid for x in selected}), 512)
            self.assertEqual([x.ordinal for x in selected], sorted(x.ordinal for x in selected))
            self.assertTrue(all(x.ordinal >= 4096 for x in focus))
            exploration_sets.append(random_rows)
        self.assertTrue(exploration_sets[0] <= exploration_sets[1] <= exploration_sets[2] <= exploration_sets[3])

    def test_ties_do_not_favor_source_ordinal_and_replay_exactly(self):
        candidates = questions(64)
        p = np.ones((64, 6)) / 6
        state = np.random.default_rng(55).bit_generator.state
        first = select_questions(candidates, p, selected_questions=16, exploration_questions=4,
                                 rng=np.random.default_rng(55), tie_seed="same")
        rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(state)
        repeated = select_questions(candidates[::-1], p[::-1], selected_questions=16,
                                    exploration_questions=4, rng=rng, tie_seed="same")
        # Exploration indexes follow presented order; focus hashes do not. Exact
        # replay therefore binds the immutable candidate order, just as runtime does.
        replay_rng = np.random.default_rng(); replay_rng.bit_generator.state = copy.deepcopy(state)
        replay = select_questions(candidates, p, selected_questions=16, exploration_questions=4,
                                 rng=replay_rng, tie_seed="same")
        self.assertEqual(first, replay)
        self.assertNotEqual([x.ordinal for x in first], list(range(16)))
        pure_a = select_questions(candidates, p, selected_questions=16, exploration_questions=0,
                                 rng=np.random.default_rng(55), tie_seed="same")
        pure_b = select_questions(candidates[::-1], p[::-1], selected_questions=16, exploration_questions=0,
                                 rng=np.random.default_rng(55), tie_seed="same")
        self.assertEqual(pure_a, pure_b)

    def test_duplicates_rejected_before_generation(self):
        pool = QuestionPool(3)
        self.assertEqual(pool.reserve("a", "same   question"), 0)
        self.assertIsNone(pool.reserve("b", "same question"))
        self.assertIsNone(pool.reserve("a", "other question"))
        self.assertEqual(pool.reserve("c", "unique question"), 1)
        self.assertEqual(len(pool.questions), 2)
        with self.assertRaises(ValueError):
            select_questions([PredictQuestion(0, "a", "same"), PredictQuestion(1, "b", "same")],
                             np.ones((2, 6))/6, selected_questions=1, exploration_questions=0,
                             rng=np.random.default_rng(42))

    def test_invalid_simplex_and_budget_rejected(self):
        for args in ((5120, 512, -1), (5120, 512, 513), (128, 512, 0), (5120, True, 0), (5120, 512, True)):
            with self.assertRaises(ValueError): validate_budget(*args)
        for p in ([[0]*6], [[float("nan")]*6], [[-.1, 0, 0, 0, 0, 1.1]], [[1, 0, 0]]):
            with self.assertRaises(ValueError): count_predictions(p)

    def test_random_audit_missing_at_zero_exploration_not_fabricated_zero(self):
        metrics = prequential_metrics([[0, 0, 1, 0, 0, 0]], [[1, 1, 0, 0, 0]], ["predicted_diversity"])
        self.assertEqual(metrics["random_exploration"], {"questions": 0})
        self.assertEqual(metrics["predicted_diversity"]["std_mae"], 0)
        self.assertEqual(metrics["all"]["mean_mae"], 0)
        with self.assertRaises(ValueError):
            prequential_metrics([[0, 0, 1, 0, 0, 0]], [[1, 1, 0, 0]], ["predicted_diversity"])


if __name__ == "__main__": unittest.main()
