"""CPU checks for the sampler and objective metrics; no GPU/model gates."""
import unittest

from .train_epoch import natural_plan, rank_slice, auroc


class HeadMathTests(unittest.TestCase):
    def test_fullbatch_exact_trajectory_coverage_and_disjoint_eight_rank_shards(self):
        plan = natural_plan(25600, 25600, 8, 1729, 1)
        self.assertEqual(len(plan), 1)
        parts = [rank_slice(plan[0], rank, 8) for rank in range(8)]
        self.assertEqual([len(part) for part in parts], [3200] * 8)
        self.assertEqual(sorted(item for part in parts for item in part), list(range(25600)))
        self.assertEqual(len({item for part in parts for item in part}), 25600)

    def test_remainder_keeps_every_sample_equal_rank_means_no_distributed_padding(self):
        plan = natural_plan(25600, 8192, 8, 1729, 2)
        self.assertEqual([len(batch) for batch in plan], [8192, 8192, 8192, 1024])
        self.assertEqual(sorted(item for batch in plan for item in batch), list(range(25600)))
        for batch in plan:
            self.assertEqual(len({len(rank_slice(batch, rank, 8)) for rank in range(8)}), 1)

    def test_epoch_rng_is_deterministic_and_preserves_global_python_rng(self):
        import random
        random.seed(9)
        before = random.getstate()
        first = natural_plan(25600, 25600, 8, 1729, 1)
        self.assertEqual(first, natural_plan(25600, 25600, 8, 1729, 1))
        self.assertNotEqual(first, natural_plan(25600, 25600, 8, 1729, 2))
        self.assertEqual(before, random.getstate())

    def test_invalid_batch_does_not_silently_pad_or_drop(self):
        for size, batch in ((25600, 255), (25599, 256), (8, 4)):
            with self.assertRaises(ValueError):
                natural_plan(size, batch, 8, 1729, 1)

    def test_auroc_uses_logits_and_half_credit_for_ties(self):
        self.assertEqual(auroc([-1000, 1000], [0, 1]), 1)
        self.assertEqual(auroc([1000, -1000], [0, 1]), 0)
        self.assertEqual(auroc([1000, 1000], [0, 1]), .5)
        self.assertIsNone(auroc([0, 1], [1, 1]))


if __name__ == '__main__':
    unittest.main()
