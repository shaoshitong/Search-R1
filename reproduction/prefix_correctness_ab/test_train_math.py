"""Focused CPU-only sampling and probability math checks."""
import random
import unittest
from .train_ddp import batch_plan, binary_metrics, EPOCHS, STEPS_PER_EPOCH


class TrainingMathTests(unittest.TestCase):
    def test_balanced_global_and_rank_partition(self):
        labels = [1] * 3000 + [0] * 22600
        a = batch_plan(labels, 'balanced', random.Random(1729))
        b = batch_plan(labels, 'balanced', random.Random(1729))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 100)
        for batch in a:
            self.assertEqual(sum(labels[i] for i in batch), 128)
            chunks = [batch[r * 64:(r + 1) * 64] for r in range(4)]
            self.assertEqual(sum(chunks, []), batch)
            self.assertTrue(all(len(chunk) == 64 for chunk in chunks))
        self.assertEqual(EPOCHS * STEPS_PER_EPOCH, 2000)

    def test_natural_epoch_exactly_once(self):
        labels = [1] * 3000 + [0] * 22600
        batches = batch_plan(labels, 'natural', random.Random(1729))
        self.assertEqual(sorted(sum(batches, [])), list(range(25600)))
        self.assertEqual(len(batches), 100)

    def test_prior_correction_direction_and_finite_extremes(self):
        import math
        raw = binary_metrics([0, 0], [0, 1])
        corrected = binary_metrics([0, 0], [0, 1], math.log(.2 / .8))
        self.assertAlmostEqual(raw['probability_mean'], .5)
        self.assertAlmostEqual(corrected['probability_mean'], .2)
        self.assertEqual(corrected['tn'], 1)
        extreme = binary_metrics([-1000, 1000], [0, 1])
        self.assertEqual(extreme['accuracy'], 1)
        self.assertTrue(math.isfinite(extreme['bce']))


if __name__ == '__main__':
    unittest.main()
