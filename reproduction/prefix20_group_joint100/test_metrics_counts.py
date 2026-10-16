"""CPU regression for the explicit complete-G5 metric contracts.

The period10 half-pool uses2560 questions. Its metrics must use the unchanged
joint losses and identity checks already used by512 diagnostic/5120 replay.
"""
import math
import unittest

from .train_epoch import evaluate_predictions


def complete_predictions(question_count):
    index, predictions = {}, []
    for question in range(question_count):
        correct_count = (1, 0, 5, 2)[question % 4]
        for sibling in range(5):
            tid = 'q%d-t%d' % (question, sibling)
            label = int(sibling < correct_count)
            index[tid] = dict(qid=str(question), question_hash='hash-%d' % question,
                              strict_em=label)
            predictions.append((tid, 3.0 if label else -3.0, label))
    # Exercise ID-based reconstruction rather than relying on contiguous G5.
    return index, [row for rank in range(4) for row in predictions[rank::4]]


class CompleteG5MetricsCountTest(unittest.TestCase):
    def test2560_candidate_and_existing512_5120_counts_keep_the_same_metrics(self):
        p = 1 / (1 + math.exp(-3))
        group_bces = []
        for k in (1, 0, 5, 2):
            q = 1 - p**k * (1 - p)**(5 - k) - (1 - p)**k * p**(5 - k)
            group_bces.append(-math.log(q if 0 < k < 5 else 1 - q))
        expected_trajectory_bce = math.log1p(math.exp(-3))
        expected_group_bce = math.fsum(group_bces) / 4
        for count in (512, 2560, 5120):
            with self.subTest(expected_questions=count):
                index, predictions = complete_predictions(count)
                metrics, groups = evaluate_predictions(predictions, index, count)
                self.assertEqual(metrics['count'], count)
                self.assertEqual(len(groups), count)
                self.assertEqual((metrics['mixed_count'], metrics['nonmixed_count']),
                                 (count // 2, count // 2))
                self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']),
                                 (count // 2, 0, 0, count // 2))
                self.assertEqual(metrics['accuracy'], 1.0)
                self.assertEqual(metrics['balanced_accuracy'], 1.0)
                self.assertEqual(metrics['threshold'], .5)
                self.assertEqual(metrics['group_weight'], 1.0)
                self.assertAlmostEqual(metrics['trajectory_bce'], expected_trajectory_bce, places=13)
                self.assertAlmostEqual(metrics['group_bce'], expected_group_bce, places=13)
                self.assertAlmostEqual(metrics['total_loss'],
                                       expected_trajectory_bce + expected_group_bce, places=13)
                self.assertTrue(all(len(row['trajectory_ids']) == 5 for row in groups))

    def test2560_still_rejects_missing_duplicate_or_foreign_trajectories(self):
        index, predictions = complete_predictions(2560)
        for broken in (predictions[:-1], predictions[:-1] + [predictions[0]],
                       [('foreign-id', predictions[0][1], predictions[0][2])] + predictions[1:]):
            with self.subTest(kind='prediction identity'), self.assertRaises(ValueError):
                evaluate_predictions(broken, index, 2560)
        missing_index = dict(index)
        missing_index.pop(next(iter(missing_index)))
        with self.assertRaises(ValueError):
            evaluate_predictions(predictions, missing_index, 2560)

    def test2560_keeps_own_label_finite_logit_and_complete_question_identity_guards(self):
        index, predictions = complete_predictions(2560)
        tid, logit, label = predictions[0]
        for broken_row in ((tid, logit, 1 - label), (tid, logit, bool(label)),
                           (tid, float('nan'), label), (tid, float('inf'), label)):
            with self.subTest(row=broken_row), self.assertRaises(ValueError):
                evaluate_predictions([broken_row] + predictions[1:], index, 2560)
        conflicting_hash = dict(index)
        conflicting_hash[tid] = dict(index[tid], question_hash='another-question')
        with self.assertRaises(ValueError):
            evaluate_predictions(predictions, conflicting_hash, 2560)
        merged_questions = dict(index)
        merged_questions[tid] = dict(index[tid], qid='1')
        with self.assertRaises(ValueError):
            evaluate_predictions(predictions, merged_questions, 2560)

    def test_no_other_or_mismatched_complete_count_is_admitted(self):
        index, predictions = complete_predictions(2560)
        for count in (0, 256, 511, 1024, 2559, 2561, 512, 5120):
            with self.subTest(expected_questions=count), self.assertRaises(ValueError):
                evaluate_predictions(predictions, index, count)


if __name__ == '__main__':
    unittest.main()
