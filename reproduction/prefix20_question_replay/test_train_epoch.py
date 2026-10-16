"""CPU checks for replay coordinates, source Adam continuity, and fixed G5 grouping."""
import unittest

from .train_epoch import (SOURCE_LEARNING_RATE, TARGET_LEARNING_RATE,
                          binding_digest, expected_adam_steps, mixed_metrics,
                          natural_plan, rank_slice, validate_replay_checkpoint, validate_source)


class ReplayTrainingTest(unittest.TestCase):
    def test_source_epoch_and_replay_iteration_have_separate_adam_coordinates(self):
        self.assertEqual(SOURCE_LEARNING_RATE, 3e-5)
        self.assertEqual(TARGET_LEARNING_RATE, 3e-6)
        source_binding = dict(prefix_tokens=20, question_hidden_passed_to_head=False,
                              learning_rate=SOURCE_LEARNING_RATE)
        saved = dict(epoch=72, binding=source_binding, binding_sha256=binding_digest(source_binding),
                     actual_adam_steps_per_rank=[7200] * 8,
                     rank_rng=[dict(rank=rank) for rank in range(8)])
        steps = validate_source(saved, 72, 'a' * 64)
        self.assertEqual(expected_adam_steps(steps, 1), [7300] * 8)
        self.assertEqual(expected_adam_steps(steps, 70), [14200] * 8)
        with self.assertRaises(ValueError):
            validate_source(saved, 70, 'a' * 64)

    def test_replay_checkpoint_cannot_reset_counter_or_skip_iteration(self):
        binding = dict(source_epoch=72, source_checkpoint_sha256='a' * 64,
                       source_adam_steps_per_rank=[7200] * 8, learning_rate=TARGET_LEARNING_RATE)
        saved = dict(replay_iteration=9, binding=binding, binding_sha256=binding_digest(binding),
                     actual_adam_steps_per_rank=[8100] * 8)
        receipt = {key: saved[key] for key in ('replay_iteration', 'binding_sha256', 'actual_adam_steps_per_rank')}
        self.assertEqual(validate_replay_checkpoint(saved, receipt, binding, 10, [7200] * 8), [8100] * 8)
        with self.assertRaises(ValueError):
            validate_replay_checkpoint(dict(saved, actual_adam_steps_per_rank=[900] * 8), receipt,
                                       binding, 10, [7200] * 8)
        with self.assertRaises(ValueError):
            validate_replay_checkpoint(saved, receipt, binding, 11, [7200] * 8)
        with self.assertRaises(ValueError):
            validate_replay_checkpoint(dict(saved, binding=dict(binding, learning_rate=SOURCE_LEARNING_RATE)),
                                       receipt, binding, 10, [7200] * 8)

    def test_natural_25600_pass_is100_batches32_per_rank_without_repetition(self):
        plan = natural_plan(25600, 256, 8, 1729, 1)
        self.assertEqual(len(plan), 100)
        self.assertEqual(sorted(index for batch in plan for index in batch), list(range(25600)))
        for batch in plan:
            slices = [rank_slice(batch, rank, 8) for rank in range(8)]
            self.assertEqual([len(part) for part in slices], [32] * 8)
            self.assertEqual([index for part in slices for index in part], batch)
        self.assertNotEqual(plan, natural_plan(25600, 256, 8, 1729, 2))

    def test_mixing_uses_real_qid_groups_after_rank_strided_gather(self):
        fixed_index, predictions = {}, []
        for question in range(512):
            for sibling in range(5):
                tid = 'q%d-t%d' % (question, sibling)
                label = int(question % 2 == 0 and sibling == 0)
                fixed_index[tid] = dict(qid=str(question), question_hash=str(question), strict_em=label)
                predictions.append((tid, 10.0 if label else -10.0, label))
        gathered = [row for rank in range(8) for row in predictions[rank::8]]
        metrics, rows = mixed_metrics(gathered, fixed_index)
        self.assertEqual(metrics['count'], 512)
        self.assertEqual(metrics['mixed_count'], 256)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual((metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']), (256, 256, 0, 0))
        self.assertTrue(all(len(row['trajectory_ids']) == 5 for row in rows))
        broken = list(gathered)
        tid, logit, label = broken[0]
        broken[0] = (tid, logit, 1 - label)
        with self.assertRaises(ValueError):
            mixed_metrics(broken, fixed_index)

    def test_complete_train_g5_grouping_uses5120_questions_and_inclusive_threshold(self):
        train_index, predictions = {}, []
        for question in range(5120):
            for sibling in range(5):
                tid = 'train-q%d-t%d' % (question, sibling)
                label = int(question % 2 == 0 and sibling == 0)
                train_index[tid] = dict(qid=str(question), question_hash=str(question), strict_em=label)
                # p=.5 for the sole correct sibling and exact float64 p=0
                # for the others produces q=.5, exercising q>=.5 unchanged.
                predictions.append((tid, 0.0 if label else -1000.0, label))
        gathered = [row for rank in range(8) for row in predictions[rank::8]]
        metrics, groups = mixed_metrics(gathered, train_index, expected_questions=5120)
        self.assertEqual(metrics['count'], 5120)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual((metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']), (2560, 2560, 0, 0))
        self.assertTrue(all(row['mixed_probability'] == .5 and row['predicted_mixed'] == 1
                            for row in groups if row['mixed_label']))
        with self.assertRaises(ValueError):
            mixed_metrics(gathered, train_index, expected_questions=512)
        with self.assertRaises(ValueError):
            mixed_metrics(gathered[:-1], train_index, expected_questions=5120)


if __name__ == '__main__':
    unittest.main()
