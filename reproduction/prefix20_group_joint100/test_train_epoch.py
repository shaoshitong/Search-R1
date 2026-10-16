"""CPU checks for whole-G5 natural batches, unequal-rank means and fresh clocks."""
import unittest

import torch

from .loss import joint_loss
from .train_epoch import (DATA_SCHEDULE, cosine_learning_rate, ddp_mean_scale, digest,
                          evaluate_predictions, natural_question_plan, question_groups,
                          rank_question_slice, schedule_state, validate_saved_checkpoint)


class JointTrainingTest(unittest.TestCase):
    def test100_updates_cover_every_question_once_and_balance_all8_ranks(self):
        plan = natural_question_plan(5120, 1729, 1)
        self.assertEqual(len(plan), 100)
        self.assertEqual([batch['global_questions'] for batch in plan].count(51), 80)
        self.assertEqual([batch['global_questions'] for batch in plan].count(52), 20)
        seen = [[] for _ in range(8)]
        for batch in plan:
            parts = [rank_question_slice(batch, rank) for rank in range(8)]
            self.assertTrue(all(len(part) in (6, 7) for part in parts))
            self.assertEqual([question for part in parts for question in part], batch['question_indices'])
            for rank, part in enumerate(parts):
                seen[rank].extend(part)
        self.assertEqual([len(part) for part in seen], [640] * 8)
        self.assertEqual(sorted(question for part in seen for question in part), list(range(5120)))
        self.assertNotEqual(plan, natural_question_plan(5120, 1729, 2))

    def test_unequal_rank_joint_mean_scaling_matches_the_true_global_gradient(self):
        for size in (51, 52):
            x = torch.arange(size * 5, dtype=torch.float64).reshape(size, 5) / 173 - .7
            labels = (torch.arange(size * 5).reshape(size, 5) % 3 == 0).double()
            counts = [6] * 8
            for rank in range(size - 48):
                counts[rank] += 1
            whole = torch.tensor(.23, dtype=torch.float64, requires_grad=True)
            joint_loss(whole * x, labels)['loss'].backward()
            averaged = 0.0
            start = 0
            for count in counts:
                parameter = torch.tensor(.23, dtype=torch.float64, requires_grad=True)
                local = joint_loss(parameter * x[start:start + count], labels[start:start + count])['loss']
                (local * ddp_mean_scale(count, size)).backward()
                averaged += float(parameter.grad) / 8
                start += count
            self.assertAlmostEqual(averaged, float(whole.grad), places=13)

    def test_cosine_uses_new_actual_updates_and_rejects_old_probe_counters(self):
        rates = [cosine_learning_rate(update) for update in range(10000)]
        self.assertEqual(rates[0], 3e-5)
        self.assertEqual(rates[-1], 3e-6)
        self.assertTrue(all(left >= right for left, right in zip(rates, rates[1:])))
        self.assertEqual(DATA_SCHEDULE, list(range(1, 71)) + list(range(1, 31)))
        binding = dict(role='fresh_joint', run_id='new-run')
        saved = dict(epoch=1, binding=binding, binding_sha256=digest(binding),
                     actual_adam_steps_per_rank=[100] * 8, fresh_run_initial_adam_steps=0,
                     cosine_schedule=schedule_state(100, 10000, 3e-5, 3e-6))
        receipt = {key: saved[key] for key in ('epoch', 'binding_sha256', 'actual_adam_steps_per_rank')}
        self.assertEqual(validate_saved_checkpoint(saved, receipt, binding, 2), 100)
        self.assertEqual(saved['cosine_schedule']['next_update_learning_rate'], rates[100])
        with self.assertRaises(ValueError):
            validate_saved_checkpoint(dict(saved, actual_adam_steps_per_rank=[7200] * 8), receipt, binding, 2)
        with self.assertRaises(ValueError):
            validate_saved_checkpoint(saved, receipt, dict(binding, run_id='old-probe'), 2)

    def test_real_qid_groups_and_post_update_mixed_metrics_keep_both_class_recalls(self):
        index, predictions = {}, []
        for question in range(512):
            for sibling in range(5):
                tid = 'q%d-t%d' % (question, sibling)
                label = int(question < 256 and sibling == 0)
                predicted = question < 128 or 256 <= question < 384
                logit = 0.0 if predicted and sibling == 0 else -1000.0
                index[tid] = dict(qid=str(question), question_hash=str(question), strict_em=label)
                predictions.append((tid, logit, label))
        gathered = [row for rank in range(8) for row in predictions[rank::8]]
        metrics, rows = evaluate_predictions(gathered, index, 512)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['balanced_accuracy'], .5)
        self.assertEqual((metrics['mixed_recall'], metrics['nonmixed_recall']), (.5, .5))
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']), (128, 128, 128, 128))
        self.assertTrue(torch.isfinite(torch.tensor(metrics['total_loss'])))
        groups = question_groups(index, [row[0] for row in gathered], 512)
        self.assertEqual({tuple(group['trajectory_ids']) for group in groups},
                         {tuple(row['trajectory_ids']) for row in rows})
        self.assertTrue(all(len(group['labels']) == 5 for group in groups))


if __name__ == '__main__':
    unittest.main()
