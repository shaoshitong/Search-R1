"""Small synthetic receipt tests; no model, features or remote access."""
import json
from pathlib import Path
import tempfile
import unittest

from .report import COMPLETE_STATUS, FORMULA, TRUE_MIXED, checked_accuracy, collect


def group(count, tp, tn, positive):
    return dict(count=count, tp=tp, tn=tn, fn=positive - tp,
                fp=count - positive - tn, mixed_count=positive,
                nonmixed_count=count - positive, accuracy=(tp + tn) / count,
                threshold=.5, formula=FORMULA, true_mixed=TRUE_MIXED)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    current, legacy = root / 'current', root / 'legacy'
    cfg = dict(run_id='same-run', source_epoch=71, source_adam_steps_per_rank=[7100] * 8,
               source_checkpoint_sha256='a' * 64, iterations=70)
    for base in (current, legacy):
        write_json(base / 'preparation/launch-config.json', cfg)
    for iteration in range(1, 71):
        old = iteration <= 6
        test = group(512, 150, 200 if iteration in (37, 70) else 150, 183)
        coordinate = dict(replay_iteration=iteration, source_data_epoch=iteration, source_epoch=71,
                          new_adam_steps_per_rank=100,
                          actual_adam_steps_per_rank=[7100 + iteration * 100] * 8,
                          mixed_group_accuracy=test['accuracy'])
        metrics = dict(coordinate, mixed_group=test, test_manifest_sha256='fixed-test')
        complete = dict(coordinate, status=COMPLETE_STATUS)
        if not old:
            train = group(5120, 1400 + iteration, 2000, 1800)
            metrics.update(train_mixed_group=train, train_mixed_group_accuracy=train['accuracy'],
                           test_mixed_group=test, test_mixed_group_accuracy=test['accuracy'],
                           mixed_metric_semantics='complete train and fixed test eval/no_grad passes of the same post-update head')
            complete.update(train_mixed_group_accuracy=train['accuracy'],
                            test_mixed_group_accuracy=test['accuracy'])
        base = legacy if old else current
        path = base / 'train/iterations' / ('iteration-%04d' % iteration)
        write_json(path / 'metrics.json', metrics)
        write_json(path / 'complete.json', complete)
    return current, legacy


class AggregateReportTests(unittest.TestCase):
    def test_group_accuracy_is_question_confusion_not_precision(self):
        actual = group(512, 101, 204, 183)
        self.assertEqual(checked_accuracy(actual, 512), 305 / 512)
        invalid = dict(actual, accuracy=101 / (101 + actual['fp']))
        with self.assertRaisesRegex(ValueError, 'confusion matrix'):
            checked_accuracy(invalid, 512)

    def test_complete70_keeps_missing_legacy_train_and_all_best_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            current, legacy = fixture(Path(directory))
            result = collect(current, legacy)
        self.assertEqual(result['status'], 'QUESTION20_REPLAY70_AGGREGATE_COMPLETE')
        self.assertEqual(len(result['rows']), 70)
        self.assertEqual(result['complete_iterations'], 70)
        self.assertTrue(all(r['train_mixed_group_accuracy'] is None for r in result['rows'][:6]))
        self.assertTrue(all(r['train_mixed_group_accuracy'] is not None for r in result['rows'][6:]))
        self.assertEqual(result['best_test']['replay_iterations'], [37, 70])
        self.assertEqual(result['latest_complete']['replay_iteration'], 70)
        self.assertFalse(result['full_prediction_files_read'])

    def test_missing_final_completion_is_pending_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            current, legacy = fixture(Path(directory))
            (current / 'train/iterations/iteration-0070/complete.json').unlink()
            result = collect(current, legacy)
        self.assertEqual(result['status'], 'PENDING_INCOMPLETE_REPLAY')
        self.assertEqual(result['complete_iterations'], 69)
        self.assertEqual(result['rows'][-1]['status'], 'PENDING')
        self.assertIsNone(result['rows'][-1]['test_mixed_group_accuracy'])


if __name__ == '__main__':
    unittest.main()
