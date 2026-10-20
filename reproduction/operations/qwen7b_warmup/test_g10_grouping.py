"""Identifier/EM-only synthetic tests for authentic G10 subgroup contracts."""
import copy
from fractions import Fraction
import unittest

from .g10_grouping import build_training_views, fixed_g5


def question(k, qid=None):
    qid = qid or 'k%d' % k
    return dict(qid=qid, trajectory_ids=[qid+'-t%d' % i for i in range(10)], labels=[1]*k+[0]*(10-k))


class G10GroupingTests(unittest.TestCase):
    def assert_real(self, original, result):
        source = {g['qid']:g for g in original}
        weights = {}
        for row in result['groups']:
            own = source[row['qid']]
            self.assertEqual(len(row['trajectory_ids']), 5)
            self.assertEqual(len(set(row['trajectory_ids'])), 5)
            self.assertEqual(row['labels'], [own['labels'][i] for i in row['source_indices']])
            self.assertEqual(row['trajectory_ids'], [own['trajectory_ids'][i] for i in row['source_indices']])
            self.assertEqual(row['true_mixed'], 0 < sum(row['labels']) < 5)
            weights[row['qid']] = weights.get(row['qid'], Fraction(0))+Fraction(row['weight_numerator'], row['weight_denominator'])
        self.assertEqual(set(weights), set(source))
        self.assertTrue(all(weight == 1 for weight in weights.values()))

    def test_full252_coverage_k0_1_5_9_10(self):
        groups = [question(k) for k in (0, 1, 5, 9, 10)]
        result = build_training_views(groups, views_per_question=252)
        self.assert_real(groups, result)
        for group in groups:
            views = [r for r in result['groups'] if r['qid'] == group['qid']]
            self.assertEqual(len({tuple(r['source_indices']) for r in views}), 252)
        self.assertEqual([row['mixed_views'] for row in sorted(result['per_question'], key=lambda r: int(r['qid'][1:]))], [0, 126, 250, 126, 0])

    def test_duplicate_ids_and_bad_em_rejected(self):
        row = question(5)
        row['trajectory_ids'][9] = row['trajectory_ids'][0]
        with self.assertRaises(ValueError):
            build_training_views([row])
        row = question(5)
        row['labels'][0] = True
        with self.assertRaises(ValueError):
            build_training_views([row])

    def test_per_question_balance_and_impossible_homogeneous(self):
        groups = [question(k) for k in (1, 5, 9)]
        result = build_training_views(groups, mode='per-question-balanced', views_per_question=4)
        self.assert_real(groups, result)
        self.assertTrue(result['summary']['count_balanced'])
        self.assertTrue(result['summary']['weighted_class_mass_balanced'])
        self.assertTrue(all(row['mixed_views'] == row['nonmixed_views'] == 2 for row in result['per_question']))
        with self.assertRaises(ValueError):
            build_training_views([question(0)], mode='per-question-balanced')
        report = build_training_views([question(0), question(5)], mode='per-question-balanced', views_per_question=6, on_unbalance='report')
        self.assert_real([question(0), question(5)], report)
        self.assertEqual(len(report['summary']['impossibilities']), 2)
        self.assertFalse(report['summary']['count_balanced'])
        self.assertEqual(next(r for r in report['per_question'] if r['qid']=='k5')['views'], 4)

    def test_global_balance_coverage_and_weighted_class_difference(self):
        groups = [question(0), question(1), question(5), question(9), question(10)]
        result = build_training_views(groups, mode='global-balanced', views_per_question=4, seed=13)
        self.assert_real(groups, result)
        self.assertEqual(result['summary']['mixed_views'], 10)
        self.assertEqual(result['summary']['nonmixed_views'], 10)
        self.assertTrue(result['summary']['count_balanced'])
        self.assertFalse(result['summary']['weighted_class_mass_balanced'])
        with self.assertRaises(ValueError):
            build_training_views([question(0), question(10)], mode='global-balanced')

    def test_seed_epoch_and_input_order_determinism(self):
        groups = [question(1), question(5), question(9)]
        for mode in ('random', 'per-question-balanced', 'global-balanced'):
            a = build_training_views(groups, mode=mode, views_per_question=4, seed=17, epoch=2)
            b = build_training_views(list(reversed(groups)), mode=mode, views_per_question=4, seed=17, epoch=2)
            self.assertEqual(a, b)
            c = build_training_views(groups, mode=mode, views_per_question=4, seed=17, epoch=3)
            self.assertNotEqual(a['summary']['selected_views_sha256'], c['summary']['selected_views_sha256'])

    def test_final_is_label_invariant_and_preserves_original_five(self):
        groups = [question(0), question(5)]
        changed = copy.deepcopy(groups)
        for group in changed:
            group['labels'] = [1-y for y in group['labels']]
        for selection in ('first5', 'random'):
            self.assertEqual(fixed_g5(groups, selection=selection), fixed_g5(changed, selection=selection))
        first = fixed_g5(groups)
        self.assertEqual(first['groups'][0]['trajectory_ids'], groups[0]['trajectory_ids'][:5])
        self.assertTrue(first['summary']['labels_consulted'] is False)
        for group in changed:
            del group['labels']
        self.assertEqual(first, fixed_g5(changed))
        designated = {g['qid']:g['trajectory_ids'][1:6] for g in groups}
        chosen = fixed_g5(groups, fixed_by_qid=designated)
        self.assertEqual(chosen['groups'][0]['trajectory_ids'], designated['k0'])
        with self.assertRaises(ValueError):
            build_training_views(groups, role='final')


if __name__ == '__main__':
    unittest.main()
