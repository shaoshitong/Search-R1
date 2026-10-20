import copy
import unittest
from . import paired_question_stability as s


def row(i, mixed, p):
    labels = [0, 0, 1, 1, 1] if mixed else [0] * 5
    ps = [p] * 5
    return dict(qid='q%02d' % i, trajectory_ids=['q%02d-g%d' % (i, j) for j in range(5)],
                strict_em=labels, p_j=ps, predicted_mixed=1-p**5-(1-p)**5)


class PairedStabilityTest(unittest.TestCase):
    def setUp(self):
        self.rows = [row(i, i % 2 == 0, .5 if i % 2 == 0 else .99) for i in range(32)]

    def test_label_blind_exact_equal_strata(self):
        ids = [r['qid'] for r in self.rows]
        a = s.freeze_strata(ids); b = s.freeze_strata(list(reversed(ids)))
        self.assertEqual(a, b); self.assertEqual([len(x) for x in a['strata']], [8] * 4)
        self.assertEqual(set(sum(a['strata'], [])), set(ids)); self.assertFalse(a['labels_consulted'])

    def test_identical_models_have_zero_paired_differences(self):
        a = s.analyse(self.rows, list(reversed(self.rows)), copy.deepcopy(self.rows), replicates=40, expected_questions=32)
        for scope in a['scopes'].values():
            for pair in scope['comparisons'].values():
                self.assertTrue(all(v == 0 for v in pair['candidate_minus_baseline'].values()))
                self.assertTrue(all(v == [0, 0] for v in pair['percentile_95_CI'].values()))
        self.assertEqual(a['scopes']['all_questions']['models']['candidate']['top_quarter_mixed_count'], 8)

    def test_deterministic_bootstrap_and_label_identity_fail_closed(self):
        a = s.analyse(self.rows, self.rows, self.rows, replicates=20, expected_questions=32)
        self.assertEqual(a, s.analyse(self.rows, self.rows, self.rows, replicates=20, expected_questions=32))
        other = copy.deepcopy(self.rows); other[0]['strict_em'] = [0] * 5
        with self.assertRaisesRegex(ValueError, 'EM differ'):
            s.analyse(self.rows, other, self.rows, replicates=0, expected_questions=32)
        other = copy.deepcopy(self.rows); other[0]['trajectory_ids'][0] = 'wrong-own-id'
        with self.assertRaisesRegex(ValueError, 'trajectories/EM differ'):
            s.analyse(self.rows, other, self.rows, replicates=0, expected_questions=32)

    def test_fixed_quarter_threshold_and_formula_reject(self):
        m = s.metrics(s._validated(self.rows).values())
        self.assertEqual(m['mixed_balanced_accuracy'], 1); self.assertEqual(m['selected_questions'], 8)
        self.assertEqual(m['threshold'], .5)
        bad = copy.deepcopy(self.rows); bad[0]['predicted_mixed'] = 0
        with self.assertRaisesRegex(ValueError, 'q formula'):
            s.analyse(bad, self.rows, self.rows, replicates=0, expected_questions=32)

    def test_known_direction_and_original_question_resampling(self):
        bad = [row(i, i % 2 == 0, .99 if i % 2 == 0 else .5) for i in range(32)]
        x = s.analyse(self.rows, bad, bad, replicates=40, expected_questions=32)
        d = x['scopes']['all_questions']['comparisons']['E2']
        self.assertEqual(d['candidate_minus_baseline']['mixed_balanced_accuracy'], 1)
        self.assertEqual(d['candidate_minus_baseline']['top_quarter_mixed_count'], 8)
        self.assertEqual(d['percentile_95_CI']['mixed_balanced_accuracy'], [1, 1])
        self.assertTrue(x['question_resampling_with_replacement'])
        self.assertTrue(x['selection_recomputed_per_resample'])


if __name__ == '__main__':
    unittest.main()
