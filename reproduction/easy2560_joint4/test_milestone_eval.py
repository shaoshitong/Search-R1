import unittest
from reproduction.easy2560_joint4.milestone_eval import aggregate


class EvaluationScores(unittest.TestCase):
    def test_repeated_occurrences_and_source_macro_weighted(self):
        rows = [dict(question_id='same', data_source='a', em=1, f1=1),
                dict(question_id='same', data_source='a', em=0, f1=.5),
                dict(question_id='last', data_source='b', em=1, f1=1)]
        result = aggregate(rows, ['same', 'same', 'last'], ['a', 'a', 'b'])
        self.assertEqual(result['question_count'], 3)
        self.assertEqual(result['weighted_em'], 2 / 3)
        self.assertEqual(result['macro_em'], .75)
        self.assertFalse(result['deduplicated'])

    def test_reject_missing_last_singleton_and_reordered_sources(self):
        row = dict(question_id='last', data_source='b', em=0, f1=0)
        with self.assertRaises(ValueError):
            aggregate([row], ['other', 'last'], ['a', 'b'])
        with self.assertRaises(ValueError):
            aggregate([row], ['last'], ['a'])


if __name__ == '__main__':
    unittest.main()
