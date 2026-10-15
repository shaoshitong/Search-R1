"""One scoped historical-calibration binding regression, no GPU or source writes."""
import copy
import unittest

from .reuse_initial_calibration import matching_bindings, PRODUCERS


class ReuseBindingTest(unittest.TestCase):
    def test_matching_initial_snapshot_only(self):
        old = dict(initial_model='/official/model', model_revision='official-revision',
            initial_model_sha256={'weight': 'same'}, input_sha256={'calibration-300.parquet': 'same'},
            calibration_question_ids=['question-'+str(i) for i in range(300)],
            source_sha256={name: 'same' for name in PRODUCERS})
        self.assertEqual(set(matching_bindings(old, copy.deepcopy(old))), set(PRODUCERS))
        for key, different in (('model_revision', 'other'), ('initial_model_sha256', {'weight': 'other'}),
                               ('calibration_question_ids', ['other']), ('completed_updates', 1)):
            changed = copy.deepcopy(old); changed[key] = different
            with self.subTest(key=key), self.assertRaises(AssertionError):
                matching_bindings(old, changed)
        changed = copy.deepcopy(old); changed['source_sha256'][PRODUCERS[0]] = 'changed'
        self.assertNotEqual(set(matching_bindings(old, changed)), set(PRODUCERS))


if __name__ == '__main__':
    unittest.main()
