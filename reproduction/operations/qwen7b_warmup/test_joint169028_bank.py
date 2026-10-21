"""No model data: regression checks for same-ID/different-execution mixing."""
import copy
import unittest

from . import joint169028_bank as bank


class PhysicalSourceJoinTests(unittest.TestCase):
    def fixture(self, tid='q-t0'):
        chosen = dict(trajectory_id=tid, qid='q', chosen_execution='qualified_execution_A',
            physical_prefix=dict(path='/remote/A.jsonl.gz', bytes=17, sha256='a' * 64),
            physical_seal=dict(path='/remote/A-seal.json', bytes=13, sha256='b' * 64),
            prefix_body_sha256='c' * 64, canonical_prefix_row_sha256='d' * 64)
        feature = dict(qid='q', prefix_sha256='c' * 64, retained_sha256='e' * 64,
            manifest=dict(path='/remote/manifest.json', bytes=19, sha256='f' * 64),
            index=dict(path='/remote/index.jsonl', bytes=23, sha256='0' * 64))
        bound = dict(chosen, feature_retained_row_sha256=feature['retained_sha256'],
                     feature_manifest=feature['manifest'], feature_index=feature['index'])
        return chosen, feature, bound

    def test_correct_tid_join_accepts_different_projection_and_capture_order(self):
        chosen, feature, bound = self.fixture()
        chosen2, feature2, bound2 = self.fixture('q-t1')
        sources = {v['trajectory_id']: v for v in (chosen, chosen2)}
        indices = {'q-t0': feature, 'q-t1': feature2}
        for row in (bound2, bound):
            bank.validate_link(row, sources[row['trajectory_id']], indices[row['trajectory_id']])
        with self.assertRaises(ValueError):
            bank.validate_link(bound2, chosen, feature)

    def test_same_protocol_tid_cannot_substitute_different_physical_execution(self):
        chosen, feature, bound = self.fixture()
        for key, value in [('chosen_execution', 'qualified_execution_B'),
                           ('physical_prefix', dict(path='/remote/B.jsonl.gz', bytes=17, sha256='a' * 64)),
                           ('canonical_prefix_row_sha256', '1' * 64),
                           ('feature_retained_row_sha256', '2' * 64),
                           ('feature_manifest', dict(path='/remote/other-manifest', bytes=19, sha256='f' * 64))]:
            bad = copy.deepcopy(bound)
            bad[key] = value
            with self.assertRaises(ValueError):
                bank.validate_link(bad, chosen, feature)

    def test_disjoint_expected_role_counts_cover_full_target(self):
        self.assertEqual(list(bank.COUNTS.values()), [124194, 9344, 14080, 14592, 6818])
        self.assertEqual(sum(bank.COUNTS.values()), 169028)


if __name__ == '__main__':
    unittest.main()
