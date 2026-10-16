import unittest
from .diversity import select_groups, summarize


def fixture(counts):
    groups, labels = [], {}
    for i, count in enumerate(counts):
        ids = ['q%d-r%d' % (i, j) for j in range(5)]
        groups.append(dict(qid=str(i), candidate_ordinal=i, q=0.75, trajectory_ids=ids))
        for j, tid in enumerate(ids):
            labels[tid] = dict(qid=str(i), strict_em=int(j < count))
    return dict(selected=groups), labels


class SelectionTests(unittest.TestCase):
    def test_variance_ranks_k2_k3_before_k1_k4_before_extremes(self):
        selection, labels = fixture([0]*256+[1]*256+[2]*256+[3]*256)
        chosen, all_rows = select_groups(selection, labels)
        self.assertEqual({r['correct_count'] for r in chosen}, {2,3})
        self.assertEqual(summarize(chosen)['mixed_fraction'], 1)
        self.assertAlmostEqual(summarize(chosen)['mean_population_em_variance'], .24)
        self.assertEqual(len({tid for g in chosen for tid in g['trajectory_ids']}), 2560)
        self.assertEqual(len(all_rows), 1024)

    def test_insufficient_mixed_keeps_exact512_no_resampling(self):
        selection, labels = fixture([1]*100+[0]*462+[5]*462)
        chosen, _ = select_groups(selection, labels)
        self.assertEqual(len(chosen), 512)
        self.assertEqual(sum(r['mixed'] for r in chosen), 100)
        self.assertEqual(chosen[100]['candidate_ordinal'], 100)

    def test_own_outcome_identity_required(self):
        selection, labels = fixture([2]*1024)
        labels['q0-r0']['qid'] = 'another'
        with self.assertRaises(ValueError): select_groups(selection, labels)

    def test_probability_tie_break_precedes_candidate_order(self):
        selection, labels = fixture([2]*1024)
        selection['selected'][-1]['q'] = .9
        chosen, _ = select_groups(selection, labels)
        self.assertEqual(chosen[0]['candidate_ordinal'], 1023)


if __name__ == '__main__': unittest.main()
