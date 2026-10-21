import copy
import unittest
from .joint124194_bank import merge_groups


def group(n, labels=None, h=None):
    return dict(qid='["nq",%d]' % n, question_hash=h or ('%064x' % n),
                trajectory_ids=['own-%d-%d' % (n, j) for j in range(10)],
                labels=labels if labels is not None else [0] * 5 + [1] * 5)


class CohortTests(unittest.TestCase):
    def test_retains_canonical_repeated_text_and_original_held_rows(self):
        banks=dict(old=[group(1)], recovered=[group(2, h='%064x' % 1)])
        result, proof=merge_groups(banks, [], dict(old=1, recovered=1), dict(nq=2), 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(proof['trajectories'], 20)
        self.assertTrue(proof['former_internal5120_is_training'])
        again, same=merge_groups(dict(old=list(reversed(banks['old'])), recovered=banks['recovered']), [], dict(old=1, recovered=1), dict(nq=2), 1)
        self.assertEqual(same['canonical_ordered_qid_hash_own_tids_sha256'], proof['canonical_ordered_qid_hash_own_tids_sha256'])

    def test_true_external_qid_and_hash_exclusions_both_remain_hard(self):
        banks=dict(one=[group(1)])
        for blocked in ([dict(qid=group(1)['qid'], question_hash='%064x' % 2)],
                        [dict(qid='["hotpotqa",99]', question_hash=group(1)['question_hash'])]):
            with self.assertRaisesRegex(ValueError, 'external evaluation'):merge_groups(banks, blocked, dict(one=1))

    def test_no_crossbank_qid_or_tid_reuse_and_no_fake_labels(self):
        original=dict(a=[group(1)], b=[group(2)])
        bad=copy.deepcopy(original);bad['b'][0]['trajectory_ids'][0]=bad['a'][0]['trajectory_ids'][0]
        with self.assertRaises(ValueError):merge_groups(bad, [], dict(a=1,b=1))
        bad=copy.deepcopy(original);bad['b'][0]['qid']=bad['a'][0]['qid']
        with self.assertRaises(ValueError):merge_groups(bad, [], dict(a=1,b=1))
        bad=copy.deepcopy(original);bad['b'][0]['labels'][0]=True
        with self.assertRaises(ValueError):merge_groups(bad, [], dict(a=1,b=1))

    def test_natural_K_labels_and_fixed_cohort_counts_preserved(self):
        banks=dict(one=[group(1, [0]*10), group(2, [1]*10), group(3)])
        result, proof=merge_groups(banks, [], dict(one=3))
        self.assertEqual([sum(g['labels']) for g in result], [0,10,5])
        self.assertEqual(proof['Mixed_capable_questions'], 1)
        with self.assertRaises(ValueError):merge_groups(banks, [], dict(one=4))
        with self.assertRaises(ValueError):merge_groups(banks, [], dict(one=3), dict(nq=2))


if __name__ == '__main__':unittest.main()
