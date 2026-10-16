import itertools
import unittest

from search_g1.prefix20_contract import mixed_probability, probe_batch_plan, top_questions


class PrefixContractTest(unittest.TestCase):
    def test_independent_probability_matches_enumerated_mixed_events(self):
        p = [.01, .9, .4, .3, .99]
        exact = sum(__import__('math').prod(p[j] if labels[j] else 1-p[j] for j in range(5))
                    for labels in itertools.product((0, 1), repeat=5) if 0 < sum(labels) < 5)
        self.assertAlmostEqual(mixed_probability(p), exact, places=14)
        self.assertEqual(mixed_probability([0]*5), 0)
        self.assertEqual(mixed_probability([1]*5), 0)
        self.assertEqual(mixed_probability([0, 0, 1, 1, 1]), 1)

    def test_top512_and_stable_ties_have_no_random_exploration(self):
        rows = [dict(qid=str(q), trajectory_ids=[f'{q}:{g}' for g in range(5)],
                     em_probabilities=[.5 if q >= 4608 else 0]*5) for q in range(5120)]
        selected = top_questions(rows)
        self.assertEqual([r['qid'] for r in selected], list(map(str, range(4608, 5120))))
        rows[0]['strict_em'] = [1]*5
        with self.assertRaisesRegex(ValueError, 'FORBIDS_GOLD'): top_questions(rows)

    def test_one_new_pass_no_replay_no_drop_no_repeat(self):
        ids = [f'trajectory{i}' for i in range(2560)]
        plan = probe_batch_plan(ids, list(reversed(range(2560))))
        self.assertEqual(len(plan), 10)
        self.assertTrue(all(len(step) == 4 and all(len(rank) == 64 for rank in step) for step in plan))
        self.assertEqual([tid for step in plan for rank in step for tid in rank], list(reversed(ids)))


if __name__ == '__main__': unittest.main()
