import unittest
from search_g1.mixed_selection import MixedFirstSelector, InsufficientEligibleGroups


def fill(categories, target=2, limit=4, order=None):
    s = MixedFirstSelector(target, limit, 5)
    for i in range(len(categories)):
        assert s.reserve(str(i), 'question '+str(i)) == i
    for i in (order if order is not None else range(len(categories))):
        vec = {'mixed': [1,0,1,0,1], 'all_correct': [1]*5, 'all_wrong': [0]*5}[categories[i]]
        s.complete(i, [str(i)+'-'+str(j) for j in range(5)], vec)
    return s


class SelectionTests(unittest.TestCase):
    def test_fallback_preserves_ordinal_not_worker_order(self):
        s = fill(['all_correct','all_wrong','mixed','all_correct'], order=[3,2,1,0])
        self.assertEqual([x.ordinal for x in s.select()], [0,2])
        self.assertEqual(s.audit()['selected_all_correct_fallback'], 1)

    def test_all_correct_and_exact_eligible(self):
        self.assertEqual([x.ordinal for x in fill(['all_correct']*4).select()], [0,1])
        self.assertEqual([x.ordinal for x in fill(['all_wrong','mixed','all_correct','all_wrong']).select()], [1,2])

    def test_no_wrong_fill_or_early_fallback(self):
        with self.assertRaises(InsufficientEligibleGroups):
            fill(['mixed','all_wrong','all_wrong','all_wrong']).select()
        with self.assertRaises(ValueError):
            fill(['all_correct','mixed']).select()

    def test_early_mixed_and_inflight_cost(self):
        s = fill(['mixed','mixed','mixed','all_correct'])
        self.assertEqual([x.ordinal for x in s.select()], [0,1])
        self.assertIsNone(s.reserve('new','new question'))
        self.assertEqual(s.audit()['attempted_candidates'], 4)

    def test_dedup_pre_generation_and_partial_failure(self):
        s = MixedFirstSelector(2,4,5)
        self.assertEqual(s.reserve('a','same   question'), 0)
        self.assertIsNone(s.reserve('b','same question'))
        self.assertIsNone(s.reserve('a','different question'))
        with self.assertRaises(ValueError):s.complete(0,['a']*5,[1]*5)
        with self.assertRaises(ValueError):s.select()

    def test_512_2048_contract(self):
        s = fill(['mixed']*400+['all_correct']*112+['all_wrong']*1536,target=512,limit=2048)
        self.assertEqual(s.audit()['selected_mixed'],400)
        self.assertEqual(s.audit()['selected_all_correct_fallback'],112)
        self.assertEqual(len(s.select()),512)

    def test_fixed_pool_completes_remaining_candidates_after_enough_mixed(self):
        s=MixedFirstSelector(target=2,limit=4,fixed_pool=True)
        for i in range(2):
            self.assertEqual(s.reserve(str(i),'question '+str(i)),i)
            s.complete(i,[str(i)+'-'+str(j) for j in range(5)],[1,0,1,0,1])
        with self.assertRaises(ValueError):s.select()
        for i in (2,3):
            self.assertEqual(s.reserve(str(i),'question '+str(i)),i)
        for i in (3,2):s.complete(i,[str(i)+'-'+str(j) for j in range(5)],[1]*5)
        self.assertEqual([x.ordinal for x in s.select()],[0,1])
        self.assertEqual(s.audit()['attempted_candidates'],4)

    def test_fixed2560_excludes_wrong_and_uses_correct_only_as_fallback(self):
        s=MixedFirstSelector(target=512,limit=2560,fixed_pool=True)
        for i in range(2560):s.reserve(str(i),'question '+str(i))
        for i in reversed(range(2560)):
            values=[1,0,0,0,0] if 100<=i<500 else [1]*5 if 1000<=i<1200 else [0]*5
            s.complete(i,[str(i)+'-'+str(j) for j in range(5)],values)
        self.assertEqual([x.ordinal for x in s.select()],list(range(100,500))+list(range(1000,1112)))
        self.assertEqual(s.audit()['selected_all_correct_fallback'],112)

if __name__ == '__main__':unittest.main()
