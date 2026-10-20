import copy,unittest
from .test_peer_expanded_l26 import fixture
from .peer_expanded_l26_merge import join_groups

class MergeTest(unittest.TestCase):
    def truth(self):return [dict(g,labels=[0,1,0,1,0,1,0,1,0,1]) for g in fixture()['groups']]
    def test_prelabel_plan_joins_own_em_without_reselection(self):
        p=fixture();truth=self.truth();got=join_groups(p,list(reversed(truth)),existing=2,added=8)
        self.assertEqual([g['qid'] for g in got],[g['qid'] for g in p['groups']]);self.assertEqual(len(got),10)
    def test_wrong_ids_or_nonbinary_own_em_rejected(self):
        for key,value in [('trajectory_ids',['bad']*10),('labels',[.5]*10)]:
            truth=self.truth();truth[3][key]=value
            with self.assertRaises(ValueError):join_groups(fixture(),truth,existing=2,added=8)
    def test_duplicate_source_question_rejected(self):
        truth=self.truth();truth.append(copy.deepcopy(truth[0]))
        with self.assertRaises(ValueError):join_groups(fixture(),truth,existing=2,added=8)
if __name__=='__main__':unittest.main()
