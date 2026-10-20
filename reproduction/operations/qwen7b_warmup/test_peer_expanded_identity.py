import copy,unittest
from .peer_expanded_identity import expand,digest
class Poison(dict):
    def __getitem__(self,k):
        if k in ('labels','strict_em','p_j','score'):raise AssertionError('Labels/scores accessed during identity selection')
        return super().__getitem__(k)
def rows(n):
    return [Poison(qid='q%d'%i,question_hash=digest(['question',i]),trajectory_ids=['q%d-t%d'%(i,j) for j in range(10)],labels=[1]*10) for i in range(n)]
def ordered(g):return sorted(g,key=lambda r:(digest(['peer-newseed-independent8192',1729,r['qid']]),r['qid']))
class TestExpandedIdentity(unittest.TestCase):
    def test_identity_only_preserved_fullbatch_tail(self):
        g=rows(99);prior=ordered(g)[:32];v=expand(g,prior,[],[])
        self.assertEqual(v['summary']['questions'],96);self.assertEqual(v['summary']['new_questions'],64);self.assertEqual(v['summary']['tail_questions_omitted_for_complete_batch'],3)
        self.assertEqual([r['qid'] for r in v['groups'][:32]],[r['qid'] for r in prior]);self.assertEqual(set(v['groups'][0]),{'qid','question_hash','trajectory_ids'})
    def test_exclusion_by_qid_or_text_and_permutation(self):
        g=rows(100);controlled=[{'qid':g[0]['qid'],'question_hash':g[0]['question_hash']}];exclude=[{'qid':'other-id','question_hash':g[1]['question_hash']}]
        valid=[r for r in g if r['qid']!=g[0]['qid'] and r['question_hash']!=g[1]['question_hash']];prior=ordered(valid)[:32]
        a=expand(g,prior,controlled,[exclude]);b=expand(list(reversed(g)),prior,controlled,[exclude]);self.assertEqual(a,b)
        self.assertNotIn(g[0]['qid'],[x['qid'] for x in a['groups']]);self.assertNotIn(g[1]['qid'],[x['qid'] for x in a['groups']])
    def test_later_duplicate_hash_removed(self):
        g=rows(100);o=ordered(g);o[-1]['question_hash']=o[0]['question_hash'];v=expand(g,o[:32],[],[])
        self.assertEqual(v['summary']['duplicated_question_text_excluded'],1);self.assertEqual(v['summary']['unique_eligible_questions'],99)
    def test_reject_changed_old_prefix_ids_and_bad_trajectories(self):
        g=rows(99);p=ordered(g)[:32]
        with self.assertRaises(ValueError):expand(g,list(reversed(p)),[],[])
        bad=copy.deepcopy(g);bad[0]['trajectory_ids']=['same']*10
        with self.assertRaises(ValueError):expand(bad,p,[],[])
        bad=copy.deepcopy(g);o=ordered(bad);o[-1]['trajectory_ids'][0]=o[0]['trajectory_ids'][0]
        with self.assertRaises(ValueError):expand(bad,p,[],[])
if __name__=='__main__':unittest.main()
