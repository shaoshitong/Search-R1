import copy,unittest
from .test_peer_audited122_l26 import fixture
from .peer_audited122_l26_merge import join_groups
class DeltaMergeTests(unittest.TestCase):
 def test_own_em_join_preserves_all_ordered_ids(self):
  g=fixture();p={'groups':[{k:v for k,v in x.items() if k!='labels'} for x in g]};out=join_groups(p,list(reversed(g)),8)
  self.assertEqual(out,g)
 def test_bad_own_em_or_authentic_ids_rejected(self):
  g=fixture();p={'groups':copy.deepcopy(g)}
  for field,value in [('trajectory_ids',['wrong']*10),('labels',[0]*9+[.5]),('question_hash','a'*64)]:
   bad=copy.deepcopy(g);bad[0][field]=value
   with self.assertRaises(ValueError):join_groups(p,bad,8)
 def test_duplicate_missing_truth_rejected(self):
  g=fixture();p={'groups':g}
  with self.assertRaises(ValueError):join_groups(p,g[:-1],8)
  with self.assertRaises(ValueError):join_groups(p,[g[0]]*8,8)
if __name__=='__main__':unittest.main()
