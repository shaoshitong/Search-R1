import itertools,unittest
from fractions import Fraction
from .train_g10_domain_statistics import expected_mixed5,summarize,overlap

class StatisticsTests(unittest.TestCase):
 def test_uniform_G5_exact_against_all252_own_subsets(self):
  for k in range(11):
   ys=[1]*k+[0]*(10-k);mix=sum(0<sum(ys[j]for j in ix)<5 for ix in itertools.combinations(range(10),5))
   self.assertEqual(expected_mixed5(k),Fraction(mix,252))
  self.assertEqual(expected_mixed5(5),Fraction(125,126));self.assertEqual(expected_mixed5(1),Fraction(1,2))
 def test_G10_heterogeneity_not_G5_mix_and_identity_alias_count(self):
  m=summarize([dict(labels=[1]+[0]*9),dict(labels=[1]*5+[0]*5)])
  self.assertEqual(m['G10_heterogeneous_fraction'],1.)
  self.assertAlmostEqual(m['uniform_distinct_G5_expected_mixed_fraction'],float((Fraction(1,2)+Fraction(125,126))/2))
  a=[dict(qid='a',question_hash='x'),dict(qid='b',question_hash='x')];b=[dict(qid='c',question_hash='x')];r=overlap(a,b)
  self.assertEqual(r['qid_overlap'],0);self.assertEqual(r['left_questions_with_any_hash_overlap'],2);self.assertEqual(r['shared_distinct_question_hashes'],1)
if __name__=='__main__':unittest.main()
