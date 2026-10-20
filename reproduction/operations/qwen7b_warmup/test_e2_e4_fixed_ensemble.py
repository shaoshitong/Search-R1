import unittest
from . import e2_e4_fixed_ensemble as m
class FixedSix(unittest.TestCase):
 def test_distinct_score_formulas(self):
  a=[.9]*5;b=[.1]*5
  qp,ps=m.combine(a,b,.5,'trajectory_probability_mix');qq,_=m.combine(a,b,.5,'direct_Mixed_score_mix')
  self.assertAlmostEqual(qp,.9375);self.assertAlmostEqual(qq,.4095);self.assertTrue(all(abs(p-.5)<1e-12 for p in ps))
 def test_known_labels_fixed_threshold_and_ranking(self):
  rows=[dict(qid='a',strict_em=[0]*5,score=.2),dict(qid='b',strict_em=[1,0,0,0,0],score=.9),dict(qid='c',strict_em=[1]*5,score=.1)]
  v=m.metrics(rows);self.assertEqual(v['mixed_balanced_accuracy'],1);self.assertEqual(v['mixed_roc_auc'],1);self.assertEqual(v['k_histogram'],[1,1,0,0,0,1])
 def test_grid_rejection(self):
  with self.assertRaises(ValueError):m.combine([.5]*5,[.5]*5,.4,'trajectory_probability_mix')
  with self.assertRaises(ValueError):m.combine([.5]*4,[.5]*5,.5,'trajectory_probability_mix')
if __name__=='__main__':unittest.main()
