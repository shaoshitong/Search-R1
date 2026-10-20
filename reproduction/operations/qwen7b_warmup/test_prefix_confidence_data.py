import math,unittest
from . import prefix_confidence_data as d
from .test_prefix_logprob_availability import data
class DataTest(unittest.TestCase):
 def test_six_only_actual_policy_and_short_prefix(self):
  x=d.six_features([-1.,-2.,-3.]);self.assertEqual(x[0],-2.);self.assertAlmostEqual(x[1],math.sqrt(2/3));self.assertEqual(x[2:],[-3.,-2.,-2.,.15])
  self.assertEqual(d.six_features([0.]*20),[0.,0.,0.,0.,0.,1.]) # legal native probability rounding after causal verification
 def test_projection_matches_causal_seal_and_six_definition(self):
  p,f=data();x=d.projection(p,f);self.assertEqual(len(x['features']),6);self.assertEqual(len(x['lp']),20);self.assertEqual(x['features'],d.six_features(x['lp']))
 def test_qid_split_fixed_whole_question_label_free(self):
  q=['q%d'%i for i in range(8192)];x=d.split_ids(q);self.assertEqual(len(x['fit_qids']),6144);self.assertEqual(len(x['held_qids']),2048);self.assertFalse(set(x['fit_qids']) & set(x['held_qids']));y=d.split_ids(list(reversed(q)));self.assertEqual(set(x['held_qids']),set(y['held_qids']))
 def test_quality_and_same_input_numeric_difference_not_semantics(self):
  p,f=data();x=d.projection(p,f);y=dict(x,lp=[v+1e-7 for v in x['lp']]);rows=[x]*5+[y]*5;n=d.noise();d.add_noise(n,rows);self.assertEqual(n['all']['pairs'],45);self.assertEqual(n['cross_event']['pairs'],25);self.assertGreater(n['all']['max_absolute_difference'],0);q=d.quality();d.add_quality(q,[x]*5);r=d.quality_summary(d.merge([q,q]));self.assertEqual(r['trajectories'],10);self.assertEqual(r['within_G5_constant_feature_fraction'],[1.]*6)
if __name__=='__main__':unittest.main()
