import unittest
from .headlr_fixed_prior_diagnostic import describe,descriptive
class Tests(unittest.TestCase):
 def test_fixed_description(self):
  self.assertEqual(describe([.2]*5)['std'],0.)
  rows=[dict(score=q,strict_em=[int(j<k) for j in range(5)],count_probabilities=[int(i==k) for i in range(6)]) for k,q in enumerate([0.,.1,.3,.5,.8,1.])]
  out=descriptive(rows)
  self.assertEqual(sum(b['questions'] for b in out['fixed_ten_equal_width_bins']),6)
  self.assertEqual(out['predicted_argmax_K'],[1]*6)
  self.assertEqual(out['fixed_ten_equal_width_bins'][-1]['questions'],1)
  self.assertFalse(out['bin_or_quantile_used_to_choose_threshold'])
if __name__=='__main__':unittest.main()
