"""Pure toy arrays validate tie handling and paired-count bootstrap arithmetic."""
import unittest
import numpy as np
from .compare_viewed2048_pilot_depth import auc_layout,weighted_auc


class AUCMath(unittest.TestCase):
 def test_half_credit_ties(self):
  y=np.array([False,True,False,True]);z=np.array([0.,0.,1.,2.])
  # Correct>worse pairs: .5+0+1+1 over4 =.625.
  self.assertEqual(weighted_auc(auc_layout(y,z),np.ones(4,dtype=int)),.625)
 def test_weighted_counts_equal_expanded_bootstrap(self):
  y=np.array([False,True,False,True]);z=np.array([0.,0.,1.,2.]);idx=np.array([0,0,1,3,3,2])
  actual=weighted_auc(auc_layout(y,z),np.bincount(idx,minlength=4))
  yy=y[idx];zz=z[idx];brute=sum(float(p>n)+.5*float(p==n)for p in zz[yy]for n in zz[~yy])/(yy.sum()*(~yy).sum())
  self.assertAlmostEqual(actual,brute)


if __name__=='__main__':unittest.main()
