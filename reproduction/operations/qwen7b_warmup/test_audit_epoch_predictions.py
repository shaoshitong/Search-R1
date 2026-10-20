import unittest
from .audit_epoch_predictions import mixed_metrics
class MixedMetricTest(unittest.TestCase):
 def test_confusion_and_auc_with_ties(self):
  rows=[dict(strict_em=y,predicted_mixed=q) for y,q in [([0]*5,.1),([1]*5,.8),([1,0,0,0,0],.8),([0,1,1,0,0],.9)]]
  m=mixed_metrics(rows);self.assertEqual(m['confusion'],dict(tp=2,fp=1,fn=0,tn=1));self.assertEqual(m['mixed_balanced_accuracy'],.75);self.assertEqual(m['roc_auc'],.875)
  self.assertEqual(m['classwise']['all_correct']['predicted_mixed_fraction'],1.)
if __name__=='__main__':unittest.main()
