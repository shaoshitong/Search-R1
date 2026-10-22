import unittest
from .observe_full169028_MSE_metadata import six_plus_next

def rows(n=7):return [dict(Adam_and_EMA_step_performed=True,actual_Adam_updates=i,EMA_updates=i,cursor=i,native_gradient_L2=.1,head_gradient_L2=.2,prefill_rows_equal_groups=True,time=float(i))for i in range(1,n+1)]
class ProgressTest(unittest.TestCase):
 def test_six_requires_next_actual_step(self):self.assertFalse(six_plus_next(rows(6)));self.assertTrue(six_plus_next(rows()))
 def test_gap_or_zero_gradient_is_not_continuous_progress(self):
  x=rows();x[-1]['actual_Adam_updates']+=1;self.assertFalse(six_plus_next(x));x=rows();x[3]['native_gradient_L2']=0.;self.assertFalse(six_plus_next(x))
if __name__=='__main__':unittest.main()
