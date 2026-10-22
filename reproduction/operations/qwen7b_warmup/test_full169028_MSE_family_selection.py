import unittest
from reproduction.operations.qwen7b_warmup.select_compare_full169028_P80_MSE import choose_seed1

class Selection(unittest.TestCase):
 def candidates(self):return [dict(epoch=e,state=s,Top512_Mixed=250)for e in range(1,11)for s in('raw','EMA')]
 def test_predeclared_earliest_then_EMA_tie(self):
  self.assertEqual(choose_seed1(self.candidates()),dict(epoch=1,state='EMA',Top512_Mixed=250))
 def test_count_precedes_epoch_and_state(self):
  rows=self.candidates();rows[-2]['Top512_Mixed']=251
  self.assertEqual(choose_seed1(rows),dict(epoch=10,state='raw',Top512_Mixed=251))
 def test_incomplete_run_rejected(self):
  with self.assertRaises(ValueError):choose_seed1(self.candidates()[:-1])
 def test_duplicate_epoch_state_rejected(self):
  rows=self.candidates();rows[-1]=dict(rows[0])
  with self.assertRaises(ValueError):choose_seed1(rows)

if __name__=='__main__':unittest.main()
