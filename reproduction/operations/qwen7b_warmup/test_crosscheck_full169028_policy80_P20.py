"""Tiny asset-free checks for exact P20/P80 alignment fingerprinting."""
import unittest
from .crosscheck_full169028_policy80_P20 import normalized_sha
class P20AlignmentTest(unittest.TestCase):
 def test_same_prompt_prefix_and_own_label_match(self):
  r=dict(qid='q',trajectory_id='t',prompt=[1,2]);self.assertEqual(normalized_sha(r,[3,4],1),normalized_sha(dict(r),[3,4,5][:2],1))
 def test_label_prompt_and_prefix_each_bound(self):
  r=dict(qid='q',trajectory_id='t',prompt=[1,2]);h=normalized_sha(r,[3,4],1)
  self.assertNotEqual(h,normalized_sha(r,[3,4],0));self.assertNotEqual(h,normalized_sha(dict(r,prompt=[2,1]),[3,4],1));self.assertNotEqual(h,normalized_sha(r,[4,3],1))
if __name__=='__main__':unittest.main()
