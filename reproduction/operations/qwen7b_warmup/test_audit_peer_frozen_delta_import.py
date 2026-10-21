import unittest
from . import audit_peer_frozen_delta_import as a

class OwnLabels(unittest.TestCase):
 def test_own_binary_order_and_ids(self):
  a.own(dict(trajectory_ids=list("abcde"),labels=[0,1,0,1,0]),5)
  for v in (dict(trajectory_ids=list("abcda"),labels=[0]*5),dict(trajectory_ids=list("abcde"),labels=[0.0]*5),dict(trajectory_ids=list("abcde"),labels=[0,1])):
   with self.assertRaises(ValueError):a.own(v,5)

if __name__=="__main__":unittest.main()
