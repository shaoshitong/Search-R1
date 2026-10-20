import unittest
from .prefix_confidence_pipeline import gate_decision

def m(ba,top):return dict(balanced_accuracy=ba,Top512=dict(Mixed=top))
class GateTest(unittest.TestCase):
 def test_raw_requires_both_strict(self):
  self.assertFalse(gate_decision(m(.55,180),m(.54,179),m(.55,181))['passed']);self.assertFalse(gate_decision(m(.55,180),m(.54,179),m(.56,180))['passed'])
 def test_control_no_decrease_plus_one_strict(self):
  self.assertTrue(gate_decision(m(.55,180),m(.56,181),m(.56,182))['passed']);self.assertFalse(gate_decision(m(.55,180),m(.57,181),m(.56,182))['passed']);self.assertFalse(gate_decision(m(.55,180),m(.56,182),m(.56,182))['passed'])
if __name__=='__main__':unittest.main()
