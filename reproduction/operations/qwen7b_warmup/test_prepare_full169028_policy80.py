"""Small CPU checks of the actual retained-token extraction, without assets."""
import unittest
from .prepare_full169028_policy80 import policy80,compact_raw,row_fingerprint
class Policy80Test(unittest.TestCase):
 def test_only_kept_policy_in_order(self):
  row=dict(qid='q',trajectory_id='t',initial_prompt_token_ids=[11,12],behavior_turns=[dict(raw_generated_token_ids=list(range(70)),kept_token_count=60,observation_token_ids=[999]*10),dict(raw_generated_token_ids=list(range(100,170)),kept_token_count=40)])
  tokens,total=policy80(row);self.assertEqual(tokens,list(range(60))+list(range(100,120)));self.assertEqual(total,100);self.assertNotIn(999,tokens)
 def test_true_short_answer_is_not_fabricated(self):
  row=dict(qid='q',trajectory_id='t',initial_prompt_token_ids=[11],behavior_turns=[dict(raw_generated_token_ids=[1,2,3,999],kept_token_count=3)])
  compact,total=compact_raw(row);self.assertEqual(compact['policy_token_ids'],[1,2,3]);self.assertEqual(total,3);self.assertNotIn('strict_em',compact)
  self.assertEqual(row_fingerprint(compact),row_fingerprint(dict(reversed(list(compact.items())))))
 def test_invalid_kept_count_rejected(self):
  with self.assertRaises(AssertionError):policy80(dict(behavior_turns=[dict(raw_generated_token_ids=[1],kept_token_count=2)]))
if __name__=='__main__':unittest.main()
