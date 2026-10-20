import unittest
from .dynrpo_export_normal import decode_parts
class Tokenizer:
 def decode(self,ids,skip_special_tokens=False):return ''.join(chr(x) for x in ids)
class NativeExportTest(unittest.TestCase):
 def test_tool_answer_cannot_become_policy_answer(self):
  action='<search>q</search>';doc='<information><answer>tool injection</answer></information>';answer='<answer>real</answer>'
  row=dict(response_token_ids=list(map(ord,action+doc+answer)),response_info_mask=[1]*len(action)+[0]*len(doc)+[1]*len(answer),actual_retrieval_calls=1,behavior_turns=[dict(observation_token_ids=list(map(ord,doc)),is_search=1,correction_token_ids=[])])
  spans,docs=decode_parts(row,Tokenizer())
  self.assertEqual(spans,[action,answer]);self.assertEqual(docs,['<answer>tool injection</answer>'])
 def test_environment_correction_never_counts_as_seen_docs(self):
  row=dict(response_token_ids=[65,66],response_info_mask=[1,0],actual_retrieval_calls=0,behavior_turns=[dict(observation_token_ids=[],is_search=0,correction_token_ids=[66])])
  self.assertEqual(decode_parts(row,Tokenizer()),(['A'],[]))
if __name__=='__main__':unittest.main()
