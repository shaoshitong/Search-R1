import copy,unittest
from . import audit_prefix_logprob_availability as a
from .test_peer_long_prefix_diagnostic import fixture,sealed

def data():
 f=fixture(first=3,second=25);alllp=[];journal=[]
 for turn in f['behavior_turns']:
  lp=[-.01*(i+1) for i in range(len(turn['raw_generated_token_ids']))];turn['selected_logprobs']=lp;n=0
  for c in turn['actual_sampling_chunks']:
   m=len(c['raw_generated_token_ids']);c['selected_logprobs']=lp[n:n+m];c['sampling']=dict(temperature=1.,top_p=1.,top_k=-1,logprobs=1);n+=m
  alllp+=lp[:turn['kept_token_count']]+[0.]*(len(turn['observation_token_ids'])+len(turn['correction_token_ids']));journal+=turn['actual_sampling_chunks']
 f['response_log_probs']=alllp;f['sampling_journal']=journal;p=sealed(f);return p,f
class TestLogprob(unittest.TestCase):
 def test_native_to_retained_prefix_excludes_environment_zeros(self):
  p,f=data();r=a.project(p,f);self.assertEqual(r['kept_prefix_policy_tokens'],20);self.assertEqual(r['prefix_nonzero_chosen_logprobs'],20);self.assertEqual(r['prefix_exact_zero_chosen_logprobs'],0);self.assertTrue(r['native_turn_response_chosen_logprobs_exact']);self.assertEqual(r['prefix_json_has_logprob_fields'],[])
 def test_reject_clock_or_value_misalignment(self):
  p,f=data();f['behavior_turns'][0]['selected_logprobs'][0]=0
  with self.assertRaises(AssertionError):a.project(p,f)
 def test_future_policy_suffix_does_not_change_prefix_logprob_summary(self):
  p,f=data();r=a.project(p,f);last=f['behavior_turns'][-1];last['selected_logprobs'][-1]=-99;last['actual_sampling_chunks'][-1]['selected_logprobs'][-1]=-99
  self.assertEqual(r,a.project(p,f))
 def test_reject_missing_nonfinite_positive_or_wrong_token_native_values(self):
  for change in (float('nan'),1.,None):
   p,f=data();f['behavior_turns'][0]['actual_sampling_chunks'][0]['selected_logprobs'][0]=change
   with self.assertRaises(AssertionError):a.project(p,f)
if __name__=='__main__':unittest.main()
