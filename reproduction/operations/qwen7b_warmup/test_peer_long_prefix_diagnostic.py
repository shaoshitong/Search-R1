"""CPU-only sealed causal horizon and aggregate boundary tests; no model/data load."""
import copy,unittest
from . import peer_long_prefix_diagnostic as d


def fixture(first=25,second=90,observation=(800,801)):
 prompt=[11,12,13];turns=[];rebuilt=[];mask=[];journal=[]
 for i,(raw,kept,obs) in enumerate([(list(range(100,100+first+3)),first,list(observation)),(list(range(300,300+second+2)),second,[]) ]):
  request=prompt+rebuilt;chunks=[dict(actual_request_token_ids=request,raw_generated_token_ids=raw[:10]),dict(actual_request_token_ids=request+raw[:10],raw_generated_token_ids=raw[10:])]
  turns.append(dict(turn_index=i,actual_request_token_ids=request,raw_generated_token_ids=raw,kept_token_count=kept,prefix_cut=dict(kept_token_ids=raw[:kept],kept_token_count=kept),actual_sampling_chunks=chunks,observation_token_ids=obs,correction_token_ids=[]))
  journal+=chunks;rebuilt+=raw[:kept]+obs;mask += [1]*kept+[0]*len(obs)
 return dict(qid='q',question_hash='h',trajectory_id='t',strict_em=1,initial_prompt_token_ids=prompt,behavior_turns=turns,sampling_journal=journal,response_token_ids=rebuilt,response_info_mask=mask,terminal_reason='answer')


def sealed(full):
 x=d.reconstruct(full,20);p=dict(qid=full['qid'],question_hash=full['question_hash'],trajectory_id=full['trajectory_id'],initial_prompt_token_ids=full['initial_prompt_token_ids'],policy_token_ids=x['tokens'],policy_position_ids=x['positions'],prefix_segments=x['segments'],retained_policy_length=len(x['tokens']),terminal=x['short'],terminal_reason=full['terminal_reason'] if x['short'] else None,future_information_included=False)
 p['prefix_sha256']=d.digest(p);full['prefix_sha256']=p['prefix_sha256'];p['feature_locator']='ignored';return p


class HorizonTest(unittest.TestCase):
 def test_prefix_cut_excludes_future_observation_and_discarded_tail(self):
  f=fixture();x=d.reconstruct(f,20)
  self.assertEqual(x['tokens'],list(range(100,120)));self.assertEqual(x['positions'],list(range(3,23)));self.assertEqual(len(x['segments']),1)
  self.assertEqual(x['segments'][0]['context_tokens'],[11,12,13]+list(range(100,120)))
  self.assertEqual(x['past_retrieval_tokens'],0);self.assertFalse(x['multi_turn'])
  changed=copy.deepcopy(f);changed['behavior_turns'][0]['raw_generated_token_ids'][-1]=9999;changed['behavior_turns'][0]['actual_sampling_chunks'][1]['raw_generated_token_ids'][-1]=9999
  changed['sampling_journal']=sum([v['actual_sampling_chunks'] for v in changed['behavior_turns']],[])
  self.assertEqual(d.reconstruct(changed,20)['causal_signature'],x['causal_signature'])
 def test_long_prefix_has_actual_past_context_and_correct_gather(self):
  f=fixture();x=d.reconstruct(f,40)
  self.assertEqual(x['tokens'],list(range(100,125))+list(range(300,315)))
  self.assertEqual(x['positions'],list(range(3,28))+list(range(30,45)))
  self.assertEqual(x['segments'][1]['context_tokens'],[11,12,13]+list(range(100,125))+[800,801]+list(range(300,315)))
  self.assertEqual(x['past_retrieval_tokens'],2);self.assertEqual(x['past_retrieval_events'],1);self.assertEqual(x['later_turn_policy_tokens'],15);self.assertTrue(x['multi_turn'])
 def test_short_terminal_and_no_padding_is_claimed_actual(self):
  f=fixture(first=3,second=4);x=d.reconstruct(f,80)
  self.assertTrue(x['short']);self.assertEqual(x['policy_tokens'],7);self.assertEqual(x['terminal_reason'],'answer')
  self.assertEqual(len(x['tokens']),7)
 def test_sealed20_exact_and_future_changes_rejected(self):
  f=fixture();p=sealed(f);wanted=dict(qid='q',question_hash='h',trajectory_id='t',label=1);x=d.qualify_row(p,f,wanted)
  self.assertEqual(set(x['values']),{'5','20','40','80','160'});self.assertNotIn('tokens',x['values']['20'])
  p['policy_position_ids'][0]+=1
  with self.assertRaises(ValueError):d.qualify_row(p,f,wanted)
 def test_identical_policy_different_causal_context_is_distinguishable(self):
  f=fixture(first=3);g=fixture(first=3,observation=(900,901));a=d.reconstruct(f,20);b=d.reconstruct(g,20)
  self.assertEqual(a['token_signature'],b['token_signature']);self.assertNotEqual(a['causal_signature'],b['causal_signature'])
 def test_reject_broken_native_request_chain(self):
  f=fixture();f['behavior_turns'][1]['actual_sampling_chunks'][1]['actual_request_token_ids'][-1]=0
  with self.assertRaises(ValueError):d.reconstruct(f,40)
 def test_aggregate_counts_and_merge_equal_single_pass(self):
  f=fixture();p=sealed(f);a=d.qualify_row(p,f,dict(qid='q',question_hash='h',trajectory_id='t',label=1));b=copy.deepcopy(a);b['label']=0
  c=copy.deepcopy(a)
  for x in c['values'].values():x['causal_signature']='different';x['token_signature']='different'
  group=[a,b,c,a,b];single=d.empty_group(5);d.add(group,single);self.assertEqual(single['correct_wrong_pairs'],6)
  s=single['lengths']['20']['representations']['full_causal_input'];self.assertEqual(s['duplicate_pairs'],6);self.assertEqual(s['conflict_pairs'],4);self.assertEqual(s['distinct_histogram'][2],1)
  merged=d.combine(single,single);twice=d.empty_group(5);d.add(group,twice);d.add(group,twice);self.assertEqual(merged,twice)
  r=d.report({'test':single})['test']['lengths']['20']['representations']['full_causal_input'];self.assertAlmostEqual(r['distinguishable_correct_wrong_pair_fraction'],1/3)

if __name__=='__main__':unittest.main()
