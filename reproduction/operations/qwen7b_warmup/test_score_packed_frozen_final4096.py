"""Three pure synthetic checks; no final data, model or checkpoint loading."""
import copy,unittest
from . import score_packed_frozen_final4096 as s


class PackedFinalTests(unittest.TestCase):
 def freeze(self):
  ref=dict(path='/remote/fake',bytes=1,sha256='a'*64)
  return dict(status='CANDIDATE_FROZEN_BEFORE_FINAL_SCORES',freeze_epoch=1.,final_labels_or_scores_used_for_selection=False,further_candidate_selection_on_final_allowed=False,primary_validation_questions=2048,baseline=dict(checkpoint=s.E1,state_key='ema',layer=27,prefix=20),candidate=dict(checkpoint=ref,state_key='raw',layer=21,prefix=20,fixed_M=5,architecture='one_question_M3to10_policy20_packed_variance_v1',selection_score='predicted_population_EM_variance'),candidate_variance_threshold=.08,two_original_2048_G5_plans_unchanged=True,viewed_validation_evaluation=ref,scorer_publication=ref)
 def test_explicit_packed_freeze(self):
  value=self.freeze();self.assertTrue(s.freeze_contract(value));value['candidate']['state_key']='EMA';self.assertTrue(s.freeze_contract(value))
  for key,bad in(('fixed_M',10),('prefix',80),('state_key','ema'),('selection_score','Mixed_logit')):
   value=self.freeze();value['candidate'][key]=bad
   with self.assertRaises(ValueError):s.freeze_contract(value)
  value=self.freeze();value['status']='TEMPLATE_ONLY_NOT_CANDIDATE_FREEZE'
  with self.assertRaises(ValueError):s.freeze_contract(value)
 def test_compact_exact_20_and_short(self):
  records={str(i):dict(qid='q',prompt=[101,102],label=i%2,segments=[dict(policy_length=12,request_length=2,context_tokens=[101,102]+list(range(12))),dict(policy_length=12,request_length=3,context_tokens=[7,8,9]+list(range(12,24)))])for i in range(5)}
  records['0']['segments']=records['0']['segments'][:1]
  compact=s.compact_records(records)
  self.assertEqual(compact['0']['policy20'],list(range(12)));self.assertEqual(compact['1']['policy20'],list(range(20)))
  altered=copy.deepcopy(records)
  for row in altered.values():row['label']=1-row['label']
  self.assertEqual(compact,s.compact_records(altered))
 def test_same_decisions_and_order_zero_paired_difference(self):
  rows=[]
  for i in range(2048):
   labels=[0,1,0,0,0]if i%3 else[0]*5;mean=sum(labels)/5;p=(i%31)/120
   rows.append(dict(qid='%04d'%i,labels=labels,true_mixed=bool(i%3),target_variance=mean*(1-mean),candidate_variance=p,baseline_logit=(p-.08)*20))
  result=s.paired_summary(rows,repeats=25);difference=result['paired_difference_candidate_minus_E1_EMA']
  self.assertEqual(difference['Mixed_balanced_accuracy'],0);self.assertEqual(difference['Top512_true_Mixed_fraction'],0)
  self.assertEqual(difference['percentile95_question_bootstrap']['Mixed_balanced_accuracy'],[0,0]);self.assertEqual(difference['selected512_overlap'],512)
  self.assertTrue(result['candidate_Mixed_BCE_not_defined']);s.old.canonical(result)

if __name__=='__main__':unittest.main()
