"""Pure metadata guards only; no large arrays, model/checkpoint or GPU loads."""
import copy,unittest
from .accept_full169028_L15P80_MSE_checkpoint import A,FIXTURE,PARENTS,recipe_for,verify_saved_phase
from .online_packed_full169028_MSE_P80 import full_acceptance_alignment_ref,full_acceptance_physical_source_ref,validate_physical_source_contract
class FullAcceptanceMetadata(unittest.TestCase):
 def binding(self,parent):
  init,plan=PARENTS[parent];return dict(architecture='one_question_packed_matched_objective_v1',training_scope='full169028_fresh_official_actor_not_pilot_resume',objective='mse',fixed_m=5,layer=15,capture_layer=14,prefix=80,head_max_positions=2048,EMA_decay=.9995,lr=5e-6,global_batch=128,training_questions=169028,seed=plan,init_seed=init,train_plan_seed=plan,diagnostic_seed=2026102228,seed_source='explicit_CLI',configured_local_batch=64,configured_micro=8,configured_accumulation=8,tail_local_batch=34,tail_global_batch=68,updates_per_epoch=1321,class_weighting=False,sampling_independent_of_EM=True,question_prompt_occurrences_per_group=1,fixture=dict(path=str(FIXTURE/'fixture-complete.json')),initial_head_sha256='a'*64)
 def test_two_explicit_runroots_and_full_not_pilot(self):
  for parent in PARENTS:
   binding=self.binding(parent);r=recipe_for(A/parent,binding);self.assertEqual(r['train_plan_seed'],binding['train_plan_seed'])
   for change in(dict(training_questions=8192),dict(training_scope='pilot'),dict(head_max_positions=1024),dict(tail_global_batch=128),dict(init_seed=1)):
    with self.assertRaises(AssertionError):recipe_for(A/parent,dict(binding,**change))
 def test_epoch_phase_clock_and_required_states(self):
  for parent in PARENTS:
   r=recipe_for(A/parent,self.binding(parent));blob=dict(r,epoch=3,cursor=3963,batch_attempts=3963,actual_Adam_updates=3963,EMA_updates=3963,margin_satisfied_batches=0,actor_updates=0,epoch_complete=True,next_epoch=4,within_next_epoch_question_position=0,execution_phase='epoch_complete_pre_evaluation',RNG_by_rank=[dict(python=1,numpy=2,torch=3,cuda=4)]*2,raw={},EMA={},Adam={});verify_saved_phase(blob,3,r)
   for changes in(dict(actual_Adam_updates=3962),dict(epoch_complete=False),dict(next_epoch=3),dict(within_next_epoch_question_position=1),dict(diagnostic_seed=0)):
    with self.assertRaises(AssertionError):verify_saved_phase(dict(blob,**changes),3,r)
   bad=copy.copy(blob);del bad['Adam']
   with self.assertRaises(AssertionError):verify_saved_phase(bad,3,r)
 def test_physical_and_P20_gates_are_both_required(self):
  for value in({},dict(actual_physical_execution_and_seal_source_exact=False)):
   with self.assertRaises(ValueError):full_acceptance_physical_source_ref(value)
  with self.assertRaises(ValueError):full_acceptance_alignment_ref({})
  valid=dict(status='ALL169028_OWN_G10_PHYSICAL_SEAL_EXECUTION_AND_OWN_EM_RESOLVED',all_selected_own_EM_matches_canonical=True,whole_G5_not_mixed=True,selection_used_EM=False);validate_physical_source_contract(valid)
  for k,v in [('whole_G5_not_mixed',False),('all_selected_own_EM_matches_canonical',False),('selection_used_EM',True)]:
   with self.assertRaises(ValueError):validate_physical_source_contract(dict(valid,**{k:v}))
if __name__=='__main__':unittest.main()
