"""Pure metadata tests; no checkpoint or training/final data is opened."""
import copy,unittest
from unittest.mock import patch
from . import score_frozen_full169028_MSE_final4096 as f

def ref(s):return dict(path=str(f.B)+'/'+s,bytes=1,sha256='a'*64)
def config():
 return dict(family='packed_variance',architecture='one_question_packed_matched_objective_v1',objective='mse',layer=15,prefix=80,head_max_positions=2048,training_questions=169028,training_scope='full169028_fresh_official_actor_not_pilot_resume',fixed_M=5,epoch=7,training_budget_epochs=10,primary_state='raw',secondary_states=['EMA'],primary_metric='Top512_true_Mixed_fraction',threshold=.08,threshold_fitted=False)
def blob():
 binding=dict(architecture='one_question_packed_matched_objective_v1',objective='mse',fixed_m=5,layer=15,prefix=80,head_max_positions=2048,training_questions=169028,training_scope='full169028_fresh_official_actor_not_pilot_resume',updates_per_epoch=1321,configured_local_batch=64,configured_micro=8,configured_accumulation=8,tail_local_batch=34,tail_global_batch=68,EMA_decay=.9995,lr=5e-6,init_seed=9137,train_plan_seed=2026102227,diagnostic_seed=2026102228,initial_head_sha256='c'*64,fixture=ref('fixture'),physical_source_contract=ref('physical-source-contract'))
 return dict(epoch=7,actor_updates=0,cursor=9247,batch_attempts=9247,actual_Adam_updates=9247,EMA_updates=9247,margin_satisfied_batches=0,epoch_complete=True,next_epoch=8,within_next_epoch_question_position=0,execution_phase='epoch_complete_pre_evaluation',raw={},EMA={},Adam={},RNG_by_rank=[{},{}],binding=binding,**{k:binding[k]for k in('init_seed','train_plan_seed','diagnostic_seed','initial_head_sha256')})
def descriptor():
 d=f.future_descriptor();d.update(status=f.STATUS,freeze_epoch=1,scorer_publication=ref('scorer'));d['configuration']=config()
 for i,r in enumerate(d['replications']):
  r.update(init_seed=9137+i*10000,train_plan_seed=[2026102227,2026102302][i],train_plan_sha256=str(i+1)*64,checkpoint_binding={})
  for key in('checkpoint','source_publication','training_complete','CPU_acceptance','viewed_validation_evaluation'):r[key]=ref(key+str(i))
  r['viewed_G5_identity_proof']=ref('same-viewed-G5')
 return d
class FullMetadataTests(unittest.TestCase):
 def test_full_geometry_and_epoch_boundary_pass_without_reads(self):
  with patch.object(f.old,'read',side_effect=AssertionError('No source/final reads in metadata tests')):
   self.assertEqual(f.dispatch(config()),'packed');self.assertEqual(f.checkpoint_metadata(blob(),config()),blob()['binding']);self.assertTrue(f.freeze_contract(descriptor()))
 def test_pilot_budget_or_incomplete_restore_rejected(self):
  for path,value in [('cursor',448),('epoch_complete',False),('next_epoch',7),('within_next_epoch_question_position',1),('actual_Adam_updates',9246),('RNG_by_rank',[{}])]:
   b=blob();b[path]=value
   with self.assertRaises(ValueError):f.checkpoint_metadata(b,config())
  b=blob();b['binding']['training_questions']=8192
  with self.assertRaises(ValueError):f.checkpoint_metadata(b,config())
 def test_physical_ref_and_seed_must_be_actual_binding(self):
  for key,value in [('physical_source_contract',None),('fixture',None)]:
   b=blob();b['binding'][key]=value
   with self.assertRaises(ValueError):f.checkpoint_metadata(b,config())
  b=blob();b['train_plan_seed']=1
  with self.assertRaises(ValueError):f.checkpoint_metadata(b,config())
 def test_future_template_stays_closed_and_full_config_is_explicit(self):
  with patch.object(f.old,'read',side_effect=AssertionError('No reads before freeze')):
   with self.assertRaises(ValueError):f.freeze_contract(f.future_descriptor())
   d=descriptor();d['configuration']['training_questions']=8192
   with self.assertRaises(ValueError):f.freeze_contract(d)
 def test_inference_geometry_and_score_semantics_unchanged(self):
  import ast,inspect
  from . import score_frozen_architectures_final4096 as original
  for name in('compact','score_replica','paired_summary'):
   self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(f,name)))),ast.dump(ast.parse(inspect.getsource(getattr(original,name)))))
if __name__=='__main__':unittest.main()
