"""Small metadata-only final architecture tests; no final path is opened."""
import copy,unittest
from unittest.mock import patch
from . import score_frozen_architectures_final4096 as f

def config(family='packed_topk'):
 return dict(family=family,layer=15,prefix=80,architecture='one_question_packed_global64_detached_window_topk_v1',objective='topk_ramp',head_max_positions=2048,fixed_M=5,epoch=8,training_budget_epochs=10,primary_state='raw',secondary_states=['EMA'],primary_metric='Top512_true_Mixed_fraction',threshold=.5,threshold_fitted=False)
def ref(tag):return dict(path=str(f.B)+'/'+tag,bytes=1,sha256='a'*64)
def descriptor():
 replicas=[]
 for i in range(2):replicas.append(dict(seed_id='seed'+str(i+1),init_seed=9137+i*10000,train_plan_seed=2026102227+i,train_plan_sha256=str(i+1)*64,checkpoint_binding={},**{name:ref(name+str(i))for name in('checkpoint','source_publication','training_complete','CPU_acceptance','viewed_validation_evaluation')},viewed_G5_identity_proof=ref('fixed-same-G5')))
 return dict(status=f.STATUS,final_labels_or_scores_used_for_selection=False,further_candidate_selection_on_final_allowed=False,two_original_2048_G5_plans_unchanged=True,primary_validation_questions=2048,freeze_epoch=1,baseline=dict(checkpoint=f.old.E1,state_key='ema',layer=27,prefix=20),configuration=config(),replications=replicas,epoch_state_selection_rule='seed1_viewed_Top512_then_same_epoch_state_for_both_seeds',selection_seed_id='seed1',scorer_publication=ref('scorer'))
class MetadataOnlyTests(unittest.TestCase):
 def test_dispatch_exact_native_and_packed(self):
  for layer in(15,18,21):self.assertEqual(f.dispatch(dict(family='native_direct',layer=layer,prefix=80,architecture='NativeMixedGroupProbe',objective='mixed_bce')),'native')
  self.assertEqual(f.dispatch(config()),'packed')
  for family,obj in(('packed_variance','mse'),('packed_direct','mixed_bce')):
   c=config();c.update(family=family,objective=obj,architecture='one_question_packed_matched_objective_v1');self.assertEqual(f.dispatch(c),'packed')
  self.assertEqual(f.dispatch(dict(family='packed_variance',layer=21,prefix=20,head_max_positions=1024,architecture='one_question_M3to10_policy20_packed_variance_v1',objective='z_mse_anchor')),'legacy_packed')
 def test_malformed_geometry_and_head_rejected(self):
  for key,val in(('prefix',20),('layer',18),('head_max_positions',1024),('objective','mixed_bce')):
   c=config();c[key]=val
   with self.assertRaises(ValueError):f.dispatch(c)
 def test_freeze_gate_never_reads_and_requires_two_distinct_plans(self):
  with patch.object(f.old,'read',side_effect=AssertionError('No reads in metadata gate')):
   self.assertTrue(f.freeze_contract(descriptor()))
   for mutation in('future','same_seed','same_plan','different_G5'):
    d=descriptor()
    if mutation=='future':d['status']='FUTURE_ONLY_NOT_FROZEN'
    elif mutation=='same_seed':d['replications'][1].update(init_seed=d['replications'][0]['init_seed'],train_plan_seed=d['replications'][0]['train_plan_seed'])
    elif mutation=='same_plan':d['replications'][1]['train_plan_sha256']=d['replications'][0]['train_plan_sha256']
    else:d['replications'][1]['viewed_G5_identity_proof']=ref('other')
    with self.assertRaises(ValueError):f.freeze_contract(d)
 def test_actual_cp_binding_strict_and_state_keys(self):
  c=config();blob=dict(actor_updates=0,epoch=8,raw={},EMA={},binding=dict(architecture=c['architecture'],layer=15,prefix=80,head_max_positions=2048,objective='topk_ramp',cache_window_capacity=2018));self.assertEqual(f.checkpoint_metadata(blob,c),blob['binding']);bad=copy.deepcopy(blob);bad['binding']['head_max_positions']=1024
  with self.assertRaises(ValueError):f.checkpoint_metadata(bad,c)
  bad=copy.deepcopy(blob);del bad['EMA']
  with self.assertRaises(ValueError):f.checkpoint_metadata(bad,c)
 def test_actual_legacy_source_seed_extraction(self):
  self.assertEqual(f.literal_seeds('SEED=2026102227\ndef run():\n torch.manual_seed(9137)\n'),(9137,2026102227))
  with self.assertRaises(ValueError):f.literal_seeds('SEED=1\ndef run():\n torch.manual_seed(9137)\n torch.manual_seed(19137)\n')
if __name__=='__main__':unittest.main()
