"""Narrow CPU synthetic headLR-only controls, no assets or GPU."""
import copy,unittest
from unittest.mock import patch
import torch
from . import direct_count_headlr_train as t,probe_ablation_train as u
from .direct_count_headlr_update import bounded_update
from .test_direct_hidden_count import setup,Reader,forward,restore,hashes,update

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_only_declared_head_LR_changes_config(self):
  c=dict(source_cursor=0,source_adam_steps=10464,update_budget=938,global_questions=32,fit_questions=30016,
   new_head=dict(learning_rate=5e-6,EMA_decay=.99),grouping=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'),
   recipe=dict(id='hidden-direct-count6-from-E1',trajectory_BCE_weight=1.,count_NLL_weight=1.,product_group_loss_weight=0.),
   source_commit='501',code_hashes={},plan=dict(sha256='same'),evaluations=['same'],source_checkpoint={'sha256':'E1'})
  with patch.object(t.a,'proof',return_value={'path':'control','sha256':'fixed'}):new=t.configured(c,dict(commit='new',files={}))
  for key in c:
   if key not in ('new_head','source_commit','code_hashes'):self.assertEqual(c[key],new[key])
  self.assertEqual(new['new_head']['learning_rate'],1e-4);self.assertEqual(new['new_head']['EMA_decay'],.99)
 def test_real_toy_first_update_old_state_same_and_new_replay(self):
  values=setup();update(values);control=hashes(values);restore(values)
  m,e,o,h,he,ho,own,*_=values
  def candidate():return bounded_update(m,e,o,h,he,ho,Reader(),own,torch.device('cpu'),grouping_kwargs=dict(mode='random',views_per_question=4,seed=1729,epoch=0),micro_trajectories=3,forward_adapter=forward)
  with self.assertRaisesRegex(ValueError,'head1e-4'):candidate()
  for p in ho.param_groups:p['lr']=1e-4
  candidate();expected=hashes(values)
  self.assertEqual(control['legacy'],expected['legacy']);self.assertNotEqual(control['new']['raw'],expected['new']['raw'])
  restore(values)
  for p in ho.param_groups:p['lr']=1e-4
  candidate();self.assertEqual(hashes(values),expected);self.assertEqual(u.steps(o),10465)
if __name__=='__main__':unittest.main()
