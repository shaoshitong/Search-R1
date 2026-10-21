"""Narrow metadata-only inherited fullstate and frozen real-G5 identity tests."""
import copy,unittest
from .audit_direct_hidden_terminal import inherit_state,frozen_groups,KEYS
class Tests(unittest.TestCase):
 def test_receipt_and_file_binding(self):
  actual=dict(path='/remote/cp.pt',bytes=7,sha256='a'*64)
  saved=dict(actual,cursor=938,legacy_Adam_steps=11402,new_head_Adam_steps=938,
   full_CPU_reload_passed=True,full_file_SHA256_verified=True,source_all_four_RNG_preserved=True,
   accepted_hashes={k:'b'*64 for k in KEYS})
  receipt=dict(checkpoint=saved,before_nine_component_hashes=saved['accepted_hashes'],after_nine_component_hashes=saved['accepted_hashes'],training_nine_components_unchanged=True)
  out=inherit_state(saved,dict(checkpoint=saved),receipt,actual)
  self.assertFalse(out['fresh_checkpoint_tensor_reload'])
  bad=copy.deepcopy(receipt);bad['after_nine_component_hashes']['rng']='c'*64
  with self.assertRaises(ValueError):inherit_state(saved,dict(checkpoint=saved),bad,actual)
  with self.assertRaises(ValueError):inherit_state(saved,dict(checkpoint=saved),receipt,dict(actual,sha256='d'*64))
 def test_fixed_real_ids_only(self):
  groups=[dict(qid='q',trajectory_ids=list('abcdefghij'),labels=[0,1]*5)]
  plan=dict(summary=dict(role='validation',labels_consulted=False),groups=[dict(qid='q',trajectory_ids=list('fghij'))])
  self.assertEqual(frozen_groups(groups,plan)[0]['labels'],[1,0,1,0,1])
  plan['summary']['labels_consulted']=True
  with self.assertRaises(ValueError):frozen_groups(groups,plan)
if __name__=='__main__':unittest.main()
