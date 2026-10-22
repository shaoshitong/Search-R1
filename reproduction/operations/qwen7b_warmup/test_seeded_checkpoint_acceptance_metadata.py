"""Only exact seed/parent/plan integration; no checkpoint/model tensors loaded."""
import unittest
from pathlib import Path
from .accept_seeded_matched_window_checkpoint import A,SEED2_PARENTS,EXPECTED_SEEDS,seed_recipe,verify_seed_stamp,actual_epoch_plan
class SeededAcceptanceMetadata(unittest.TestCase):
 def binding(self,batch,capacity):
  v=dict(EXPECTED_SEEDS,seed=2026102302,seed_source='explicit_CLI',fixed_m=5,global_batch=batch,head_max_positions=2048,initial_head_sha256='a'*64,objective='mse'if capacity is None else'topk_ramp')
  if capacity is not None:v['cache_window_capacity']=capacity
  return v
 def test_exact_actual_parents_and_seed_stamps(self):
  for parent,(module,cell,batch,capacity)in SEED2_PARENTS.items():
   binding=self.binding(batch,capacity);recipe=seed_recipe(A/parent/cell,binding);self.assertTrue(recipe['module'].endswith(module));verify_seed_stamp(dict(EXPECTED_SEEDS,initial_head_sha256='a'*64),recipe,True)
   with self.assertRaises(AssertionError):seed_recipe(A/parent/'wrong-cell',binding)
   with self.assertRaises(AssertionError):seed_recipe(A/parent/cell,dict(binding,train_plan_seed=2026102227))
   with self.assertRaises(AssertionError):verify_seed_stamp(dict(EXPECTED_SEEDS,initial_head_sha256='b'*64),recipe,True)
   with self.assertRaises(KeyError):verify_seed_stamp({},recipe,True)
 def test_selected_runtime_reproduces_explicit_seed_plan(self):
  groups={str(i):dict(qid=str(i),trajectory_ids=[str(i)+'-t'+str(j)for j in range(10)],labels=[j%2 for j in range(10)])for i in range(8192)}
  plans=[]
  for parent,(module,cell,batch,capacity)in SEED2_PARENTS.items():
   binding=self.binding(batch,capacity);recipe=seed_recipe(A/parent/cell,binding);p=actual_epoch_plan(groups,1,binding,recipe);self.assertEqual(len({r['qid']for r in p}),8192);self.assertEqual(p,actual_epoch_plan(groups,1,binding,recipe));plans.append(p)
  self.assertEqual(plans[0],plans[1]);self.assertEqual(plans[0],plans[2])
  from .online_packed_matched_objective import epoch_plan as old_plan
  self.assertNotEqual(plans[0],old_plan(groups,1,5))
if __name__=='__main__':unittest.main()
