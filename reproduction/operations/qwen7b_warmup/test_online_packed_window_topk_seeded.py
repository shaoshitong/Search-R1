"""Actual seed plumbing with 8192 identities, without rollout/model data."""
import tempfile,unittest
from pathlib import Path
from .online_packed_window_topk_seeded import epoch_plan,seed_model_rng,head_state_sha256,enforce_start_mode
from .online_packed_matched_objective import epoch_plan as original_epoch_plan
from .packed_variance_probe import PackedVarianceHead

class SeededCurrent64Test(unittest.TestCase):
 def test_actual_plan_owns_seed_and_changes_order_and_views(self):
  groups={str(i):dict(qid=str(i),trajectory_ids=[str(i)+'-t'+str(j)for j in range(10)],labels=[j%2 for j in range(10)])for i in range(8192)}
  old=epoch_plan(groups,1,5,train_plan_seed=2026102227);new=epoch_plan(groups,1,5,train_plan_seed=2026102302)
  self.assertEqual(old,original_epoch_plan(groups,1,5));self.assertEqual(new,epoch_plan(groups,1,5,train_plan_seed=2026102302))
  self.assertEqual(len(new),len({r['qid']for r in new}));self.assertEqual({r['qid']for r in old},set(groups));self.assertEqual({r['qid']for r in new},set(groups))
  self.assertNotEqual([r['qid']for r in old],[r['qid']for r in new]);a={r['qid']:r for r in old};b={r['qid']:r for r in new}
  self.assertGreater(sum(a[q]['trajectory_ids']!=b[q]['trajectory_ids']for q in a),8100)
  second=epoch_plan(groups,2,5,train_plan_seed=2026102302);self.assertNotEqual([r['qid']for r in new],[r['qid']for r in second])
  self.assertTrue(all(r['M']==5 and len(set(r['trajectory_ids']))==5 for r in new))

 def test_fresh_head_seed_changes_and_repeats(self):
  seed_model_rng(9137);old=head_state_sha256(PackedVarianceHead(max_positions=2048))
  seed_model_rng(19137);new=head_state_sha256(PackedVarianceHead(max_positions=2048))
  seed_model_rng(19137);again=head_state_sha256(PackedVarianceHead(max_positions=2048))
  self.assertNotEqual(old,new);self.assertEqual(new,again)

 def test_fresh_is_not_implicit_resume(self):
  with tempfile.TemporaryDirectory()as tmp:
   enforce_start_mode(tmp,'fresh')
   with self.assertRaisesRegex(ValueError,'requires this replica'):enforce_start_mode(tmp,'resume')
   (Path(tmp)/'latest.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'cannot resume'):enforce_start_mode(tmp,'fresh')
   enforce_start_mode(tmp,'resume')

if __name__=='__main__':unittest.main()
