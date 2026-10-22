"""CPU-only seed plumbing and fresh-start boundary checks; no model asset read."""
import json
from pathlib import Path
import tempfile
import unittest
from .packed_variance_probe import PackedVarianceHead,plan
from .online_packed_variance_seeded import seed_model_rng,head_state_sha256,training_plan,enforce_start_mode


def groups_fixture():
 return {'qid%04d'%i:dict(qid='qid%04d'%i,trajectory_ids=['qid%04d-t%d'%(i,j)for j in range(10)],labels=[j%2 for j in range(10)])for i in range(320)}


class SeededVarianceTest(unittest.TestCase):
 def test_explicit_seed_reaches_actual_question_order_and_own_views(self):
  groups=groups_fixture();first=training_plan(groups,1,2026102227);second=training_plan(groups,1,2026102302)
  self.assertEqual(first,plan(groups,1,seed=2026102227));self.assertEqual(second,training_plan(groups,1,2026102302))
  self.assertNotEqual([r['qid']for r in first],[r['qid']for r in second])
  a={r['qid']:r for r in first};b={r['qid']:r for r in second}
  self.assertGreater(sum(a[q]['trajectory_ids']!=b[q]['trajectory_ids']for q in a),300)
  self.assertGreater(sum(a[q]['M']!=b[q]['M']for q in a),200)
  self.assertEqual(set(a),set(b));self.assertEqual(len(second),len({r['qid']for r in second}))
  third=training_plan(groups,2,2026102302);self.assertNotEqual([r['qid']for r in second],[r['qid']for r in third])

 def test_new_initialization_seed_changes_head_and_is_repeatable(self):
  seed_model_rng(9137);one=head_state_sha256(PackedVarianceHead())
  seed_model_rng(19137);two=head_state_sha256(PackedVarianceHead())
  seed_model_rng(19137);again=head_state_sha256(PackedVarianceHead())
  self.assertNotEqual(one,two);self.assertEqual(two,again)

 def test_fresh_mode_cannot_silently_resume(self):
  with tempfile.TemporaryDirectory()as tmp:
   enforce_start_mode(tmp,'fresh')
   with self.assertRaisesRegex(ValueError,'requires this replica'):enforce_start_mode(tmp,'resume')
   (Path(tmp)/'latest.json').write_text(json.dumps({'not_a_real_checkpoint':True}))
   with self.assertRaisesRegex(ValueError,'cannot resume'):enforce_start_mode(tmp,'fresh')
   enforce_start_mode(tmp,'resume')


if __name__=='__main__':unittest.main()
