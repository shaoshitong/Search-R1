import copy,unittest
from . import peer_newseed_frozen_subset_export as f

def fixture(n=2):
 rows=[dict(input='input-%03d'%i,questions=2,trajectories=10,audit=dict(status='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND')) for i in range(n)]
 return dict(status='FROZEN_SOURCE_AUDITED%d_AFTER_ACCEPTED3_METADATA_ONLY'%n,subset='existing-audited',source_base=str(f.BASE),source_run_id=f.e.RUN_ID,seed=f.e.SEED,model_revision=f.e.REVISION,sampling_attempt_id=f.e.ATTEMPT,batches=n,questions=2*n,trajectories=10*n,domain_counts={'hotpotqa':2*n},selected_batches=rows,batch_records_sha256=f.digest(rows),prior_snapshots=[dict(complete=dict(path='/source/prior1'),expected_batches=1,expected_questions=2,expected_trajectories=10),dict(complete=dict(path='/source/prior2'),expected_batches=2,expected_questions=4,expected_trajectories=20)])
class FrozenDeltaTests(unittest.TestCase):
 def test_variable_explicit_counts_not_historical_budget(self):
  for n in (1,2,7):self.assertEqual(f.validate_selection(fixture(n))['questions'],n*2)
 def test_record_digest_and_counts_cannot_drift(self):
  for key,value in [('questions',99),('trajectories',99),('batches',99),('batch_records_sha256','a'*64),('status','FROZEN_SOURCE_AUDITED2_AFTER_ACCEPTED244_METADATA_ONLY')]:
   v=fixture();v[key]=value
   with self.assertRaises(ValueError):f.validate_selection(v)
 def test_sorted_proof_vs_ordered_plan_at_decimal_width_boundary(self):
  ids=['run-c%04d-g%d'%(i,j) for i in (99998,99999,100000,100001) for j in range(5)]
  self.assertNotEqual(f.digest(ids),f.digest(sorted(ids)))
  f.verify_id_digests(ids,f.digest(ids),f.digest(sorted(ids)))
  with self.assertRaises(ValueError):f.verify_id_digests(ids,f.digest(ids),f.digest(ids))
  with self.assertRaises(ValueError):f.verify_id_digests(ids+ids[:1],f.digest(ids+ids[:1]),f.digest(sorted(ids+ids[:1])))
 def test_prior_and_own_identity_recurrence_rejected(self):
  old=dict(qids={'old'},hashes={'oldh'},tids={'oldt'});seen=dict(qids=set(),hashes=set(),tids=set());g=dict(qid='new',question_hash='newh',trajectory_ids=['t%d'%i for i in range(5)])
  f.identity_guard(old,[g],seen)
  with self.assertRaises(ValueError):f.identity_guard(old,[g],seen)
  for field,value in [('qid','old'),('question_hash','oldh'),('trajectory_ids',['oldt','a','b','c','d'])]:
   bad=copy.deepcopy(g);bad[field]=value
   with self.assertRaises(ValueError):f.identity_guard(old,[bad],dict(qids=set(),hashes=set(),tids=set()))
if __name__=='__main__':unittest.main()
