"""Small receipt fixtures only; never starts or tests a training process."""
import copy
from pathlib import Path
import runpy
import unittest

READER = Path(__file__).parent/'once_records_20261018/read_progressive_period_pair.py.txt'
API = runpy.run_path(str(READER))


class PeriodPairAdoptionTests(unittest.TestCase):
    def fixture(self, n=100):
        before = dict(run_id='B', completed_outer=n,
                      actual_adam_steps_per_rank=[1400]*4, learning_rate=5e-6,
                      ema_updates=1400, selection_model_kind='ema', sha256='old')
        after = dict(before, sha256='expanded')
        migration = dict(before=before, after=after, migration=dict(
            old_prefix=API['stage_prefix'](n), new_prefix=API['stage_prefix'](n+1),
            block_reset=False, optimizer_reset=False, ema_reset=False,
            rank_rng_reset=False, new_linear_coordinates_zero=True))
        joint = dict(probe=before, actor=dict(actor_version='actor%d'%n))
        adoption = dict(outer=n+1, actor_version='actor%d'%n,
                        selection_model_kind='ema', outcome_labels_used=False,
                        probe_before_checkpoint_sha256='expanded', ema_updates=1400)
        return joint, adoption, migration

    def test_unchanged_stage_uses_exact_completed_payload(self):
        j, a, _ = self.fixture(90)
        a['probe_before_checkpoint_sha256']='old'
        self.assertEqual(API['following_probe'](j,a,90)['adopted_payload_sha256'],'old')

    def test_both_dimension_boundaries_follow_the_bound_new_payload(self):
        for n in (100,150):
            j,a,m=self.fixture(n)
            result=API['following_probe'](j,a,n,m)
            self.assertEqual(result['adopted_payload_sha256'],'expanded')
            self.assertIn('NOT_TENSOR_READBACK',result['evidence_kind'])

    def test_dimension_change_cannot_use_an_unrelated_receipt(self):
        j,a,m=self.fixture()
        with self.assertRaises(ValueError):API['following_probe'](j,a,100)
        for target,key,value in [('before','sha256','unrelated'),
                                 ('after','ema_updates',0),
                                 ('after','run_id','other'),
                                 ('after','actual_adam_steps_per_rank',[0]*4),
                                 ('migration','optimizer_reset',True),
                                 ('migration','new_prefix',80)]:
            altered=copy.deepcopy(m);altered[target][key]=value
            with self.assertRaises(ValueError):API['following_probe'](j,a,100,altered)

    def test_actual_scoring_must_use_expanded_ema_without_outcomes(self):
        j,a,m=self.fixture()
        for key,value in [('probe_before_checkpoint_sha256','old'),
                          ('outcome_labels_used',True),('selection_model_kind','raw'),
                          ('ema_updates',0),('actor_version','actor99'),('outer',102)]:
            altered=dict(a,**{key:value})
            with self.assertRaises(ValueError):API['following_probe'](j,altered,100,m)


if __name__=='__main__':unittest.main()
