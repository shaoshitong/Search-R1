"""CPU tests of the audit's finite-window theorem and no-grad/no-cache-write contract."""
import copy
import random
import unittest
import torch
from .group_topk_losses import precision_at_k_ramp
from .score_window_audit import audit_score_window


def entries_for(scores, labels):
    return [dict(qid='q%d'%i, trajectory_ids=['q%d-t%d'%(i,j)for j in range(5)],
                 labels=[1,0,0,0,0]if y else[0,0,0,0,0],true_mixed=bool(y),
                 cached_score=s,scored_step=10-i%4,scored_batch_attempt=20-i%4,score_model='raw')
            for i,(s,y)in enumerate(zip(scores,labels))]


class ScoreWindowAuditTest(unittest.TestCase):
    def test_identity_only_callback_no_grad_and_unchanged_cache(self):
        cache=entries_for([.8,.1,-.3,.2],[1,0,1,0]);before=copy.deepcopy(cache)
        parameter=torch.nn.Parameter(torch.tensor([.7,.2,-.1,.4],dtype=torch.float64));saved=parameter.detach().clone()
        def scorer(views):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(set(view)=={'qid','trajectory_ids','M'}for view in views))
            return parameter*1.0
        result=audit_score_window(cache,current_step=15,score_fn=scorer,k=2)
        self.assertTrue(torch.is_grad_enabled());self.assertEqual(cache,before)
        self.assertTrue(torch.equal(parameter,saved));self.assertIsNone(parameter.grad)
        self.assertAlmostEqual(result['max_abs_score_drift'],.2)
        self.assertTrue(result['loss_drift_bound_holds']);self.assertTrue(result['fresh_regret_bound_holds'])
        self.assertEqual(result['age_actual_updates_histogram'],{'5':1,'6':1,'7':1,'8':1})
        self.assertFalse(any(torch.is_tensor(value)for value in result.values()))
        self.assertNotIn('fresh_scores',result);self.assertNotIn('labels',result)

    def test_exact_oracle_when_positives_fewer_than_k_and_unbounded_loss(self):
        cache=entries_for([-20.,0.,1.,2.],[1,0,0,0])
        result=audit_score_window(cache,current_step=10,score_fn=lambda _:torch.tensor([-20.,0.,1.,2.]),k=3)
        self.assertEqual(result['oracle_precision'],1/3);self.assertEqual(result['fresh_P_at_K'],0)
        self.assertEqual(result['fresh_attainable_regret'],1/3)
        self.assertGreater(result['cached_L'],1)
        self.assertEqual(result['cached_L'],result['fresh_L'])
        self.assertEqual(result['fresh_P_at_K_lower_bound'],0)
        negative=entries_for([0.,1.,2.],[0,0,0])
        empty_positive=audit_score_window(negative,current_step=10,score_fn=lambda _:[0.,1.,2.],k=2)
        self.assertEqual(empty_positive['cached_L'],0)
        self.assertEqual(empty_positive['oracle_precision'],0)
        self.assertEqual(empty_positive['fresh_P_at_K'],0)

    def test_random_finite_drift_certificate_including_short_windows(self):
        rng=random.Random(20261023)
        for iteration in range(100):
            n=rng.randrange(1,24);k=rng.randrange(1,n+1);gamma=rng.choice([.2,1.,3.])
            old=[rng.uniform(-6,6)for _ in range(n)];new=[value+rng.uniform(-.4,.4)for value in old]
            labels=[rng.randrange(2)for _ in range(n)];cache=entries_for(old,labels)
            result=audit_score_window(cache,current_step=15,score_fn=lambda _,values=new:values,k=k,gamma=gamma,
                                      loss_fn=precision_at_k_ramp)
            self.assertTrue(result['loss_drift_bound_holds'],iteration)
            self.assertTrue(result['fresh_regret_bound_holds'],iteration)
            self.assertTrue(result['fresh_ramp_bounds_measured_regret'],iteration)
            self.assertTrue(result['cached_ramp_bounds_measured_regret'],iteration)
            self.assertGreaterEqual(result['fresh_P_at_K']+1e-10,result['fresh_P_at_K_lower_bound'])

    def test_empty_window_never_calls_forward(self):
        def never(_):raise AssertionError('Empty cache must not run inference')
        result=audit_score_window([],current_step=0,score_fn=never,k=0)
        self.assertEqual(result['status'],'EMPTY_CACHE_WINDOW_NO_FORWARD')

    def test_callback_cache_mutation_is_detected(self):
        cache=entries_for([0.,1.],[1,0])
        def corrupt(_):
            cache[0]['cached_score']=3.
            return [0.,1.]
        with self.assertRaisesRegex(RuntimeError,'cache changed'):
            audit_score_window(cache,current_step=10,score_fn=corrupt,k=1)

    def test_unrelated_same_name_loss_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'exact imported'):
            audit_score_window(entries_for([0.],[1]),current_step=10,score_fn=lambda _:[0.],k=1,
                               loss_fn=lambda *a,**k:0.)


if __name__=='__main__':unittest.main()
