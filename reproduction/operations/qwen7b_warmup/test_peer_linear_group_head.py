"""Small synthetic-only CPU tests; no real tensor/data artifact local."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from . import peer_linear_group_head as m

class LinearGroupHeadTests(unittest.TestCase):
    def test_policy_mean_last_real_length_and_zero(self):
        values=np.asarray([[1.,2.],[3.,4.]],dtype='<f4');payload=(values.view('<u4')>>16).astype('<u2').tobytes();p=m.pool_policy(payload,2,2)
        np.testing.assert_array_equal(p['prefix_mean'],[2.,3.]);np.testing.assert_array_equal(p['prefix_last'],[3.,4.]);np.testing.assert_array_equal(m.pool_policy(b'',0,2)['prefix_last'],[0.,0.])
        with self.assertRaises(ValueError):m.pool_policy(payload[:-2],2,2)
    def test_group_permutation_invariance_and_real_five(self):
        x=np.arange(30,dtype=np.float32).reshape(10,3);lens=np.arange(10);a=m.group_vector(x,[0,1,2,4,8],lens);b=m.group_vector(x,[8,0,4,2,1],lens);np.testing.assert_allclose(a,b);self.assertEqual(len(a),9)
    def test_whole_qid_split_deterministic_not_label_dependent(self):
        values=[m.split_question('q%d'%i) for i in range(200)];self.assertEqual(values,[m.split_question('q%d'%i) for i in range(200)]);self.assertGreater(sum(values),20);self.assertLess(sum(values),70)
        for i in range(200):self.assertEqual(len({m.split_question('q%d'%i) for _ in range(4)}),1)
    def test_auc_ties_threshold_and_top512_metrics(self):
        s=np.asarray([.1,.2,.3,.4,.5,.6]);k=np.asarray([0,5,1,4,2,3]);q=list('abcdef');v=m.metric(s,k,q,.3,4);self.assertEqual(v['balanced_accuracy'],1);self.assertEqual(v['AUC'],1);self.assertEqual(v['Top4']['Mixed'],4);self.assertEqual(v['allwrong_FPR'],0);self.assertEqual(v['allcorrect_FPR'],0)
        t=m.choose_threshold(s,k);self.assertEqual(m.metric(s,k,q,t)['balanced_accuracy'],1);self.assertEqual(m.metric(np.zeros(6),k,q,0)['AUC'],.5)
    def test_logistic_real_update_finds_linear_synthetic_signal(self):
        x=np.asarray([[-2,-1],[-1,-2],[-1,0],[0,1],[1,0],[2,1]],dtype=np.float32);y=np.asarray([0,0,0,1,1,1],dtype=np.float32);history=[];w,b,fit=m.fit_logistic(x,y,.01,lambda *v:history.append(v),max_iter=40);self.assertLess(fit['final_regularized_loss'],fit['initial_regularized_loss']);self.assertGreater(m.metric(x@w+b,np.asarray([0,5,0,1,2,3]),list('abcdef'),0)['AUC'],.9);self.assertTrue(history)
    def test_streaming_standardizer_matches_training_only_moments(self):
        x=np.random.default_rng(1729).normal(size=(1300,12)).astype(np.float32);x[:,0]=1.;mean,scale=m.standardizer(x);np.testing.assert_allclose(mean,x.mean(axis=0),atol=1e-6);expected=x.std(axis=0);expected[0]=1.;np.testing.assert_allclose(scale,expected,atol=1e-6)
    def test_mmap_bounded_take_preserves_order_and_data(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.lib.format.open_memmap(Path(d)/'synthetic.npy',mode='w+',shape=(1100,7),dtype=np.float32)
            try:
                a[:]=np.arange(7700).reshape(1100,7)
                if not hasattr(a._mmap,'madvise'):self.skipTest('Linux madvise-only production path')
                actual=m.take_rows(a,np.arange(1100)%5==0);np.testing.assert_array_equal(actual,np.arange(7700).reshape(1100,7)[np.arange(1100)%5==0])
            finally:a._mmap.close()
    def test_grid_budget_is_finite_and_two_methods_only(self):
        self.assertEqual(m.METHODS,('prefix_mean','prefix_last'));self.assertEqual(m.GRID,(.001,.01,.1));self.assertLess(m.LIMIT_SECONDS,30*60)

if __name__=='__main__':unittest.main()
