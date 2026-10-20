"""Focused mathematical tests on synthetic data only."""
import itertools
import io
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from . import platt_g10_calibration as m
from .g5_event_scores_dp import group_event_scores_dp
from .probe_ablation_loss import group_event_scores


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_dp_equals_32_patterns_value_and_gradient(self):
        z = (torch.randn(17, 5, generator=torch.Generator().manual_seed(11), dtype=torch.float64)*4).requires_grad_()
        old, new = group_event_scores(z), group_event_scores_dp(z)
        for key in old:
            torch.testing.assert_close(new[key], old[key], atol=2e-14, rtol=2e-14)
        a = torch.autograd.grad(sum(x.sum() for x in old.values()), z, retain_graph=True)[0]
        b = torch.autograd.grad(sum(x.sum() for x in new.values()), z)[0]
        torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-13)

    def test_dp_extremes_and_half_probabilities(self):
        z = torch.tensor([[0.]*5, [1000.]*5, [-1000.]*5, [1000., -1000., 1000., -1000., 1000.]], dtype=torch.float64, requires_grad=True)
        new = group_event_scores_dp(z)
        self.assertAlmostEqual(float(new['mixed_probability'][0].detach()), .9375)
        for value in new.values(): self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(torch.logsumexp(torch.stack([new['log_allwrong'], new['log_mixed'], new['log_allcorrect']], 1), 1), torch.zeros(4,dtype=torch.float64), atol=1e-14, rtol=1e-14)
        gradient = torch.autograd.grad(new['log_mixed'].sum(), z)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_all252_objective_and_calibration_gradient_against_brute(self):
        z = torch.randn(2,10,generator=torch.Generator().manual_seed(32),dtype=torch.float64)
        y = torch.tensor([[0,0,1,0,1,0,1,0,1,1],[1,1,1,1,1,1,1,1,1,1]],dtype=torch.float64)
        raw = torch.tensor([.13,-.07],dtype=torch.float64,requires_grad=True)
        transformed = m.calibrated(z,raw); terms=[]
        for i in range(2):
            for ix in itertools.combinations(range(10),5):
                vz=transformed[i,list(ix)].reshape(1,5);vy=y[i,list(ix)].reshape(1,5)
                scores=group_event_scores(vz);k=int(vy.sum())
                key='log_allwrong' if k==0 else 'log_allcorrect' if k==5 else 'log_mixed'
                terms.append(F.binary_cross_entropy_with_logits(vz,vy)-scores[key].mean())
        expected=torch.stack(terms).mean();actual=m.chunk_objective(z,y,raw,'trajectory+three-event')
        torch.testing.assert_close(actual,expected,atol=2e-14,rtol=2e-14)
        g1=torch.autograd.grad(expected,raw,retain_graph=True)[0];g2=torch.autograd.grad(actual,raw)[0]
        torch.testing.assert_close(g1,g2,atol=2e-13,rtol=2e-13)

    def test_question_split_and_fixed5_independent_of_labels(self):
        qids=['q%d'%i for i in range(8192)];values=m.question_split(qids)
        self.assertEqual(int(values.sum()),2048)
        reverse=m.question_split(list(reversed(qids)))
        np.testing.assert_array_equal(values,reverse[::-1])
        for i in range(100):
            q='q%d'%i;ix=m.fixed_indices(q)
            self.assertEqual(len(set(ix)),5);self.assertEqual(ix,sorted(ix));self.assertEqual(ix,m.fixed_indices(q))
            self.assertEqual(len({bool(values[i]) for _ in range(252)}),1)

    def test_real_two_parameter_fit_reduces_shifted_trajectory_loss(self):
        rng=np.random.default_rng(11);z=rng.normal(size=(64,10))-1.
        p=1/(1+np.exp(-(2*z+2)));y=(rng.random(z.shape)<p).astype(np.float64)
        before=float(m.chunk_objective(torch.tensor(z),torch.tensor(y),torch.zeros(2,dtype=torch.float64),'trajectory'))
        history=[];raw,report=m.fit(z,y,'trajectory',0.,lambda *x:history.append(x),max_iter=25,max_eval=40)
        after=float(m.chunk_objective(torch.tensor(z),torch.tensor(y),raw,'trajectory'))
        self.assertLess(after,before-.05);self.assertTrue(history);self.assertGreater(report['optimizer_iterations'],0)
        self.assertTrue(float(m.parameters(raw)[0])>0)

    def test_identical_probability_interface_and_monotonicity(self):
        p=np.asarray([0.,1e-8,.01,.2,.8,.99,1.-1e-8,1.]);z=m.probability_logits(p)
        self.assertTrue(np.isfinite(z).all());self.assertTrue((np.diff(z)>=0).all())
        calibrated=m.calibrated(torch.tensor(z),torch.tensor([.2,-.1],dtype=torch.float64)).sigmoid().numpy()
        self.assertTrue((np.diff(calibrated)>=0).all())
        with self.assertRaises(ValueError):m.probability_logits([np.nan])
        with self.assertRaises(ValueError):m.probability_logits([-1.])

    def test_unclipped_baseline_metrics_preserved(self):
        p=np.asarray([[0.,0.,0.,0.,0.],[1.,1.,1.,1.,1.],[1.,0.,1.,0.,1.],[.5,.5,.5,.5,.5]])
        y=np.asarray([[0]*5,[1]*5,[1,0,1,0,1],[0,1,0,1,0]])
        s=m.raw_probability_summary(p,y,list('abcd'))
        self.assertEqual(s['balanced_accuracy'],1.);self.assertEqual(s['AUC'],1.)

    def test_finite_registered_search(self):
        self.assertEqual(len(m.COMBINATIONS),252);self.assertEqual(len(m.OBJECTIVES)*len(m.REGULARIZERS),6)
        self.assertEqual(m.MAX_ITER,40);self.assertLess(m.LIMIT_SECONDS,30*60)

    def test_full_LBFGS_checkpoint_restores_same_next_update(self):
        z=np.random.default_rng(17).normal(size=(32,10));y=(z>.3).astype(np.float64);states=[]
        m.fit(z,y,'trajectory',.01,lambda *v:None,max_iter=2,max_eval=5,
              checkpoint_callback=lambda state: states.append(state) or {'synthetic_checkpoint':True})
        stream=io.BytesIO();torch.save(states[0],stream);stream.seek(0)
        loaded=torch.load(stream,weights_only=False)
        self.assertTrue(m.same_state(states[0],loaded))
        outputs=[]
        for state in (states[0],loaded):
            raw=state['raw'].clone().requires_grad_();opt=torch.optim.LBFGS([raw]);opt.load_state_dict(state['optimizer'])
            def closure():
                opt.zero_grad();loss=m.chunk_objective(torch.tensor(z),torch.tensor(y),raw,'trajectory')
                _,b,loga=m.parameters(raw);loss=loss+.01*(loga.square()+b.square());loss.backward();return loss
            opt.step(closure);outputs.append((raw.detach().clone(),opt.state_dict()))
        self.assertTrue(m.same_state(outputs[0],outputs[1]))


if __name__=='__main__':unittest.main()
