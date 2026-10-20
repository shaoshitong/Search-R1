"""Synthetic tensor fixtures: execute only in the remote existing CPU runtime."""
import copy
import io
import itertools
import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from . import prefix_confidence_calibration as m
from .probe_ablation_loss import group_event_scores


def features(q=4):
    rng=np.random.default_rng(31);lp=-rng.uniform(.01,4,size=(q,10,20))
    return np.stack((lp.mean(-1),lp.std(-1),lp.min(-1),lp[...,:5].mean(-1),lp[...,-5:].mean(-1),np.ones((q,10))),-1)


def binding(mode='prefix_full8'):
    return dict(qualification_only=True,availability_qualified=True,
                availability_receipt=dict(path='synthetic-only',bytes=0,sha256='0'*64),source_kind='CPU_TOY_NO_REAL_DATA',calibration_mode=mode)


class PrefixCalibrationTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_identity_feature_normalizer_and_fit_only_split(self):
        x=features(4);n=m.normalizer_fit(x,list('abcd'));z=torch.randn(4,10,dtype=torch.float64)
        self.assertTrue(torch.equal(m.calibrated(z,x,torch.zeros(8,dtype=torch.float64),n),z))
        np.testing.assert_allclose(n['mean'].numpy(),x.reshape(-1,6).mean(0),rtol=1e-14)
        self.assertEqual(float(n['scale'][5]),1e-6)
        mask=m.question_split(['q%d'%i for i in range(8192)])
        self.assertEqual(mask.sum(),2048)
        np.testing.assert_array_equal(mask,m.question_split(['q%d'%i for i in reversed(range(8192))])[::-1])
        with self.assertRaises(ValueError):m.normalizer_fit(x,['a','a','b','c'])

    def test_all252_value_gradient_against_independent32_patterns(self):
        x=features(2);n=m.normalizer_fit(x,['a','b']);z=torch.randn(2,10,generator=torch.Generator().manual_seed(7),dtype=torch.float64)
        y=torch.tensor([[0,0,1,0,1,0,1,1,0,1],[1]*10],dtype=torch.float64)
        raw=torch.tensor([.1,-.03,.04,-.02,.01,.03,-.02,.06],dtype=torch.float64,requires_grad=True)
        out=m.calibrated(z,x,raw,n);terms=[]
        for i in range(2):
            for ix in itertools.combinations(range(10),5):
                vz=out[i,list(ix)].reshape(1,5);vy=y[i,list(ix)].reshape(1,5);k=int(vy.sum())
                events=group_event_scores(vz);key='log_allwrong' if k==0 else 'log_allcorrect' if k==5 else 'log_mixed'
                terms.append(F.binary_cross_entropy_with_logits(vz,vy)-events[key].mean())
        want=torch.stack(terms).mean();got=m.chunk_objective(z,y,x,raw,n)
        torch.testing.assert_close(got,want,atol=3e-14,rtol=3e-14)
        a=torch.autograd.grad(got,raw,retain_graph=True)[0];b=torch.autograd.grad(want,raw)[0]
        torch.testing.assert_close(a,b,atol=5e-13,rtol=5e-13)

    def test_real_eight_scalar_fit_full_state_reload_and_same_next_update(self):
        x=features(24);z=np.random.default_rng(3).normal(size=(24,10))
        y=(z+.8*(x[...,0]-x[...,0].mean())>0).astype(float);states=[]
        state,report=m.fit(z,y,x,['q%d'%i for i in range(24)],binding(),qualification_only=True,max_iter=3,max_eval=12,
            checkpoint_callback=lambda s:states.append(s) or dict(synthetic_durable=True))
        self.assertEqual(state['phase'],m.PHASE);self.assertFalse(state['holdout_assessed']);self.assertEqual(len(states),1)
        self.assertLess(report['accepted_final_regularized_loss'],report['initial_loss'])
        stream=io.BytesIO();torch.save(state,stream);stream.seek(0);loaded=torch.load(stream,weights_only=False)
        self.assertTrue(m.same_state(state,loaded));results=[]
        for source in (state,loaded):
            restored=m.restore(source);raw=restored['raw'];opt=restored['optimizer'];n=restored['normalizer']
            def closure():
                opt.zero_grad();loss=m.chunk_objective(z,y,x,raw,n)+m.regularization(raw);loss.backward();return loss
            opt.step(closure)
            results.append(dict(raw=raw.detach().clone(),optimizer=opt.state_dict(),rng=m.capture_rng(),
                                draws=[random.random(),float(np.random.random()),float(torch.rand(()))]))
        self.assertTrue(m.same_state(results[0],results[1]))

    def test_availability_real_budget_shape_and_feature_fail_closed(self):
        x=features(2);z=np.zeros((2,10));y=np.zeros_like(z)
        with self.assertRaisesRegex(ValueError,'qualified'):m.fit(z,y,x,['a','b'],{})
        with self.assertRaisesRegex(ValueError,'6144'):m.fit(z,y,x,['a','b'],dict(binding(),qualification_only=False))
        bad=x.copy();bad[...,5]=0
        with self.assertRaises(ValueError):m.normalizer_fit(bad,['a','b'])
        bad=x.copy();bad[...,0]=np.nan
        with self.assertRaises(ValueError):m.normalizer_fit(bad,['a','b'])
        n=m.normalizer_fit(x,['a','b'])
        with self.assertRaises(ValueError):m.chunk_objective(z,np.full_like(z,.5),x,torch.zeros(8),n)

    def test_identity_clipping_interface_and_frozen_g5_inference(self):
        p=np.array([[0,1,.2,.5,.9]]*4);z=m.probability_logits(p)
        state=dict(raw=torch.zeros(8,dtype=torch.float64),normalizer=m.normalizer_fit(features(4),list('abcd')),use_prefix=True)
        x=features(4)[:,:5];scores=m.score(z,x,state)
        pp=np.clip(p,1e-6,1-1e-6);expected=1-pp.prod(1)-(1-pp).prod(1)
        np.testing.assert_allclose(scores,expected,atol=2e-15,rtol=2e-15)
        self.assertTrue(np.isfinite(z).all());self.assertEqual(m.MAX_ITER,40);self.assertEqual(m.MAX_EVAL,60)

    def test_matched2_ignores_features_identity_mode_binding_and_timeout_state(self):
        x=features(8);z=np.random.default_rng(4).normal(size=(8,10));y=(z>.2).astype(float);qids=['q%d'%i for i in range(8)]
        raw=torch.zeros(8,dtype=torch.float64)
        self.assertTrue(torch.equal(m.calibrated(z,None,raw,None,False),torch.tensor(z)))
        state,report=m.fit(z,y,None,qids,binding('matched_platt2'),use_prefix=False,qualification_only=True,max_iter=2,max_eval=8)
        self.assertIsNone(state['normalizer']);self.assertTrue(torch.equal(state['raw'][2:],torch.zeros(6,dtype=torch.float64)))
        restored=m.restore(state);self.assertFalse(restored['use_prefix'])
        bad=copy.deepcopy(state);bad['mode']='prefix_full8'
        with self.assertRaisesRegex(ValueError,'mode'):m.restore(bad)
        partial=[]
        with self.assertRaises(TimeoutError):
            m.fit(z,y,x,qids,binding(),qualification_only=True,deadline_monotonic=0,checkpoint_callback=lambda v:partial.append(v))
        self.assertEqual(len(partial),1);self.assertFalse(partial[0]['accepted']);self.assertFalse(partial[0]['holdout_assessed'])
        with self.assertRaises(ValueError):m.restore(partial[0])


if __name__=='__main__':unittest.main()
