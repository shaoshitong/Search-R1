"""Numerical objective, actual gradient, and checkpoint tests for loss ablation."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
import torch
from search_g1.qprobe_model import (CountMomentProbe, OnlineQProbe, QProbeConfig,
                                    count_moment_loss, per_group_loss)


class Sigma3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def config(self, **kwargs):
        return QProbeConfig(loss_mode='sigma3', mean_loss_weight=0.,std_loss_weight=1.,
                            auxiliary_count_loss_weight=.1, **kwargs)

    def test_loss_matches_direct_probability_formula_for_every_k(self):
        logits=torch.tensor([[.2,-.8,1.3,.4,-1.1,2.]]*6,requires_grad=True)
        prediction=CountMomentProbe(3).moments(logits)
        counts=torch.arange(6);p=prediction['probabilities']
        q=torch.stack([p[:,0]+p[:,5],p[:,1]+p[:,4],p[:,2]+p[:,3]],-1)
        c=torch.tensor([0,1,2,2,1,0]);std=(counts.float()*(5-counts.float())/20).sqrt()
        expected=-q[torch.arange(6),c].log()+((prediction['expected_std']-std)/math.sqrt(.3)).square()-.1*p[torch.arange(6),counts].log()
        actual,parts=per_group_loss(prediction,counts,self.config())
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        weights=torch.tensor([1.,2.,4.,8.,16.,32.]);loss,_=count_moment_loss(prediction,counts,self.config(),weights)
        self.assertAlmostEqual(loss.item(),(expected*weights).sum().item()/weights.sum().item(),places=6)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIn('sigma_ce',parts)

    def test_primary_objective_cannot_distinguish_symmetric_counts(self):
        cfg=QProbeConfig(loss_mode='sigma3',mean_loss_weight=0.,std_loss_weight=1.,auxiliary_count_loss_weight=0.)
        logits=torch.tensor([[2.,-.5,1.,-.7,.3,-2.]]*6,requires_grad=True)
        losses,_=per_group_loss(CountMomentProbe(2).moments(logits),torch.arange(6),cfg)
        for a,b in ((0,5),(1,4),(2,3)):self.assertEqual(losses[a].item(),losses[b].item())

    def test_extreme_logits_have_finite_loss_and_gradients(self):
        logits=torch.tensor([[10000.,-10000.,-20000.,-15000.,-9000.,0.]],requires_grad=True)
        loss,_=count_moment_loss(CountMomentProbe(2).moments(logits),torch.tensor([2]),self.config())
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())

    def test_actual_optimizer_uses_sigma3_global_weighted_objective(self):
        probe=OnlineQProbe(3,self.config(head_epochs_per_outer=1,head_batch_size=6))
        probe.replay_features=torch.tensor([[1.,2.,3.],[3.,2.,1.]]*3)
        probe.replay_labels=[dict(correct_count=k,source_outer=1 if k<3 else 2) for k in range(6)]
        before=copy.deepcopy(probe.model.state_dict());expected=CountMomentProbe(3);expected.load_state_dict(before)
        optimizer=torch.optim.Adam(expected.parameters(),lr=probe.config.head_learning_rate)
        rng=torch.Generator().set_state(probe.rng.get_state());order=torch.randperm(6,generator=rng)
        counts=torch.arange(6);w=torch.tensor([2**(-(2-x['source_outer'])/4) for x in probe.replay_labels]);w=w*6/w.sum()
        p=expected(probe.replay_features[order]);logits=p['logits'];probs=logits.softmax(-1)
        q=torch.stack([probs[:,0]+probs[:,5],probs[:,1]+probs[:,4],probs[:,2]+probs[:,3]],-1)
        c=torch.tensor([0,1,2,2,1,0])[order];k=counts[order]
        std=(k.float()*(5-k.float())/20).sqrt()
        manual=-q[torch.arange(6),c].log()+((p['expected_std']-std)/math.sqrt(.3)).square()-.1*probs[torch.arange(6),k].log()
        (manual*w[order]).mean().backward();optimizer.step()
        report=probe._fit(2)
        for a,b in zip(probe.model.parameters(),expected.parameters()):self.assertTrue(torch.allclose(a,b,atol=1e-7))
        self.assertEqual(report['head_optimizer_updates'],1)
        self.assertIn('sigma_ce',report)

    def test_config_and_native_checkpoint_keep_loss_and_capacity(self):
        with self.assertRaises(ValueError):QProbeConfig(loss_mode='sigma3')
        probe=OnlineQProbe(2048,self.config(),feature_spec_sha='f'*64)
        self.assertEqual(sum(x.numel() for x in probe.model.parameters()),135622)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'head.pt';probe.save(path,actor_snapshot_id='new-run:0')
            restored=OnlineQProbe.load(path)
            self.assertEqual(restored.config,probe.config)
            self.assertEqual(restored.config.loss_mode,'sigma3')


if __name__=='__main__':unittest.main()
