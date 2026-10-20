"""Narrow CPU math/reduction tests; no model, dataset, or distributed claims."""
import unittest
import torch
import torch.nn.functional as F
from .probe_ablation_loss import probe_ablation_loss, group_event_scores
from reproduction.prefix20_group_joint100.loss import joint_loss
from .event_candidate_loss import event_candidate_loss


class ProbeAblationLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_base_modes_match_existing_losses(self):
        torch.manual_seed(1729)
        z = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0]*5, [1,0,0,0,0], [1,1,1,1,0], [1]*5], dtype=z.dtype)
        v = probe_ablation_loss(z, y, binary_mixed_weight=1.)
        old = joint_loss(z, y)
        torch.testing.assert_close(v['loss'], old['loss'], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(v['loss'], z, retain_graph=True)[0],
                                   torch.autograd.grad(old['loss'], z, retain_graph=True)[0], rtol=1e-12, atol=1e-12)
        mask = torch.zeros((4,5,5), dtype=torch.bool)
        three = probe_ablation_loss(z, y, three_event_weight=1.)
        candidate = event_candidate_loss(z, y, mask, weight=0.)
        torch.testing.assert_close(three['loss'], candidate['loss'], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(three['group_event_nll'], candidate['group_event_nll'], rtol=1e-12, atol=1e-12)
        own = probe_ablation_loss(z, y)
        torch.testing.assert_close(own['loss'], F.binary_cross_entropy_with_logits(z, y))

    def test_population_moments_and_own_em_translation(self):
        y = torch.tensor([[1.,1.,0.,0.,0.]], dtype=torch.float64)
        z = torch.zeros_like(y)
        v = probe_ablation_loss(z, y, trajectory_weight=0., stat_mean_weight=2., stat_variance_weight=3.)
        self.assertAlmostEqual(float(v['stat_mean_mse']), .01, places=12)
        self.assertAlmostEqual(float(v['target_population_variance'][0]), .24, places=12)
        self.assertAlmostEqual(float(v['stat_variance_mse']), .24**2, places=12)
        self.assertAlmostEqual(float(v['loss']), 2*.01+3*.24**2, places=12)
        exact = probe_ablation_loss(torch.where(y.bool(), 1000., -1000.), y,
                                    trajectory_weight=0., stat_mean_weight=1., stat_variance_weight=1.)
        self.assertEqual(float(exact['loss']), 0.)
        for labels in (torch.zeros_like(y), torch.ones_like(y)):
            self.assertEqual(float(probe_ablation_loss(z, labels)['target_population_variance'][0]), 0.)

    def test_within_direction_and_eligible_group_denominator(self):
        z = torch.zeros(2,5,dtype=torch.float64,requires_grad=True)
        y = torch.tensor([[1,0,0,0,0],[0]*5],dtype=z.dtype)
        mask = torch.zeros(2,5,5,dtype=torch.bool);mask[0,0,1] = True
        v = probe_ablation_loss(z,y,trajectory_weight=0.,within_rank_weight=1.,within_pair_mask=mask)
        self.assertEqual(v['within_eligible_groups'],1)
        self.assertEqual(v['within_eligible_pairs'],1)
        self.assertAlmostEqual(float(v['loss'].detach()),float(F.softplus(torch.tensor(1.,dtype=z.dtype))),places=12)
        grad = torch.autograd.grad(v['loss'],z)[0]
        self.assertLess(float(grad[0,0]),0.);self.assertGreater(float(grad[0,1]),0.)
        self.assertEqual(float(grad[1].abs().sum()),0.)

    def test_within_means_questions_before_pairs_and_hard_ties_are_stable(self):
        z = torch.tensor([[2.,0.,-2.,0.,0.],[0.,2.,-1.,0.,0.]],dtype=torch.float64)
        y = torch.tensor([[1,0,0,0,0],[1,0,0,0,0]],dtype=z.dtype)
        mask = torch.zeros(2,5,5,dtype=torch.bool)
        mask[0,0,1] = True
        mask[1,0,1] = True;mask[1,0,2] = True
        v = probe_ablation_loss(z,y,trajectory_weight=0.,within_rank_weight=1.,within_pair_mask=mask)
        one = F.softplus(z.new_tensor(-1.))
        two = (F.softplus(z.new_tensor(3.))+F.softplus(z.new_tensor(0.)))/2
        torch.testing.assert_close(v['loss'],(one+two)/2)
        self.assertEqual(v['within_eligible_groups'],2)
        self.assertEqual(v['within_eligible_pairs'],3)
        tied = torch.zeros(3,5,dtype=torch.float64)
        labels = torch.tensor([[1,0,0,0,0],[0]*5,[1]*5],dtype=tied.dtype)
        hard = probe_ablation_loss(tied,labels,inter_hard_negative_k=1)
        self.assertEqual(hard['inter_selected_negative_indices'].tolist(),[1])

    def test_inter_direction_allpairs_and_hard_negative(self):
        p = torch.tensor([[.9,.1,.1,.1,.1],[.5]*5,[.01]*5],dtype=torch.float64)
        z = torch.logit(p).requires_grad_()
        y = torch.tensor([[1,0,0,0,0],[0]*5,[1]*5],dtype=z.dtype)
        v = probe_ablation_loss(z,y,trajectory_weight=0.,inter_rank_weight=1.)
        s = v['mixed_event_logit']
        expected = F.softplus(1.-(s[0]-s[1:])).mean()
        torch.testing.assert_close(v['loss'],expected)
        self.assertEqual(v['inter_actual_pairs'],2)
        score_grad = torch.autograd.grad(v['loss'],s,retain_graph=True)[0]
        self.assertLess(float(score_grad[0]),0.);self.assertTrue(bool((score_grad[1:]>0).all()))
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(v['loss'],z)[0]).all()))
        hard = probe_ablation_loss(z,y,trajectory_weight=0.,inter_rank_weight=1.,inter_hard_negative_k=1)
        self.assertEqual(hard['inter_selected_negative_indices'].tolist(),[1])
        self.assertEqual(hard['inter_actual_pairs'],1)
        torch.testing.assert_close(hard['loss'],F.softplus(1.-(hard['mixed_event_logit'][0]-hard['mixed_event_logit'][1])))

    def test_empty_eligible_pairs_connected_zero_and_no_label_in_score(self):
        z = torch.randn(2,5,dtype=torch.float64,requires_grad=True)
        y = torch.zeros_like(z);mask = torch.zeros(2,5,5,dtype=torch.bool)
        v = probe_ablation_loss(z,y,trajectory_weight=0.,within_rank_weight=1.,inter_rank_weight=1.,within_pair_mask=mask)
        self.assertEqual(float(v['loss'].detach()),0.);self.assertEqual(v['inter_actual_pairs'],0)
        self.assertEqual(v['within_eligible_groups'],0)
        self.assertEqual(float(torch.autograd.grad(v['loss'],z)[0].abs().sum()),0.)
        other = probe_ablation_loss(z,1.-y)
        torch.testing.assert_close(v['mixed_probability'],other['mixed_probability'],rtol=0,atol=0)
        expected = 1.-z.sigmoid().prod(1)-(1.-z.sigmoid()).prod(1)
        torch.testing.assert_close(group_event_scores(z)['mixed_probability'],expected,rtol=1e-12,atol=1e-12)

    def test_shape_labels_weights_and_masks_fail_closed(self):
        z = torch.zeros(2,5);y = torch.zeros_like(z)
        for invalid in (torch.zeros(0,5),torch.zeros(2,4),torch.zeros(5),torch.full((2,5),float('nan'))):
            with self.assertRaises(ValueError):probe_ablation_loss(invalid,torch.zeros_like(invalid))
        with self.assertRaises(TypeError):probe_ablation_loss(z.long(),y)
        with self.assertRaises(ValueError):probe_ablation_loss(z,y+.2)
        with self.assertRaises(ValueError):probe_ablation_loss(z,y,binary_mixed_weight=1.,three_event_weight=1.)
        with self.assertRaises(ValueError):probe_ablation_loss(z,y,stat_mean_weight=float('nan'))
        with self.assertRaises(ValueError):probe_ablation_loss(z,y,within_rank_weight=1.)
        with self.assertRaises(ValueError):probe_ablation_loss(z,y,within_pair_mask=torch.ones(2,5,5,dtype=torch.bool))
        with self.assertRaises(ValueError):probe_ablation_loss(z,y,inter_hard_negative_k=0)


if __name__ == '__main__':
    unittest.main()
