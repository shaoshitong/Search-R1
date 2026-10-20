"""G20 own-view objective against independent per-view losses/gradients."""
import copy
import unittest
import torch
from . import g20_grouping as grouping
from .g20_question_loss import g20_question_loss
from .probe_ablation_loss import probe_ablation_loss


def groups():
    return [dict(qid='q%d'%q,trajectory_ids=['own-%d-%d'%(q,j) for j in range(20)],
        labels=labels) for q,labels in enumerate([[0]*20,[0,1]*10,[1]*20,[0]*17+[1]*3])]


class G20LossTests(unittest.TestCase):
    def test_loss_and_gradient_equal_independent_question_weighted_G5_reference(self):
        own=groups();views=grouping.build_training_views(own)['groups']
        z=torch.linspace(-2,2,80,dtype=torch.float64).reshape(4,20).requires_grad_(True)
        out=g20_question_loss(z,own,views,trajectory_weight=1,three_event_weight=1)
        manual=z.sum()*0
        for view in views:
            q=int(view['qid'][1:]);ix=view['source_indices'];y=torch.tensor([view['labels']],dtype=z.dtype)
            term=probe_ablation_loss(z[q,ix][None,:],y,trajectory_weight=1,three_event_weight=1)['loss']
            manual=manual+term*view['weight_numerator']/view['weight_denominator']/4
        self.assertTrue(torch.allclose(out['loss'],manual,rtol=1e-12,atol=1e-12))
        a=torch.autograd.grad(out['loss'],z,retain_graph=True)[0];b=torch.autograd.grad(manual,z)[0]
        self.assertTrue(torch.allclose(a,b,rtol=1e-12,atol=1e-12))
        self.assertTrue(out['supports_G20']);self.assertEqual(out['source_group_size'],20)
        self.assertEqual(out['per_original_question_total_weight'],[1.]*4)

    def test_repeated_view_representation_with_split_weight_is_invariant(self):
        own=groups();views=grouping.build_training_views(own)['groups'];twice=[]
        for v in views:
            for _ in range(2):
                item=copy.deepcopy(v);item['weight_denominator']*=2;twice.append(item)
        z=torch.linspace(-1,1,80,dtype=torch.float64).reshape(4,20).requires_grad_(True)
        kwargs=dict(trajectory_weight=1,three_event_weight=1,inter_rank_weight=.3,stat_mean_weight=.2,stat_variance_weight=.4)
        a=g20_question_loss(z,own,views,**kwargs);b=g20_question_loss(z,own,twice,**kwargs)
        self.assertTrue(torch.allclose(a['loss'],b['loss'],rtol=1e-12,atol=1e-12))
        ga=torch.autograd.grad(a['loss'],z,retain_graph=True)[0];gb=torch.autograd.grad(b['loss'],z)[0]
        self.assertTrue(torch.allclose(ga,gb,rtol=1e-12,atol=1e-12))

    def test_reject_G10_or_tampered_own_EM_as_G20(self):
        own=groups();views=grouping.build_training_views(own)['groups']
        for g in own:g['trajectory_ids']=g['trajectory_ids'][:10];g['labels']=g['labels'][:10]
        with self.assertRaises(ValueError):g20_question_loss(torch.zeros(4,20),own,views)
        own=groups();views[0]['labels'][0]=1-views[0]['labels'][0]
        with self.assertRaises(ValueError):g20_question_loss(torch.zeros(4,20),own,views)

    def test_reject_mass_and_wrong_mask_geometry(self):
        own=groups();views=grouping.build_training_views(own)['groups'];views[0]['weight_denominator']=8
        with self.assertRaises(ValueError):g20_question_loss(torch.zeros(4,20),own,views)
        views=grouping.build_training_views(own)['groups']
        with self.assertRaises(ValueError):g20_question_loss(torch.zeros(4,20),own,views,
            within_rank_weight=1,distinguishable_pair_mask=torch.zeros(4,10,10,dtype=torch.bool))

    def test_real_G20_within_pair_mask_and_inter_gradient_are_finite(self):
        own=groups();views=grouping.build_training_views(own)['groups']
        z=torch.linspace(-2,2,80).reshape(4,20).requires_grad_(True)
        y=torch.tensor([g['labels'] for g in own]);mask=(y[:,:,None]==1)&(y[:,None,:]==0)
        out=g20_question_loss(z,own,views,three_event_weight=1,within_rank_weight=1,
            inter_rank_weight=1,distinguishable_pair_mask=mask)
        out['loss'].backward();self.assertTrue(bool(torch.isfinite(z.grad).all()))
        self.assertEqual(out['denominators']['inter_ordered_original_question_pairs'],12)


if __name__=='__main__':unittest.main()
