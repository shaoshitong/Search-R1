"""Small CPU accounting/gradient/identity tests, not a remote/GPU qualification."""
import copy
from fractions import Fraction
import unittest
import torch
from .g10_grouping import build_training_views
from .g10_question_loss import g10_question_loss,freeze_evaluation_g5
from .g10_question_update import question_weighted_update


def groups():
    return [dict(qid='q'+str(i),trajectory_ids=['q%d:t%d'%(i,j) for j in range(10)],
                 labels=[int(j<k) for j in range(10)]) for i,k in enumerate((1,9,5))]


def duplicate(views,times):
    result=[]
    for view in views:
        for _ in range(times):
            v=copy.deepcopy(view);v['weight_denominator']*=times
            v['weight']=v['weight_numerator']/v['weight_denominator'];result.append(v)
    return result


class G10QuestionLossTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_duplicate_views_preserve_all_components_and_source_gradients(self):
        own=groups();torch.manual_seed(71)
        for mode in ('random','per-question-balanced','global-balanced'):
            views=build_training_views(own,mode=mode,views_per_question=4,seed=7)['groups']
            z=torch.randn(3,10,dtype=torch.float64,requires_grad=True)
            y=torch.tensor([g['labels'] for g in own],dtype=z.dtype)
            mask=(y[:,:,None]==1)&(y[:,None,:]==0)
            for binary,event in ((1.,0.),(0.,1.)):
                kwargs=dict(binary_mixed_weight=binary,three_event_weight=event,
                            within_rank_weight=.3,inter_rank_weight=.7,
                            stat_mean_weight=.4,stat_variance_weight=.6,
                            distinguishable_pair_mask=mask)
                first=g10_question_loss(z,own,views,**kwargs)
                repeated=g10_question_loss(z,own,duplicate(views,3),**kwargs)
                for name in first['components']:
                    torch.testing.assert_close(first['components'][name],repeated['components'][name],rtol=1e-12,atol=1e-12)
                first_grad=torch.autograd.grad(first['loss'],z,retain_graph=True)[0]
                repeated_grad=torch.autograd.grad(repeated['loss'],z,retain_graph=True)[0]
                torch.testing.assert_close(first_grad,repeated_grad,rtol=1e-12,atol=1e-12)
                self.assertEqual(first['per_original_question_total_weight'],[1.]*3)
                self.assertEqual(first['inter_nominal_weighted_pair_mass'],6)
                self.assertEqual(first['inter_eligible_weighted_pair_mass_exact'],repeated['inter_eligible_weighted_pair_mass_exact'])

    def test_same_question_never_forms_an_inter_question_pair(self):
        own=[groups()[2]]
        views=build_training_views(own,mode='per-question-balanced',views_per_question=4)['groups']
        self.assertEqual(sum(v['true_mixed'] for v in views),2)
        z=torch.randn(1,10,dtype=torch.float64,requires_grad=True)
        v=g10_question_loss(z,own,views,trajectory_weight=0.,inter_rank_weight=1.)
        self.assertEqual(float(v['loss'].detach()),0.)
        self.assertEqual(v['inter_actual_eligible_view_pairs'],0)
        self.assertEqual(v['inter_eligible_weighted_pair_mass_exact'],'0')
        self.assertEqual(float(torch.autograd.grad(v['loss'],z)[0].abs().sum()),0.)

    def test_random_and_balanced_keep_same_original_question_mass(self):
        own=groups();signatures=[]
        for mode in ('random','per-question-balanced','global-balanced'):
            result=build_training_views(own,mode=mode,views_per_question=4,seed=7)
            mass={g['qid']:Fraction(0) for g in own}
            for v in result['groups']:mass[v['qid']]+=Fraction(v['weight_numerator'],v['weight_denominator'])
            signatures.append((result['summary']['qid_sha256'],mass))
        self.assertEqual(signatures[0],signatures[1]);self.assertEqual(signatures[1],signatures[2])

    def test_eval_freezes_ids_without_accessing_labels(self):
        class NoLabelAccess(dict):
            def __getitem__(self,key):
                if key=='labels':raise AssertionError('EM read before evaluation G5 frozen')
                return super().__getitem__(key)
        own=[NoLabelAccess(g) for g in groups()]
        first=freeze_evaluation_g5(own,selection='random',role='validation',seed=19)
        changed=groups()
        for g in changed:g['labels']=[1-y for y in g['labels']]
        second=freeze_evaluation_g5(changed,selection='random',role='validation',seed=19)
        self.assertEqual(first,second)
        self.assertFalse(first['summary']['labels_consulted'])
        self.assertTrue(all('labels' not in g for g in first['groups']))

    def test_authentic_g10_ids_own_em_and_exact_weight_required(self):
        own=groups();views=build_training_views(own,views_per_question=2)['groups'];z=torch.zeros(3,10)
        for field,mutate in [('labels',lambda v:v['labels'].__setitem__(0,1-v['labels'][0])),
                             ('trajectory_ids',lambda v:v['trajectory_ids'].__setitem__(0,'foreign')),
                             ('weight',lambda v:v.__setitem__('weight_denominator',v['weight_denominator']+1))]:
            bad=copy.deepcopy(views);mutate(bad[0])
            with self.assertRaises(ValueError):g10_question_loss(z,own,bad)
        invalid=copy.deepcopy(own);invalid[0]['trajectory_ids'][0]=invalid[1]['trajectory_ids'][0]
        with self.assertRaises(ValueError):g10_question_loss(z,invalid,views)
        with self.assertRaises(ValueError):g10_question_loss(torch.zeros(3,20),own,views)

    def test_runner_adapter_retains_adam_and_updates_original_questions_once(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(2,1,dtype=torch.float64)
            def forward(self,data):return self.linear(data['features']).flatten()
        class Reader:
            def get(self,ids,device):
                values=[]
                for tid in ids:
                    a,b=tid.split(':');values.append([int(a[1:])+.1,int(b[1:])*.1])
                return dict(features=torch.tensor(values,dtype=torch.float64,device=device))
        torch.manual_seed(19);model=Probe();ema=copy.deepcopy(model).requires_grad_(False)
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-6)
        model(dict(features=torch.ones(1,2,dtype=torch.float64))).sum().backward();optimizer.step()
        self.assertEqual({int(v['step']) for v in optimizer.state.values()},{1})
        result=question_weighted_update(model,ema,optimizer,Reader(),groups(),torch.device('cpu'),
                  grouping_kwargs=dict(mode='per-question-balanced',views_per_question=4,seed=7),
                  loss_kwargs=dict(three_event_weight=1.,inter_rank_weight=.5,
                                   stat_mean_weight=.1,stat_variance_weight=.1),
                  micro_trajectories=3)
        self.assertEqual({int(v['step']) for v in optimizer.state.values()},{2})
        self.assertEqual(result['original_questions'],3)
        self.assertEqual(result['virtual_views'],12)
        self.assertEqual(result['denominators']['inter_ordered_original_question_pairs'],6)
        self.assertEqual(result['actor_updates'],0)
        self.assertFalse(result['supports_G20'])


if __name__=='__main__':unittest.main()
