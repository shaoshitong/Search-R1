"""Sampling continuity, negative supervision, and unequal-rank gradient checks."""
import unittest
import torch
from .balanced_sampling import plan,proof
from .balanced_loss import balanced_loss
from .mixed_loss import mixed_only_loss


def masks(y):return (y[:,:,None]==1)&(y[:,None,:]==0)


class BalancedTests(unittest.TestCase):
    def test_epoch_is_balanced_unique_replayable_and_not_same_subset(self):
        g=[dict(qid=i,labels=([1,0,0,0,0] if i<900 else ([0]*5 if i%3 else [1]*5))) for i in range(2400)]
        a=plan(g,1729,2,1020);b=plan(g,1729,3,1020)
        self.assertEqual(a,plan(g,1729,2,1020));self.assertNotEqual(a,b)
        p=proof(g,a);self.assertEqual((p['mixed'],p['nonmixed'],p['unique_questions']),(510,510,1020))
        self.assertTrue(all(abs(sum(0<sum(g[i]['labels'])<5 for i in batch)-len(batch)/2)<=.5 for batch in a))
        self.assertTrue(all(len(batch)>=4 for batch in a))

    def test_duplicate_question_or_invalid_budget_is_rejected(self):
        g=[dict(qid=1,labels=[1,0,0,0,0])]*5
        with self.assertRaises(ValueError):plan(g,1,1,4)
        with self.assertRaises(ValueError):plan([],1,1,5)

    def test_negative_groups_get_bce_gradient_not_ranking(self):
        z=torch.zeros(2,5,dtype=torch.float64,requires_grad=True);y=torch.tensor([[0]*5,[1]*5],dtype=torch.float64)
        v=balanced_loss(z,y,masks(y),1.,1.,.5)
        self.assertEqual(float(v['pairwise_loss']),0.)
        v['loss'].backward();self.assertTrue(bool(torch.isfinite(z.grad).all()))
        self.assertTrue(bool((z.grad[0]>0).all()));self.assertTrue(bool((z.grad[1]<0).all()))

    def test_four_rank_and_micro_gradient_equals_global_reference(self):
        torch.manual_seed(1729)
        y=torch.tensor([[1,0,0,0,0],[0]*5,[1,1,0,0,0],[1]*5,[1,0,1,0,1],[0]*5,[1,1,1,0,0],[1]*5,[1,0,0,0,1],[0]*5,[0,1,1,1,1]],dtype=torch.float64)
        z=torch.randn(len(y),5,dtype=torch.float64,requires_grad=True);pm=masks(y);pm[0]=False
        f=float(((y.sum(1)>0)&(y.sum(1)<5)).double().mean())
        value=balanced_loss(z,y,pm,1.,1.,f);ref=torch.autograd.grad(value['loss'],z)[0]
        copy=z.detach().clone().requires_grad_();pieces=[]
        for rank in range(4):
            own=list(range(rank,len(y),4))
            for i in range(0,len(own),2):
                ids=own[i:i+2]
                # DDP four-rank averaging cancels the production factor4.
                pieces.append(balanced_loss(copy[ids],y[ids],pm[ids],1.,1.,f)['loss']*len(ids)/len(y))
        actual=torch.autograd.grad(sum(pieces),copy)[0]
        torch.testing.assert_close(actual,ref,rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(float(sum(pieces)),float(value['loss']),12)

    def test_new_rank_weight_is_tenfold_on_same_mixed_input(self):
        y=torch.tensor([[1,0,1,0,0],[0,1,1,1,0]],dtype=torch.float64);z=torch.randn(2,5,dtype=torch.float64)
        old=mixed_only_loss(z,y,masks(y),.1,1.);new=balanced_loss(z,y,masks(y),1.,1.,1.)
        self.assertAlmostEqual(float(new['loss']-old['loss']),.9*float(old['pairwise_loss']),12)


if __name__=='__main__':unittest.main()
