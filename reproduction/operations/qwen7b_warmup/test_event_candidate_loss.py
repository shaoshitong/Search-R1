"""Objective tests on synthetic logits only; no diagnostic data used for fit."""
import math,unittest
import torch
from .balanced_loss import balanced_loss
from .event_candidate_loss import event_candidate_loss


def mask(y):return (y[:,:,None]==1)&(y[:,None,:]==0)


class EventCandidateTests(unittest.TestCase):
    def test_mixed_values_and_gradients_are_exactly_unchanged(self):
        torch.manual_seed(1729);z=torch.randn(4,5,dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0]],dtype=z.dtype)
        old=balanced_loss(z,y,mask(y),1.,1.,1.);new=event_candidate_loss(z,y,mask(y),1.,1.,1.)
        self.assertEqual(float(new['event_correction']),0.);self.assertTrue(torch.equal(old['loss'],new['loss']))
        self.assertTrue(torch.equal(torch.autograd.grad(old['loss'],z,retain_graph=True)[0],torch.autograd.grad(new['loss'],z)[0]))

    def test_negative_group_gradient_has_correct_direction_including_wrong_basin(self):
        z=torch.tensor([[-5.,-1.6,0.,1.6,5.],[-5.,-1.6,0.,1.6,5.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[1]*5,[0]*5],dtype=z.dtype);v=event_candidate_loss(z,y,mask(y),1.,1.,.5)
        grad=torch.autograd.grad(v['loss'],z)[0];expected=1.2*(z.sigmoid()-y)/len(y)
        torch.testing.assert_close(grad,expected,rtol=1e-12,atol=1e-12)
        self.assertTrue(bool((grad[0]<0).all()));self.assertTrue(bool((grad[1]>0).all()))
        expected_nll=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none').sum(1).mean()
        torch.testing.assert_close(v['group_event_nll'],expected_nll,rtol=1e-12,atol=1e-12)

    def test_synthetic_allcorrect_fit_escapes_current_wrong_local_minimum(self):
        def fit(fn):
            z=torch.nn.Parameter(torch.full((1,5),math.log(.2/.8),dtype=torch.float64));y=torch.ones_like(z);opt=torch.optim.Adam([z],lr=.02)
            for _ in range(350):opt.zero_grad();fn(z,y,mask(y),1.,1.,1.)['loss'].backward();opt.step()
            return float(z.sigmoid().detach().mean())
        self.assertLess(fit(balanced_loss),.3);self.assertGreater(fit(event_candidate_loss),.9)


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
