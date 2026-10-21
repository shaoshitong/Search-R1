import unittest
import torch
from . import prior_logit_event_loss as m

class Tests(unittest.TestCase):
 def test_K_mapping_and_raw_aggregation(self):
  labels=torch.tensor([[int(j<k)for j in range(5)]for k in range(6)])
  self.assertEqual(m.targets_from_own_EM(labels,'binary').tolist(),[0,1,1,1,1,0])
  self.assertEqual(m.targets_from_own_EM(labels,'three_event').tolist(),[0,1,1,1,1,2])
  torch.manual_seed(42);z=torch.randn(6,6,dtype=torch.float64);p=torch.rand(6,5,dtype=torch.float64)
  for mode in m.MODES:
   out=m.inference_outputs(z,mode,p)
   torch.testing.assert_close(out['Mixed_balanced_decision_score'],z.softmax(-1)[:,1:5].sum(-1),atol=1e-14,rtol=1e-14)
   self.assertIs(out['trajectory_probabilities'],p)
   self.assertIn('not calibrated',out['score_semantics'])
   torch.testing.assert_close(m.event_logits(z+91,mode),m.event_logits(z,mode)+91)
 def test_finite_extreme_gradients(self):
  z=torch.tensor([[900.,-900.,20.,-20.,-30.,899.]],dtype=torch.float64,requires_grad=True);labels=torch.tensor([[0,1,0,1,0]])
  for mode in m.MODES:
   pi=m.coarse_prior_from_counts([5,1,2,2,1,3],mode)
   loss=m.prior_adjusted_view_losses(z,labels,pi,mode).sum();g=torch.autograd.grad(loss,z,retain_graph=True)[0]
   self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(g).all())
 def test_constant_tau_expected_gradient_zero_and_sign(self):
  # This weighted sum is the expectation under synthetic train pi, not production class weighting.
  for mode in m.MODES:
   tau=m.desired_priors(mode);pi=torch.tensor([.8,.2]if mode=='binary'else[.6,.25,.15],dtype=torch.float64)
   six=torch.tensor([.25,.125,.125,.125,.125,.25],dtype=torch.float64);z=six.log().requires_grad_()
   Ks=[0,1]if mode=='binary'else[0,1,5];labels=torch.tensor([[int(j<k)for j in range(5)]for k in Ks])
   expected=(m.prior_adjusted_view_losses(z.expand(len(Ks),6),labels,pi,mode)*pi).sum();g=torch.autograd.grad(expected,z,retain_graph=True)[0]
   torch.testing.assert_close(m.event_logits(z,mode),tau.log(),atol=1e-14,rtol=1e-14)
   torch.testing.assert_close(g,torch.zeros_like(g),atol=1e-14,rtol=0)
   wrong=torch.nn.functional.cross_entropy(m.event_logits(z.expand(len(Ks),6),mode),m.targets_from_own_EM(labels,mode),reduction='none');self.assertGreater(torch.autograd.grad((wrong*pi).sum(),z)[0].abs().max().item(),.01)
 def test_exact_four_view_question_reduction_and_gradient(self):
  torch.manual_seed(73);labels=torch.randint(0,2,(3,4,5));a=torch.randn(3,4,6,dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
  for mode in m.MODES:
   pi=m.coarse_prior_from_counts([21,3,6,4,9,7],mode);actual=m.question_weighted_loss(a,labels,pi,mode)
   expected=sum(m.prior_adjusted_view_losses(b[q,v],labels[q,v],pi,mode)*.25/3 for q in range(3)for v in range(4))
   torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-14)
   torch.testing.assert_close(torch.autograd.grad(actual,a,retain_graph=True)[0],torch.autograd.grad(expected,b,retain_graph=True)[0],atol=1e-14,rtol=1e-14)
   with self.assertRaises(ValueError):m.question_weighted_loss(a,labels,pi,mode,torch.tensor([[.5,.5,0,0]]*3))
 def test_prior_guard_no_smoothing_or_trainable_prior(self):
  with self.assertRaises(ValueError):m.coarse_prior_from_counts([1,0,0,0,0,1],'binary')
  z=torch.randn(1,6,dtype=torch.float64);y=torch.zeros(1,5,dtype=torch.long);pi=torch.tensor([.5,.5],dtype=torch.float64,requires_grad=True)
  with self.assertRaises(ValueError):m.prior_adjusted_view_losses(z,y,pi,'binary')

if __name__=='__main__':unittest.main()
