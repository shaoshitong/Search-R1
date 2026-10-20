import hashlib,unittest
import numpy as np
import torch
from torch import nn
from .score_readout_contribution import ContributionHook,mixed,sigmoid,summarize,corr

class ReadoutTest(unittest.TestCase):
 def test_frozen_fp32_reconstruction_and_no_forward_change(self):
  torch.manual_seed(3);head=nn.Linear(12,1).eval().requires_grad_(False);x=torch.randn(7,12);expected=head(x).clone();before={k:v.clone() for k,v in head.state_dict().items()};hook=ContributionHook(head,2,2,3)
  actual=head(x);parts=hook.pop();hook.close();self.assertTrue(torch.equal(actual,expected));self.assertTrue(all(torch.equal(v,before[k]) for k,v in head.state_dict().items()));self.assertTrue(all(p.grad is None for p in head.parameters()))
  np.testing.assert_allclose(np.asarray(parts['question'])+parts['prefix'],parts['shadow_full'],atol=2e-7,rtol=2e-6)
 def test_zero_prefix_and_original_bias_only_once(self):
  head=nn.Linear(12,1).eval().requires_grad_(False)
  with torch.no_grad():head.weight.fill_(.1);head.bias.fill_(2.)
  x=torch.ones(3,12);x[:,6:]=0;hook=ContributionHook(head,2,2,3);actual=head(x).flatten().tolist();v=hook.pop();hook.close()
  self.assertEqual(v['prefix'],[0.]*3);np.testing.assert_allclose(v['question'],actual);np.testing.assert_allclose(v['question'],[2.6]*3)
 def test_hook_cannot_silently_retain_microbatches(self):
  head=nn.Linear(12,1);hook=ContributionHook(head,2,2,3);head(torch.ones(1,12))
  with self.assertRaises(ValueError):head(torch.ones(1,12))
  hook.pop();hook.close()
 def test_geometry_and_diagnostic_statistics(self):
  with self.assertRaises(ValueError):ContributionHook(nn.Linear(10,1),2,2,3)
  rows=[]
  labels=[[0]*5,[1]*5,[0,1,0,1,0],[1,1,0,0,0]]
  for i,y in enumerate(labels):
   q=np.ones(5)*(i-.5);p=np.asarray(y)*2.-1.;f=q+p;prob=sigmoid(f)
   rows.append(dict(qid=str(i),strict_em=y,legacy_logits=f.tolist(),legacy_p_j=prob.tolist(),legacy_q=float(mixed([prob])[0]),shadow_full_logits=f.tolist(),question_logits=q.tolist(),prefix_logits=p.tolist()))
  s=summarize(rows);self.assertEqual(s['components']['question']['within_question_population_variance'],0.)
  self.assertEqual(s['components']['question']['within_pair_micro_AUC'],.5);self.assertEqual(s['components']['prefix']['within_pair_micro_AUC'],1.)
  self.assertAlmostEqual(s['decomposition']['signed_total_variance_question_share']+s['decomposition']['signed_total_variance_prefix_share'],1.)
  self.assertLess(s['decomposition']['full_FP32_minus_sum_abs_max'],1e-12);self.assertIsNone(corr([1,1],[0,1]))
if __name__=='__main__':unittest.main()
