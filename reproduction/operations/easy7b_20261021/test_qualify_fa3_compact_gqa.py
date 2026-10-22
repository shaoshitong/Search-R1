"""CPU checks of qualification arithmetic and shared-KV chain-rule scope."""
import unittest
import torch

from reproduction.operations.easy7b_20261021 import qualify_fa3_compact_gqa_cuda as diagnostic
from search_g1 import flash3_attention as adapter
from search_g1.test_flash3_attention import backend,flash_attn_varlen_func


class QualificationTests(unittest.TestCase):
    def test_expansion_is_original_head_major_repeat_and_gradient_sum(self):
        value=torch.arange(3*4*8,dtype=torch.float32).view(3,4,8).requires_grad_(True)
        actual=diagnostic.expanded_kv(value)
        original=value.transpose(0,1).unsqueeze(0)
        b,h,t,d=original.shape
        expected=original[:,:,None,:,:].expand(b,h,7,t,d).reshape(b,h*7,t,d).transpose(1,2).squeeze(0)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(actual.stride(),(8,3*8,1))
        actual.sum().backward();torch.testing.assert_close(value.grad,torch.full_like(value,7))

    def test_candidate_compares_same_four_head_leaf_gradients_after_reduction(self):
        torch.manual_seed(454)
        values=[torch.randn(5,h,8,dtype=torch.bfloat16) for h in (28,4,4)]
        upstream=torch.randn(5,28,8,dtype=torch.bfloat16);cu=torch.tensor([0,2,5],dtype=torch.int32)
        b=backend();audit=adapter._Audit(b)
        kernel=adapter._Kernel(flash_attn_varlen_func,b,audit,'varlen',require_cuda=False)
        compact=diagnostic.candidate(kernel,values,cu,3,upstream,False)
        expanded=diagnostic.candidate(kernel,values,cu,3,upstream,True)
        self.assertEqual([tuple(v.shape) for v in compact],[(5,28,8),(5,28,8),(5,4,8),(5,4,8)])
        comparison=diagnostic._comparisons(expanded,compact,('output','dq','shared_dk','shared_dv'))
        self.assertTrue(comparison['passed'],comparison)

    def test_math_reference_has_packed_causal_and_shared_head_semantics(self):
        torch.manual_seed(455)
        values=[torch.randn(5,h,8,dtype=torch.bfloat16,requires_grad=True) for h in (28,4,4)]
        actual=diagnostic._math_output(values,[2,3])
        expected=flash_attn_varlen_func(*values,torch.tensor([0,2,5],dtype=torch.int32),
            torch.tensor([0,2,5],dtype=torch.int32),3,3,causal=True)
        torch.testing.assert_close(actual.to(torch.bfloat16),expected,rtol=.01,atol=.01)
        actual.sum().backward()
        self.assertEqual(values[1].grad.shape,(5,4,8));self.assertTrue(torch.isfinite(values[1].grad).all())

    def test_fixed_tolerance_rejects_large_and_nonfinite_errors(self):
        x=torch.ones(3)
        self.assertTrue(diagnostic._comparisons([x],[x],('output',))['passed'])
        self.assertFalse(diagnostic._comparisons([x],[x+.1],('output',))['passed'])
        self.assertFalse(diagnostic._comparisons([x],[x*float('nan')],('dq',))['passed'])


if __name__=='__main__':unittest.main()
