"""CPU contract tests of opt-in, actor-instance-local KV4 FA3 dispatch."""
import copy
import os
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from search_g1 import flash3_attention as fa3
from search_g1.test_flash3_attention import _flash_attention_forward, backend, run


def get_ulysses_sequence_parallel_world_size():return 1


def repeat_kv(value,n_rep):
    b,h,t,d=value.shape
    return value[:,:,None,:,:].expand(b,h,n_rep,t,d).reshape(b,h*n_rep,t,d)


class Qwen2FlashAttention2(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads=28;self.num_key_value_heads=4;self.num_key_value_groups=7;self.head_dim=2
        self.q_proj=nn.Linear(8,56,bias=False);self.k_proj=nn.Linear(8,8,bias=False)
        self.v_proj=nn.Linear(8,8,bias=False);self.o_proj=nn.Linear(56,8,bias=False)

    def forward(self,hidden_states,position_ids):
        b,t,_=hidden_states.shape
        q=self.q_proj(hidden_states).reshape(b,t,28,2).transpose(1,2)
        k=self.k_proj(hidden_states).reshape(b,t,4,2).transpose(1,2)
        v=self.v_proj(hidden_states).reshape(b,t,4,2).transpose(1,2)
        # Distinct, deterministic position-dependent Q/K operations ensure that
        # bypassing repeat does not bypass or re-order prior position handling.
        cos=(position_ids.float()/13).cos().to(q.dtype)[:,None,:,None]
        sin=(position_ids.float()/13).sin().to(q.dtype)[:,None,:,None]
        q=q*cos+q.flip(-1)*sin;k=k*cos+k.flip(-1)*sin
        k=repeat_kv(k,self.num_key_value_groups);v=repeat_kv(v,self.num_key_value_groups)
        out=_flash_attention_forward(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),None,t,
                                     position_ids=position_ids,is_causal=True,softmax_scale=.4)
        return self.o_proj(out.reshape(b,t,56))


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([Qwen2FlashAttention2() for _ in range(28)])
        self.config=types.SimpleNamespace(model_type='qwen2',hidden_size=3584,num_hidden_layers=28,
                                          _attn_implementation='flash_attention_2')
        self.bfloat16()
    def forward(self,x,position_ids):return self.layers[0](x,position_ids)


class CompactTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(431)

    def test_default_retains_expanded_heads_and_explicit_mode_keeps_four(self):
        for compact,heads in ((False,28),(True,4)):
            m=Model();b=backend();before=tuple(m.state_dict())
            result=fa3.install_flash3_actor(m,run(),backend=b,require_cuda=False,compact_gqa=compact)
            m(torch.randn(1,6,8,dtype=torch.bfloat16),torch.tensor([[0,1,2,0,1,2]]))
            call=b.module.flash_attn_varlen_func.call_args
            self.assertEqual(call.args[0].shape[-2],28)
            self.assertEqual(call.args[1].shape[-2],heads);self.assertEqual(call.args[2].shape[-2],heads)
            self.assertEqual(result['compact_gqa_enabled'],compact);self.assertEqual(before,tuple(m.state_dict()))

    def test_forward_and_projected_parameter_gradients_match_expanded_shared_kv(self):
        expanded=Model();compact=copy.deepcopy(expanded);left,right=backend(),backend()
        fa3.install_flash3_actor(expanded,run(),backend=left,require_cuda=False,compact_gqa=False)
        fa3.install_flash3_actor(compact,run(),backend=right,require_cuda=False,compact_gqa=True)
        x=torch.randn(1,6,8,dtype=torch.bfloat16);pos=torch.tensor([[0,1,2,0,1,2]])
        a=x.clone().requires_grad_(True);b=x.clone().requires_grad_(True)
        upstream=torch.randn(1,6,8,dtype=torch.bfloat16)
        outa=expanded(a,pos);outb=compact(b,pos)
        torch.testing.assert_close(outa,outb,rtol=0,atol=0)
        outa.backward(upstream);outb.backward(upstream)
        torch.testing.assert_close(a.grad,b.grad,rtol=.02,atol=.004)
        for (name,p),(other,q) in zip(expanded.named_parameters(),compact.named_parameters()):
            self.assertEqual(name,other);self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=.02,atol=.004)
        self.assertEqual(fa3.execution_report(compact)['qkv_gradient_returns'],1)

    def test_reference_global_forward_and_repeat_are_untouched_and_mode_restores(self):
        m=Model();reference=copy.deepcopy(m);function=Qwen2FlashAttention2.forward;repeat=repeat_kv
        before=torch.get_rng_state().clone();b=backend()
        fa3.install_flash3_actor(m,run(),backend=b,require_cuda=False,compact_gqa=True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertIs(Qwen2FlashAttention2.forward,function);self.assertIs(repeat_kv,repeat)
        self.assertIs(reference.layers[0].forward.__func__,function)
        x=torch.randn(1,4,8,dtype=torch.bfloat16,requires_grad=True);pos=torch.tensor([[0,1,0,1]])
        with fa3.flash3_mode(m,enabled=False):
            self.assertEqual(m.layers[0].forward.__func__,function)
            m(x,pos).float().sum().backward()
        b.module.flash_attn_varlen_func.assert_not_called()
        checkpoint(lambda a:m(a,pos),x,use_reentrant=False).float().sum().backward()
        self.assertGreaterEqual(b.module.flash_attn_varlen_func.call_count,2)
        self.assertTrue(all(row['k'][-2]==4 for row in fa3.execution_report(m)['observed_shapes']))

    def test_compact_requires_pinned_repeat_binding_and_sp_one_before_install(self):
        for mutation in ('sp','heads','function'):
            m=Model();original=m.layers[0].forward
            if mutation=='heads':m.layers[0].num_key_value_heads=8
            if mutation=='function':
                def wrong(self,x,position_ids):return x
                m.layers[0].forward=types.MethodType(wrong,m.layers[0]);original=m.layers[0].forward
            context=patch(__name__+'.get_ulysses_sequence_parallel_world_size',return_value=2) if mutation=='sp' else patch.dict(os.environ,{})
            with context,self.assertRaises(ValueError):
                fa3.install_flash3_actor(m,run(),backend=backend(),require_cuda=False,compact_gqa=True)
            self.assertEqual(m.layers[0].forward,original)

    def test_environment_is_strict_explicit_and_cannot_override_explicit_false(self):
        with patch.dict(os.environ,{'SEARCH_G1_FA3_COMPACT_GQA':'1'}):
            m=Model();fa3.install_flash3_actor(m,run(),backend=backend(),require_cuda=False)
            self.assertTrue(fa3.execution_report(m)['compact_gqa_enabled'])
            m=Model();fa3.install_flash3_actor(m,run(),backend=backend(),require_cuda=False,compact_gqa=False)
            self.assertFalse(fa3.execution_report(m)['compact_gqa_enabled'])
        with patch.dict(os.environ,{'SEARCH_G1_FA3_COMPACT_GQA':'true'}),self.assertRaisesRegex(ValueError,'ENV_MUST'):
            fa3.install_flash3_actor(Model(),run(),backend=backend(),require_cuda=False)
        with self.assertRaisesRegex(ValueError,'BOOLEAN_COMPACT'):
            fa3.install_flash3_actor(Model(),run(),backend=backend(),require_cuda=False,compact_gqa=1)

    def test_repeat_identity_preserves_exact_tensor_and_checks_changed_sp(self):
        m=Model();fn=Qwen2FlashAttention2.forward
        identity=fa3._compact_repeat(m.layers[0],fn)
        tensor=torch.randn(1,4,3,2)
        self.assertIs(identity(tensor,7),tensor)
        with self.assertRaises(ValueError):identity(tensor,6)
        with patch(__name__+'.get_ulysses_sequence_parallel_world_size',return_value=1) as getter:
            dynamic=fa3._compact_repeat(m.layers[0],fn)
            getter.return_value=2
            with self.assertRaisesRegex(ValueError,'PARALLEL_CHANGED'):dynamic(tensor,7)


if __name__=='__main__':unittest.main()
