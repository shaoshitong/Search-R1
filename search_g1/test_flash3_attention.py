"""CPU mocks of HF packing/causal/GQA semantics and instance-local FA3 dispatch."""
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from search_g1 import flash3_attention as fa3
from search_g1.easy7b_infra import flags,profile_flags


def attend(q,k,v,scale=None,causal=False,window=(-1,-1),softcap=0.0):
    k=k.repeat_interleave(q.shape[-2]//k.shape[-2],dim=-2)
    v=v.repeat_interleave(q.shape[-2]//v.shape[-2],dim=-2)
    scores=torch.einsum('qhd,khd->hqk',q.float(),k.float())*(q.shape[-1]**-.5 if scale is None else scale)
    if softcap:scores=torch.tanh(scores/softcap)*softcap
    keys=torch.arange(k.shape[0]);queries=torch.arange(q.shape[0])+k.shape[0]-q.shape[0]
    mask=torch.ones((q.shape[0],k.shape[0]),dtype=torch.bool)
    if causal:mask &= keys.unsqueeze(0)<=queries.unsqueeze(1)
    if window[0]>=0:mask &= keys.unsqueeze(0)>=queries.unsqueeze(1)-window[0]
    if window[1]>=0:mask &= keys.unsqueeze(0)<=queries.unsqueeze(1)+window[1]
    scores=scores.masked_fill(~mask.unsqueeze(0),float('-inf'))
    probability=torch.softmax(scores,dim=-1)
    return torch.einsum('hqk,khd->qhd',probability,v.float()).to(q.dtype)


def flash_attn_func(q,k,v,dropout_p=0.0,softmax_scale=None,causal=False,window_size=(-1,-1),
                    softcap=0.0,alibi_slopes=None,deterministic=False,return_attn_probs=False):
    return torch.stack([attend(a,b,c,softmax_scale,causal,window_size,softcap) for a,b,c in zip(q,k,v)])


def flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,
                          dropout_p=0.0,softmax_scale=None,causal=False,window_size=(-1,-1),softcap=0.0,
                          alibi_slopes=None,deterministic=False,return_attn_probs=False,block_table=None):
    return torch.cat([attend(q[a:b],k[c:d],v[c:d],softmax_scale,causal,window_size,softcap)
        for a,b,c,d in zip(cu_seqlens_q[:-1],cu_seqlens_q[1:],cu_seqlens_k[:-1],cu_seqlens_k[1:])])


def _flash_attention_forward(q,k,v,attention_mask,query_length,position_ids=None,dropout=0.0,
                             sliding_window=None,is_causal=True,softmax_scale=None):
    window=(-1,-1) if sliding_window is None else (sliding_window,sliding_window)
    if position_ids is not None and bool((position_ids[:,1:]<position_ids[:,:-1]).any()):
        starts=(position_ids.reshape(-1)==0).nonzero().flatten().to(torch.int32)
        cu=torch.cat((starts,torch.tensor([q.shape[1]],dtype=torch.int32)))
        maximum=int((cu[1:]-cu[:-1]).max())
        out=flash_attn_varlen_func(q.squeeze(0),k.squeeze(0),v.squeeze(0),cu,cu,maximum,maximum,
            dropout,softmax_scale,is_causal,window)
        return out.unsqueeze(0)
    return flash_attn_func(q,k,v,dropout,softmax_scale,is_causal,window)


class Qwen2FlashAttention2(nn.Module):
    def __init__(self):
        super().__init__();self.q_proj=nn.Linear(8,8,bias=False);self.k_proj=nn.Linear(8,4,bias=False)
        self.v_proj=nn.Linear(8,4,bias=False);self.o_proj=nn.Linear(8,8,bias=False)
    def forward(self,hidden_states,position_ids):
        batch,length,_=hidden_states.shape
        q=self.q_proj(hidden_states).reshape(batch,length,2,4)
        k=self.k_proj(hidden_states).reshape(batch,length,1,4)
        v=self.v_proj(hidden_states).reshape(batch,length,1,4)
        output=_flash_attention_forward(q,k,v,None,length,position_ids=position_ids,is_causal=True)
        return self.o_proj(output.reshape(batch,length,8))


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([Qwen2FlashAttention2() for _ in range(28)])
        self.config=types.SimpleNamespace(model_type='qwen2',hidden_size=3584,num_hidden_layers=28,
            _attn_implementation='flash_attention_2')
        self.bfloat16()
    def forward(self,x,position_ids):return self.layers[0](x,position_ids)


def run():
    revision='a09a35458c702b33eeacc393d103063234e8bc28'
    return dict(easy2560_selection=True,profile='common8',execution_site='hpc3',execution_backend='slurm',
        training_world_size=4,data_selection_mode='mixed_first',question_probe=None,actor_logical_micro_batch_per_rank=4,
        model_revision=revision,budget=dict(model_id='Qwen/Qwen2.5-7B-Instruct',model_revision=revision),
        infra_optimizations=profile_flags('fa3-exact-cache-v1'))


def backend():
    module=types.SimpleNamespace(_flash_attn_backward=Mock(return_value='mock-backward-return'))
    def dense(q,k,v,**options):
        output=flash_attn_func(q,k,v,**options)
        return output,torch.empty(0)
    def varlen(q,k,v,cuq,cuk,usedq,usedk,maxq,maxk,**options):
        assert usedq is None and usedk is None
        output=flash_attn_varlen_func(q,k,v,cuq,cuk,maxq,maxk,**options)
        return output,torch.empty(0)
    module.flash_attn_func=Mock(side_effect=dense);module.flash_attn_varlen_func=Mock(side_effect=varlen)
    return types.SimpleNamespace(module=module,provenance={'interface_module':'CPU-mock-private','extension_module':'CPU-mock'})


class Flash3Tests(unittest.TestCase):
    def kernel(self,kind='varlen'):
        b=backend();a=fa3._Audit(b)
        return fa3._Kernel(flash_attn_varlen_func if kind=='varlen' else flash_attn_func,b,a,kind,require_cuda=False),b,a

    def test_old_profile_outputs_are_exact_two_keys_and_new_profile_is_explicit_native7b(self):
        self.assertEqual(profile_flags('exact-cache-v1'),dict(selected_policy_logits=False,exact_retrieval_cache=True))
        self.assertEqual(profile_flags('baseline'),dict(selected_policy_logits=False,exact_retrieval_cache=False))
        self.assertTrue(flags(run())['flash_attention_3_actor'])
        wrong=run();wrong['training_world_size']=8
        with self.assertRaises((AssertionError,ValueError)):flags(wrong)
        wrong=run();wrong['budget']['model_id']='Qwen/Qwen2.5-3B-Instruct'
        wrong['model_revision']=wrong['budget']['model_revision']='aa8e72537993ba99e69dfaafa59ed015b17504d1'
        with self.assertRaises(ValueError):flags(wrong)

    def test_varlen_preserves_objects_gqa_scale_causal_window_and_tuple_autograd(self):
        kernel,b,a=self.kernel();q=torch.randn(5,2,4,dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn(5,1,4,dtype=torch.bfloat16,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
        cu=torch.tensor([0,2,5],dtype=torch.int32)
        output=kernel(q,k,v,cu,cu,3,3,softmax_scale=.25,causal=True,window_size=(1,0),deterministic=True)
        row=b.module.flash_attn_varlen_func.call_args
        self.assertIs(row.args[0],q);self.assertIs(row.args[3],cu);self.assertIs(row.args[4],cu)
        self.assertEqual(row.args[5:9],(None,None,3,3));self.assertEqual(row.kwargs['softmax_scale'],.25)
        self.assertEqual(row.kwargs['window_size'],(1,0));self.assertTrue(row.kwargs['causal']);self.assertTrue(row.kwargs['deterministic'])
        expected=flash_attn_varlen_func(q,k,v,cu,cu,3,3,softmax_scale=.25,causal=True,window_size=(1,0))
        self.assertTrue(torch.equal(output,expected));output.float().square().sum().backward()
        self.assertEqual(a.qkv_backward_returns,1);self.assertTrue(all(x.grad is not None for x in (q,k,v)))

    def test_dense_unequal_query_key_lengths_preserve_bottom_right_causal_alignment(self):
        kernel,b,a=self.kernel('dense');q=torch.randn(1,2,2,4,dtype=torch.bfloat16)
        k=torch.randn(1,5,1,4,dtype=torch.bfloat16);v=torch.randn_like(k)
        self.assertTrue(torch.equal(kernel(q,k,v,causal=True),flash_attn_func(q,k,v,causal=True)))
        self.assertEqual(a.forward_returns,{'dense':1,'varlen':0})

    def test_nonzero_dropout_alibi_probs_block_table_and_fp32_are_rejected_before_backend(self):
        kernel,b,_=self.kernel();q=torch.randn(3,2,4,dtype=torch.bfloat16);k=q[:,:1];v=k.clone();cu=torch.tensor([0,3],dtype=torch.int32)
        for values in ({'dropout_p':.1},{'alibi_slopes':torch.ones(2)},{'return_attn_probs':True},{'block_table':torch.ones(1)}):
            with self.assertRaises(ValueError):kernel(q,k,v,cu,cu,3,3,**values)
        with self.assertRaisesRegex(ValueError,'BF16'):kernel(q.float(),k.float(),v.float(),cu,cu,3,3)
        b.module.flash_attn_varlen_func.assert_not_called()

    def test_packed_sequence_boundaries_prevent_cross_question_attention(self):
        kernel,_,_=self.kernel();q=torch.randn(4,2,4,dtype=torch.bfloat16);k=torch.randn(4,1,4,dtype=torch.bfloat16);v=torch.randn_like(k)
        cu=torch.tensor([0,2,4],dtype=torch.int32);first=kernel(q,k,v,cu,cu,2,2,causal=True)
        v2=v.clone();v2[:2]+=100
        second=kernel(q,k,v2,cu,cu,2,2,causal=True)
        self.assertTrue(torch.equal(first[2:],second[2:]));self.assertFalse(torch.equal(first[:2],second[:2]))

    def test_instance_clones_leave_reference_and_global_functions_untouched_and_gradients_equal(self):
        torch.manual_seed(41);model=Model();reference=copy.deepcopy(model);b=backend()
        before=tuple(model.state_dict());function=Qwen2FlashAttention2.forward;helper=_flash_attention_forward
        rng=torch.get_rng_state().clone();report=fa3.install_flash3_actor(model,run(),backend=b,require_cuda=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(tuple(model.state_dict()),before)
        self.assertIs(Qwen2FlashAttention2.forward,function);self.assertIs(_flash_attention_forward,helper)
        self.assertIs(reference.layers[0].forward.__func__,function);self.assertIsNot(model.layers[0].forward.__func__,function)
        x=torch.randn(1,6,8,dtype=torch.bfloat16);position=torch.tensor([[0,1,2,0,1,2]])
        actual=model(x,position);expected=reference(x,position)
        self.assertTrue(torch.equal(actual,expected));actual.float().square().sum().backward();expected.float().square().sum().backward()
        for left,right in zip(model.parameters(),reference.parameters()):
            self.assertEqual(left.grad is None,right.grad is None)
            if left.grad is not None:self.assertTrue(torch.equal(left.grad,right.grad))
        self.assertEqual(report['attention_layers'],28);self.assertGreater(b.module.flash_attn_varlen_func.call_count,0)

    def test_nonreentrant_checkpoint_recomputation_uses_the_same_actor_kernel(self):
        model=Model();b=backend();fa3.install_flash3_actor(model,run(),backend=b,require_cuda=False)
        x=torch.randn(1,4,8,dtype=torch.bfloat16,requires_grad=True);position=torch.tensor([[0,1,0,1]])
        output=checkpoint(lambda value:model(value,position),x,use_reentrant=False)
        output.float().sum().backward()
        self.assertGreaterEqual(b.module.flash_attn_varlen_func.call_count,2)
        self.assertIsNotNone(x.grad)

    def test_reversible_mode_uses_original_fa2_and_restores_after_nested_failure(self):
        model=Model();b=backend();original=model.layers[0].forward
        fa3.install_flash3_actor(model,run(),backend=b,require_cuda=False)
        private=model.layers[0].forward;keys=tuple(model.state_dict());rng=torch.get_rng_state().clone()
        x=torch.randn(1,4,8,dtype=torch.bfloat16);position=torch.tensor([[0,1,0,1]])
        before=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with fa3.flash3_mode(model,enabled=False):
                self.assertEqual(model.layers[0].forward,original)
                self.assertFalse(fa3.execution_report(model)['flash3_enabled'])
                with fa3.flash3_mode(model,enabled=True):self.assertEqual(model.layers[0].forward,private)
                self.assertEqual(model.layers[0].forward,original)
                model(x,position).float().sum().backward()
                raise RuntimeError('fixture')
        self.assertEqual(model.layers[0].forward,private)
        b.module.flash_attn_varlen_func.assert_not_called()
        self.assertEqual(tuple(model.state_dict()),keys);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(fa3.set_flash3_enabled(model,False));self.assertFalse(fa3.set_flash3_enabled(model,False))
        with self.assertRaisesRegex(ValueError,'BOOLEAN_ENABLE'):fa3.set_flash3_enabled(model,1)

    def test_checkpoint_baseline_recomputes_original_fa2_without_private_kernel(self):
        model=Model();b=backend();fa3.install_flash3_actor(model,run(),backend=b,require_cuda=False)
        x=torch.randn(1,4,8,dtype=torch.bfloat16,requires_grad=True);position=torch.tensor([[0,1,0,1]])
        with fa3.flash3_mode(model,enabled=False):
            checkpoint(lambda value:model(value,position),x,use_reentrant=False).float().sum().backward()
        b.module.flash_attn_varlen_func.assert_not_called();self.assertIsNotNone(x.grad)
        self.assertTrue(fa3.execution_report(model)['flash3_enabled'])

    def test_private_backward_success_count_requires_return_and_synchronized_report(self):
        model=Model();b=backend();fa3.install_flash3_actor(model,run(),backend=b,require_cuda=False)
        x=torch.randn(1,4,8,dtype=torch.bfloat16);position=torch.tensor([[0,1,0,1]])
        model(x,position)
        self.assertEqual(b.module._flash_attn_backward(),'mock-backward-return')
        self.assertFalse(fa3.execution_report(model)['actual_flash3_training_executed'])
        with patch.object(torch.cuda,'is_initialized',return_value=True),patch.object(torch.cuda,'synchronize') as sync:
            report=fa3.execution_report(model,synchronize=True)
        sync.assert_called_once();self.assertTrue(report['actual_flash3_training_executed'])
        self.assertEqual(report['kernel_backward_success_returns'],1)

    def test_bad_return_contract_and_duplicate_install_are_rejected(self):
        kernel,b,_=self.kernel('dense');q=torch.randn(1,2,2,4,dtype=torch.bfloat16)
        b.module.flash_attn_func.side_effect=lambda *a,**kw:q
        with self.assertRaisesRegex(ValueError,'OUT_LSE_PAIR'):kernel(q,q,q)
        model=Model();fa3.install_flash3_actor(model,run(),backend=backend(),require_cuda=False)
        with self.assertRaisesRegex(ValueError,'ALREADY_INSTALLED'):fa3.install_flash3_actor(model,run(),backend=backend(),require_cuda=False)

    def test_loader_binding_requires_all_six_identity_fields_and_fresh_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();value={}
            for kind,filename in (('interface','flash_attn_interface.py'),('extension','flash_attn_3_cuda.fixture.so'),('cudart','libcudart.so.12')):
                path=root/filename;path.write_bytes(kind.encode());value[kind+'_path']=str(path);value[kind+'_sha256']=fa3._digest(path)
            spec=dict(infra_flash_attention_3=value)
            with patch.object(fa3,'INTERFACE_SHA',value['interface_sha256']):
                self.assertEqual(fa3._binding(spec),value)
                (root/'libcudart.so.12').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'CUDART_SHA'):fa3._binding(spec)
            with self.assertRaises(ValueError):fa3._binding({'infra_flash_attention_3':{}})


if __name__=='__main__':unittest.main()
