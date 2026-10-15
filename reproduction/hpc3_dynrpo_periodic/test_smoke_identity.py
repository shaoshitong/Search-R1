"""BF16 smoke still rejects wrong weights, token contexts and nonfinite data."""
import copy
from types import SimpleNamespace
import unittest

import torch

from reproduction.hpc3_dynrpo_periodic.smoke_identity import active,master_identity,qwen_weight_readback
from reproduction.common8.model_smoke import compare
from reproduction.common8.test_model_smoke import ModelSmokeTests


class SmokeIdentity(unittest.TestCase):
    def fixture(self):
        c=SimpleNamespace(hidden_size=8,num_attention_heads=2,num_key_value_heads=1,
                          intermediate_size=16,num_hidden_layers=2,tie_word_embeddings=True)
        weights={'model.embed_tokens.weight':torch.randn(16,8),'model.norm.weight':torch.randn(8)}
        params={k:torch.nn.Parameter(v.to(torch.bfloat16)) for k,v in weights.items()}
        weights['lm_head.weight']=weights['model.embed_tokens.weight']
        params['lm_head.weight']=params['model.embed_tokens.weight']
        for layer in range(2):
            prefix='model.layers.'+str(layer)+'.'
            for key in ('input_layernorm.weight','post_attention_layernorm.weight'):
                weights[prefix+key]=torch.randn(8);params[prefix+key]=torch.nn.Parameter(weights[prefix+key].to(torch.bfloat16))
            for part,width,number in [('q',8,1),('k',4,2),('v',4,3)]:
                for suffix,shape in [('weight',(width,8)),('bias',(width,))]:
                    weights[prefix+'self_attn.'+part+'_proj.'+suffix]=torch.full(shape,float(number))
            for suffix in ('weight','bias'):
                params[prefix+'self_attn.qkv_proj.'+suffix]=torch.nn.Parameter(torch.cat([
                    weights[prefix+'self_attn.'+part+'_proj.'+suffix] for part in ('q','k','v')]).to(torch.bfloat16))
            weights[prefix+'self_attn.o_proj.weight']=torch.randn(8,8)
            params[prefix+'self_attn.o_proj.weight']=torch.nn.Parameter(weights[prefix+'self_attn.o_proj.weight'].to(torch.bfloat16))
            for part,number in [('gate',4),('up',5)]:
                weights[prefix+'mlp.'+part+'_proj.weight']=torch.full((16,8),float(number))
            params[prefix+'mlp.gate_up_proj.weight']=torch.nn.Parameter(torch.cat([
                weights[prefix+'mlp.gate_proj.weight'],weights[prefix+'mlp.up_proj.weight']]).to(torch.bfloat16))
            weights[prefix+'mlp.down_proj.weight']=torch.randn(8,16)
            params[prefix+'mlp.down_proj.weight']=torch.nn.Parameter(weights[prefix+'mlp.down_proj.weight'].to(torch.bfloat16))
        model=SimpleNamespace(named_parameters=lambda **kwargs:iter(params.items()))
        return c,weights,params,model

    def test_all_fused_and_direct_actual_weights_are_checked(self):
        c,w,p,m=self.fixture();result=qwen_weight_readback(w,m,c)
        self.assertTrue(result['accepted'],result)
        self.assertEqual(result['checked_vllm_tensor_count'],len(p))
        self.assertEqual(result['actor_tensor_count'],27)

    def test_wrong_vllm_qkv_gate_and_direct_weights_are_rejected(self):
        for key,index in [('model.layers.0.self_attn.qkv_proj.weight',13),
                          ('model.layers.0.mlp.gate_up_proj.weight',27),('model.norm.weight',0)]:
            c,w,p,m=self.fixture()
            with torch.no_grad():p[key][index]+=1
            self.assertFalse(qwen_weight_readback(w,m,c)['accepted'])

    def test_original_master_weight_changes_remain_fatal(self):
        first=torch.nn.Linear(4,2);second=copy.deepcopy(first)
        self.assertEqual(master_identity(first),master_identity(second))
        with torch.no_grad():second.weight[0,0]+=1
        self.assertNotEqual(master_identity(first),master_identity(second))

    def test_bf16_diagnostics_do_not_replace_context_and_finite_checks(self):
        inputs=ModelSmokeTests().samples();inputs[4][0,0]-=1
        self.assertIsNone(compare(*inputs,require_parity=False)['tolerance'])
        with self.assertRaises(AssertionError):compare(*inputs)
        inputs[4][0,0]=float('nan')
        with self.assertRaises(AssertionError):compare(*inputs,require_parity=False)
        inputs=ModelSmokeTests().samples();inputs[3][0,0]=99
        with self.assertRaises(AssertionError):compare(*inputs,require_parity=False)

    def test_policy_is_limited_to_the_explicit_periodic_native_bf16_run(self):
        run=dict(precision_profile='native4_bf16_varlen_fp32_master',execution_backend='slurm',
            execution_site='hpc3',training_world_size=4,actor_micro_batch_per_rank=4,
            vllm_gpu_memory_utilization=.1,behavior_probability_policy='sampling_journal_no_extra_hf_forward',
            question_probe=dict(head_kind='qwen_native_periodic_lastblock_binary_v1'))
        self.assertTrue(active(run));self.assertFalse(active(None))
        run['question_probe']['head_kind']='qwen_native_lastblock_binary_v1'
        self.assertFalse(active(run))


if __name__=='__main__':unittest.main(verbosity=2)
