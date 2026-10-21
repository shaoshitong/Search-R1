"""CPU mechanics: one FA2 path, exact same-input shadows, cleanup and math."""
import copy
import json
import random
import types
import unittest

import numpy as np
import torch

from reproduction.easy2560_joint4 import fa3_layer_shadow as shadow
from search_g1 import flash3_attention as adapter
from search_g1.test_flash3_attention import Model,attend,backend,run


class AllLayers(Model):
    def __init__(self):
        super().__init__();self.config.num_key_value_heads=1
    def forward(self,x,position_ids):
        for layer in self.layers:x=x+layer(x,position_ids)
        return x


class Actor:
    def __init__(self,model):
        self.actor_module=model;self.calls=0;self.output=None
        self._actual_precision_logged=False
    def _forward_micro_batch(self,micro_batch,temperature,compute_entropy):
        assert compute_entropy is False
        self.calls+=1;self._actual_precision_logged=True
        self.output=self.actor_module(micro_batch['input_ids'],micro_batch['position_ids'])
        return None,self.output


def fixture():
    model=AllLayers();b=backend();adapter.install_flash3_actor(model,run(),backend=b,require_cuda=False)
    micro=dict(input_ids=torch.randn(1,6,8,dtype=torch.bfloat16),position_ids=torch.tensor([[0,1,2,0,1,2]]))
    return Actor(model),micro,b


def rng():
    return random.getstate(),np.random.get_state(),torch.get_rng_state().clone()


def assert_rng(case,before):
    after=rng();case.assertEqual(before[0],after[0]);case.assertEqual(before[1][0],after[1][0])
    case.assertTrue(np.array_equal(before[1][1],after[1][1]));case.assertEqual(before[1][2:],after[1][2:])
    case.assertTrue(torch.equal(before[2],after[2]))


class LayerShadowTests(unittest.TestCase):
    def test_exact28_same_input_reports_and_no_backward_rng_or_state_change(self):
        actor,micro,b=fixture();model=actor.actor_module;model.eval();model.layers[3].train(True)
        before_forwards=[layer.forward for layer in model.layers];modes=[m.training for m in model.modules()]
        state={k:v.clone() for k,v in model.state_dict().items()};before_rng=rng()
        report=shadow.run_layer_shadow(actor,micro,1.,adapter=adapter)
        self.assertEqual(actor.calls,1);self.assertEqual(report['actual_attention_layers'],28)
        self.assertEqual(report['acceptance_status'],'DIAGNOSTIC_ONLY_NOT_QUALIFICATION')
        self.assertFalse(report['numerical_qualification_accepted']);self.assertEqual(report['backward_calls'],0)
        self.assertEqual(b.module.flash_attn_varlen_func.call_count,28)
        self.assertEqual([layer.forward for layer in model.layers],before_forwards)
        self.assertEqual([m.training for m in model.modules()],modes);self.assertFalse(actor._actual_precision_logged)
        assert_rng(self,before_rng);self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,state[key]))
        for index,row in enumerate(report['layer_reports']):
            self.assertEqual(row['layer_index'],index);self.assertFalse(row['forward_grad_enabled'])
            self.assertEqual(row['q']['shape'],[6,2,4]);self.assertEqual(row['k']['shape'],[6,1,4])
            self.assertEqual(row['packed_sequences']['cu_seqlens_q'],[0,3,6])
            self.assertEqual(row['packed_sequences']['query_lengths'],[3,3])
            self.assertEqual(row['fa3_vs_same_input_fa2']['max_absolute_error'],0)
            self.assertEqual(row['fa2_repeat_output_noise']['max_absolute_error'],0)
            self.assertEqual(row['fp32_sample_reference']['query_count'],6)
            self.assertTrue(row['actual_fa2_options']['causal']);self.assertEqual(row['effective_softmax_scale'],.5)
        self.assertLess(len(json.dumps(report,allow_nan=False)),300000)

    def test_bad_shadow_output_does_not_propagate_into_original_fa2_model_trajectory(self):
        actor,micro,b=fixture();model=actor.actor_module
        with adapter.flash3_mode(model,enabled=False),torch.no_grad():expected=model(micro['input_ids'],micro['position_ids'])
        original=b.module.flash_attn_varlen_func.side_effect
        def changed(*args,**kwargs):
            result,lse=original(*args,**kwargs);return result+10,lse
        b.module.flash_attn_varlen_func.side_effect=changed
        report=shadow.run_layer_shadow(actor,micro,1.,adapter=adapter,fp32_queries_per_sequence=0)
        self.assertTrue(torch.equal(actor.output,expected));self.assertEqual(actor.calls,1)
        self.assertTrue(all(row['fa3_vs_same_input_fa2']['max_absolute_error']>9 for row in report['layer_reports']))
        self.assertTrue(all(not row['fp32_sample_reference']['enabled'] for row in report['layer_reports']))

    def test_exception_restores_forwards_mixed_modes_all_rng_and_actor_logging(self):
        actor,micro,b=fixture();model=actor.actor_module;model.eval();model.layers[2].train(True)
        forwards=[layer.forward for layer in model.layers];modes=[m.training for m in model.modules()];before=rng()
        def fail(*args,**kwargs):
            random.random();np.random.random();torch.rand(1)
            raise RuntimeError('same-input kernel fixture failure')
        b.module.flash_attn_varlen_func.side_effect=fail
        with self.assertRaisesRegex(RuntimeError,'same-input kernel fixture failure'):
            shadow.run_layer_shadow(actor,micro,1.,adapter=adapter)
        self.assertEqual([layer.forward for layer in model.layers],forwards)
        self.assertEqual([m.training for m in model.modules()],modes);self.assertFalse(actor._actual_precision_logged)
        assert_rng(self,before);self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_fp32_reference_keeps_complete_late_query_context_gqa_window_softcap_scale(self):
        q=torch.randn(7,4,8,dtype=torch.bfloat16);k=torch.randn(7,2,8,dtype=torch.bfloat16);v=torch.randn_like(k)
        options=dict(causal=True,window_size=(3,0),softmax_scale=.25,softcap=2.)
        indices=[0,3,6];original_tf32=torch.backends.cuda.matmul.allow_tf32
        reference=shadow._fp32_sample(q,k,v,indices,options)
        expected=attend(q.float(),k.float(),v.float(),.25,True,(3,0),2.)[indices]
        self.assertTrue(torch.allclose(reference,expected,rtol=1e-6,atol=1e-6))
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,original_tf32)
        # Last query's allowed window includes key3, outside a length3 prefix.
        changed=v.clone();changed[3]+=10
        altered=shadow._fp32_sample(q,k,changed,indices,options)
        self.assertTrue(torch.equal(reference[0],altered[0]));self.assertFalse(torch.equal(reference[-1],altered[-1]))

    def test_exact_sevenfold_kv_repeat_proof_and_corruption(self):
        compact=torch.randn(11,4,128,dtype=torch.bfloat16);expanded=compact.repeat_interleave(7,dim=-2)
        proof=shadow._repeat_proof(expanded,4)
        self.assertEqual(proof['current_heads'],28);self.assertEqual(proof['repeat_group_size'],7)
        self.assertTrue(proof['exact_repeats']);expanded[0,1,0]+=1
        self.assertFalse(shadow._repeat_proof(expanded,4)['exact_repeats'])

    def test_fp32_reference_disables_the_enclosing_actor_bf16_autocast(self):
        q=torch.randn(13,4,8,dtype=torch.bfloat16);k=torch.randn(13,2,8,dtype=torch.bfloat16);v=torch.randn_like(k)
        options=dict(causal=True,window_size=(-1,-1),softmax_scale=.25,softcap=0.)
        indices=[0,6,12];expected=shadow._fp32_sample(q,k,v,indices,options)
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            actual=shadow._fp32_sample(q,k,v,indices,options)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(actual.dtype,torch.float32);self.assertTrue(torch.equal(actual,expected))

    def test_pending_gradients_and_unbounded_sampling_are_rejected(self):
        actor,micro,_=fixture();parameter=next(actor.actor_module.parameters());parameter.grad=torch.zeros_like(parameter)
        with self.assertRaisesRegex(ValueError,'PENDING_GRADIENTS'):
            shadow.run_layer_shadow(actor,micro,1.,adapter=adapter)
        parameter.grad=None
        with self.assertRaisesRegex(ValueError,'BOUNDED_QUERY'):
            shadow.run_layer_shadow(actor,micro,1.,adapter=adapter,fp32_queries_per_sequence=4)


if __name__=='__main__':unittest.main()
