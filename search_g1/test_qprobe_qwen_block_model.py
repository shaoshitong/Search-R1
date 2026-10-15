"""Small native-HF CPU fixtures; no official weights or GPU are required."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import unittest

import torch

from search_g1.qprobe_qwen_block_model import (
    QwenBlockQProbeConfig, QwenBlockWarmup, formal_replay_loss, formal_replay_weights,
    native_implementation_binding,
)
from search_g1.qprobe_token_model import token_binary_loss


NATIVE_HF_AVAILABLE = importlib.util.find_spec('transformers') is not None


@unittest.skipUnless(NATIVE_HF_AVAILABLE,'actual native transformers is required; never substitute a fake block')
class QwenBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, **options):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        config = QwenBlockQProbeConfig(max_tokens=3,hidden_size=8,num_attention_heads=2,
            num_key_value_heads=1,intermediate_size=16,**options)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(43).get_state())
            native = Qwen2DecoderLayer(config.native_config(),layer_idx=35)
        state = {'model.layers.35.'+key:value.detach().to(torch.bfloat16).clone()
                 for key,value in native.state_dict().items()}
        return config,native,state

    def test_exact_native_copy_padding_frozen_inputs_and_mean_gradient_accumulation(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        config,native,state = self.fixture(batch_size=4,micro_batch_size=3)
        global_rng = torch.get_rng_state().clone()
        engine = QwenBlockWarmup(config,state)
        self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
        model = engine.model.eval()
        self.assertIs(type(model.block),Qwen2DecoderLayer)
        self.assertEqual(model.block.self_attn.num_heads,2)
        self.assertEqual(model.block.self_attn.num_key_value_heads,1)
        self.assertEqual(model.block.mlp.gate_proj.out_features,16)
        self.assertEqual(model.output.in_features,3*8)
        self.assertTrue(all(parameter.requires_grad for parameter in model.block.parameters()))
        expected = {key.removeprefix('model.layers.35.'):value.float() for key,value in state.items()}
        native.load_state_dict(expected,strict=True)
        self.assertEqual(engine.initial_block_proof['tensor_count'],len(expected))
        for key,value in model.block.state_dict().items():
            torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            proof = engine.initial_block_proof['tensors'][key]
            original = state['model.layers.35.'+key]
            self.assertTrue(proof['exact_cast_equal'])
            self.assertEqual(proof['source_dtype'],'bfloat16')
            self.assertEqual(proof['source_sha256'],hashlib.sha256(original.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            self.assertNotEqual(value.data_ptr(),original.data_ptr())
        self.assertTrue(engine.initial_readout_proof['fresh'])
        self.assertEqual((engine.update_count,engine._adam_steps(),engine.fitted),(0,0,False))
        self.assertFalse(engine.optimizer.state)
        x = torch.arange(4*3*8,dtype=torch.float32).reshape(4,3,8)/30
        x.requires_grad_(True)
        lengths = torch.tensor([2,1,2,1]); counts = torch.tensor([0,1,2,5])
        prediction = model(x,lengths)
        alternate = x.detach().clone(); mask = torch.arange(3)[None,:] < lengths[:,None]
        alternate[~mask] = float('nan')
        torch.testing.assert_close(model(alternate,lengths)['raw_outputs'],prediction['raw_outputs'],rtol=0,atol=0)
        torch.testing.assert_close(model(x[:,:2],lengths)['raw_outputs'],prediction['raw_outputs'],rtol=0,atol=0)
        changed = x.detach().clone(); changed[0,1,0] += 3.
        self.assertFalse(torch.allclose(model(changed,lengths)['raw_outputs'],prediction['raw_outputs']))
        with self.assertRaisesRegex(ValueError,'BMAX'): model(torch.zeros(1,4,8),[4])
        # Independent actual HF layer/rotary/mask utility gives the identical wrapper result.
        active = x.detach()[:,:2].masked_fill(~mask[:,:2,None],0.)
        position = torch.arange(2); rotary = Qwen2RotaryEmbedding(config=config.native_config())
        native_mask = _prepare_4d_causal_attention_mask_for_sdpa(mask[:,:2],active.shape[:2],active,0)
        hidden = native(active,attention_mask=native_mask,position_ids=position[None,:],
            position_embeddings=rotary(active,position[None,:]),cache_position=position,use_cache=False,output_attentions=False)
        if isinstance(hidden,tuple): hidden = hidden[0]
        hidden = torch.nn.functional.pad(hidden.masked_fill(~mask[:,:2,None],0.),(0,0,0,1))
        torch.testing.assert_close(prediction['raw_outputs'],model.output(hidden.flatten(1)),rtol=0,atol=0)
        loss,parts = token_binary_loss(prediction,counts,config)
        self.assertAlmostEqual(float(loss),float(parts['binary_bce']+.3*parts['regression_mse_normalized']),places=6)
        loss.backward()
        self.assertIsNone(x.grad)  # Frozen features cannot accidentally update the encoder.
        full_gradients = {name:p.grad.detach().clone() for name,p in model.named_parameters()}
        self.assertEqual(float(model.output.weight.grad[:,2*8:].abs().sum()),0.)
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            self.assertGreater(float(getattr(model.block.self_attn,name).weight.grad.abs().sum()),0.)
        model.zero_grad(set_to_none=True)
        for index in (torch.arange(3),torch.tensor([3])):
            micro = model(x[index],lengths[index])
            micro_loss,_ = token_binary_loss(micro,counts[index],config)
            (micro_loss*len(index)/4).backward()
        for name,p in model.named_parameters():
            torch.testing.assert_close(p.grad,full_gradients[name],rtol=2e-5,atol=2e-6)
        binding = native_implementation_binding(config)
        self.assertEqual(binding,json.loads(json.dumps(binding,sort_keys=True,allow_nan=False)))
        self.assertEqual(set(binding['native_config']['id2label']),{'0','1'})
        self.assertIn('Qwen2DecoderLayer',binding['implementations']['decoder_layer']['qualified_name'])
        self.assertEqual(binding['attention_implementation'],'sdpa')

    def test_actual_ten_epochs_two_hundred_adam_and_recent_eight_raw_weights(self):
        config,_,state = self.fixture(batch_size=256,micro_batch_size=256)
        config = replace(config,max_tokens=1)
        engine = QwenBlockWarmup(config,state)
        before_rng = torch.get_rng_state().clone()
        ids = [f'fixture-{i}' for i in range(5120)]
        counts = torch.arange(5120)%6
        features = torch.zeros(5120,1,8)
        features[:,0,0] = counts.float()/5
        events = []
        metrics = engine.fit(features,[1]*5120,counts,ids,progress_callback=events.append)
        self.assertTrue(torch.equal(before_rng,torch.get_rng_state()))
        self.assertEqual((engine.update_count,engine._adam_steps(),engine.fitted),(200,200,True))
        self.assertEqual(metrics['expected_head_adam_updates'],200)
        self.assertEqual(len(metrics['epoch_history']),11)
        self.assertEqual(len(metrics['minibatch_history']),200)
        self.assertEqual(len(events),211)
        for epoch,row in enumerate(metrics['epoch_history']):
            self.assertEqual((row['epoch'],row['actual_head_adam_updates']),(epoch,epoch*20))
            self.assertEqual(sum(row['binary_confusion'].values()),5120)
            self.assertAlmostEqual(row['total_loss'],row['binary_bce']+.3*row['regression_mse_normalized'],places=6)
            self.assertEqual(row['top_k']['selected_count'],512)
            self.assertTrue(math.isfinite(row['binary_accuracy']))
        for ordinal,row in enumerate(metrics['minibatch_history'],1):
            self.assertEqual((row['epoch'],row['minibatch_index'],row['batch_size']),((ordinal-1)//20+1,(ordinal-1)%20,256))
            self.assertEqual((row['head_update_ordinal'],row['head_adam_steps_before'],row['head_adam_steps_after']),
                             (ordinal,ordinal-1,ordinal))
            self.assertAlmostEqual(row['pre_step_loss'],row['binary_bce']+.3*row['regression_mse_normalized'],places=6)
            self.assertTrue(0 <= row['binary_accuracy'] <= 1)
        checkpoint = engine.state_dict()
        self.assertEqual((checkpoint['update_count'],checkpoint['fitted']),(200,True))
        self.assertTrue(torch.equal(checkpoint['private_rng'],engine.rng.get_state()))
        weights = formal_replay_weights(torch.arange(10))
        torch.testing.assert_close(weights,torch.tensor([2.,1.,.5,.25,.125,.0625,.03125,.015625,0.,0.]),rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'FUTURE'): formal_replay_weights([-1])
        prediction = engine.model(features[:3],[1,1,1]); ages = torch.tensor([0,1,8])
        raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(prediction['binary_logit'],
            ((counts[:3] > 0)&(counts[:3] < 5)).float(),reduction='none')
        target = (counts[:3].float()*(5-counts[:3].float())/20).sqrt()
        groups = raw_bce+.3*((prediction['predicted_statistic']-target)/math.sqrt(.3)).square()
        expected = (groups*torch.tensor([2.,1.,0.])).mean()
        torch.testing.assert_close(formal_replay_loss(prediction,counts[:3],ages,config),expected,rtol=0,atol=0)
        # All-current weights remain2: normalizing their sum would incorrectly halve this objective.
        unweighted,_ = token_binary_loss(prediction,counts[:3],config)
        torch.testing.assert_close(formal_replay_loss(prediction,counts[:3],[0,0,0],config),2*unweighted)
        failing = QwenBlockWarmup(config,state)
        def fail(event):
            if event['stage'] == 'head_optimizer_update': raise RuntimeError('once-only callback failure')
        with self.assertRaisesRegex(RuntimeError,'once-only callback'):
            failing.fit(features[:256],[1]*256,counts[:256],ids[:256],progress_callback=fail)
        self.assertEqual((failing.update_count,failing._adam_steps(),failing.fitted),(1,1,False))


if __name__ == '__main__':
    unittest.main()
