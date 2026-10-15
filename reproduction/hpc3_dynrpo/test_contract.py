"""Actual native-Qwen continuation parity; no GPU or external assets required."""
import copy
from dataclasses import asdict
from pathlib import Path
import json
import unittest

import torch

from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig,QwenBlockWarmup
from reproduction.hpc3_dynrpo.warmup import fit_continuation


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class NativeContinuationTests(unittest.TestCase):
    def test_real_native_fit_matches_interruption_and_true_adam_resume(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        torch.set_num_threads(1)
        config=QwenBlockQProbeConfig(max_tokens=2,hidden_size=8,num_attention_heads=2,num_key_value_heads=1,
            intermediate_size=16,epochs=2,batch_size=8,micro_batch_size=4,top_k=4)
        official=Qwen2DecoderLayer(config.native_config(),layer_idx=35).state_dict()
        official={k:v.detach().to(torch.bfloat16).clone() for k,v in official.items()}
        data=torch.randn(16,2,8,generator=torch.Generator().manual_seed(93));lengths=torch.ones(16,dtype=torch.long)*2
        counts=torch.arange(16)%6;ids=['q'+str(i) for i in range(16)]
        native=QwenBlockWarmup(config,official);expected=native.fit(data,lengths,counts,ids)
        actual=QwenBlockWarmup(config,official);history=dict(epoch_history=[],minibatch_history=[])
        def interrupt(event):
            if event['stage']=='head_optimizer_update' and event['head_adam_steps_after']==3:
                raise RuntimeError('Injected failure after real third Adam return')
        with self.assertRaisesRegex(RuntimeError,'third Adam'):
            fit_continuation(actual,data,lengths,counts,ids,history,interrupt)
        saved=actual.state_dict();actual=QwenBlockWarmup(config,official)
        actual.model.load_state_dict(saved['model']);actual.optimizer.load_state_dict(saved['optimizer'])
        actual.rng.set_state(saved['private_rng']);actual.update_count=saved['update_count']
        result=fit_continuation(actual,data,lengths,counts,ids,history)
        self.assertTrue(equal(native.model.state_dict(),actual.model.state_dict()))
        self.assertTrue(equal(native.optimizer.state_dict(),actual.optimizer.state_dict()))
        self.assertTrue(torch.equal(native.rng.get_state(),actual.rng.get_state()))
        self.assertTrue(equal(expected['minibatch_history'],result['minibatch_history']))
        self.assertTrue(equal(expected['epoch_history'],result['epoch_history']))
        self.assertEqual(actual._adam_steps(),4)

    def test_effective_recipe_is_the_requested_release(self):
        from reproduction.common8.qprobe_qwen_bind import recipe
        from reproduction.common8.qprobe_configuration import arguments
        release=json.loads((Path(__file__).parents[2]/'docs/releases/qprobe-qwen-native-block-2026-10-15-config.json').read_text())
        settings=recipe('/data/own/token-features')
        for k,v in release['question_probe'].items():
            if k in settings:self.assertEqual(settings[k],v,k)
        settings.update(initial_head_checkpoint='/data/own/initial.pt',initial_head_checkpoint_sha256='a'*64,
                        initial_head_ready='/data/own/ready.json',initial_head_ready_sha256='b'*64)
        self.assertTrue(arguments(dict(question_probe=settings),'/data/own'))


if __name__=='__main__':
    unittest.main()
