"""Tiny synthetic structure tests only; no actor, checkpoint, or real feature IO."""
import copy,unittest,tempfile,json,hashlib
from pathlib import Path
import torch
from torch import nn
from reproduction.operations.qwen7b_warmup.native_mixed_group_classifier import (
    NativeMixedGroupProbe,MixedTokenHead,block_state_hash,mixed_only_bce,
    fresh_official_training_state,optimizer_parameter_names,map_block_adam_state,OFFICIAL_REVISION)


class TinyBlock(nn.Linear):
    def forward(self,x,**kwargs):
        self.last_kwargs=kwargs
        return (super().forward(x),)


class TinyRotary(nn.Module):
    def forward(self,x,positions):return x,positions


class Legacy(nn.Module):
    def __init__(self):
        super().__init__();self.qmax=3;self.prefix=20;self.width=23;self.cache_position=True
        self.block=TinyBlock(8,8);self.rotary=TinyRotary();self.output=nn.Linear(23*8,1)


_FIXTURE=tempfile.TemporaryDirectory(prefix='official-mixed-tiny-')
def inventory():
    p=Path(_FIXTURE.name)/'official-inventory.json';raw=json.dumps(dict(revision=OFFICIAL_REVISION,synthetic_fixture=True)).encode();p.write_bytes(raw)
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def setup():
    torch.manual_seed(7);legacy=Legacy();rng=torch.get_rng_state().clone();h=block_state_hash(legacy.block)
    state=fresh_official_training_state(legacy,expected_block_hash=h,official_inventory=inventory(),head_seed=9137,hidden_size=8)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    return legacy,state['block_adam'],state['model'],state


def batch(n=10):
    x=torch.randn(n,23,8,requires_grad=True);mask=torch.ones(n,23,dtype=torch.bool);positions=torch.arange(23).expand(n,-1).clone()
    return dict(features=x,attention_mask=mask,position_ids=positions)


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_real_two_layer_256_eight_head_structure_all100_prefix_and_single_logit(self):
        _,_,m,_=setup();h=m.group_head
        self.assertEqual(len(h.layers),2);self.assertEqual(h.dimension,256)
        self.assertTrue(all(layer.self_attn.num_heads==8 and layer.dropout.p==0 for layer in h.layers))
        self.assertFalse(any(name.startswith('output.')for name,_ in m.named_parameters()))
        x=batch();z=m(x);self.assertEqual(tuple(z.shape),(2,));self.assertEqual(tuple(m.block.last_kwargs['cache_position'].shape),(23,))
        encoded=m.encode_trajectories(x);hidden=encoded['hidden'].detach().reshape(2,5,23,8).requires_grad_(True)
        z=h(hidden,x['attention_mask'].reshape(2,5,23),x['position_ids'].reshape(2,5,23));z.sum().backward()
        self.assertTrue(bool((hidden.grad[:,:,3:].abs().sum(-1)>0).all()))

    def test_group_permutation_and_shared_question_averaging(self):
        _,_,m,_=setup();x=batch();enc=m.encode_trajectories(x);enc={k:v.reshape(2,5,*v.shape[1:])for k,v in enc.items()}
        # Q hidden intentionally differ between trajectories, as qualified HF
        # floating arithmetic can; question masks and positions are shared.
        perm=[4,0,3,1,2];a=m.group_logits(enc);b=m.group_logits({k:v[:,perm]for k,v in enc.items()})
        torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
        wrong={k:v.clone()for k,v in enc.items()};wrong['position_ids'][0,1,0]=77
        with self.assertRaisesRegex(ValueError,'share question'):m.group_logits(wrong)

    def test_padding_zero_prefix_nan_erasure_and_no_hidden_pooling(self):
        _,_,m,_=setup();h=m.group_head;x=torch.randn(2,5,23,8);mask=torch.ones(2,5,23,dtype=torch.bool);mask[:,:,2]=False;mask[0,:,3:]=False;mask[1,0,8:]=False;pos=torch.arange(23).expand(2,5,-1)
        a=h(x,mask,pos);changed=x.clone();changed[~mask]=float('nan');b=h(changed,mask,pos)
        torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertTrue(bool(torch.isfinite(b).all()))

    def test_only_mixed_bce_joint_gradient_feature_detach_and_target_validation(self):
        _,_,m,_=setup();x=batch();z=m(x);ys=torch.tensor([[0,0,0,0,0],[0,1,0,1,0]])
        loss=mixed_only_bce(z,ys);ref=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([0.,1.]));torch.testing.assert_close(loss,ref,rtol=0,atol=0)
        loss.backward();self.assertIsNone(x['features'].grad);self.assertGreater(float(m.block.weight.grad.abs().sum()),0);self.assertGreater(float(m.group_head.classifier.weight.grad.abs().sum()),0)
        with self.assertRaises(ValueError):mixed_only_bce(z,torch.ones(2,10))

    def test_official_block_copy_fresh_Adam0_EMA_raw_identity_and_first_update(self):
        legacy,bo,m,state=setup();ho=state['head_adam'];ema=state['ema'];source_hash=block_state_hash(legacy.block)
        self.assertIs(m.block,legacy.block);self.assertFalse(bo.state);self.assertFalse(ho.state)
        for name,value in m.state_dict().items():torch.testing.assert_close(value,ema.state_dict()[name],rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for p in ema.parameters()))
        self.assertEqual(state['provenance']['native_Adam_steps'],0);self.assertEqual(state['provenance']['group_head_Adam_steps'],0)
        self.assertFalse(state['provenance']['posttraining_probe_checkpoint_read']);self.assertEqual(block_state_hash(m.block),source_hash)
        retired=legacy.output.weight.detach().clone();x=batch();loss=mixed_only_bce(m(x),torch.tensor([[0,1,0,1,0],[1,1,1,1,1]]));loss.backward();bo.step();ho.step()
        self.assertEqual({int(s['step'])for s in bo.state.values()},{1});self.assertEqual({int(s['step'])for s in ho.state.values()},{1})
        torch.testing.assert_close(legacy.output.weight,retired,rtol=0,atol=0)

    def test_official_revision_inventory_hash_geometry_and_Adam_mapping_fail_closed(self):
        legacy,_,m,_=setup();kw=dict(expected_block_hash=block_state_hash(legacy.block),official_inventory=inventory(),head_seed=1,hidden_size=8)
        with self.assertRaisesRegex(ValueError,'revision'):NativeMixedGroupProbe.from_official_probe(legacy,official_revision='wrong',**kw)
        bad=dict(kw);bad['expected_block_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'block hash'):NativeMixedGroupProbe.from_official_probe(legacy,**bad)
        bad=dict(kw);bad['official_inventory']=dict(kw['official_inventory'],sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'inventory SHA'):NativeMixedGroupProbe.from_official_probe(legacy,**bad)
        old=torch.optim.Adam(legacy.parameters(),lr=5e-6);names=optimizer_parameter_names(legacy,old)
        with self.assertRaises(ValueError):map_block_adam_state(old.state_dict(),names[:-1],['block.weight','block.bias'])



if __name__=='__main__':unittest.main()
