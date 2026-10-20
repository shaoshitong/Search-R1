"""CPU toy inheritance/recovery tests; no official weights or GPU touched."""
import copy
import tempfile
from pathlib import Path
import unittest
import torch
from . import probe_ablation_train as u
from .l27_readout_train import (initialize_branch,adam_count,branch_hashes,
    restore_branch,save,update,last_scope_alias_proof)
from .probe_ablation_model import TokenReadout


class Legacy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.block=torch.nn.Linear(2,2,dtype=torch.float64);self.output=torch.nn.Linear(4,1,dtype=torch.float64)
    def forward(self,data):
        h=self.block(data['features']);return self.output(torch.cat((h,h),1)).flatten()


class NewHead(torch.nn.Module):
    def __init__(self,attention=False):
        super().__init__();self.block=torch.nn.Linear(2,2,dtype=torch.float64)
        self.readout=TokenReadout(2,'learned_attention' if attention else 'mean','prefix_only',1,1).double()
    def forward(self,data):
        h=self.block(data['features']);hidden=torch.stack((h,h+.1),1)
        return self.readout(hidden,torch.ones(len(h),2,dtype=torch.bool))


class Reader:
    def get(self,ids,device):return dict(features=torch.tensor([[i/100.,(i%7)/10.] for i in ids],dtype=torch.float64,device=device))


def source():
    torch.manual_seed(19);model=Legacy();ema=copy.deepcopy(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6)
    optimizer.zero_grad();model(dict(features=torch.ones(3,2,dtype=torch.float64))).sum().backward();optimizer.step()
    for state in optimizer.state.values():state['step'].fill_(10464)
    rng=u.rng_state(torch.device('cpu'))
    return copy.deepcopy(dict(raw=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
        raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),steps=10464,
        rank_states=[dict(rank=i,rng=copy.deepcopy(rng)) for i in range(4)]))


def groups():
    return {i:dict(qid=i,trajectory_ids=list(range(i*5,i*5+5)),labels=[1,0,0,0,0] if i%2 else [0]*5) for i in range(51)}


class ReadoutRecoveryTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_named_block_moments_inherit_new_head_starts_zero_and_seed_matches(self):
        saved=source();device=torch.device('cpu')
        mean=initialize_branch(saved,lambda:NewHead(False),1729,device)
        attn=initialize_branch(saved,lambda:NewHead(True),1729,device)
        for result in (mean,attn):
            model,ema,bo,ho,proof=result
            self.assertEqual(adam_count(bo),10464);self.assertEqual(adam_count(ho),0)
            self.assertFalse(proof['whole_model_lossless_resume'])
            for name,value in model.block.state_dict().items():
                self.assertTrue(torch.equal(value,saved['raw']['block.'+name]))
                self.assertTrue(torch.equal(ema.block.state_dict()[name],saved['ema']['block.'+name]))
            for index,state in bo.state_dict()['state'].items():
                self.assertTrue(torch.equal(state['exp_avg'],saved['optimizer']['state'][index]['exp_avg']))
                self.assertTrue(torch.equal(state['exp_avg_sq'],saved['optimizer']['state'][index]['exp_avg_sq']))
        self.assertTrue(torch.equal(mean[0].readout.output.weight,attn[0].readout.output.weight))
        self.assertTrue(torch.equal(mean[0].readout.output.bias,attn[0].readout.output.bias))
        self.assertEqual(float(attn[0].readout.attention_key.detach().abs().sum()),0.)

    def test_fresh_flatten_control_keeps_block_but_drops_old_head_adam(self):
        saved=source();model,ema,bo,ho,proof=initialize_branch(saved,Legacy,1729,torch.device('cpu'))
        self.assertEqual((adam_count(bo),adam_count(ho)),(10464,0))
        self.assertFalse(torch.equal(model.output.weight,saved['raw']['output.weight']))
        self.assertTrue(torch.equal(model.output.weight,ema.output.weight))
        self.assertTrue(torch.equal(model.block.weight,saved['raw']['block.weight']))
        self.assertFalse(proof['source_old_head_inherited'])

    def test_real_toy_update_replay_and_checkpoint_restore_separate_counters(self):
        device=torch.device('cpu');saved=source()
        model,ema,bo,ho,_=initialize_branch(saved,lambda:NewHead(True),1729,device)
        binding=dict(config_sha256='config',matched_binding_sha256='matched')
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);initial=save(base,model,ema,bo,ho,device,0,binding,saved['rank_states'])
            blob=torch.load(u.checked(initial),map_location='cpu',mmap=True,weights_only=False)
            config=dict(micro_groups=2)
            update(model,ema,bo,ho,Reader(),groups(),list(range(51)),config,device)
            first=branch_hashes(model,ema,bo,ho,device)
            self.assertEqual((adam_count(bo),adam_count(ho)),(10465,1))
            restore_branch(blob,model,ema,bo,ho,device)
            update(model,ema,bo,ho,Reader(),groups(),list(range(51)),config,device)
            self.assertEqual(first,branch_hashes(model,ema,bo,ho,device))
            receipt=save(base,model,ema,bo,ho,device,1,binding,saved['rank_states'])
            restored=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
            update(model,ema,bo,ho,Reader(),groups(),list(range(51)),config,device)
            restore_branch(restored,model,ema,bo,ho,device)
            self.assertEqual(first,branch_hashes(model,ema,bo,ho,device))
            self.assertEqual((adam_count(bo),adam_count(ho)),(10465,1))
            self.assertEqual(int(saved['optimizer']['state'][0]['step']),10464)

    def test_last_scope_alias_requires_nonempty_prefix_in_all_three_roles(self):
        class Metadata:
            def __init__(self,empty=False):self.empty=empty
            def alias_role(self,ids):return dict(trajectories=len(ids),empty_prefix=int(self.empty),minimum_prefix=0 if self.empty else 1,maximum_prefix=20)
        group=[dict(trajectory_ids=list(range(5)))]
        self.assertTrue(last_scope_alias_proof(Metadata(),Metadata(),group,group,group)['eligible'])
        self.assertFalse(last_scope_alias_proof(Metadata(),Metadata(True),group,group,group)['eligible'])
        self.assertFalse(last_scope_alias_proof(Metadata(),Metadata(),group,group,group)['real_duplicate_GPU_execution_claimed'])


if __name__=='__main__':unittest.main()
