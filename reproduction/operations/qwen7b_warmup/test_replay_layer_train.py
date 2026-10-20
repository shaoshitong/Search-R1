"""Tiny CPU-only fresh-state, padding, plan and durable-evaluation proofs."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import torch
from . import probe_ablation_train as u
from .replay_layer_probe import FlattenReadout,ReplayLayerProbe
from .replay_layer_train import (TARGET,build_plan,adam_steps,initialize,restore,phase,
    checkpoint_blob,save,update,evaluate_epoch,render_sbatch)


class TinyProbe(torch.nn.Module):
    def __init__(self,layer):
        super().__init__();self.block=torch.nn.Linear(2,2,dtype=torch.float64)
        with torch.no_grad():self.block.weight.fill_(layer/100.);self.block.bias.fill_(layer/1000.)
        self.readout=FlattenReadout(2,1,1,seed=1729).double();self.dropout=torch.nn.Dropout(.2)
        self.register_buffer('native_clock',torch.zeros(1,dtype=torch.float64),persistent=False)
    def forward(self,batch):
        self.native_clock.add_(1)
        h=self.dropout(self.block(batch['features'].detach()))
        return self.readout(h,batch['attention_mask'])+self.native_clock.clone()*.0001


class TinyReader:
    def get(self,ids,device):
        values=[]
        for tid in ids:
            q,t=tid.split(':');values.append([[int(q[1:])*.1+.2,int(t[1:])*.03],[.4,.7]])
        return dict(features=torch.tensor(values,dtype=torch.float64,device=device),
            attention_mask=torch.ones(len(ids),2,dtype=torch.bool,device=device))


def fixture():
    device=torch.device('cpu');model,ema,opt=initialize(lambda:TinyProbe(21),1729,device)
    groups=[dict(qid='q'+str(q),trajectory_ids=['q%d:t%d'%(q,t) for t in range(5)],labels=[int(t<k) for t in range(5)]) for q,k in enumerate((1,5))]
    return model,ema,opt,{g['qid']:g for g in groups},device


class ReplayLayerTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_fresh_three_blocks_same_head_ema_copy_adam0(self):
        heads=[];blocks=[]
        for layer in (21,24,27):
            m,e,o=initialize(lambda:TinyProbe(layer),1729,torch.device('cpu'))
            self.assertEqual(adam_steps(o),0);self.assertFalse(o.state);self.assertEqual(u.model_hash(m),u.model_hash(e))
            heads.append(u.model_hash(m.readout));blocks.append(u.model_hash(m.block))
        self.assertEqual(len(set(heads)),1);self.assertEqual(len(set(blocks)),3)
        with self.assertRaises(ValueError):ReplayLayerProbe('/not-an-asset',20)

    def test_flatten_padding_is_zero_and_hidden_gradients_remain(self):
        readout=FlattenReadout(2,2,2).double();hidden=torch.randn(2,4,2,dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([[1,1,1,0],[1,0,0,0]],dtype=torch.bool)
        changed=hidden.detach().clone();changed[~mask]=float('nan')
        self.assertTrue(torch.equal(readout(hidden,mask),readout(changed,mask)))
        readout(hidden,mask).sum().backward();self.assertEqual(float(hidden.grad[~mask].abs().sum()),0.)
        self.assertGreater(float(hidden.grad[mask].abs().sum()),0.)

    def test_full_4epoch_plan_same_cohort_once_each_and_seed(self):
        qids=['q%05d'%i for i in range(6528)];reference=dict(cohort_qids=qids,batches=[qids[i:i+51] for i in range(0,6528,51)])
        plan=build_plan(reference);self.assertEqual(len(plan['batches']),512);self.assertEqual(plan,build_plan(reference))
        self.assertFalse(plan['labels_used'])
        for epoch in range(4):
            own=plan['batches'][epoch*128:(epoch+1)*128]
            self.assertTrue(all(len(b)==51 for b in own));self.assertEqual(sorted(q for b in own for q in b),qids)
        self.assertNotEqual(plan['batches'][:128],plan['batches'][128:256])
        self.assertEqual([n for n in range(1,513) if phase(n)=='EPOCH_TRAINED_EVALUATION_PENDING'],[128,256,384,512])

    def test_actual_toy_next_update_double_replay_and_fullstate_CP0_CP1(self):
        model,ema,opt,lookup,device=fixture();binding=dict(config_sha256='toy-fresh-config')
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);save(base,checkpoint_blob(model,ema,opt,device,0,binding))
            start=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',weights_only=False)
            self.assertEqual(start['steps'],0);self.assertFalse(start['optimizer']['state']);self.assertFalse(start['E1_checkpoint_inherited'])
            update(model,ema,opt,TinyReader(),lookup,list(lookup),dict(micro_groups=1),device)
            expected=u.state_hashes(model,ema,opt,device)
            restore(start,model,ema,opt,device)
            update(model,ema,opt,TinyReader(),lookup,list(lookup),dict(micro_groups=1),device)
            self.assertEqual(u.state_hashes(model,ema,opt,device),expected);self.assertEqual(adam_steps(opt),1)
            receipt=save(base,checkpoint_blob(model,ema,opt,device,1,binding));saved=torch.load(u.checked(receipt),map_location='cpu',weights_only=False)
            self.assertTrue(saved['optimizer']['state']);self.assertEqual(saved['next_batch_index'],1)
            update(model,ema,opt,TinyReader(),lookup,list(lookup),dict(micro_groups=1),device);next_hash=u.state_hashes(model,ema,opt,device)
            restore(saved,model,ema,opt,device)
            update(model,ema,opt,TinyReader(),lookup,list(lookup),dict(micro_groups=1),device)
            self.assertEqual(u.state_hashes(model,ema,opt,device),next_hash);self.assertEqual(adam_steps(opt),2)

    def test_epoch_evaluation_requires_prior_fullstate_and_restores_RNG(self):
        model,ema,opt,lookup,device=fixture();binding=dict(config_sha256='toy-boundary')
        update(model,ema,opt,TinyReader(),lookup,list(lookup),dict(micro_groups=1),device)
        # Synthetic count-only fixture for the epoch boundary, not128 real updates.
        for state in opt.state.values():state['step'].fill_(128)
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);u.atomic(base/'latest.json',dict(cursor=127,steps=127))
            with mock.patch.object(u,'evaluate') as called:
                with self.assertRaisesRegex(ValueError,'Durable'):evaluate_epoch(base,1,model,ema,opt,TinyReader(),[],[],device,binding)
                called.assert_not_called()
            save(base,checkpoint_blob(model,ema,opt,device,128,binding));before=u.state_hashes(model,ema,opt,device)
            def evaluate(_model,_reader,_groups,_device,_dest):
                self.assertEqual(u.read(base/'latest.json')['cursor'],128)
                self.assertTrue(Path(u.read(base/'latest.json')['path']).is_file());torch.rand(3)
                return dict(pool=dict(mixed_balanced_accuracy=.5))
            with mock.patch.object(u,'evaluate',side_effect=evaluate) as called:
                value=evaluate_epoch(base,1,model,ema,opt,TinyReader(),[],[],device,binding)
                self.assertEqual(called.call_count,2)
            self.assertEqual(u.state_hashes(model,ema,opt,device),before);self.assertEqual(value['checkpoint']['cursor'],128)

    def test_normal_one_GPU_two_hour_sbatch(self):
        for layer in (21,24,27):
            script=render_sbatch(TARGET+'/layer'+str(layer),TARGET+'/code-test',layer)
            subprocess.run(['bash','-n'],input=script,text=True,check=True)
            self.assertIn('#SBATCH --gres=gpu:1\n',script);self.assertIn('#SBATCH --time=02:00:00\n',script)
            self.assertIn('#SBATCH --qos=normal\n',script);self.assertNotIn('--dependency',script)


if __name__=='__main__':unittest.main()
