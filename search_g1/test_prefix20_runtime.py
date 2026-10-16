"""CPU transactions and real tiny Adam snapshots; no GPU/model acceptance claim."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from search_g1.prefix20_runtime import Prefix20Runtime,validate_config,validate_saved_state
from search_g1.training_state import cpu_tree


def settings(base):
    return validate_config(dict(model_dir=str(base/'official'),head_safetensors=str(base/'head.safetensors'),
        probe_config=str(base/'probe-config.json'),output_dir=str(base/'artifacts/question-probe'),
        eligible_pool_path=str(base/'eligible-question-pool.json')))


def rows():
    return [dict(trajectory_id='t-%d'%i,qid='q-%d'%(i//5),response_index=i%5,strict_em=int(i%5<2)) for i in range(2560)]


class TinyFourRankTransport:
    def __init__(self):
        self.parameter=torch.nn.Parameter(torch.tensor([.2]))
        self.optimizer=torch.optim.Adam([self.parameter],lr=1e-6,weight_decay=0)
        self.outer=0;self.steps=0;self.updates=[];self.saved=None

    def states(self):
        return [dict(rank=rank,completed_outer=self.outer,actual_steps=self.steps,
            model={'weight':self.parameter.detach().clone()},optimizer=cpu_tree(self.optimizer.state_dict()),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.tensor([rank],dtype=torch.uint8))) for rank in range(4)] if self.saved is None else copy.deepcopy(self.saved)

    def __call__(self,operation,payloads):
        if operation=='init':
            return [dict(rank=r,world_size=4,completed_outer=0,actual_steps=0,pretrained_head_loaded=True) for r in range(4)]
        if operation=='predict':
            probability=float(torch.sigmoid(self.parameter.detach()))
            return [dict(rank=r,completed_outer=self.outer,actual_steps=self.steps,
                predictions=[dict(trajectory_id=row['trajectory_id'],em_probability=probability) for row in p['rows']])
                for r,p in enumerate(payloads)]
        if operation=='update':
            self.updates.append(copy.deepcopy(payloads))
            for step in range(10):
                labels=torch.tensor([row['strict_em'] for p in payloads for row in p['batches'][step]],dtype=torch.float32)
                self.optimizer.zero_grad()
                torch.nn.functional.binary_cross_entropy_with_logits(self.parameter.expand(256),labels).backward()
                self.optimizer.step();self.steps+=1
            self.outer=payloads[0]['outer']
            return [dict(rank=r,completed_outer=self.outer,actual_steps=self.steps,local_trajectories=640) for r in range(4)]
        if operation=='state_dict':return self.states()
        if operation=='load_state_dict':
            self.saved=[copy.deepcopy(p['state']) for p in payloads]
            self.outer=self.saved[0]['completed_outer'];self.steps=self.saved[0]['actual_steps']
            return [dict(rank=r,completed_outer=self.outer,actual_steps=self.steps,model_optimizer_rng_restored=True,
                actual_readback_equal=True,actor_rng_unchanged=True) for r in range(4)]
        raise AssertionError(operation)


class Prefix20RuntimeTests(unittest.TestCase):
    def runtime(self,base):
        transport=TinyFourRankTransport()
        return Prefix20Runtime(settings(base),base,transport=transport),transport

    def test_one_new_global_pass_ten_steps_and_training_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,transport=self.runtime(Path(directory));data=rows()
            before=runtime.score([{k:v for k,v in row.items() if k!='strict_em'} for row in data])
            runtime.begin_outer(1,'1'*32,data,before,'actor-before-1')
            event=runtime.complete_outer(1,'actor-after-1',dict(attempt_id='1'*32))
            flattened=[row['trajectory_id'] for p in transport.updates[0] for batch in p['batches'] for row in batch]
            self.assertEqual(len(flattened),2560);self.assertEqual(set(flattened),set(before))
            self.assertEqual((runtime.completed_outer,transport.steps),(1,10))
            self.assertEqual(event['head_update']['selected_mixed_fraction'],1.)
            self.assertAlmostEqual(event['head_update']['selected_mean_true_sample_std'],np.sqrt(.3))
            self.assertEqual(event['head_update']['before_positive_fraction'],.4)
            self.assertEqual(event['head_update']['before_binary_accuracy'],.4)
            self.assertIsNone(runtime.pending)

    def test_abort_does_not_fit_and_labels_or_group_reordering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime,transport=self.runtime(Path(directory));data=rows();prob={r['trajectory_id']:.5 for r in data}
            with self.assertRaisesRegex(ValueError,'LABEL_LEAK'):runtime.score(data)
            bad=copy.deepcopy(data);bad[1]['qid']='wrong-group'
            with self.assertRaisesRegex(ValueError,'GROUP_ORDER'):runtime.begin_outer(1,'1'*32,bad,prob,'old')
            runtime.begin_outer(1,'1'*32,data,prob,'old');runtime.abort_outer(1,dict(attempt_id='1'*32))
            self.assertEqual((transport.steps,len(transport.updates),runtime.completed_outer),(0,0,0))

    def test_four_saved_states_fresh_sha_and_actual_restore_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);runtime,transport=self.runtime(base);data=rows()
            runtime.begin_outer(1,'1'*32,data,{r['trajectory_id']:.5 for r in data},'before')
            with self.assertRaisesRegex(ValueError,'PARTIAL'):runtime.save_checkpoint(base/'cp',1,{},'reward',actor_snapshot_id='actor:1')
            runtime.complete_outer(1,'actor:1')
            receipt=runtime.save_checkpoint(base/'cp',1,{'cursor':7},'reward',actor_snapshot_id='actor:1')
            self.assertEqual(len(receipt['rank_files']),4)
            restored,other=self.runtime(base/'restored')
            restored.restore(receipt['path'],'actor:1')
            self.assertEqual((restored.completed_outer,other.steps),(1,10))
            path=base/'cp/prefix20-rank-3.pt';raw=bytearray(path.read_bytes());raw[-30]^=1;path.write_bytes(raw)
            state=torch.load(receipt['path'],map_location='cpu',weights_only=False)
            with self.assertRaisesRegex(ValueError,'FILE_SHA'):validate_saved_state(state,receipt['path'],'actor:1')

    def test_resolved_four_gpu_config_has_new_namespace_and_virtual_zero(self):
        from reproduction.common8.test_configuration import ConfigurationTests
        from reproduction.common8.configuration import resolved
        from reproduction.hpc3_qprobe.test_native4 import NativeFourTests
        from reproduction.common8.formal_contract import formal_mathematics
        with tempfile.TemporaryDirectory() as directory:
            base,run=ConfigurationTests().fixture(directory);run.update(NativeFourTests().binding())
            run['budget']['world_size']=4
            run['budget'].update(formal_mathematics(run));run['question_probe']=settings(base)
            run['data_selection_mode']='prefix20_probe'
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,config,args=resolved(base)
            self.assertEqual(config.data.selection_mode,'prefix20_probe')
            self.assertTrue(config.algorithm.all_correct_virtual_zero)
            self.assertEqual((config.trainer.n_gpus_per_node,config.prefix20_probe.head_batch_size,
                config.prefix20_probe.exploration_questions),(4,256,0))
            self.assertEqual(config.actor_rollout_ref.actor.ppo_micro_batch_size,16)
            self.assertEqual(len(args),len({a.split('=',1)[0] for a in args}))

    def test_cpu_global_gradient_matches_four_equal_local64_means(self):
        inputs=torch.linspace(-2,2,256);labels=(torch.arange(256)%3==0).float()
        whole=torch.tensor(.3,requires_grad=True)
        torch.nn.functional.binary_cross_entropy_with_logits(whole*inputs,labels).backward()
        gradients=[]
        for rank in range(4):
            parameter=torch.tensor(.3,requires_grad=True);sl=slice(rank*64,(rank+1)*64)
            torch.nn.functional.binary_cross_entropy_with_logits(parameter*inputs[sl],labels[sl]).backward()
            gradients.append(parameter.grad)
        torch.testing.assert_close(whole.grad,torch.stack(gradients).mean())


if __name__=='__main__':unittest.main()
