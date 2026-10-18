import copy
import unittest
import os
import json
import subprocess
import sys
import torch
from .progressive_prefix import (actor_prefix,after_boundary,boundary,before_fit,
                                 expand_probe_state,DIM,QMAX)


class ProgressivePrefixTests(unittest.TestCase):
    def test_short_prefix_padding_is_masked_at_every_new_length(self):
        code='''
import torch
from reproduction.prefix_selection200.contract import WIDTH,PREFIX_TOKENS,QMAX,DIM
from reproduction.prefix_selection200.prefix5_model import validate_online_layout
n=3;plen=7
features=torch.zeros(n,WIDTH,DIM);mask=torch.zeros(n,WIDTH,dtype=torch.bool)
question=mask.clone();policy=mask.clone();position=torch.zeros(n,WIDTH,dtype=torch.long)
question[:,:10]=True;position[:,:10]=torch.arange(10)
policy[:,QMAX:QMAX+plen]=True;position[:,QMAX:QMAX+plen]=torch.arange(50,50+plen)
batch=dict(features=features,attention_mask=question|policy,position_ids=position,
 question_mask=question,policy_mask=policy,question_lengths=torch.full((n,),10,dtype=torch.long),
 policy_lengths=torch.full((n,),plen,dtype=torch.long),
 policy_segments=[[dict(turn_index=0,policy_start=0,policy_length=plen,position_start=50)] for _ in range(n)])
validate_online_layout(batch,32768)
assert not policy[:,QMAX+plen:].any() and not position[:,QMAX+plen:].any()
print('PAD_MASK_PASS')
'''
        for prefix in (20,40,80):
            env=dict(os.environ,DYNRPO_PROGRESSIVE_PREFIX='cp50_prefix20_40_80_three_epoch_v1',DYNRPO_PREFIX_TOKENS=str(prefix))
            self.assertEqual(subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip(),'PAD_MASK_PASS')

    def test_all_stage_processes_have_three_complete_g5_epochs(self):
        code='''
import json
from reproduction.prefix_selection200.contract import PREFIX_TOKENS,WIDTH,PROBE_EPOCHS,PROBE_UPDATES
from reproduction.prefix_selection200.probe_plan import complete_pool_plan,rank_question_slice
from reproduction.prefix_selection200.periodic import probe_boundary
p=complete_pool_plan(1729,50)
assert len(p)==150
for epoch in range(1,4):
 batches=[r for r in p if r['epoch']==epoch]
 assert sorted(i for b in batches for i in b['question_indices'])==list(range(2560))
 assert all(sum(len(rank_question_slice(b,r)) for b in batches)==640 for r in range(4))
print(json.dumps([PREFIX_TOKENS,WIDTH,PROBE_EPOCHS,PROBE_UPDATES,probe_boundary(50)['actual_adam_per_rank']]))
'''
        for prefix in (20,40,80):
            env=dict(os.environ,DYNRPO_PROGRESSIVE_PREFIX='cp50_prefix20_40_80_three_epoch_v1',DYNRPO_PREFIX_TOKENS=str(prefix))
            result=subprocess.check_output([sys.executable,'-c',code],env=env,text=True)
            self.assertEqual(json.loads(result),[prefix,273+prefix,3,150,400])

    def test_boundary_lengths_do_not_shift_actor_history(self):
        self.assertEqual([actor_prefix(n) for n in [1,50,51,100,101,150,151,200]],
                         [5,5,20,20,40,40,80,80])
        self.assertEqual([after_boundary(n) for n in [49,50,99,100,149,150]],
                         [5,20,20,40,40,80])

    def test_three_epochs_retain_old250_clock(self):
        self.assertEqual(before_fit(50)['actual_adam_per_rank'],250)
        self.assertEqual([boundary(n)['actual_adam_per_rank'] for n in [50,59,60,100,150,190,200]],
                         [400,400,550,1150,1900,2500,2500])
        self.assertEqual(boundary(200)['last_update_outer'],190)

    def test_expansion_preserves_block_moments_bias_clock_and_rng(self):
        model={'block.weight':torch.randn(3,3),'output.weight':torch.randn(1,(QMAX+5)*DIM),
               'output.bias':torch.randn(1)}
        opt=torch.optim.Adam([model[k].requires_grad_() for k in model],lr=5e-6)
        for value in model.values():value.grad=torch.ones_like(value)
        opt.step()
        saved=dict(model={k:v.detach().clone() for k,v in model.items()},optimizer=opt.state_dict(),
                   optimizer_parameter_names=list(model),rank_rng=[{'rank':r,'rng':'retained'} for r in range(4)])
        previous=copy.deepcopy(saved)
        changed=expand_probe_state(saved,20)
        for name in ['block.weight','output.bias']:
            self.assertTrue(torch.equal(changed['model'][name],previous['model'][name]))
        self.assertTrue(torch.equal(changed['model']['output.weight'][:,:278*DIM],previous['model']['output.weight']))
        self.assertEqual(torch.count_nonzero(changed['model']['output.weight'][:,278*DIM:]),0)
        for i in [0,2]:
            for key in ['step','exp_avg','exp_avg_sq']:
                self.assertTrue(torch.equal(changed['optimizer']['state'][i][key],previous['optimizer']['state'][i][key]))
        for key in ['exp_avg','exp_avg_sq']:
            self.assertTrue(torch.equal(changed['optimizer']['state'][1][key][:,:278*DIM],previous['optimizer']['state'][1][key]))
            self.assertEqual(torch.count_nonzero(changed['optimizer']['state'][1][key][:,278*DIM:]),0)
        self.assertEqual(changed['rank_rng'],previous['rank_rng'])
        self.assertEqual(tuple(saved['model']['output.weight'].shape),(1,278*DIM))


if __name__=='__main__':unittest.main()
