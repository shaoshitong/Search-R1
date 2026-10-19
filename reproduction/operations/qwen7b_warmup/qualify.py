"""CPU invariants using real packer, Prefix5 forward, G5 loss and Adam."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from contextlib import ExitStack
import numpy as np
import torch
from torch import nn
from .packed_features import NativeFeatureWriter,FeatureReader,DIM,QMAX,PREFIX,WIDTH
from .model import Probe
from .train import balanced_plan
from .sampling import prefix_record,canonical
import hashlib
from .download_model import atomic
from reproduction.prefix20_group_joint100.loss import joint_loss


class Echo(nn.Module):
    def forward(self,x,**kwargs):return (x,)


class Rotary(nn.Module):
    def forward(self,x,positions):return None


def main():
    base=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
    torch.manual_seed(19);random.seed(19);np.random.seed(19)
    with tempfile.TemporaryDirectory(prefix='qualification-',dir=str(base/'tmp')) as temporary, ExitStack() as cleanup:
        root=Path(temporary);tids=['q%d-r%d'%(q,j) for q in range(2) for j in range(5)]
        writer=NativeFeatureWriter(root,0,tids);expected={}
        for q in range(2):
            length=min(QMAX,4+q);question=torch.randn(length,DIM).bfloat16()
            for j in range(5):
                tid='q%d-r%d'%(q,j);plen=[0,2,5,12,20][j]
                features=torch.zeros(WIDTH,DIM,dtype=torch.bfloat16)
                features[:length]=(question.float()+.125*j).bfloat16()
                features[QMAX:QMAX+plen]=torch.randn(plen,DIM).bfloat16()
                positions=torch.zeros(WIDTH,dtype=torch.long);positions[:length]=torch.arange(length)
                positions[QMAX:QMAX+plen]=torch.arange(plen)+length+17
                writer.write(dict(trajectory_id=tid,qid=str(q),initial_prompt_token_ids=list(range(length)),actor_version='official7b-test'),
                    dict(features=features,position_ids=positions,question_length=length,policy_length=plen))
                expected[tid]=(features,positions)
        manifest=writer.finish();reader=FeatureReader([str(manifest)]);cleanup.callback(reader.close);data=reader.get(tids,'cpu')
        for i,tid in enumerate(tids):
            assert torch.equal(data['features'][i],expected[tid][0])
            assert torch.equal(data['position_ids'][i],expected[tid][1])
        probe=Probe.__new__(Probe);nn.Module.__init__(probe)
        probe.qmax=QMAX;probe.prefix=5;probe.width=QMAX+5;probe.block=Echo();probe.rotary=Rotary()
        probe.cache_position=False
        probe.output=nn.Linear((QMAX+5)*DIM,1)
        with torch.autocast('cpu',dtype=torch.bfloat16):before=probe(data).detach()
        future={k:v.clone() for k,v in data.items()}
        future['features'][:,QMAX+5:]=1000;future['position_ids'][:,QMAX+5:]=99999
        future['attention_mask'][:,QMAX+5:]=~future['attention_mask'][:,QMAX+5:]
        with torch.autocast('cpu',dtype=torch.bfloat16):after=probe(future).detach()
        assert torch.equal(before,after),'Later15 tokens leaked into Prefix5'
        groups=[dict(labels=([0]*5 if i<7 else [0,1,0,1,0] if i<11 else [1]*5)) for i in range(16)]
        batches,info=balanced_plan(groups,17,1);flattened=[i for batch in batches for i in batch]
        assert set(flattened)==set(range(16)) and len(flattened)==21
        assert balanced_plan(groups,17,1)==(batches,info)
        for n in (4,5,47,51):
            z=torch.randn(n,5,dtype=torch.float64,requires_grad=True)
            y=torch.randint(0,2,(n,5)).double();loss=joint_loss(z,y)['loss']
            full=torch.autograd.grad(loss,z)[0];local=torch.zeros_like(z)
            for rank in range(4):
                ix=list(range(rank,n,4));v=z.detach()[ix].clone().requires_grad_()
                value=joint_loss(v,y[ix])['loss']*(4*len(ix)/n)
                local[ix]=torch.autograd.grad(value,v)[0]/4
            assert torch.allclose(full,local,rtol=1e-12,atol=1e-12)
        model=nn.Linear(3,1).double();adam=torch.optim.Adam(model.parameters(),lr=5e-6)
        def step():
            x=torch.randn(5,3,dtype=torch.float64);y=torch.randn(5,1,dtype=torch.float64)
            adam.zero_grad();((model(x)-y)**2).mean().backward();adam.step()
        step();saved=copy.deepcopy(dict(model=model.state_dict(),adam=adam.state_dict(),rng=torch.get_rng_state()))
        path=root/'small-complete-state.pt';torch.save(saved,path);step()
        uninterrupted=copy.deepcopy(dict(model=model.state_dict(),adam=adam.state_dict()))
        restored=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(restored['model']);adam.load_state_dict(restored['adam']);torch.set_rng_state(restored['rng']);step()
        assert all(torch.equal(v,uninterrupted['model'][k]) for k,v in model.state_dict().items())
        assert all(torch.equal(v,uninterrupted['adam']['state'][k][name]) for k,s in adam.state_dict()['state'].items() for name,v in s.items())
        fixture=dict(run_id='fixture',outer=1,actor_version='fixture',trajectory_id='fixture-g0',
            candidate_ordinal=0,response_index=0,prompt=[1,2],turns=[],terminal=False,terminal_reason=None,
            action=dict(actual_request_token_ids=[1,2],raw_generated_token_ids=list(range(3,23)),turn_index=0),
            group=dict(qid='fixture',data_source='fixture',question_hash='0'*64,split='diagnostic'))
        prefix=prefix_record(fixture);assert prefix['split']=='diagnostic'
        digest=prefix.pop('prefix_sha256');assert hashlib.sha256(canonical(prefix)).hexdigest()==digest
        report=dict(status='CPU_INVARIANTS_PASSED',lossless_nonidentical_question_features=True,
            short_answer_and_position_gaps=True,prefix5_future15_invariance=True,whole_G5_balance=True,
            unequal_DDP_rank_gradient_matches_global_mean=True,Adam_RNG_next_step_exact=True,
            diagnostic_split_bound_in_prefix_hash=True,
            feature_layer=26,head_layer=27,layer_indexing='zero_based_model.layers',fit_prefix=5,capture_prefix=20,
            gpu_native_capture_and_four_rank_resume='NOT_YET_TESTED')
        atomic(base/'preparation/cpu-qualification.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
