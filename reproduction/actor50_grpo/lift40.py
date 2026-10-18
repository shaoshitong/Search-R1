"""At outer100 extend the head and Adam moments; preserve all existing state."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def extend(value,width):
    import torch
    if value.ndim!=2 or value.shape[0]!=1 or value.shape[1]>=width:
        raise ValueError('Only a strictly wider flattened Linear head may be extended')
    result=torch.zeros((1,width),dtype=value.dtype,device=value.device)
    result[:,:value.shape[1]].copy_(value)
    assert torch.equal(result[:,:value.shape[1]],value) and torch.count_nonzero(result[:,value.shape[1]:])==0
    return result


def lift(saved):
    import torch
    result=copy.deepcopy(saved);old_width=293*2048;new_width=313*2048
    assert tuple(result['model']['output.weight'].shape)==(1,old_width)
    result['model']['output.weight']=extend(result['model']['output.weight'],new_width)
    optimizer=result['optimizer'];assert optimizer is not None and len(optimizer['param_groups'])==1
    params=optimizer['param_groups'][0]['params'];assert len(params)==14
    # Native model registration order is block's12 parameters, then output W,b.
    state=optimizer['state'][params[12]]
    assert tuple(state['exp_avg'].shape)==tuple(state['exp_avg_sq'].shape)==(1,old_width)
    for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        if key in state:state[key]=extend(state[key],new_width)
    assert all(int(v['step'])==saved['actual_adam_steps_per_rank'][0] for v in optimizer['state'].values())
    result['binding']=dict(saved['binding'],prefix_tokens=40,head_max_tokens=313)
    result['binding_sha256']=hashlib.sha256(json.dumps(result['binding'],sort_keys=True,
        separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return result


def main():
    import os,time,torch
    from reproduction.prefix_selection200.probe import read,sha,once,atomic_checkpoint
    from reproduction.prefix_selection200.prefix5_model import validate_model_state,validate_native_buffers
    p=argparse.ArgumentParser();p.add_argument('--state-dir',required=True);p.add_argument('--outer',type=int,required=True);a=p.parse_args()
    assert a.outer==100 and os.environ['SEARCH_G1_GRPO_PREFIX']=='40'
    root=Path(a.state_dir);receipt=read(root/'latest.json');assert receipt['completed_outer']==90 and receipt['actual_adam_steps_per_rank']==[200]*4
    assert (root/'latest.pt').stat().st_size==receipt['bytes'] and sha(root/'latest.pt')==receipt['sha256']
    saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    assert saved['binding']['prefix_tokens']==20 and saved['completed_outer']==90
    before=root/'prefix20-before-outer100.pt';assert not before.exists();os.link(root/'latest.pt',before)
    result=lift(saved);validate_model_state(result['model']);validate_native_buffers(result['model_nonpersistent_buffers'])
    proof=dict(kind='Prefix20_to40_before_outer100',source_sha256=receipt['sha256'],
        source_completed_outer=90,actual_adam_steps_per_rank=receipt['actual_adam_steps_per_rank'],
        original_model_and_moments_bit_preserved=True,new20_head_columns_and_moments_zero=True,
        optimizer_reset=False,rank_RNG_preserved=True,checked_unix=time.time())
    result['dimension_transition']=proof;after=atomic_checkpoint(root,result,90)
    once(root/'prefix40-transition.json',dict(proof,checkpoint=after))
