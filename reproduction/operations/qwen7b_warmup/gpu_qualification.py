"""Actual four-rank recovery check on a separate shadow probe/Adam only."""
import copy
import gc
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .download_model import atomic,fingerprints


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    return copy.deepcopy(value)


def comparison(actual,expected,*,parameter=False):
    maximum=0.;passed=True;exact=True
    for name,before in expected.items():
        after=actual[name].detach().cpu()
        if before.numel():maximum=max(maximum,float((after-before).abs().max()))
        exact=exact and torch.equal(after,before)
        passed=passed and torch.allclose(after,before,rtol=0 if parameter else 1e-5,atol=1e-7)
    return dict(passed=passed,bit_exact=exact,max_absolute_error=maximum,rtol=0 if parameter else 1e-5,atol=1e-7)


def qualify(base,c,reader,groups,production,rank,device,config_sha,bank_sha,update,save,restore,evaluate):
    untouched=cpu_tree(production.state_dict());original_rng=rng();start=time.monotonic()
    root=base/('qualification-'+os.environ['SLURM_JOB_ID']);root.mkdir(exist_ok=True)
    shadow=copy.deepcopy(production).train();ddp=DDP(shadow,device_ids=[device.index],broadcast_buffers=False)
    ema=copy.deepcopy(shadow).eval().requires_grad_(False);optimizer=torch.optim.Adam(shadow.parameters(),lr=5e-6,weight_decay=0)
    # Test the real native block, not an echo module: later15 features/masks/
    # positions cannot affect Prefix5. Only training-bank questions are used.
    ids=[tid for g in groups[rank:rank+2] for tid in g['trajectory_ids']]
    data=reader.get(ids,device);altered={k:v.clone() for k,v in data.items()}
    q=c['question_max_tokens'];altered['features'][:,q+5:]=1234
    altered['attention_mask'][:,q+5:]=~altered['attention_mask'][:,q+5:];altered['position_ids'][:,q+5:]=9999
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        first=shadow(data);second=shadow(altered)
    assert torch.equal(first,second),'Real GPU native block Prefix5 leaked future15'
    del data,altered,first,second
    evaluation=evaluate(ema,reader,groups[:51],rank,device,root,0,'qualification')
    assert evaluation['groups']==51
    update(ddp,shadow,ema,optimizer,reader,groups,list(range(51)),rank,device)
    save(root,shadow,ema,optimizer,1,1,1,config_sha,bank_sha,None,rank,qualification_only=True)
    receipt=json.loads((root/'latest.json').read_text());assert fingerprints(receipt['path'])['sha256']==receipt['sha256']
    update(ddp,shadow,ema,optimizer,reader,groups,list(range(51,102)),rank,device)
    expected=dict(raw=cpu_tree(shadow.state_dict()),ema=cpu_tree(ema.state_dict()),optimizer=cpu_tree(optimizer.state_dict()))
    loaded=torch.load(receipt['path'],map_location='cpu',weights_only=False)
    assert loaded['qualification_only'] and loaded['steps']==1
    restore(shadow,ema,optimizer,loaded,rank,device);del loaded
    update(ddp,shadow,ema,optimizer,reader,groups,list(range(51,102)),rank,device)
    observed=dict(rank=rank,raw=comparison(shadow.state_dict(),expected['raw'],parameter=True),
        ema=comparison(ema.state_dict(),expected['ema'],parameter=True))
    moments={};reference={}
    for key,state in optimizer.state_dict()['state'].items():
        for field,value in state.items():
            moments[str(key)+':'+field]=value;reference[str(key)+':'+field]=expected['optimizer']['state'][key][field]
    observed['Adam']=comparison(moments,reference)
    observed['production_unchanged']=all(torch.equal(v.detach().cpu(),untouched[k]) for k,v in production.state_dict().items())
    all_ranks=[None]*4;dist.all_gather_object(all_ranks,observed)
    passed=all(x['production_unchanged'] and all(x[k]['passed'] for k in ('raw','ema','Adam')) for x in all_ranks)
    if rank==0:
        proof=dict(status='FOUR_RANK_GPU_RECOVERY_QUALIFICATION_PASSED' if passed else 'FOUR_RANK_GPU_RECOVERY_QUALIFICATION_FAILED',
            qualification_only=True,production_optimizer_updates=0,real_prefix5_future15_invariance=True,
            real_uneven_rank_evaluation_groups=evaluation['groups'],
            shadow_saved_step=1,shadow_compared_next_step=2,rank_results=all_ranks,checkpoint=receipt,
            elapsed_seconds=time.monotonic()-start,epoch=time.time(),fit_code_sha256=loaded_sha(base))
        (base/'probe-qualification').mkdir(exist_ok=True);atomic(base/'probe-qualification/proof.json',proof);atomic(root/'proof.json',proof)
        print(json.dumps(proof),flush=True)
    random.setstate(original_rng['python']);np.random.set_state(original_rng['numpy']);torch.set_rng_state(original_rng['torch']);torch.cuda.set_rng_state(original_rng['cuda'])
    del shadow,ddp,ema,optimizer,expected,untouched,moments,reference
    gc.collect();torch.cuda.empty_cache();dist.barrier()
    assert passed,'GPU recovery qualification failed; production probe remains unmodified atstep0'


def loaded_sha(base):
    return fingerprints(Path(__file__).with_name('train.py'))['sha256']
