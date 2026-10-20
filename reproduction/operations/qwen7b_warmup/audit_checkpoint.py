"""Independent CPU acceptance of a hard-linked, immutable own fit checkpoint.

Run separately from the trainer. Never rewrite active state or load an actor.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import time
import torch
from reproduction.operations.qwen7b_warmup.download_model import atomic, fingerprints


def state_hash(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def accept(base,receipt=None):
    base=Path(base);root=base/'probe'
    if receipt is None:receipt=json.loads((root/'latest.json').read_text())
    assert receipt['steps']>0 and not receipt['qualification_only']
    audit=base/'checkpoint-audit';audit.mkdir(exist_ok=True)
    path=audit/('epoch-%04d-step-%08d.pt'%(receipt['epoch'],receipt['steps']))
    # Trainer atomically replaces inflight.pt. A hard link preserves the inode
    # being accepted without copying or modifying the active checkpoint.
    if not path.exists():os.link(receipt['path'],path)
    actual=fingerprints(path);assert actual['sha256']==receipt['sha256'] and actual['bytes']==receipt['bytes']
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert state['steps']==receipt['steps'] and state['cursor']==receipt['cursor'] and not state['qualification_only']
    assert state['epoch']==receipt['epoch'] and state['completed_epoch']==receipt['completed_epoch']
    assert state['actor_updates']==receipt['actor_updates']==0
    assert state['config_sha256']==fingerprints(base/'config.json')['sha256']
    assert state['bank_sha256']==fingerprints(base/'bank/complete.json')['sha256']
    config=json.loads((base/'config.json').read_text())
    for key in ('raw','ema'):
        assert tuple(state[key]['block.self_attn.q_proj.weight'].shape)==(3584,3584)
        assert tuple(state[key]['block.self_attn.k_proj.weight'].shape)==(512,3584)
        assert tuple(state[key]['block.mlp.gate_proj.weight'].shape)==(18944,3584)
        assert tuple(state[key]['output.weight'].shape)==(1,(config['question_max_tokens']+5)*3584)
    source=json.loads((base/'preparation/postprocess-source-publication.json').read_text())
    assert state['fit_source_commit']==source['commit']
    fit=Path(source['code_root'])/'reproduction/operations/qwen7b_warmup/train.py'
    assert state['fit_code_sha256']==fingerprints(fit)['sha256']
    ranks=state['rank_states'];assert [r['rank'] for r in ranks]==list(range(4))
    for key,record in [('raw','model_sha256'),('ema','ema_sha256')]:
        digest=state_hash(state[key]);assert all(r[record]==digest for r in ranks)
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    for key in ('raw_buffers','ema_buffers'):
        assert isinstance(state[key],dict)
        assert all(bool(torch.isfinite(v).all()) for v in state[key].values())
    optimizer=state['optimizer'];h=hashlib.sha256();steps=set()
    for group in optimizer['param_groups']:
        assert group['lr']==5e-6 and group['weight_decay']==0
        for parameter in group['params']:
            item=optimizer['state'][parameter];steps.add(int(item['step']))
            for name,value in sorted(item.items()):
                assert bool(torch.isfinite(value).all());h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
    assert steps=={state['steps']} and all(r['optimizer_sha256']==h.hexdigest() and r['steps']==state['steps'] for r in ranks)
    for rank in ranks:
        rng=rank['rng'];assert set(rng)=={'python','numpy','torch','cuda'} and rng['torch'].numel()>0 and rng['cuda'].numel()>0
    report=dict(status='INDEPENDENT_CPU_FULL_CHECKPOINT_SHA_MODELS_ADAM_FOUR_RANK_RNG_VERIFIED',
        immutable_checkpoint=str(path),checkpoint=receipt,fit_source_commit=source['commit'],
        bank_sha256=state['bank_sha256'],epoch=state['epoch'],cursor=state['cursor'],steps=state['steps'],
        actor_updates=state['actor_updates'],checked_ranks=4,optimizer_parameter_tensors=len(optimizer['state']),
        epoch_time=time.time(),next_update_proof=str(base/'probe-qualification/proof.json'))
    atomic(audit/('step-%08d.json'%state['steps']),report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'))
    parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    receipt=json.loads(args.receipt.read_text()) if args.receipt else None
    print('RESULT_JSON='+json.dumps(accept(args.base,receipt)),flush=True)


if __name__=='__main__':main()
