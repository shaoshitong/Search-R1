"""CPU-only actual AB joint savepoint readback; never a training gate."""
import argparse
import gc
import json
from pathlib import Path

from . import ab_policy as policy
from .checkpoint_cpu_once import need, tensor_cpu, validate_adam, validate_rng, digest
from .prefix5_model import PARAMETER_SHAPES, NATIVE_BUFFER_LAYOUTS


def probe_layout(config,outer):
    """A saved joint belongs to its completed stage, not the next live stage."""
    from . import b_progressive
    need(type(outer) is int and 1<=outer<=200,'Require an actual completed outer1..200')
    if b_progressive.enabled(config):
        b_progressive.validate(config)
        prefix=b_progressive.prefix(outer)
    else:
        prefix=5
    shapes=dict(PARAMETER_SHAPES)
    shapes['output.weight']=(1,(273+prefix)*2048)
    return prefix,shapes


def validate_probe_state(saved,receipt,config,outer):
    import torch
    expected=policy.validate_receipt(config,receipt,outer)
    steps=expected['actual_adam_per_rank'];arm=policy.mode(config)
    need(saved['binding_sha256']==digest(saved['binding'])==receipt['binding_sha256']
         and saved['run_id']==config['run_id'] and saved['completed_outer']==expected['last_update_outer']
         and saved['actual_adam_steps_per_rank']==[steps]*4,'AB saved state/receipt mismatch')
    b=saved['binding'];prefix,shapes=probe_layout(config,outer)
    need(b['selection_probe_experiment']==arm and b['learning_rate']==5e-6
         and b['group_weight']==config['probe_group_loss_weight'] and b['ema_rate']==config['probe_ema_rate']
         and b['prefix_tokens']==prefix and b['head_max_tokens']==273+prefix and b['question_max_tokens']==273
         and b['source_probe_sha256']==config['probe_source_sha256']
         and b['source_epoch']==5 and b['source_model_kind']=='ema'
         and b['warmup_optimizer_imported'] is False and b['world_size']==4,'AB source/architecture/optimizer recipe differs')
    def model_tensors(model,buffers):
        need(set(model)==set(shapes),'Missing copied block/Linear parameters')
        for name,shape in shapes.items():tensor_cpu(model[name],dtype=torch.float32,shape=shape,finite=True)
        layout=next((layout for layout in NATIVE_BUFFER_LAYOUTS if set(layout)==set(buffers)),None)
        need(layout is not None,'Missing native RoPE buffers')
        for name,shape in layout.items():tensor_cpu(buffers[name],dtype=torch.float32,shape=shape,finite=True)
    model_tensors(saved['model'],saved['model_nonpersistent_buffers'])
    if arm==policy.BALANCED:
        model_tensors(saved['ema_model'],saved['ema_nonpersistent_buffers'])
        need(saved['ema_updates']==steps==receipt['ema_updates'],'EMA actual write count differs')
        if outer:
            sample=saved['sampling'];n=2560 if outer<200 and outer%10==0 else 512
            if sample['status']=='SKIPPED_MISSING_CLASS':
                need(not sample['sampled_indices'] and 0 in sample['original_counts'],'Missing-class skip must be real')
            else:
                need(len(sample['sampled_indices'])==n and sample['sampled_counts']==[n//2,n//2]
                     and sum(sample['group_multiplicities'].values())==n,'Saved balancing sequence differs')
            rng=sample['rng_after'];probe_rng=__import__('random').Random();probe_rng.setstate(rng)
    else:
        need(saved['optimizer'] is None and saved['rank_rng'] is None and 'ema_model' not in saved,
             'Frozen experiment cannot have online optimizer/EMA')
    if steps:
        adam=validate_adam(saved['optimizer'],[v.shape for v in saved['model'].values()],steps,5e-6,weight_decay=0.)
    else:
        need(saved['optimizer'] is None,'Actual0 cannot contain an Adam state');adam=0
    if saved.get('rank_rng') is not None:
        need(len(saved['rank_rng'])==4,'Missing four probe RNG states')
        for rank,row in enumerate(saved['rank_rng']):
            need(row['rank']==rank and row['actual_adam_steps']==steps,'Probe RNG actual clock differs');validate_rng(row)
    elif steps:raise ValueError('Actual trained probe requires rank RNG')
    return dict(actual_adam_steps_per_rank=[steps]*4,ema_updates=saved.get('ema_updates',0),
                raw_model_verified=True,ema_model_verified=arm==policy.BALANCED,adam_states=adam,
                prefix_tokens=prefix,head_max_tokens=273+prefix)


def inspect(base,outer):
    from .checkpoint_native_pending_probe import Seals,_hf
    from .i1_i6_checkpoint_cpu import actor_state,actor_ledger,validate_driver_probe_receipt
    from .candidate_schedule import consumed_candidates_through
    from .storage import actual_actor_boundary
    from reproduction.prefix20_online.launch import read,sha,once
    base=Path(base);config=read(base/'preparation/launch-config.json');policy.validate(config)
    output=base/'preparation'/('ab-joint-%04d-independent-full-sha-cpu.json'%outer)
    if output.exists():return dict(status='EXISTING_CPU_PROOF_NO_REPLAY',proof=str(output),sha256=sha(output))
    root=base/'outers'/('outer-%06d'%outer)
    if not (root/'complete.json').exists():return dict(status='PENDING_NO_COMPLETE_JOINT',outer=outer)
    joint=read(root/'complete.json');actor=joint['actor'];probe=joint['probe']
    need(actual_actor_boundary(actor,outer) and joint['run_id']==config['run_id'],'Incomplete actor boundary')
    from .ab_lifecycle import check_update
    check_update(config,joint,joint['probe_metrics'])
    intent=output.with_suffix('.intent.json')
    need(not intent.exists(),'Existing CPU audit intent must be reconciled, never replayed')
    once(intent,dict(run_id=config['run_id'],outer=outer,joint_sha256=sha(root/'complete.json')))
    allowed=[Path(config['official_model_dir'])/'config.json']
    seals=Seals(base,allowed=allowed);hf=_hf(seals,actor,config);ranks=[]
    for rank,item in enumerate(actor['rank_states']):
        saved,file=seals.mmap_torch(item['path'],item)
        ranks.append(dict(actor_state(saved,actor,config,outer,rank),file=file,ledger=actor_ledger(seals,actor,outer,rank)))
        del saved;gc.collect()
    driver,file=seals.mmap_torch(actor['driver_state']['path'],actor['driver_state'])
    pool=seals.json(joint['pool_snapshot']);consumed=consumed_candidates_through(config,outer)
    need(driver['run_id']==config['run_id'] and driver['completed_outer']==outer
         and driver['candidate_pool_state']['value']==pool and driver['grounding_state']==joint['grounding'],
         'Driver/pool/grounding actual boundary differs')
    need(pool['next_outer']==outer+1 and pool['consumed_candidates']==consumed
         and pool['cycle']*len(pool['order'])+pool['cursor']==consumed
         and sorted(pool['order'])==list(range(len(pool['order']))),'Pool permutation/cursor differs')
    score=read(root/'probe-score/manifest.json')
    previous=read(base/'outers'/('outer-%06d'%(outer-1))/'complete.json')['probe'] if outer>1 else read(base/'probe/initialization-complete.json')['receipt']
    before=policy.validate_receipt(config,previous,outer-1)
    validate_driver_probe_receipt(driver['prefix_probe_receipt'],score,before);validate_rng(driver['rng'])
    del driver;gc.collect()
    saved,file=seals.mmap_torch(joint['probe_snapshot'],probe)
    head=validate_probe_state(saved,probe,config,outer);del saved;gc.collect()
    result=dict(status='AB_JOINT_FULL_SHA_CPU_PASS',run_id=config['run_id'],outer=outer,
        actor_hf=hf,actor_rank_states=ranks,probe=head,files=seals.finish(),
        actor_actual_adam_steps_per_rank=actor['actual_adam_steps_per_rank'],
        probe_actual_adam_steps_per_rank=probe['actual_adam_steps_per_rank'],
        actual_cpu_deserialization=True,gpu_forward_performed=False,independent_backup_performed=False)
    once(output,result);return dict(status=result['status'],proof=str(output),sha256=sha(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--outer',type=int,required=True)
    a=p.parse_args();print(json.dumps(inspect(a.base,a.outer)))
