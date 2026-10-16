"""Normal CPU/full-file checkpoint readback for actor-hidden periodic probes.

Not a training gate, no GPU forward, no repeated existing proof, no backup claim.
Supports the actual latest complete outer as well as normal milestones.
"""
import argparse
import gc
import json
import os
from pathlib import Path

from .periodic import probe_boundary, is_probe_update
from .checkpoint_cpu_once import (
    read, need, digest, owned, hash_file, load_torch_cpu, validate_actor_state,
    head_shapes, tensor_cpu, validate_adam, validate_rng,
)


def inspect(base,outer):
    import torch
    base=Path(base);config=read(base/'preparation/launch-config.json')
    proof=base/'preparation'/('joint-native-%04d-independent-full-sha-cpu.json'%outer)
    if proof.exists():return dict(status='EXISTING_CPU_PROOF_NO_REPLAY',proof=str(proof),sha256=hash_file(proof)['sha256'])
    root=base/'outers'/('outer-%06d'%outer)
    if not (root/'complete.json').exists():return dict(status='PENDING_NO_COMPLETE_CHECKPOINT',outer=outer)
    joint=read(root/'complete.json');actor=joint['actor'];probe=joint['probe'];boundary=probe_boundary(outer)
    need(config['actor_hidden_native'] and joint['run_id']==config['run_id'] and joint['outer']==outer,
         'Inspect the current native-actor run only')
    need(actor['completed_outer']==outer and actor['actual_adam_steps_per_rank']==[20*outer]*8
         and actor['scheduler_completed_outer']==outer,'Actual actor clock differs')
    need(probe['completed_outer']==boundary['last_update_outer']
         and probe['actual_adam_steps_per_rank']==[boundary['actual_adam_per_rank']]*8,
         'Probe may advance only100 writes at20-outer boundaries')
    files=[];ranks=[]
    for name,item in actor['hf_files'].items():
        files.append(hash_file(owned(base,Path(actor['hf_model_dir'])/name),item))
    for rank,item in enumerate(actor['rank_states']):
        state,meta=load_torch_cpu(owned(base,item['path']),item)
        ranks.append(validate_actor_state(state,actor,config,outer,rank));files.append(meta)
        del state;gc.collect()
    driver,meta=load_torch_cpu(owned(base,actor['driver_state']['path']),actor['driver_state']);files.append(meta)
    selected_probe=driver['prefix_probe_receipt']['value'];before=probe_boundary(outer-1)
    need(selected_probe['completed_outer']==before['last_update_outer']
         and selected_probe['actual_adam_steps_per_rank']==[before['actual_adam_per_rank']]*8,
         'Actor must retain the probe actually used before this outer update')
    del driver;gc.collect()
    head,meta=load_torch_cpu(owned(base,joint['probe_snapshot']),probe);files.append(meta)
    need(head['run_id']==config['run_id'] and head['completed_outer']==boundary['last_update_outer']
         and head['binding_sha256']==digest(head['binding'])==probe['binding_sha256'],
         'Probe checkpoint version/binding differs')
    rate=config['probe_learning_rate']
    b=head['binding'];need(b['representation_source']=='current_sampling_actor_native_layer27_postblock'
         and b['independent_encoder_forward'] is False and b['probe_update_period']==20
         and b['updates_per_period']==100 and b['learning_rate']==rate,'Wrong periodic native probe recipe')
    model=head['model'];shapes=head_shapes();need(set(model)==set(shapes),'Missing original probe tensors')
    for name,value in model.items():tensor_cpu(value,dtype=torch.float32,shape=shapes[name],finite=True)
    steps=boundary['actual_adam_per_rank']
    if steps:
        adam=validate_adam(head['optimizer'],[v.shape for v in model.values()],steps,rate,weight_decay=0.)
        need(len(head['rank_rng'])==8,'Missing periodic probe rank RNG')
        for rank,rng in enumerate(head['rank_rng']):
            need(rng['rank']==rank and rng['actual_adam_steps']==steps,'Wrong periodic RNG clock');validate_rng(rng)
    else:
        need(head['optimizer'] is None and head['rank_rng'] is None,'No probe optimizer/RNG stream before first update20')
        adam=0
    del head;gc.collect()
    pool=read(owned(base,joint['pool_snapshot']));binding=read(base/'question-pool/binding.json')
    size=binding['eligible_questions']
    need(sorted(pool['order'])==list(range(size)) and pool['next_outer']==outer+1
         and pool['consumed_candidates']==5120*outer and pool['cycle']*size+pool['cursor']==5120*outer,
         'Saved question pool/permutation/cycle/cursor differs from completed outer')
    files.append(hash_file(owned(base,joint['pool_snapshot'])))
    result=dict(status='NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS',run_id=config['run_id'],outer=outer,
        actor_actual_adam_steps_per_rank=[20*outer]*8,probe_actual_adam_steps_per_rank=[steps]*8,
        probe_last_update_outer=boundary['last_update_outer'],actor_rank_cpu_states=ranks,
        probe_fp32_tensors=14,probe_fp32_adam_states=adam,files=files,
        pool_next_outer=pool['next_outer'],pool_consumed_candidates=pool['consumed_candidates'],
        gpu_forward_performed=False,cuda_rng_applied=False,independent_backup_performed=False)
    with proof.open('x') as f:json.dump(result,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    return dict(status=result['status'],proof=str(proof),sha256=hash_file(proof)['sha256'],outer=outer,
                actor_actual_adam_steps_per_rank=[20*outer]*8,probe_actual_adam_steps_per_rank=[steps]*8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--outer',type=int,required=True)
    a=p.parse_args();print(json.dumps(inspect(a.base,a.outer)))


if __name__=='__main__':main()
