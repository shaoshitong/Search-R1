"""Stopped FULL joint continuation, preserving both optimizers and every RNG.

This is distinct from the historical human rerunFROM50. Failed, uncommitted
sampling stays in the parent namespace; no accepted actor/probe update is redone.
"""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback

from .bootstrap import read,sha,once,publish,digest,validate_config
from .schedule import probe_boundary


def validate_boundary(base,c):
    base=Path(base);q=read(base/'preparation/full-joint-qualified.json')
    n=c['resume_joint']['completed_outer'];a=read(base/'actor/latest.json')
    p=read(base/'probe/latest.json');pool=read(base/'question-pool/state.json')
    assert q['status']=='ACTOR50_ACTUAL_FULL_JOINT_QUALIFIED'
    assert q['source_commit']==c['source_commit'] and q['completed_outer']==n
    assert a['completed_outer']==a['scheduler_completed_outer']==n
    assert a['actual_adam_steps_per_rank']==q['actor_actual_Adam']
    expected=probe_boundary(n)
    assert p['completed_outer']==expected['last_update_outer']
    assert p['actual_adam_steps_per_rank']==[expected['actual_adam_per_rank']]*4
    assert pool['next_outer']==n+1 and pool['consumed_candidates']==q['pool_consumed_candidates']
    assert a['binding']==c['retained_actor50_binding']
    for row in q['sealed_metadata']:
        assert sha(row['path'])==row['sha256'],'Imported metadata changed'
    for row in q['payload_seals']:
        s=Path(row['path']).stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['dev'],row['inode'],row['bytes'],row['mtime_ns'])
    return n



def prepare_reuse(base,c):
    """Rebind an existing full SHA/finite proof without rereading any payload."""
    import copy
    item=c['reuse_qualified_full_joint'];old=Path(item['base'])
    path=old/'preparation/full-joint-qualified.json'
    assert sha(path)==item['sha256']
    prior=read(path);oldc=read(old/'preparation/launch-config.json')
    assert read(old/'preparation/prepare-full-exit.json')['success']
    assert not (old/'preparation/submit-intent.json').exists()
    validate_boundary(old,oldc)
    # Resource, source, ports and own metadata paths may differ; science cannot.
    allowed={'base','execution_id','source_commit','pool_state_path','prefix_probe_receipt',
        'retrieval_port','search_url','grounding_probe_base_port','auxiliary_gpu_indices','auxiliary_gpu_uuids',
        'retrieval_gpu_shards','approved_single_auxiliary','retrieval_consistency_questions','reuse_qualified_full_joint'}
    assert {k:v for k,v in oldc.items() if k not in allowed}=={k:v for k,v in c.items() if k not in allowed},'Qualified scientific configuration changed'
    from reproduction.operations.a800_i1_i6_boundary_handoff import owned_present
    parent=Path(c['resume_joint']['source_base']);terminal=read(c['resume_joint']['terminal_path'])
    assert not owned_present(parent) and sha(c['resume_joint']['terminal_path'])==c['resume_joint']['terminal_sha256']
    assert terminal['owned_writers_gone'] and sha(parent/'latest-joint.json')==prior['source_joint_sha256']
    for rel in ('actor/latest.json','probe/latest.json','question-pool/state.json','question-pool/binding.json','latest-joint.json'):
        once(base/rel,read(old/rel))
    os.link(old/'probe/latest.pt',base/'probe/latest.pt')
    n=prior['completed_outer']
    for ordinal in range(1,n+1):
        for rel in ('complete.json','probe-update/complete.json','probe-update/metrics.json','validation/complete.json'):
            source=old/'outers'/('outer-%06d'%ordinal)/rel
            if source.is_file():
                target=base/'outers'/('outer-%06d'%ordinal)/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    for rel in ('probe/prefix40-transition.json','probe/initialization-complete.json','logs/qa-pending.json'):
        if (old/rel).exists():once(base/rel,read(old/rel))
    proof=copy.deepcopy(prior);proof.update(source_commit=c['source_commit'],checked_unix=time.time(),
        qualification_reused=dict(base=str(old),sha256=item['sha256'],payload_SHA_finite_scanned_again=False,all_19_payload_stat_seals_reverified=True),
        sealed_metadata=[dict(path=str(base/rel),sha256=sha(base/rel)) for rel in
            ('actor/latest.json','probe/latest.json','question-pool/state.json','question-pool/binding.json','latest-joint.json','preparation/launch-config.json')])
    once(base/'preparation/full-joint-qualified.json',proof);validate_boundary(base,c)
    return dict(status=proof['status'],completed_outer=n,actor_actual_Adam=proof['actor_actual_Adam'],
        probe_actual_Adam=proof['probe_actual_Adam'],qualification_reused=proof['qualification_reused'],checked_unix=time.time())


def prepare(base):
    base=Path(base);c=read(base/'preparation/launch-config.json');validate_config(c,base)
    if c.get('reuse_qualified_full_joint'):return prepare_reuse(base,c)
    import torch
    from safetensors import safe_open
    from reproduction.prefix_selection200.checkpoint_cpu_once import hash_file,tensor_cpu,validate_adam,validate_rng
    from reproduction.prefix_selection200.actor import learning_rate
    from reproduction.prefix_selection200.prefix5_model import validate_model_state,validate_native_buffers
    from reproduction.operations.a800_i1_i6_boundary_handoff import owned_present
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    base=Path(base);c=read(base/'preparation/launch-config.json');validate_config(c,base)
    parent=Path(c['resume_joint']['source_base']);n=c['resume_joint']['completed_outer']
    terminal=read(c['resume_joint']['terminal_path'])
    assert sha(c['resume_joint']['terminal_path'])==c['resume_joint']['terminal_sha256']
    assert terminal['termination_cause']=='FAULT_EXIT' and terminal['owned_writers_gone'] and not owned_present(parent)
    assert terminal['completed_outer']==n and terminal['uncommitted_actor_Adam']==terminal['uncommitted_probe_Adam']==[0]*4
    assert sha(parent/'latest-joint.json')==terminal['source_joint_sha256']
    j=read(parent/'latest-joint.json');a=j['actor'];p=j['probe'];expected=probe_boundary(n)
    assert j['outer']==a['completed_outer']==a['scheduler_completed_outer']==n
    assert a['binding']==c['retained_actor50_binding'] and a['actual_adam_steps_per_rank']==terminal['actor_actual_Adam']
    assert p['completed_outer']==expected['last_update_outer'] and p['actual_adam_steps_per_rank']==[expected['actual_adam_per_rank']]*4
    seals=[];ranks=[]
    for rank,row in enumerate(a['rank_states']):
        publish(base/'preparation/prepare-progress.json',dict(stage='QUALIFY_FULL_JOINT_ACTOR_ADAM_RNG',rank=rank,outer=n,checked_unix=time.time()))
        seals.append(hash_file(row['path'],row));s=torch.load(row['path'],map_location='cpu',weights_only=False,mmap=True)
        steps=a['actual_adam_steps_per_rank'][rank]
        assert s['rank']==rank and s['completed_outer']==n and s['actual_adam_steps']==steps
        assert s['binding']==a['binding'] and s['grounding_state']==j['grounding']
        assert s['optimizer_layout']=='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4'
        shapes=[v['shape'] for group in s['optimizer_parameter_layout'] for v in group]
        count=validate_adam(s['optimizer'],shapes,steps,learning_rate(n),weight_decay=.01)
        assert s['scheduler']['last_epoch']==n and s['scheduler']['_step_count']==n+1
        assert s['scheduler']['_last_lr']==[learning_rate(n)]
        ranks.append(dict(rank=rank,Adam=steps,scheduler=n,adam_states=count,rng=validate_rng(s['rng'])))
        del s;gc.collect()
    for name,row in a['hf_files'].items():
        path=Path(a['hf_model_dir'])/name;seals.append(hash_file(path,row))
        if name.endswith('.safetensors'):
            with safe_open(str(path),framework='pt',device='cpu') as f:
                for key in f.keys():tensor_cpu(f.get_tensor(key),dtype=torch.float32,finite=True)
    driver=a['driver_state'];seals.append(hash_file(driver['path'],driver))
    s=torch.load(driver['path'],map_location='cpu',weights_only=False);pool=read(j['pool_snapshot'])
    assert s['completed_outer']==n and s['actor_version']==a['actor_version'] and s['grounding_state']==j['grounding']
    assert s['candidate_pool_state']['value']==pool and sha(j['pool_snapshot'])==j['pool_sha256']
    assert pool['next_outer']==n+1 and pool['cycle']*len(pool['order'])+pool['cursor']==pool['consumed_candidates']
    from reproduction.prefix_selection200.candidate_schedule import consumed_candidates_through
    assert pool['consumed_candidates']==consumed_candidates_through(c,n)
    assert sorted(pool['order'])==list(range(len(pool['order'])))
    driver_rng=validate_rng(s['rng']);del s;gc.collect()
    publish(base/'preparation/prepare-progress.json',dict(stage='QUALIFY_FULL_JOINT_PROBE_ADAM_RNG',outer=n,checked_unix=time.time()))
    seals.append(hash_file(j['probe_snapshot'],p));s=torch.load(j['probe_snapshot'],map_location='cpu',weights_only=False,mmap=True)
    assert s['completed_outer']==p['completed_outer'] and s['actual_adam_steps_per_rank']==p['actual_adam_steps_per_rank']
    assert s['binding_sha256']==p['binding_sha256']==digest(s['binding'])
    assert s['binding']['prefix_tokens']==40 and s['binding']['head_max_tokens']==313
    assert s['binding']['source_probe_sha256']==c['probe_source_sha256'] and s['binding']['probe_total_actual_adam_per_rank']==700
    validate_model_state(s['model']);validate_native_buffers(s['model_nonpersistent_buffers'])
    for value in s['model'].values():tensor_cpu(value,dtype=torch.float32,finite=True)
    for value in s['model_nonpersistent_buffers'].values():tensor_cpu(value,dtype=torch.float32,finite=True)
    adam=validate_adam(s['optimizer'],[v.shape for v in s['model'].values()],expected['actual_adam_per_rank'],5e-6,weight_decay=0.)
    assert len(s['rank_rng'])==4 and [v['rank'] for v in s['rank_rng']]==list(range(4))
    assert all(v['actual_adam_steps']==expected['actual_adam_per_rank'] for v in s['rank_rng'])
    probe_rng=[validate_rng(v) for v in s['rank_rng']];del s;gc.collect()
    assert not torch.cuda.is_initialized() and not owned_present(parent)
    once(base/'actor/latest.json',a);once(base/'probe/latest.json',p)
    os.link(j['probe_snapshot'],base/'probe/latest.pt')
    once(base/'question-pool/state.json',pool);once(base/'question-pool/binding.json',read(parent/'question-pool/binding.json'))
    once(base/'latest-joint.json',j)
    for ordinal in range(1,n+1):
        source=parent/'outers'/('outer-%06d'%ordinal)/'complete.json'
        dest=base/'outers'/('outer-%06d'%ordinal)/'complete.json';dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest)
        # Retain existing scalar diagnostics/QA receipts without rereading any
        # raw trajectory, probability, feature or tensor as a monitoring input.
        for rel in ('probe-update/complete.json','probe-update/metrics.json','validation/complete.json'):
            source_meta=source.parent/rel
            if source_meta.is_file():
                target=dest.parent/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(source_meta,target)
    transition=parent/'probe/prefix40-transition.json'
    if transition.exists():once(base/'probe/prefix40-transition.json',read(transition))
    # Prefix40 transition and previous scheduled diagnostics retain old paths;
    # only metadata is linked. Model/Adam payloads remain on the execution host.
    for rel in ('probe/initialization-complete.json','logs/qa-pending.json'):
        if (parent/rel).exists():once(base/rel,read(parent/rel))
    metadata=[base/x for x in ('actor/latest.json','probe/latest.json','question-pool/state.json','question-pool/binding.json','latest-joint.json','preparation/launch-config.json')]
    proof=dict(status='ACTOR50_ACTUAL_FULL_JOINT_QUALIFIED',source_commit=c['source_commit'],completed_outer=n,
        source_base=str(parent),source_joint_sha256=terminal['source_joint_sha256'],actor_actual_Adam=a['actual_adam_steps_per_rank'],
        probe_actual_Adam=p['actual_adam_steps_per_rank'],pool_consumed_candidates=pool['consumed_candidates'],
        payload_seals=seals,sealed_metadata=[dict(path=str(f),sha256=sha(f)) for f in metadata],
        actor_ranks=ranks,driver_rng=driver_rng,probe_adam_states=adam,probe_rank_rng=probe_rng,
        all_model_Adam_finite=True,checkpoint_payloads_SHA_verified_once=True,CUDA_restore='PENDING_ACTUAL_NEXT_ACTOR',checked_unix=time.time())
    once(base/'preparation/full-joint-qualified.json',proof);validate_boundary(base,c)
    return dict(status=proof['status'],completed_outer=n,actor_actual_Adam=a['actual_adam_steps_per_rank'],probe_actual_Adam=p['actual_adam_steps_per_rank'],checked_unix=time.time())


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();b=Path(a.base)
    try:result=prepare(b);once(b/'preparation/prepare-full-exit.json',dict(success=True,**result))
    except BaseException as error:
        once(b/'preparation/prepare-full-exit.json',dict(success=False,error=repr(error),traceback=traceback.format_exc(),checked_unix=time.time()));raise


if __name__=='__main__':main()
