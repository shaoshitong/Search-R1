"""Explicit CP181 eight-to-four state merge, without an optimizer update.

FSDP uses the same 37 flat parameters. Independently measured HF numels must
have no padding at either world size; adjacent shards concatenate in rank order.
Historical eight-rank journals remain historical. Four RNG streams inherit
source ranks 0/2/4/6; this changes execution partitioning, not an exact executor
resume. The original eight streams and all source files remain retained.
"""
import copy
import gc
import os
from pathlib import Path
import time
from .diverse_resume import read, sha, once, digest, flat_numels
MODE = 'diverse_eight_to_four_cp181_v1'
STATUS = 'DIVERSE_CP181_EIGHT_TO_FOUR_FULL_STATE'
# Only the three narrow hardware/boundary dispatches differ in scientific files.
UNCHANGED = ('actor.py','actor_safety.py','probe.py','ab_probe.py','ab_policy.py',
             'probe_plan.py','rollout.py','actor_hidden.py','prefix_diversity.py',
             'interaction_protocol.py','pool.py','launch.py')
DISPATCH = ('storage.py','grounding.py','b_progressive.py')


def validate_compatibility(c, p):
    root=Path(c['base'])/'repo/reproduction/prefix_selection200'
    if (sha(c['runtime_compatibility']['path'])!=c['runtime_compatibility']['sha256']
        or p['status']!=STATUS or p['runtime_commit']!=c['source_commit']
        or p['binding_commit']!=c['actor_checkpoint_binding_commit']
        or c['world_size']!=4 or c['actor_global_micro_batch_size']!=4
        or c['physical_gpu_indices']!=[0,1,2,3]
        or set(p['runtime_sha256'])!=set(UNCHANGED+DISPATCH)):
        raise ValueError('Explicit authorized four-rank full-state profile differs')
    for name, value in p['runtime_sha256'].items():
        if sha(root/name)!=value:raise ValueError('Published runtime differs: '+name)
    for name in UNCHANGED:
        if sha(Path(p['source_base'])/'repo/reproduction/prefix_selection200'/name)!=p['runtime_sha256'][name]:
            raise ValueError('Scientific implementation changed: '+name)
    return p


def seal_qualified(q):
    for row in q['files']:
        st=Path(row['path']).stat()
        if (st.st_size,st.st_ino,st.st_mtime_ns)!=(row['bytes'],row['inode'],row['mtime_ns']):
            raise ValueError('Qualified immutable file changed: '+row['path'])


def merge_optimizer(left, right, counts):
    import torch
    if left['optimizer']['param_groups']!=right['optimizer']['param_groups']:
        raise ValueError('Source Adam hyperparameters/parameter order differ')
    if left['optimizer_parameter_layout']!=right['optimizer_parameter_layout']:
        raise ValueError('Source flat layouts differ')
    result=copy.deepcopy(left['optimizer']);layout=copy.deepcopy(left['optimizer_parameter_layout'])
    ids=[p for group in result['param_groups']for p in group['params']]
    flat=[p for group in layout for p in group]
    if len(ids)!=37 or len(flat)!=37:raise ValueError('Same native root+36-layer wrap required')
    comparisons=[]
    for i,(pid,item)in enumerate(zip(ids,flat)):
        n=counts[i]//8
        if counts[i]%8 or item!={'shape':[n],'numel':n,'dtype':'torch.float32'}:
            raise ValueError('Independent HF counts must prove unpadded rank order')
        a=left['optimizer']['state'][pid];b=right['optimizer']['state'][pid]
        if set(a)!=set(b) or set(a)!=set(('step','exp_avg','exp_avg_sq')):
            raise ValueError('Only original native AdamW state admitted')
        if int(a['step'])!=3620 or not torch.equal(a['step'],b['step']):
            raise ValueError('Require identical direct CP181 Adam steps')
        for key in ('exp_avg','exp_avg_sq'):
            if any(t.dtype!=torch.float32 or list(t.shape)!=[n]or not torch.isfinite(t).all()for t in(a[key],b[key])):
                raise ValueError('Original Adam moment shape/dtype/finiteness differs')
            merged=torch.cat((a[key],b[key]))
            if not torch.equal(merged[:n],a[key])or not torch.equal(merged[n:],b[key]):
                raise ValueError('Bitwise moment preservation failed')
            result['state'][pid][key]=merged
        item.update(shape=[2*n],numel=2*n)
        comparisons.append(dict(flat_parameter=i,source_numel=n,target_numel=2*n,bitwise_moments_preserved=True))
    return result,layout,comparisons


def imported_actor_boundary(a,n):
    try:
        ref=a['imported_8to4_cp181'];p=read(ref['path']);j=read(p['source_joint_path'])
        return (n==181 and sha(ref['path'])==ref['sha256'] and p['status']==STATUS
                and sha(p['source_joint_path'])==p['source_joint_sha256']
                and p['new_optimizer_steps']==0 and p['all_moments_bitwise_preserved'] is True
                and a['rank_states']==p['rank_states'] and a['rank_receipts']==j['actor']['rank_receipts']
                and len(a['rank_receipts'])==8 and len(a['rank_states'])==4
                and a['actual_adam_steps_per_rank']==[3620]*4
                and j['actor']['actual_adam_steps_per_rank']==[3620]*8)
    except(KeyError,TypeError,ValueError,OSError):return False


def validate_imported(base,c):
    b=Path(base);r=read(b/'preparation/resume-joint-receipt.json');j=read(b/'latest-joint.json')
    q=read(r['qualification']);seal_qualified(q)
    if (r['mode']!=MODE or q['status']!='DIVERSE_NATIVE_FULL_STATE_CPU_PASS' or q['outer']!=181
        or sha(r['qualification'])!=r['qualification_sha256']
        or q['joint_sha256']!=r['source_joint_sha256']
        or not imported_actor_boundary(j['actor'],181)
        or j['actor']!=read(b/'actor/latest.json')or j['probe']!=read(b/'probe/latest.json')
        or read(b/'question-pool/state.json')!=r['pool_state']
        or j['probe']['actual_adam_steps_per_rank']!=[2530]*4):
        raise ValueError('Four-rank imported actor/probe/data boundary differs')
    return 181


def prepare(source,base,c,qualification):
    import torch
    from .actor import _state_binding
    from search_g1.training_state import atomic_save
    from .probe import atomic_checkpoint
    from .pool import reserve_for_outer,state_sha
    from .diverse_native_resume import stopped
    s=Path(source);b=Path(base);stopped(s);q=read(qualification);jp=s/'latest-joint.json';j=read(jp)
    if (j['outer']!=181 or q['outer']!=181 or q['status']!='DIVERSE_NATIVE_FULL_STATE_CPU_PASS'
        or sha(jp)!=q['joint_sha256'] or j['actor']['actual_adam_steps_per_rank']!=[3620]*8
        or j['probe']['actual_adam_steps_per_rank']!=[2530]*8):raise ValueError('Qualified complete181 required')
    seal_qualified(q)
    once(b/'preparation/resume-joint-intent.json',dict(mode=MODE,source_joint_sha256=sha(jp),created_unix=time.time()))
    target=b/'actor/imported-cp181-four';target.mkdir(parents=True)
    binding=_state_binding(dict(c,grounding_state=j['actor']['grounding_state']))
    counts=flat_numels(j['actor']['hf_model_dir']);states=[];checks=[]
    for rank in range(4):
        pair=[torch.load(j['actor']['rank_states'][i]['path'],map_location='cpu',mmap=True,weights_only=False)for i in(2*rank,2*rank+1)]
        for i,saved in enumerate(pair):
            if(saved['rank']!=2*rank+i or saved['actual_adam_steps']!=3620 or saved['scheduler_completed_outer']!=181
                or saved['scheduler']['last_epoch']!=181 or saved['binding']!=j['actor']['binding']):raise ValueError('Direct source rank boundary differs')
        if pair[0]['scheduler']!=pair[1]['scheduler']:raise ValueError('Schedulers differ')
        optimizer,layout,comparison=merge_optimizer(*pair,counts)
        out={k:v for k,v in pair[0].items()if k not in('optimizer','optimizer_parameter_layout')}
        out.update(rank=rank,binding=binding,optimizer=optimizer,optimizer_parameter_layout=layout,
                   optimizer_layout='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4')
        receipt=atomic_save(target/('rank-%d.pt'%rank),out)
        # Real saved readback: verify serialized target moments against source slices.
        actual=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False)
        for pid,state in actual['optimizer']['state'].items():
            for key in('exp_avg','exp_avg_sq'):
                value=state[key];n=value.numel()//2
                assert torch.equal(value[:n],pair[0]['optimizer']['state'][pid][key])
                assert torch.equal(value[n:],pair[1]['optimizer']['state'][pid][key])
            assert int(state['step'])==3620
        states.append(receipt);checks.append(dict(target_rank=rank,source_ranks=[2*rank,2*rank+1],parameters=comparison,saved_cpu_readback=True))
        print(__import__('json').dumps(dict(event='CP181_FOUR_RESHARD_RANK_COMPLETE',rank=rank,adam=3620,bytes=receipt['bytes'],sha256=receipt['sha256'])),flush=True)
        del actual,out,optimizer,layout,pair;gc.collect()
    proof=dict(status=STATUS,source_joint_path=str(jp),source_joint_sha256=sha(jp),rank_states=states,
        rank_checks=checks,new_optimizer_steps=0,all_moments_bitwise_preserved=True,source_cpu_qualified=True,
        rng_mapping=[0,2,4,6],rng_scope='four streams inherit source even ranks; all eight originals retained; not exact executor resume',completed_unix=time.time())
    migration=b/'preparation/actor-cp181-8to4-reshard.json';once(migration,proof)
    actor=copy.deepcopy(j['actor']);actor.update(binding=binding,source_commit=c['source_commit'],rank_states=states,
        state_dir=str(target),state_path=str(target),actual_adam_steps_per_rank=[3620]*4,new_adam_steps_per_rank=[0]*4,
        skipped_update_opportunities=[0]*4,imported_8to4_cp181=dict(path=str(migration),sha256=sha(migration)),
        historical_rank_receipts_world=8,new_four_rank_updates_at_import=0)
    saved=torch.load(j['probe_snapshot'],map_location='cpu',mmap=True,weights_only=False)
    original=copy.deepcopy(saved)
    saved['binding'].update(world_size=4,rank_questions_per_period=640,rank_trajectories_per_period=3200)
    saved['binding_sha256']=digest(saved['binding']);saved['actual_adam_steps_per_rank']=[2530]*4
    saved['rank_rng']=[dict(original['rank_rng'][2*r],rank=r)for r in range(4)]
    for key in('model','ema_model'):
        for name,value in saved[key].items():assert torch.equal(value,original[key][name])and torch.isfinite(value).all()
    for pid,state in saved['optimizer']['state'].items():
        assert int(state['step'])==2530
        for key in('exp_avg','exp_avg_sq'):assert torch.equal(state[key],original['optimizer']['state'][pid][key])and torch.isfinite(state[key]).all()
    (b/'probe').mkdir();probe=atomic_checkpoint(b/'probe',saved,181,dimension_migration=True)
    snapshot=b/'probe/head-imported-cp181-four.pt';os.link(b/'probe/latest.pt',snapshot)
    actual=torch.load(snapshot,map_location='cpu',mmap=True,weights_only=False)
    for key in('model','ema_model'):
        for name,value in original[key].items():assert torch.equal(value,actual[key][name])
    for pid,state in original['optimizer']['state'].items():
        for key,value in state.items():assert torch.equal(value,actual['optimizer']['state'][pid][key])
    del actual,saved,original;gc.collect()
    def link(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    for n in range(1,182):
        root=s/'outers'/('outer-%06d'%n)
        for name in('complete.json','pool-state-at-completion.json','grounding-state-at-completion.json','grounding-refresh.json'):
            if(root/name).exists():link(root/name,b/'outers'/root.name/name)
    for name in('binding.json','provenance.json'):
        if(s/'question-pool'/name).exists():link(s/'question-pool'/name,b/'question-pool'/name)
    before=read(j['pool_snapshot']);current=read(s/'question-pool/state.json');pool=before
    if current!=before:
        old=read(s/'preparation/launch-config.json');root=s/'outers/outer-000182';reservation=read(root/'pool-reservation.json')
        ids,cycles,after=reserve_for_outer(before,old,182)
        assert reservation['before_sha256']==state_sha(before)and reservation['after_sha256']==state_sha(after)
        assert reservation['indices']==ids and reservation['cycles']==cycles and reservation['after']==after and current==after
        pool=after
        for name in('pool-reservation.json','selection.json','questions.json'):
            if(root/name).exists():link(root/name,b/'outers'/root.name/name)
    once(b/'question-pool/state.json',pool)
    fork=copy.deepcopy(j);fork.update(actor=actor,probe=probe,probe_snapshot=str(snapshot),imported_boundary=True)
    # Replace only our imported metadata link, never mutate the source inode.
    (b/'outers/outer-000181/complete.json').unlink()
    once(b/'outers/outer-000181/complete.json',fork);once(b/'latest-joint.json',fork)
    once(b/'actor/latest.json',actor)
    for path in(s/'preparation').glob('prefix-migration-*.json'):link(path,b/'preparation'/path.name)
    r=dict(mode=MODE,completed_outer=181,qualification=str(qualification),qualification_sha256=sha(qualification),
        source_joint_sha256=sha(jp),pool_state=pool,actor_adam=[3620]*4,probe_adam=[2530]*4,
        pending_reservation_preserved=current!=before,optimizer_reset=False,new_optimizer_steps=0,
        source_base=str(s),rng_mapping=[0,2,4,6],completed_unix=time.time())
    once(b/'preparation/resume-joint-receipt.json',r);validate_imported(b,c);return r
