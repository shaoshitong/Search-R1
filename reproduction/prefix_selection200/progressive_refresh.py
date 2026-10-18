"""Post-actor boundary50/100/150 new-length data and readout migration.

This namespace has its own deterministic question reservation. The actor pool
and completed actor updates never move during this extra probe-only collection.
Ordinary P10 nodes retain their original aligned full-pool data, now3epochs.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import random
import time

from .progressive_prefix import MODE,after_boundary,before_fit,expand_probe_state
from reproduction.prefix20_online.pool import read,digest,immutable,identity


def migrate(base,outer):
    import torch
    from .contract import PREFIX_TOKENS,WIDTH,PROBE_UPDATES
    from .probe import atomic_checkpoint,digest as binding_digest,actual_adam_steps
    base=Path(base);directory=base/'probe';latest=read(directory/'latest.json')
    marker=directory/('prefix-migration-%03d.json'%outer)
    if marker.exists():
        proof=read(marker)
        if latest['binding_sha256']!=proof['migrated_receipt']['binding_sha256']:
            raise ValueError('A completed migration cannot be replayed over another probe binding')
        return proof
    path=directory/'latest.pt'
    if path.stat().st_size!=latest['bytes'] or digest(path)!=latest['sha256']:
        raise ValueError('Migration must load the exact retained prior probe checkpoint')
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    expected=before_fit(outer)
    if (saved['completed_outer']!=expected['last_update_outer']
            or saved['actual_adam_steps_per_rank']!=[expected['actual_adam_per_rank']]*4):
        raise ValueError('Boundary migration may not reset or substitute the prior probe clock')
    names=list(saved['model'])
    # The immutable source's real CPU audit validated Adam against state_dict
    # parameter order. This architecture has14 learned tensors and only explicitly
    # separate nonpersistent buffers; do not infer optimizer identity by shape.
    if len(names)!=14 or len(saved['optimizer']['param_groups'])!=1:
        raise ValueError('Unexpected native block/readout parameter layout')
    ids=saved['optimizer']['param_groups'][0]['params']
    if len(ids)!=len(names):raise ValueError('Saved Adam parameter layout differs')
    for name,param_id in zip(names,ids):
        state=saved['optimizer']['state'][param_id]
        if (tuple(state['exp_avg'].shape)!=tuple(saved['model'][name].shape)
                or tuple(state['exp_avg_sq'].shape)!=tuple(saved['model'][name].shape)
                or int(state['step'])!=expected['actual_adam_per_rank']):
            raise ValueError('Prior full-state Adam layout/clock differs: '+name)
    saved['optimizer_parameter_names']=names
    target=after_boundary(outer)
    if target!=PREFIX_TOKENS:raise ValueError('Migration process prefix differs from its boundary')
    changed=expand_probe_state(saved,target)
    unchanged=[name for name in names if name!='output.weight']
    if not all(torch.equal(saved['model'][name],changed['model'][name]) for name in unchanged):
        raise ValueError('Only flattened output.weight may change shape')
    if not all(torch.equal(value,changed['model_nonpersistent_buffers'][name])
               for name,value in saved['model_nonpersistent_buffers'].items()):
        raise ValueError('Native transformer buffers changed during readout expansion')
    changed['binding']=dict(saved['binding'],prefix_tokens=target,head_max_tokens=WIDTH,
        updates_per_period=PROBE_UPDATES,epochs_per_update=3,probe_total_actual_adam_per_rank=2500,
        rank_questions_per_period=1920,rank_trajectories_per_period=9600,
        rank_questions_per_epoch=640,rank_trajectories_per_epoch=3200,
        prefix_schedule_mode=MODE)
    changed['binding_sha256']=binding_digest(changed['binding']);changed['prefix_tokens']=target
    result=atomic_checkpoint(directory,changed,saved['completed_outer'],milestone_alias=False)
    proof=dict(outer=outer,source_receipt=latest,migrated_receipt=result,
        source_actual_adam=saved['actual_adam_steps_per_rank'],target_actual_adam=changed['actual_adam_steps_per_rank'],
        all_block_parameters_equal=True,all_native_buffers_retained=True,
        optimizer_parameter_names=names,readout_migration=changed['prefix_migration'],epoch=time.time())
    immutable(marker,proof);del changed,saved;gc.collect()
    return proof


def prepare(base,outer):
    from .contract import PREFIX_TOKENS,HIDDEN_CAPTURE_CONTRACT
    from .pool import fixed_test_identities
    if outer not in (50,100,150) or PREFIX_TOKENS!=after_boundary(outer):
        raise ValueError('Use only the exact scheduled new-prefix transition')
    base=Path(base);c=read(base/'preparation/launch-config.json');actor=read(base/'actor/latest.json')
    if actor['completed_outer']!=outer or actor['scheduler_completed_outer']!=outer:
        raise ValueError('Save the completed student before sampling its new-prefix probe data')
    data=base/'probe-refreshes'/('round-%06d'%outer);target=data/'outers'/('outer-%06d'%outer)
    target.mkdir(parents=True,exist_ok=True);(data/'preparation').mkdir(exist_ok=True)
    conf=data/'preparation/launch-config.json'
    if conf.exists():return dict(config=str(conf),root=str(target),actor_version=actor['actor_version'])
    source=Path(c['source_question_pool'])
    if digest(source)!=c['source_question_pool_sha256']:raise ValueError('Eligible source pool changed')
    pool=read(source);indices=random.Random(c['seed']+outer*1000003+907).sample(range(len(pool['records'])),2560)
    records=[dict(pool['records'][index],pool_index=index,candidate_ordinal=i) for i,index in enumerate(indices)]
    fixed=fixed_test_identities(c);fixed_q={r['qid'] for r in fixed};fixed_h={r['normalized_question_hash'] for r in fixed}
    if any(r['qid'] in fixed_q or r['normalized_question_hash'] in fixed_h for r in records):
        raise ValueError('Probe refresh training questions overlap fixed512')
    import pyarrow.parquet as pq
    table=pq.read_table(c['canonical_train'],columns=['prompt','reward_model','data_source','extra_info'])
    originals=table.take([r['source_index'] for r in records]).to_pylist();questions=[]
    for record,original in zip(records,originals):
        actual=identity(original)
        if any(record[k]!=v for k,v in actual.items()):raise ValueError('Probe question identity changed')
        questions.append(dict(record,**original))
    immutable(target/'questions.json',dict(run_id=c['run_id'],outer=outer,questions=2560,rows=questions))
    immutable(target/'probe-only-reservation.json',dict(run_id=c['run_id'],outer=outer,indices=indices,
        source_sha256=c['source_question_pool_sha256'],seed=c['seed']+outer*1000003+907,
        student_checkpoint_outer=outer,student_actor_version=actor['actor_version'],
        prefix_tokens=PREFIX_TOKENS,actor_pool_advanced=False))
    immutable(target/'actor-binding.json',dict(run_id=c['run_id'],outer=outer,
        actor_model_dir=actor['hf_model_dir'],actor_version=actor['actor_version'],
        actor_model_files_sha256={name:item['sha256'] for name,item in actor['hf_files'].items()}))
    scoped=dict(c,base=str(data),prefix_policy_tokens=PREFIX_TOKENS,hidden_capture_contract=HIDDEN_CAPTURE_CONTRACT,
        progressive_probe_refresh=True,probe_refresh_tag='probe-refresh-p%d'%PREFIX_TOKENS,
        actor_model_dir=actor['hf_model_dir'],actor_version=actor['actor_version'])
    for name in ['adaptive_candidates','candidate_schedule']:
        scoped.pop(name,None)
    immutable(conf,scoped)
    return dict(config=str(conf),root=str(target),actor_version=actor['actor_version'])


def bootstrap_joint(base):
    """New typed joint50 after real3epoch Prefix20 fit; actor50 is never rerun."""
    import os
    from .probe import sha
    from .periodic import probe_boundary
    from reproduction.prefix20_online.launch import publish
    base=Path(base);c=read(base/'preparation/launch-config.json');root=base/'outers/outer-000050'
    old=read(root/'inherited-complete.json');probe=read(base/'probe/latest.json')
    metrics=read(root/'probe-update/metrics.json')
    if (probe['actual_adam_steps_per_rank']!=[400]*4 or metrics['new_adam_steps_per_rank']!=150
            or metrics['epochs']!=3 or metrics['prefix_tokens']!=20
            or metrics['actor_version']!=old['actor']['actor_version']):
        raise ValueError('Do not publish CP50 bootstrap before all3epochs actually finish')
    snapshot=root/'probe-state-at-completion.pt';os.link(base/'probe/latest.pt',snapshot)
    joint=dict(old,execution_id=c['execution_id'],source_commit=c['source_commit'],
        runtime_source_commit=c['source_commit'],science_source_commit=c['science_source_commit'],
        probe=probe,probe_snapshot=str(snapshot),probe_metrics=metrics,probe_boundary=probe_boundary(50),
        pool_snapshot=str(root/'pool-state-at-completion.json'),
        progressive_bootstrap=dict(actor_outer_reexecuted=False,actor_prefix_tokens=5,probe_prefix_tokens=20,
            probe_source_actor_version=old['actor']['actor_version'],source_joint_sha256=c['resume_joint']['source_joint_sha256'],
            inherited_probe_updates=250,new_probe_updates=150,epochs=3),observed_unix=time.time())
    reservation=base/'probe-refreshes/round-000050/outers/outer-000050/probe-only-reservation.json'
    joint['probe_refresh_reservation']=dict(path=str(reservation),sha256=sha(reservation),actor_pool_advanced=False)
    immutable(root/'complete.json',joint);publish(base/'latest-joint.json',joint)
    immutable(base/'preparation/progressive-bootstrap-complete.json',dict(status='CP50_PREFIX20_THREE_EPOCH_BOOTSTRAP_COMPLETE',
        joint_sha256=sha(root/'complete.json'),probe=probe,actor=old['actor'],actor_updates_reexecuted=0))
    return joint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['migrate','prepare','bootstrap-joint'])
    parser.add_argument('--base',required=True);parser.add_argument('--outer',type=int,required=True);args=parser.parse_args()
    result=bootstrap_joint(args.base) if args.action=='bootstrap-joint' else {'migrate':migrate,'prepare':prepare}[args.action](args.base,args.outer)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
