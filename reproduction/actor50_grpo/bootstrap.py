"""CPU qualification and explicit branch seed; no CUDA allocation or actor writes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .schedule import BRANCH_SCHEDULE,validate_branch

SOURCE_BASE='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116-resume-a4a0c2a18cca40bc8c3850d40eed8554'
SOURCE_JOINT_SHA='80c5b4cdbafed1c92441352cd725cc7b2819319da37f7672498f58524ec79f2c'
RUN='54ac33dcebe64e62a11fe4177f41f116'


def read(p):return json.loads(Path(p).read_text())


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(v):return hashlib.sha256(canonical(v)).hexdigest()


def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):d.update(block)
    return d.hexdigest()


def once(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)


def publish(p,value):
    p=Path(p);temp=p.with_name(p.name+'.partial');once(temp,value);os.replace(temp,p)


def validate_config(c,base):
    validate_branch(c)
    required=dict(run_id=RUN,base=str(base),outers=200,candidate_questions=2560,
        selected_questions=512,responses_per_question=5,prefix_policy_tokens=20,
        exploration_questions=0,probe_learning_rate=5e-6,probe_epochs_per_update=1,
        probe_update_period=10,probe_updates_per_period=50,probe_global_batch_average=256,
        actor_global_mini_batch=128,actor_ppo_epochs=1,world_size=4,
        actor_hidden_native=True,probe_update_at_final=False,
        nonmixed_advantage_mode='zero_anchor_all_correct_only')
    if any(c.get(k)!=v for k,v in required.items()):raise ValueError('Actor50 GRPO branch recipe differs')
    resumed=c['resume_joint'].get('adapter')=='actor50_full_joint_v1'
    if (not resumed and (c['resume_joint']['adapter']!='actor50_prefix2040_v1'
            or c['resume_joint']['completed_outer']!=50 or c['resume_joint']['source_base']!=SOURCE_BASE)):
        raise ValueError('Bind the actual original actor50 boundary')
    if (c['source_joint50_sha256']!=SOURCE_JOINT_SHA
            or c['probe_source_model_kind']!='ema' or c['probe_source_epoch']<1
            or len(c['probe_source_sha256'])!=64 or c['actual_em_selection_fallback']!='probe_score_fill'
            or c['model_revision']!='aa8e72537993ba99e69dfaafa59ed015b17504d1'):
        raise ValueError('Bind actual actor50 and the latest complete warmup EMA source')
    if resumed:
        if not 51<=c['resume_joint']['completed_outer']<200 or not c.get('full_joint_fault_recovery'):
            raise ValueError('Require an explicit stopped full-joint fault recovery')
    from reproduction.prefix_selection200.actor import OBJECTIVE
    from reproduction.prefix_selection200.actor_safety import POLICY_SAFEGUARDS
    from reproduction.prefix_selection200.interaction_protocol import CONTRACT
    from reproduction.prefix_selection200.grounding import validate_config as check_grounding
    if c['actor_loss_reduction']!=OBJECTIVE or c['policy_safeguards']!=POLICY_SAFEGUARDS or c['interaction_protocol']!=CONTRACT:
        raise ValueError('I1-I6 actor science and shared BF16 interaction must remain unchanged')
    check_grounding(c)
    from .single_auxiliary import validate as single_auxiliary
    if not single_auxiliary(c) and (len(set(c['physical_gpu_uuids']+c['auxiliary_gpu_uuids']))!=6 or c['auxiliary_gpu_indices']!=[5,6]):
        raise ValueError('Retain four actual actor UUIDs and two approved auxiliary UUIDs')
    for key in ('canonical_train','official_model_dir','source_question_pool','probe_source_checkpoint','validation_questions','final_questions'):
        if not str(c[key]).startswith('/data/'):raise ValueError('Large data must remain remote')
    return c


def validate_imported_boundary(base,c):
    base=Path(base);validate_config(c,base)
    if c['resume_joint'].get('adapter')=='actor50_full_joint_v1':
        from .full_joint_resume import validate_boundary
        return validate_boundary(base,c)
    q=read(base/'preparation/branch-qualified.json')
    if q['source_commit']!=c['source_commit'] or q['status']!='ACTOR50_WARMUP_EMA_BRANCH_CPU_QUALIFIED':
        raise ValueError('Missing immutable actor50/warmup CPU import qualification')
    for row in q['sealed_metadata']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Imported small boundary changed')
    a=read(base/'actor/latest.json');p=read(base/'probe/latest.json');pool=read(base/'question-pool/state.json')
    if (a['completed_outer']!=50 or a['scheduler_completed_outer']!=50
            or a['actual_adam_steps_per_rank']!=[1000]*4
            or p['completed_outer']!=50 or p['actual_adam_steps_per_rank']!=[0]*4
            or pool['next_outer']!=51 or pool['consumed_candidates']!=230400
            or a['binding']!=c['retained_actor50_binding']):
        raise ValueError('The branch must start exactly at actor50/pool51/new online probe0')
    for row in q['actor50_files']:
        s=Path(row['path']).stat()
        if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=(row['dev'],row['inode'],row['bytes'],row['mtime_ns']):
            raise ValueError('CPU-qualified source state changed before CUDA restore')
    return 50


def prepare_actor(base):
    import torch
    torch.set_num_threads(4)
    from reproduction.prefix_selection200.checkpoint_cpu_once import hash_file,tensor_cpu,validate_rng,validate_adam
    from reproduction.prefix_selection200.actor import learning_rate
    base=Path(base);source=Path(SOURCE_BASE);sealed=[];rank_proofs=[]
    assert not torch.cuda.is_initialized()
    jointpath=source/'outers/outer-000050/complete.json';assert sha(jointpath)==SOURCE_JOINT_SHA
    joint=read(jointpath);actor=joint['actor'];assert actor['completed_outer']==50 and actor['actual_adam_steps_per_rank']==[1000]*4
    for rank,row in enumerate(actor['rank_states']):
        publish(base/'preparation/prepare-progress.json',dict(stage='ACTOR50_RANK_FULL_SHA_CPU_FINITE',rank=rank,checked_unix=time.time()))
        file=hash_file(row['path'],row);sealed.append(file)
        saved=torch.load(row['path'],map_location='cpu',weights_only=False,mmap=True)
        assert saved['rank']==rank and saved['completed_outer']==50 and saved['actual_adam_steps']==1000
        assert saved['binding']==actor['binding'] and saved['scheduler_completed_outer']==50
        assert saved['optimizer_layout']=='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4'
        shapes=[r['shape'] for group in saved['optimizer_parameter_layout'] for r in group]
        count=validate_adam(saved['optimizer'],shapes,1000,learning_rate(50),weight_decay=.01)
        sch=saved['scheduler'];assert sch['last_epoch']==50 and sch['_last_lr']==[learning_rate(50)]
        rank_proofs.append(dict(rank=rank,Adam=1000,scheduler=50,optimizer_states=count,rng=validate_rng(saved['rng'])))
        del saved;gc.collect()
    from safetensors import safe_open
    for name,row in actor['hf_files'].items():
        path=Path(actor['hf_model_dir'])/name;sealed.append(hash_file(path,row))
        if name.endswith('.safetensors'):
            publish(base/'preparation/prepare-progress.json',dict(stage='ACTOR50_HF_FP32_FULL_SHA_FINITE',file=name,checked_unix=time.time()))
            with safe_open(str(path),framework='pt',device='cpu') as f:
                for key in f.keys():tensor_cpu(f.get_tensor(key),dtype=torch.float32,finite=True)
    driver=actor['driver_state'];sealed.append(hash_file(driver['path'],driver))
    saved=torch.load(driver['path'],map_location='cpu',weights_only=False)
    assert saved['completed_outer']==50 and saved['actor_version']==actor['actor_version']
    driver_rng=validate_rng(saved['rng']);pool=read(joint['pool_snapshot'])
    assert saved['candidate_pool_state']['value']==pool
    assert pool['next_outer']==51 and pool['consumed_candidates']==230400
    assert pool['cycle']*len(pool['order'])+pool['cursor']==230400
    assert sorted(pool['order'])==list(range(len(pool['order'])))
    result=dict(status='ACTOR50_FULL_SHA_CPU_FINITE_PASS',checked_unix=time.time(),source_base=SOURCE_BASE,
        source_joint_sha256=SOURCE_JOINT_SHA,actor50_files=sealed,rank_proofs=rank_proofs,
        driver_rng=driver_rng,pool_consumed=230400,actor_code_sha256=sha(source/'repo/reproduction/prefix_selection200/actor.py'),
        all_large_assets_remain_remote=True,CUDA_restore='PENDING_REAL_ACTOR51')
    once(base/'preparation/actor50-qualified.json',result)
    assert not torch.cuda.is_initialized();return result


def seed_branch(base):
    import torch
    from reproduction.prefix_selection200.prefix5_model import validate_model_state,validate_native_buffers
    from reproduction.prefix_selection200.probe import atomic_checkpoint
    from reproduction.prefix_selection200.checkpoint_cpu_once import hash_file,validate_adam,validate_rng
    base=Path(base);source=Path(SOURCE_BASE);q=read(base/'preparation/actor50-qualified.json')
    warmup=read(base/'preparation/warmup-parent-handle.json');wb=Path(warmup['base'])
    cleanup=read(wb/'logs/cleanup-status.json');assert cleanup['cleaned'] and cleanup['owned_births_all_gone']
    stop=read(wb/'preparation/human-handoff-to-grpo50.json');latest=read(wb/'probe/latest.json')
    assert latest==stop['latest'] and latest['actual_adam_steps_per_rank']==[latest['ema_updates']]*4
    file=hash_file(latest['checkpoint'],latest);saved=torch.load(latest['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    assert saved['run_id']==warmup['run_id'] and saved['completed_epochs']==latest['completed_epochs']
    assert saved['actor_updates']==0 and saved['prefix_policy_tokens']==20
    validate_model_state(saved['raw_model']);validate_model_state(saved['ema_model'])
    validate_native_buffers(saved['raw_buffers']);validate_native_buffers(saved['ema_buffers'])
    validate_adam(saved['optimizer'],[v.shape for v in saved['raw_model'].values()],latest['ema_updates'],5e-6,weight_decay=0.)
    assert len(saved['rank_rng'])==4
    for rng in saved['rank_rng']:validate_rng(rng)
    joint=read(source/'outers/outer-000050/complete.json');actor=joint['actor'];c=read(base/'preparation/launch-config.json')
    c.update(probe_source_checkpoint=latest['checkpoint'],probe_source_sha256=latest['sha256'],
        probe_source_epoch=latest['completed_epochs'],probe_source_model_kind='ema',probe_source_run=warmup['run_id'],
        retained_actor50_binding=actor['binding'],source_joint50_sha256=SOURCE_JOINT_SHA,actor50_pool_consumed_candidates=230400)
    validate_config(c,base);publish(base/'preparation/launch-config.json',c)
    for path,value in ((base/'actor/latest.json',actor),(base/'question-pool/state.json',read(joint['pool_snapshot'])),
            (base/'question-pool/binding.json',read(source/'question-pool/binding.json'))):once(path,value)
    binding=dict(run_id=c['run_id'],role='actor50_grpo_prefix2040_online_correctness_probe',
        prefix_tokens=20,head_max_tokens=293,question_max_tokens=273,world_size=4,learning_rate=5e-6,
        source_probe_sha256=latest['sha256'],source_epoch=latest['completed_epochs'],source_model_kind='ema',
        warmup_model_weights_only=True,warmup_optimizer_imported=False,online_initial_adam_steps=0,
        probe_update_period=10,updates_per_period=50,actor_start_outer=50,
        group_weight=1.,loss='mean_trajectory_BCE_plus_mean_G5_mixed_BCE',
        candidate_questions=2560,selected_questions=512,probe_total_actual_adam_per_rank=700)
    proof=dict(source_warmup_run=warmup['run_id'],source_epoch=latest['completed_epochs'],source_model_kind='ema',
        source_sha256=latest['sha256'],learned_tensor_count=14,warmup_model_weights_only=True,
        warmup_optimizer_imported=False,warmup_rng_imported=False)
    initial=dict(schema_version=1,run_id=c['run_id'],binding=binding,binding_sha256=digest(binding),completed_outer=50,
        model={k:v.clone() for k,v in saved['ema_model'].items()},
        model_nonpersistent_buffers={k:v.clone() for k,v in saved['ema_buffers'].items()},
        warmup_model_import=proof,optimizer=None,rank_rng=None,actual_adam_steps_per_rank=[0]*4,learning_rate=5e-6)
    (base/'probe').mkdir();receipt=atomic_checkpoint(base/'probe',initial,50)
    os.link(base/'probe/latest.pt',base/'probe/head-outer-0050.pt')
    once(base/'probe/initialization-complete.json',dict(proof,receipt=receipt,warmup_updates_reexecuted=0))
    start=dict(joint,execution_id=c['execution_id'],source_commit=c['source_commit'],probe=receipt,
        probe_snapshot=str(base/'probe/head-outer-0050.pt'),branch_start=True,
        periodic_probe_update=False,probe_boundary=dict(last_update_outer=50,actual_adam_per_rank=0),
        probe_metrics=dict(status='LATEST_COMPLETE_WARMUP_EMA_MODEL_ONLY_SEED',
            source_epoch=latest['completed_epochs'],test=latest['metrics']['after']['test']),
        source_lineage=dict(actor50_joint_sha256=SOURCE_JOINT_SHA,warmup_checkpoint_sha256=latest['sha256'],
            source_actor50_unchanged=True,new_online_probe_Adam0=True,retained_pool_consumed_candidates=230400))
    once(base/'latest-joint.json',start)
    # Immutable old history is provenance; it is not relabeled as new probe optimization.
    for n in range(1,51):
        src=source/'outers'/('outer-%06d'%n)/'complete.json'
        target=base/'outers'/('outer-%06d'%n)/'complete.json';target.parent.mkdir(parents=True,exist_ok=True);os.link(src,target)
    metadata=[base/x for x in ('actor/latest.json','probe/latest.json','question-pool/state.json','question-pool/binding.json','latest-joint.json','preparation/launch-config.json')]
    result=dict(status='ACTOR50_WARMUP_EMA_BRANCH_CPU_QUALIFIED',checked_unix=time.time(),source_commit=c['source_commit'],
        actor50_files=q['actor50_files'],source_joint_sha256=SOURCE_JOINT_SHA,warmup_source=file,
        warmup_complete_epochs=latest['completed_epochs'],warmup_actual_Adam_EMA=latest['ema_updates'],
        actor_actual_Adam_per_rank=[1000]*4,scheduler50=True,new_online_probe_Adam_per_rank=[0]*4,
        source_warmup_all_states_preserved=True,actor_optimizer_reset=False,probe_initialization=proof,
        sealed_metadata=[dict(path=str(p),sha256=sha(p)) for p in metadata],CUDA_restore='PENDING_REAL_ACTOR51')
    once(base/'preparation/branch-qualified.json',result);assert not torch.cuda.is_initialized();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('actor','seed'));p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    try:
        result=(prepare_actor if a.phase=='actor' else seed_branch)(base)
        once(base/'preparation'/('prepare-'+a.phase+'-exit.json'),dict(success=True,checked_unix=time.time(),status=result['status']))
    except BaseException as error:
        once(base/'preparation'/('prepare-'+a.phase+'-exit.json'),dict(success=False,checked_unix=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
