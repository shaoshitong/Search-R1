"""Genuine once-only CPU/full-SHA qualification of a stopped I1–I6 joint.

Run only at the authorized execution-host boundary. This reads real saved
weights/Adam/RNG once, uses CPU mmap and finite chunks, never CUDA/forward/eval.
A pre-existing successful proof is checked by immutable stat seals without a
second payload scan. An unfinished intent is never silently replayed.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .actor import _state_binding, learning_rate, REVISION
from .candidate_schedule import candidate_count, consumed_candidates_through
from .checkpoint_cpu_once import need, tensor_cpu, validate_adam, validate_rng, digest
from .contract import WORLD, PROBE_LR, PROBE_UPDATES
from .grounding import expected_round
from .i1_i6_joint_resume import (QUALIFIED, canonical_path, read, sha, _once, _writer_barrier,
    _joint, _grounding, verify_source_compatibility, scientific_binding,pending_sampling_boundary)
from .periodic import probe_boundary
from .pool import state_sha


def actor_state(saved,actor,config,outer,rank):
    binding_config=dict(config,grounding_state=actor['grounding_state'])
    expected=(scientific_binding(binding_config) if config.get('source_compatibility') else _state_binding(binding_config))
    steps=actor['actual_adam_steps_per_rank'][rank]
    need(saved['run_id']==config['run_id'] and saved['completed_outer']==outer and saved['rank']==rank and
         saved['binding']==actor['binding']==expected and saved['actual_adam_steps']==steps and
         saved['scheduler_completed_outer']==outer and saved['grounding_state']==actor['grounding_state'] and
         saved['grounding_probe_version']==actor['grounding_probe_version'] and
         saved['original_rollout_sha256']==actor['original_rollout_sha256'], 'Actual actor rank/state/science/grounding boundary differs')
    need(saved['model_master_dtype']=='torch.float32' and
         saved['optimizer_layout']=='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4','Actual FP32 master/FSDP layout differs')
    groups=saved['optimizer']['param_groups'];layout=saved['optimizer_parameter_layout'];shapes=[]
    need(len(groups)==len(layout),'Saved optimizer group/flat-shard layout differs')
    for group,rows in zip(groups,layout):
        need(len(group['params'])==len(rows),'Saved FSDP parameter ordering differs')
        for row in rows:
            need(row['dtype']=='torch.float32' and isinstance(row['shape'],list) and
                 all(type(n) is int and n>=0 for n in row['shape']) and math.prod(row['shape'])==row['numel'],
                 'Saved actor flat-shard shape/dtype is invalid')
            shapes.append(row['shape'])
    adam=validate_adam(saved['optimizer'],shapes,steps,learning_rate(outer),weight_decay=.01)
    schedule=saved['scheduler']
    need(schedule['last_epoch']==outer and schedule['_step_count']==outer+1 and
         schedule['base_lrs']==[1e-6] and schedule['_last_lr']==[learning_rate(outer)], 'Actual outer scheduler clock/LR differs')
    return dict(rank=rank,actual_adam_steps=steps,adam_states=adam,scheduler_completed_outer=outer,rng=validate_rng(saved['rng']))


def actor_ledger(seals,actor,outer,rank):
    row=next(value for value in actor['rank_receipts'] if value['rank']==rank)
    file=seals.file(row['ledger_path']);need(file['sha256']==row['ledger_sha256'],'Actual rank update ledger SHA differs')
    with Path(row['ledger_path']).open() as stream:events=[json.loads(line) for line in stream if line.strip()]
    writes=[value for value in events if value['event']=='ACTUAL_ADAM_WRITE']
    opportunities=[value for value in events if value['event']=='OPPORTUNITY']
    before=row['actual_adam_steps']-row['new_adam_steps']
    need(len(writes)==row['new_adam_steps'] and len(opportunities)==20 and
         [value['opportunity'] for value in opportunities]==list(range(20)) and
         [value['status'] for value in opportunities]==row['opportunity_statuses'], 'Actual20-opportunity ledger is incomplete')
    last=-1
    for index,value in enumerate(writes):
        need(value['rank']==rank and value['outer']==outer and value['writes_this_attempt']==index+1 and
             value['actual_adam_steps']==before+index+1 and last<value['opportunity']<20,
             'Actual returned Adam writes are not sequential retained updates')
        last=value['opportunity']
    cursor=before
    for value in opportunities:
        need(value['rank']==rank and value['outer']==outer and value['processed'] is True and
             value['actual_adam_before']==cursor and value['actual_adam_after']-cursor in (0,1) and
             value['automatic_rollback_performed'] is False,'Actual opportunity counter/rollback semantics differs')
        cursor=value['actual_adam_after']
    need(cursor==row['actual_adam_steps'],'Actual retained Adam count differs from its ledger')
    return dict(rank=rank,returned_writes=len(writes),opportunities=20,file=file)


def probe_state(saved,receipt,config,outer,history):
    import torch
    from .prefix5_model import PARAMETER_SHAPES,NATIVE_BUFFER_LAYOUTS
    expected=probe_boundary(outer);completed=expected['last_update_outer'];steps=expected['actual_adam_per_rank']
    need(saved['run_id']==config['run_id'] and saved['completed_outer']==completed and
         saved['actual_adam_steps_per_rank']==[steps]*WORLD and saved['learning_rate']==PROBE_LR and
         saved['binding_sha256']==digest(saved['binding'])==receipt['binding_sha256'],'Actual periodic probe state/steps/binding differs')
    required=dict(role='prefix_selection200_actorhidden_period10_question5_correctness_probe',run_id=config['run_id'],
        representation_source='current_sampling_actor_native_layer27_postblock',independent_encoder_forward=False,
        probe_update_period=10,updates_per_period=50,encoder_revision=REVISION,question_max_tokens=273,
        head_max_tokens=278,prefix_tokens=5,feature_layer=27,feature_boundary='post_block',
        source_probe_sha256=config['probe_source_sha256'],source_epoch=config['probe_source_epoch'],source_model_kind='ema',
        source_warmup_run=config['probe_source_run'],warmup_model_weights_only=True,warmup_optimizer_imported=False,
        online_initial_adam_steps=0,learning_rate=PROBE_LR,group_weight=1.,
        loss='mean_trajectory_BCE_plus_mean_G5_mixed_BCE',updates_per_outer=0,global_batch_average=256,
        actual_global_batches=[255,260],world_size=WORLD,accumulation=1,rank_questions_per_period=640,
        rank_trajectories_per_period=3200,selected_questions=512,candidate_questions=2560,exploration=0,
        diagnostic_manifest_sha256=config['fixed_test_manifest_sha256'],final_outer_probe_update=False,
        probe_total_actual_adam_per_rank=950)
    need(all(saved['binding'].get(key)==value for key,value in required.items()),'Actual retained head optimizer/science recipe differs')
    model=saved['model'];need(set(model)==set(PARAMETER_SHAPES),'Native copied block/binary-head tensors missing')
    for name,shape in PARAMETER_SHAPES.items():tensor_cpu(model[name],dtype=torch.float32,shape=shape,finite=True)
    buffers=saved['model_nonpersistent_buffers'];layout=next((value for value in NATIVE_BUFFER_LAYOUTS if set(value)==set(buffers)),None)
    need(layout is not None,'Actual native RoPE buffers missing')
    for name,shape in layout.items():tensor_cpu(buffers[name],dtype=torch.float32,shape=shape,finite=True)
    if completed==0:
        need(saved['optimizer'] is None and saved['rank_rng'] is None,'Never manufacture online probe optimizer/RNG at actual0')
        imported=saved['warmup_model_import']
        need(imported['source_epoch']==config['probe_source_epoch'] and imported['source_model_kind']=='ema' and
             imported['source_warmup_run']==config['probe_source_run'] and imported['warmup_model_weights_only'] is True and
             imported['warmup_optimizer_imported'] is False and imported['warmup_rng_imported'] is False,
             'Initial head is not its original model-only warmup import')
        return dict(actual_adam_steps_per_rank=[0]*WORLD,optimizer='correctly_absent_at_actual0',rank_rng='correctly_absent_at_actual0',finite_model=True)
    actual=history[completed]
    need(actual['probe']==receipt and saved['actor_version']==actual['rollout']['actor_version'] and
         saved['scored_features_manifest_sha256']==actual['probe_metrics']['scored_features_manifest_sha256'] and
         saved['own_em_sha256']==actual['probe_metrics']['own_em_sha256'],'Saved head binds another actual sampling actor/outcome batch')
    adam=validate_adam(saved['optimizer'],[value.shape for value in model.values()],steps,PROBE_LR,weight_decay=0.)
    need(len(saved['rank_rng'])==WORLD and [value['rank'] for value in saved['rank_rng']]==list(range(WORLD)),
         'Keep all four true head RNG streams')
    rng=[]
    for value in saved['rank_rng']:
        need(value['actual_adam_steps']==steps,'Head RNG belongs to another Adam boundary');rng.append(validate_rng(value))
    return dict(actual_adam_steps_per_rank=[steps]*WORLD,adam_states=adam,rank_rng=rng,finite_model=True)


def seal_pending_sampling(seals,base,pending):
    """Full-SHA each accepted prefix/score file once, including native hidden."""
    if pending is None:return
    root=base/'outers'/('outer-%06d'%pending['outer'])
    for name in ('pool-reservation.json','selection.json','questions.json','actor-binding.json'):seals.json(root/name)
    for phase in pending['phases']:seals.tree(root/phase)
    prefix=seals.json(root/'prefix/complete.json');seals.verify_producer_references(prefix)
    scan=seals.json(root/'prefix/scan/scan-complete.json')
    for name,producer in scan['files'].items():
        need(Path(name).name==name,'Accepted pending scan filename escaped its original phase')
        seals.file(root/'prefix/scan'/name,producer)
    score=seals.json(root/'probe-score/manifest.json')
    for name,key in (('index.json','index_sha256'),('candidate-scores.jsonl','candidate_scores_sha256')):
        path=root/'probe-score'/name;seals.file(path,dict(bytes=path.stat().st_size,sha256=score[key]))


def qualify(base,outer,terminal_proof,terminal_sha,output):
    from .checkpoint_native_pending_probe import Seals,_hf
    base=canonical_path(base);output=canonical_path(output,exists=False);config=read(base/'preparation/launch-config.json')
    terminal=read(terminal_proof);ledger=read(base/'logs/process-ledger.json')
    need(type(outer) is int and 1<=outer<=200 and sha(terminal_proof)==terminal_sha,'Exact terminal/current joint selection required')
    need(terminal['process_ledger_sha256']==sha(base/'logs/process-ledger.json'),'Actual terminal ownership ledger changed')
    authorization=terminal.get('human_intent_sha256') if terminal['termination_cause']=='AUTHORIZED_BOUNDARY_HANDOFF' else None
    _writer_barrier(config,terminal,ledger,authorized_handoff_sha=authorization,completed_outer=outer)
    need(terminal.get('uncommitted_after_joint',{}).get('actor_actual_writes_per_rank')==[0]*WORLD and
         terminal['uncommitted_after_joint'].get('probe_actual_writes_per_rank')==[0]*WORLD and
         terminal['uncommitted_after_joint'].get('optimizer_work_in_flight') is False and
         terminal['uncommitted_after_joint'].get('all_updates_accounted') is True,'Do not qualify a discarded/unknown partial update as the earlier joint')
    joint=read(base/'latest-joint.json');pending=pending_sampling_boundary(base,config,joint)
    if output.exists():
        result=read(output)
        need(result['status']==QUALIFIED and result['completed_outer']==outer and result['terminal_proof_sha256']==terminal_sha and
             result['source_joint_sha256']==sha(base/'latest-joint.json'),'Existing qualification belongs to a different actual boundary')
        need(result.get('pending_sampling')==pending,'Accepted pending sampling inventory changed after its once-only qualification')
        for row in result['verified_files']:
            info=canonical_path(row['path']).stat()
            need((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)==(row['dev'],row['inode'],row['bytes'],row['mtime_ns']),
                 'An already-qualified immutable file changed; no blind rescan')
        return dict(result,existing_proof_reused_without_payload_rescan=True)
    intent=output.with_name(output.name+'.intent.json')
    need(not intent.exists(),'Existing incomplete once-only qualification requires reconciliation')
    actor=joint['actor'];state=joint['grounding']
    _once(intent,dict(status='ACTUAL_CPU_QUALIFICATION_STARTED',run_id=config['run_id'],outer=outer,
        source_joint_sha256=sha(base/'latest-joint.json'),terminal_proof_sha256=terminal_sha,started_unix=time.time()))
    allowed=[Path(config['official_model_dir'])/'config.json',terminal_proof,Path(actor['hf_model_dir'])]
    allowed.extend(Path(row['path']) for row in actor['rank_states']);allowed.append(Path(actor['driver_state']['path']))
    allowed.extend(Path(actor['hf_model_dir'])/name for name in actor['hf_files'])
    history={}
    for n in range(1,outer+1):
        item=read(base/'outers'/('outer-%06d'%n)/'complete.json');history[n]=item
        value=item['grounding'];allowed.extend(Path(value[key]) for key in ('manifest_path','commit','boundary','adoption_receipt'))
        allowed.extend((Path(item['grounding_snapshot']),Path(item['pool_snapshot']),Path(item['probe_snapshot'])))
    due=base/'grounding/rounds'/('round-%06d'%outer)/'fit-complete.json'
    if due.exists():
        value=read(due);allowed.extend(Path(value[key]) for key in ('manifest_path','commit','boundary'))
        if 'adoption_receipt' in value:allowed.append(Path(value['adoption_receipt']))
    class OnceSeals(Seals):
        def file(self,path,expected=None):
            path=self.path(path);info=path.stat();identity=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
            if str(path) not in self.files:
                for existing in self.files.values():
                    if identity==(existing['dev'],existing['inode'],existing['bytes'],existing['mtime_ns']):
                        self.files[str(path)]=dict(existing,path=str(path),full_sha_reused_same_inode=True,
                            actual_full_sha_source_path=existing.get('actual_full_sha_source_path',existing['path']))
                        break
            return super().file(path,expected)
    seals=OnceSeals(base,allowed=allowed);seals.file(terminal_proof)
    for name in ('preparation/launch-config.json','logs/process-ledger.json','latest-joint.json','actor/latest.json',
                 'probe/latest.json','probe/initialization-complete.json','question-pool/binding.json','question-pool/state.json'):
        seals.json(base/name)
    previous=[0]*WORLD
    for n,item in history.items():
        _joint(config,item,n,previous_actual=previous);previous=item['actor']['actual_adam_steps_per_rank']
        seals.json(base/'outers'/('outer-%06d'%n)/'complete.json');seals.json(item['grounding_snapshot']);seals.json(item['pool_snapshot'])
        value=item['grounding'];seals.json(value['manifest_path']);seals.json(value['adoption_receipt'])
        for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
            path=Path(value[key]);seals.file(path,dict(bytes=path.stat().st_size,sha256=value['head_sha256'][name]))
    need(joint==history[outer] and read(base/'actor/latest.json')==actor and read(base/'probe/latest.json')==joint['probe'],
         'Actual actor/probe/latest joint are not one complete boundary')
    if config.get('source_compatibility'):verify_source_compatibility(config)
    hf=_hf(seals,actor,config)
    ranks=[]
    for rank in range(WORLD):
        saved,file=seals.mmap_torch(actor['rank_states'][rank]['path'],actor['rank_states'][rank])
        ranks.append(dict(actor_state(saved,actor,config,outer,rank),file=file,ledger=actor_ledger(seals,actor,outer,rank)))
        del saved;gc.collect();print('I1_I6_CPU_QUALIFIED_ACTOR_RANK='+str(rank),flush=True)
    pool=seals.json(joint['pool_snapshot']);size=len(pool['order']);consumed=consumed_candidates_through(config,outer)
    need(pool['next_outer']==outer+1 and pool['consumed_candidates']==consumed and
         pool['cycle']*size+pool['cursor']==consumed and sorted(pool['order'])==list(range(size)) and
         pool['seed']==int(config.get('seed',1729)) and sha(joint['pool_snapshot'])==joint['pool_sha256'], 'Actual pool permutation/cursor/schedule differs')
    driver,file=seals.mmap_torch(actor['driver_state']['path'],actor['driver_state']);captured=driver['candidate_pool_state']
    need(driver['run_id']==config['run_id'] and driver['completed_outer']==outer and driver['actor_version']==actor['actor_version'] and
         driver['grounding_state']==state and driver['grounding_probe_version']==state['version'] and captured['value']==pool,
         'Actual driver RNG/pool/grounding state differs')
    from reproduction.prefix20_online.pool import canonical as pool_canonical
    need(captured['sha256']==hashlib.sha256(pool_canonical(pool)+b'\n').hexdigest(),'Actual driver captured pool bytes differ from producer serialization')
    selected=driver['prefix_probe_receipt'];score=seals.json(base/'outers'/('outer-%06d'%outer)/'probe-score/manifest.json')
    before=probe_boundary(outer-1)
    need(selected['value']['completed_outer']==before['last_update_outer'] and
         selected['value']['actual_adam_steps_per_rank']==[before['actual_adam_per_rank']]*WORLD and
         selected['sha256']==score['probe_before_checkpoint_sha256'],'Actual driver selection head snapshot differs')
    driver_proof=dict(file=file,rng=validate_rng(driver['rng']),candidate_pool_state_exact=True,prefix_probe_receipt=selected)
    del driver;gc.collect()
    saved,file=seals.mmap_torch(base/'probe/latest.pt',joint['probe'])
    probe=dict(probe_state(saved,joint['probe'],config,outer,history),file=file);del saved;gc.collect()
    snapshot=seals.file(joint['probe_snapshot'],joint['probe'])
    need((snapshot['dev'],snapshot['inode'])==(file['dev'],file['inode']),'Actual joint probe snapshot inode differs')
    for name in ('grounding/latest.json',):
        if (base/name).exists():seals.json(base/name)
    if due.exists():
        fitted=seals.json(due);_grounding(config,fitted,outer,adopted=False)
        need(fitted['source_actor_version']==actor['actor_version'],'Completed due fit belongs to another actor; never refit/reuse incorrectly')
        seals.json(fitted['manifest_path'])
        for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
            path=Path(fitted[key]);seals.file(path,dict(bytes=path.stat().st_size,sha256=fitted['head_sha256'][name]))
        if 'adoption_receipt' in fitted:seals.json(fitted['adoption_receipt'])
    seal_pending_sampling(seals,base,pending)
    files=seals.finish();_writer_barrier(config,read(terminal_proof),read(base/'logs/process-ledger.json'),
        authorized_handoff_sha=authorization,completed_outer=outer)
    result=dict(schema_version=1,status=QUALIFIED,run_id=config['run_id'],source_execution_id=config['execution_id'],
        source_commit=config['source_commit'],completed_outer=outer,source_joint_sha256=sha(base/'latest-joint.json'),
        terminal_proof_sha256=terminal_sha,actual_adam_steps_per_rank=actor['actual_adam_steps_per_rank'],
        scheduler_completed_outer=outer,probe_adam_steps_per_rank=joint['probe']['actual_adam_steps_per_rank'],
        consumed_candidates=consumed,actor_rank_optimizer_rng_scheduler_verified=WORLD,actor_driver_rng_verified=True,
        probe_model_optimizer_rank_rng_verified=True,pool_exact_permutation_verified=True,payload_full_sha_read_once=True,
        pending_sampling=pending,phase_hashes={name:value['sha256'] for name,value in pending['phases'].items()} if pending else {},
        accepted_phase_trees_full_sha_read_once=bool(pending),
        actor_hf=hf,actor_rank_states=ranks,actor_driver=driver_proof,probe=probe,verified_files=files,
        actual_cpu_deserialization=True,cuda_restore_verified=False,gpu_forward_performed=False,
        qa_or_new_rollout_performed=False,optimizer_reset=False,automatic_rollback_performed=False,completed_unix=time.time())
    _once(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--outer',type=int,required=True)
    parser.add_argument('--terminal-proof',required=True);parser.add_argument('--terminal-proof-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(qualify(args.base,args.outer,args.terminal_proof,args.terminal_proof_sha256,args.output),allow_nan=False))


if __name__=='__main__':main()
