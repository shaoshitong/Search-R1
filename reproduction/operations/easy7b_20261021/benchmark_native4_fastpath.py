"""Independent native4 CP50 actor/Adam benchmark: legacy versus lean+KV4.

Run each mode in a separate torchrun4 Slurm job. The production job is never
changed. Both modes restore the same CP50 HF weights, rank-local Adam moments,
fixed57 scheduler and global Python/NumPy/Torch RNG, then use the same original
selected512x5 contexts, masks and sampling LPs for twenty real Adam writes.
Advantages are fixed five-EM test advantages, NOT production grounding rewards.
No rollout/reference/retriever/teacher model is initialized, and this does not
claim a complete training-engine/data-stream/private-rollout-RNG restoration.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
import types

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE))


def emit(root,phase,**fields):
    row=dict(phase=phase,epoch=time.time(),rank=int(os.environ.get('RANK','0')),**fields)
    print('NATIVE4_FASTPATH='+json.dumps(row,sort_keys=True,allow_nan=False),flush=True)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/('progress-rank-%s.jsonl'%row['rank'])).open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def sample_indices(size,count=64):
    import torch
    count=min(count,size)
    return torch.zeros(count,dtype=torch.long) if count<=1 else torch.arange(count,dtype=torch.long)*(size-1)//(count-1)


def parameter_samples(module):
    """Small deterministic real FP32-shard observations, no full-weight scan."""
    rows=[]
    for name,parameter in module.named_parameters():
        flat=parameter.detach().reshape(-1)
        sample=flat[sample_indices(flat.numel()).to(flat.device)].float().cpu()
        rows.append(dict(name=name,shape=list(parameter.shape),dtype=str(parameter.dtype),
                         elements=parameter.numel(),samples=sample.tolist(),sample_sha256=tensor_sha(sample)))
    return rows


def actual_selection(source,checkpoint,selection_step):
    """Only small complete-step/audit metadata; raw tensors stay on shared disk."""
    complete=read(checkpoint/'complete.json');step=int(complete['step'])
    if step!=50:raise ValueError('Benchmark requires the sealed CP50 boundary')
    updates=[]
    with (source/'logs/actual-updates.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):break
            row=json.loads(line)
            if row['step']==selection_step:updates.append(row)
    update=updates[0] if len(updates)==1 else None
    if update:
        audit_path=source/'artifacts/mixed-selection'/(update['attempt_id']+'.json');audit=read(audit_path)
    else:
        # Selection is durably saved before its actor update. It need not wait
        # for that production update to finish before independent benchmarking.
        audit_path=audit=None
        for path in sorted((source/'artifacts/mixed-selection').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
            candidate=read(path)
            if candidate.get('step')==selection_step:audit_path,audit=path,candidate;break
        if audit is None:raise ValueError('Requested original saved selection is not ready: '+str(selection_step))
    groups=[g for g in audit['groups'] if g['selected']]
    if len(groups)!=512 or audit['step']!=selection_step:raise ValueError('Require actual saved selected512 groups')
    pool=Path(audit['pool_path']).resolve()
    if source not in pool.parents:raise ValueError('Candidate pool must belong to the explicit source')
    return groups,pool,dict(source_checkpoint_step=50,selection_step=selection_step,source_attempt_id=audit_path.stem,
        sampling_policy_completed_step=selection_step-1,
        checkpoint_minus_sampling_policy_updates=50-(selection_step-1),
        selection_audit=str(audit_path),selection_audit_sha256=sha(audit_path),
        actual_source_reward=update['metrics'].get('critic/rewards/mean') if update else None,
        selected_questions=512,selected_trajectories=2560,
        selected_ordinals=[int(g['ordinal']) for g in groups],raw_pool=str(pool))


def local_batch(groups,pool,rank,pad):
    import torch
    from tensordict import TensorDict
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
    requested={int(g['ordinal']):g for g in groups};parts=[];ordinals=[]
    for path in sorted((p for p in pool.glob('*.pt') if p.stem.isdecimal()),key=lambda p:int(p.stem)):
        raw=torch.load(path,map_location='cpu',weights_only=False)
        take=[(i,int(ordinal)) for i,ordinal in enumerate(raw['ordinals']) if int(ordinal) in requested]
        if take:
            ids=torch.tensor([i*5+j for i,_ in take for j in range(5)],dtype=torch.long)
            original=raw['output'].batch;rw=original['responses'].shape[1]
            piece={key:original[key].index_select(0,ids).clone() for key in
                   ('input_ids','attention_mask','position_ids','responses')}
            piece['loss_mask']=original['info_mask'].index_select(0,ids)[:,-rw:].clone()
            piece['old_log_probs']=original['sampling_log_probs'].index_select(0,ids).clone()
            labels=[]
            for i,ordinal in take:
                expected=list(requested[ordinal]['strict_correctness'])
                if list(raw['strict_correctness'][i])!=expected:raise ValueError('Raw/audit EM labels differ')
                labels.append(expected);ordinals.append(ordinal)
            scores=torch.tensor(labels,dtype=torch.float32)
            standardized=(scores-scores.mean(1,keepdim=True))/(scores.std(1,correction=1,keepdim=True)+1e-6)
            piece['advantages']=standardized.flatten()[:,None].expand(-1,rw).clone()*piece['loss_mask']
            parts.append(piece)
        del raw
    if ordinals!=sorted(requested) or sum(p['responses'].shape[0] for p in parts)!=2560:
        raise ValueError('Original selected trajectory order/membership differs')
    maximum=max(p['responses'].shape[1] for p in parts)
    for part in parts:
        extra=maximum-part['responses'].shape[1]
        if extra:
            for key in part:
                part[key]=torch.nn.functional.pad(part[key],(0,extra),value=pad if key in ('input_ids','responses') else 0)
    global_batch={key:torch.cat([p[key] for p in parts]) for key in parts[0]}
    lengths=global_batch['attention_mask'].sum(-1).tolist()
    partitions=get_seqlen_balanced_partitions(lengths,k_partitions=4,equal_size=True)
    ids=torch.tensor(partitions[rank],dtype=torch.long)
    local={key:value.index_select(0,ids).contiguous() for key,value in global_batch.items()}
    identity={key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value)) for key,value in local.items()}
    return TensorDict(local,batch_size=[640]),dict(local_rows=partitions[rank],tensor_identity=identity,
        local_authored_tokens=int(local['loss_mask'].sum()),local_context_tokens=int(local['attention_mask'].sum()),
        physical_micro=8,logical_micro=4,local_mini=32,gradient_accumulation=4,
        advantages='fixed diagnostic five-EM sample-standardized original labels, NOT production grounding rewards',
        original_sampling_lp_and_info_masks=True,partition='same deterministic original Karmarkar-Karp four-rank equal-size partition')


def adam_samples(optimizer):
    """Small real moment readbacks, with exact Adam counters and layouts."""
    import torch
    rows=[]
    for index,(parameter,state) in enumerate(optimizer.state.items()):
        row=dict(index=index,parameter_shape=list(parameter.shape),states={})
        for name,value in state.items():
            if torch.is_tensor(value):
                flat=value.detach().reshape(-1)
                if name=='step':row['step']=float(flat[0])
                sample=flat[sample_indices(flat.numel()).to(flat.device)].cpu() if flat.numel() else flat.cpu()
                row['states'][name]=dict(shape=list(value.shape),dtype=str(value.dtype),sample_sha256=tensor_sha(sample))
            else:row['states'][name]=value
        rows.append(row)
    return rows


def prepare_frozen(actor,batch,temperature,source):
    import torch
    mask=batch['loss_mask'];ordered=torch.argsort(mask.sum(-1),stable=True)
    indices=ordered[torch.linspace(0,len(batch)-1,8,device=ordered.device).long()]
    frozen=batch[indices].clone()
    actor._monitor_identity=dict(selection_indices=indices.cpu().tolist(),
        actor_token_lengths=mask[indices].sum(-1).cpu().tolist(),
        tensor_sha256={name:tensor_sha(value) for name,value in frozen.items()})
    actor._monitor_temperature=temperature
    calibration=source/'logs/policy-safety-calibration.json'
    actor._safety_calibration_sha=sha(calibration) if calibration.exists() else '0'*64
    actor._policy_execution_role='production'
    return frozen


def restore_actor_state(worker,checkpoint):
    import torch
    import numpy as np
    from reproduction.common8.scheduler import fixed_outer_warmup_scheduler
    path=checkpoint/'training_state'/('rank-%d.pt'%worker.rank)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['rank']!=worker.rank or saved['world_size']!=4 or saved['completed_outer']!=50:
        raise ValueError('Require actual native4 rank-local CP50 Adam/RNG state')
    # The generic worker constructor is used without production admission. Give
    # it the original warmup function before loading its actual saved fields.
    for group in worker.actor_optimizer.param_groups:
        group.pop('initial_lr',None);group['lr']=1e-6
    worker.actor_lr_scheduler=fixed_outer_warmup_scheduler(worker.actor_optimizer,warmup_steps=57)
    worker.actor_optimizer.load_state_dict(saved['optimizer'])
    worker.actor_lr_scheduler.load_state_dict(saved['scheduler'])
    rng=saved['rng'];torch.set_rng_state(rng['torch_cpu'])
    torch.cuda.set_rng_state(rng['torch_cuda'],device=torch.cuda.current_device())
    np.random.set_state(rng['numpy']);random.setstate(rng['python'])
    actor=worker.actor
    actor._optimizer_writes=saved['actual_optimizer_writes'];actor._optimizer_calls=saved.get('optimizer_calls',actor._optimizer_writes)
    actor._optimizer_rollbacks=saved.get('optimizer_rollbacks',0);actor._complete_outers=50
    actor._full_budget_outers=saved.get('full_budget_outers',50)
    actor._actual_monitor_forward_count=actor._reused_monitor_forward_count=0
    actor._recoverable_guard_mode=True;actor._invalidate_monitor_forward_cache()
    actual_samples=adam_samples(worker.actor_optimizer)
    # Compare the same bounded, deterministic samples of every saved state.
    expected=[]
    for index,((_,state),parameter) in enumerate(zip(saved['optimizer']['state'].items(),worker.actor_optimizer.state)):
        row=dict(index=index,parameter_shape=list(parameter.shape),states={})
        for name,value in state.items():
            if torch.is_tensor(value):
                flat=value.reshape(-1)
                if name=='step':row['step']=float(flat[0])
                sample=flat[sample_indices(flat.numel())] if flat.numel() else flat
                row['states'][name]=dict(shape=list(value.shape),dtype=str(value.dtype),sample_sha256=tensor_sha(sample))
            else:row['states'][name]=value
        expected.append(row)
    if actual_samples!=expected:raise ValueError('Actual restored Adam state samples/layout/counters differ')
    from search_g1.full_state_restore import equal_tree
    equal_tree(worker.actor_lr_scheduler.state_dict(),saved['scheduler'])
    if not torch.equal(torch.get_rng_state(),rng['torch_cpu']) or not torch.equal(torch.cuda.get_rng_state(),rng['torch_cuda']):
        raise ValueError('Saved global Torch RNG readback differs')
    result=dict(path=str(path),rank=worker.rank,actual_optimizer_writes=actor._optimizer_writes,
        actual_scheduler_last_epoch=worker.actor_lr_scheduler.last_epoch,
        actual_adam_state_entries=len(worker.actor_optimizer.state),actual_adam_samples=actual_samples,
        adam_readback_scope='all state layouts/step counters and64 deterministic elements per state tensor; not full rehash',
        scheduler_exact_readback=True,global_torch_cpu_rng_sha256=tensor_sha(rng['torch_cpu']),
        global_torch_cuda_rng_sha256=tensor_sha(rng['torch_cuda']),global_python_numpy_rng_restored=True,
        rollout_private_rng_restored=False,driver_data_stream_restored=False,
        saved_actor_private_cuda_rng_sha256=tensor_sha(rng['actor_cuda_saved']),
        saved_rollout_private_cuda_rng_sha256=tensor_sha(rng['rollout_cuda_saved']) if rng['rollout_cuda_saved'] is not None else None)
    del saved;gc.collect()
    return result


def run(args):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf,open_dict
    from search_g1.easy7b_infra import profile_flags
    source=Path(args.source_base).resolve(strict=True);checkpoint=Path(args.checkpoint).resolve(strict=True)
    output=Path(args.output).absolute();output.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or not 1<=args.steps<=20:raise ValueError('Need native torchrun4 and1..20 real writes')
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():dist.init_process_group(backend='nccl')
    groups,pool,binding=actual_selection(source,checkpoint,args.selection_step)
    parent=read(source/'logs/run-manifest.json');manifest=copy.deepcopy(parent)
    operator_commit=SOURCE.name
    if re.fullmatch('[0-9a-f]{40}',operator_commit) is None:
        import subprocess
        operator_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    manifest.update(source_training_commit=parent['source_commit'],
        source_training_job=parent.get('slurm_job_id') or (source/'job.id').read_text().strip(),
        source_training_sha256=parent.get('source_sha256'),source_commit=operator_commit,
        source_checkpoint=str(checkpoint),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        source_sha256={name:sha(SOURCE/name) for name in (
            'reproduction/operations/easy7b_20261021/benchmark_native4_fastpath.py',
            'search_g1/flash3_attention.py','search_g1/actor_performance.py','verl/workers/actor/dp_actor.py')},
        source_sha256_scope='four direct benchmark/actor entry files; published operator commit binds remaining source')
    manifest['diagnostic_role']='native4_fixed_work_actor_benchmark_NOT_production'
    manifest['benchmark_mode']=args.mode
    if args.mode=='leancompact':
        manifest['performance_profile']='lean_v1';manifest['extra_policy_monitor_forwards']=False
        manifest['infra_optimizations']=profile_flags('fa3-lean-v1')
    else:
        manifest.pop('performance_profile',None);manifest['extra_policy_monitor_forwards']=True
        manifest['infra_optimizations']=profile_flags('fa3-exact-cache-v1')
    if rank==0:write(output/'logs/run-manifest.json',manifest)
    dist.barrier()
    # Use the actual resolved production global configuration, not a fresh
    # Hydra training/admission pipeline or hardcoded model/optimizer recipe.
    document=read(source/'logs/resolved-training-config.json')
    config=OmegaConf.create(document.get('resolved_config',document))
    with open_dict(config.actor_rollout_ref.model):config.actor_rollout_ref.model.actor_checkpoint_path=str(checkpoint)
    with open_dict(config.actor_rollout_ref.actor):
        config.actor_rollout_ref.actor.performance_profile='lean_v1' if args.mode=='leancompact' else 'legacy'
        config.actor_rollout_ref.actor.policy_safety.extra_policy_monitor_forwards=args.mode=='legacy'
        config.actor_rollout_ref.actor.optim.completed_updates=0
        config.actor_rollout_ref.actor.optim.total_training_steps=201
    # Only constructor admission is bypassed. Actual actor math below retains
    # the production legacy finite scans/monitor forwards/CPU snapshots, or
    # explicitly chooses the lean_v1 path; both use identical real losses.
    os.environ.update(SEARCH_G1_BASE=str(output),SEARCH_G1_TRAINING_WORLD_SIZE='4',
                      SEARCH_G1_FA3_COMPACT_GQA='1' if args.mode=='leancompact' else '0')
    os.environ.pop('SEARCH_G1_PROFILE',None)
    for key in ('WG_PREFIX','WG_BACKEND','DISABLE_WORKER_INIT'):os.environ.pop(key,None)
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
    emit(output,'ACTOR_ONLY_INITIALIZING',mode=args.mode,checkpoint=str(checkpoint),gpu=torch.cuda.get_device_name())
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    worker=ActorRolloutRefWorker(config.actor_rollout_ref,role='actor');worker.init_model()
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad,load_fsdp_optimizer
    load_fsdp_param_and_grad(worker.actor_module_fsdp,device_id=torch.cuda.current_device(),load_grad=False)
    restore=restore_actor_state(worker,checkpoint)
    load_fsdp_optimizer(worker.actor_optimizer,device_id=torch.cuda.current_device())
    actor=worker.actor;actor.actor_module.train(True)
    if actor.config.ppo_micro_batch_size!=8 or actor.config.ppo_mini_batch_size!=32:raise ValueError('Actual physical8/mini32 rank geometry differs')
    actor.gradient_accumulation=4
    batch,data=local_batch(groups,pool,rank,worker.tokenizer.pad_token_id)
    temperature=float(config.actor_rollout_ref.rollout.temperature)
    frozen=prepare_frozen(actor,batch,temperature,source) if args.mode=='legacy' else None
    safety=actor.config.policy_safety
    os.environ['SEARCH_G1_PROFILE']='common8'
    initial=parameter_samples(actor.actor_module)
    # Capture small, comparable training-only LP observations from real forwards.
    original_forward=actor._forward_micro_batch;forward_records=[]
    def measured_forward(self,*pos,**kw):
        entropy,lp=original_forward(*pos,**kw)
        if torch.is_grad_enabled():
            micro=kw.get('micro_batch',pos[0] if pos else None);mask=micro['loss_mask'].bool()
            values=lp.detach()[mask];sample=values[torch.linspace(0,values.numel()-1,min(32,values.numel()),device=values.device).long()]
            forward_records.append(dict(authored_tokens=values.numel(),mean=float(values.double().mean()),
                sum_squares=float(values.double().square().sum()),min=float(values.min()),max=float(values.max()),
                samples=sample.float().cpu().tolist()))
        return entropy,lp
    actor._forward_micro_batch=types.MethodType(measured_forward,actor)
    torch.cuda.synchronize();dist.barrier();torch.cuda.reset_peak_memory_stats()
    before=actor._optimizer_writes;calls_before=actor._optimizer_calls;timings=[];started=time.monotonic()
    emit(output,'REAL_ADAM_BENCHMARK_BEGIN',mode=args.mode,source_writes=before,steps=args.steps,
         data_identity_sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),restore=restore)
    for index,mini in enumerate(batch.split(32)):
        if index>=args.steps:break
        actor._gradient_diagnostic_context=dict(logical_update=51,minibatch_index=index,
            prior_optimizer_writes=actor._optimizer_writes,attempt_id=binding['source_attempt_id'],
            attempt_index=51,behavior_probability_gate_sha256='0'*64,diagnostic_only=True)
        actor._pending_optimizer_transaction=None;actor._pending_write_receipt=None
        metrics={};forward_before=len(forward_records);torch.cuda.synchronize();began=time.monotonic()
        actor._run_minibatch(mini,temperature=temperature,safety_enabled=True,safety=safety,
                             frozen=frozen,tail_limit=float(safety.negative_tail_limit),metrics=metrics)
        actor._commit_optimizer_write(worker.actor_module_fsdp.process_group)
        torch.cuda.synchronize()
        row=dict(index=index,seconds=time.monotonic()-began,optimizer_writes=actor._optimizer_writes,
            optimizer_calls=actor._optimizer_calls,metrics=metrics,
            train_forwards=forward_records[forward_before:])
        timings.append(row);emit(output,'REAL_ADAM_STEP_COMPLETE',**row)
    torch.cuda.synchronize();dist.barrier();elapsed=time.monotonic()-started
    if actor._optimizer_writes-before!=args.steps or actor._optimizer_calls-calls_before!=args.steps:
        raise ValueError('Real restored Adam write/call delta differs')
    scheduler_before=copy.deepcopy(worker.actor_lr_scheduler.state_dict())
    if args.steps==20:worker.actor_lr_scheduler.step()
    scheduler_after=copy.deepcopy(worker.actor_lr_scheduler.state_dict())
    final=parameter_samples(actor.actor_module)
    from search_g1.flash3_attention import execution_report
    report=dict(mode=args.mode,rank=rank,hostname=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        source_binding=binding,checkpoint=str(checkpoint),input_data=data,restored_actor_state=restore,
        actual_optimizer_writes=args.steps,actual_optimizer_calls=args.steps,elapsed_seconds=elapsed,
        actual_training_forwards=len(forward_records),actual_monitor_forwards=actor._actual_monitor_forward_count,
        reused_monitor_forwards=actor._reused_monitor_forward_count,
        initial_parameter_samples=initial,final_parameter_samples=final,final_adam_samples=adam_samples(worker.actor_optimizer),
        scheduler_before=scheduler_before,scheduler_after=scheduler_after,scheduler_advances=int(args.steps==20),
        flash3=execution_report(worker.actor_module,synchronize=True),timings=timings,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),production_modified=False,
        production_grounding_reward_equivalence=False,full_training_engine_restore=False,
        final_parameter_tensors_saved=False,scope='actual native4 actor+restored Adam fixed-work timing/numerical diagnostics; '
            'same original selected512x5 saved data and fixed EM test advantages; no fresh rollout/QA/full outer claim')
    write(output/('rank-%d.json'%rank),report)
    gathered=[None]*4;dist.all_gather_object(gathered,dict(rank=rank,elapsed_seconds=elapsed,
        actual_optimizer_writes=args.steps,
        actual_training_forwards=len(forward_records),actual_monitor_forwards=actor._actual_monitor_forward_count,
        rank_report=str(output/('rank-%d.json'%rank))))
    if rank==0:
        write(output/'result.json',dict(status='NATIVE4_REAL_ADAM_BENCHMARK_COMPLETED',mode=args.mode,
            actual_world_size=4,ranks=gathered,checkpoint=str(checkpoint),steps=args.steps,
            source_binding=binding,production_modified=False,full_outer_speedup_proven=False,
            full_training_restore_proven=False,grounding_reward_equivalence=False))
        emit(output,'NATIVE4_BENCHMARK_COMPLETE',mode=args.mode,slowest_rank_seconds=max(r['elapsed_seconds'] for r in gathered))
    dist.barrier();dist.destroy_process_group()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-base',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--mode',choices=('legacy','leancompact'),required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--selection-step',type=int,default=51,
                        help='Saved policy50 selection51 by default;50 is an explicit one-update-stale fallback')
    args=parser.parse_args(argv)
    try:run(args)
    except Exception as error:
        emit(args.output,'NATIVE4_BENCHMARK_FAILED',mode=args.mode,error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
