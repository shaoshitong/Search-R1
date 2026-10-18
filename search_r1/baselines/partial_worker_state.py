"""Matched refusal evidence: raw local worker state, never rollback or resume."""
import copy
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .policy_safety_runtime import PolicySafetyRefusal
from .training_state import atomic_save, cpu_copy, digest, finite, rng_state
from .tree_guard_continue import enabled, RESULT_PROTOCOL, validate_ledger

PROTOCOL = 'tree_partial_state_v1'
IDENTITY_KEYS = {'job_id','run_id','code_commit','config_sha256','data_identity','model_identity',
                 'training_protocol','attempted_outer','completed_outer'}
ENV_KEYS = {'job_id':('RBA_H100_SHARED_EXECUTION_ID' if os.environ.get('RBA_H100_SHARED_EXECUTION_ID') else 'SLURM_JOB_ID'),'run_id':'SEARCH_R1_BASELINE_RUN_ID','code_commit':'SEARCH_R1_BASELINE_COMMIT',
            'config_sha256':'SEARCH_R1_BASELINE_CONFIG_SHA256','data_identity':'SEARCH_R1_BASELINE_DATA_IDENTITY',
            'model_identity':'SEARCH_R1_MODEL_IDENTITY'}


def _gather(value):
    if not dist.is_initialized():return [value]
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value);return values


def _validated_context(worker,data):
    if worker.config.actor.get('fresh200_protocol')!='tree_fresh200_five_leaf_v1':
        raise ValueError('Partial worker persistence requires explicit fresh200 profile')
    context=data.meta_info['tree_partial_context']
    keys = IDENTITY_KEYS | ({'attempt_id'} if enabled(worker.config.actor) else set())
    if context['protocol']!=PROTOCOL or set(context['identity'])!=keys:
        raise ValueError('Partial context protocol/complete driver identity differs')
    identity=context['identity']
    if (identity['training_protocol']!='tree_fresh200_five_leaf_v1'
            or type(identity['attempted_outer']) is not int or type(identity['completed_outer']) is not int
            or identity['attempted_outer']!=identity['completed_outer']+1 or not 0<=identity['completed_outer']<200
            or data.meta_info['baseline_outer_step']!=identity['attempted_outer']):
        raise ValueError('Partial attempted/completed outer identity invalid')
    if enabled(worker.config.actor) and (type(identity['attempt_id']) is not int
            or identity['attempt_id'] < identity['attempted_outer']
            or data.meta_info.get('baseline_attempt_id') != identity['attempt_id']):
        raise ValueError('Partial attempt identity invalid')
    if any(identity[key]!=os.environ[env] for key,env in ENV_KEYS.items()):
        raise ValueError('Partial source/run identity differs from actual worker environment')
    directory=Path(context['directory'])
    own_run=Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']).resolve().parent
    name = (f'partial_attempt_{identity["attempt_id"]:06d}_outer_{identity["attempted_outer"]:06d}'
            if enabled(worker.config.actor) else f'partial_outer_{identity["attempted_outer"]:06d}')
    if (not directory.is_absolute() or not directory.resolve().is_relative_to(own_run)
            or directory.name!=name or directory.parent.name!='training_state'):
        raise ValueError('Partial state directory must remain in this own run and match attempted outer')
    # Metadata only, fixed before native update. No model/PT read occurs here.
    json.dumps(context,sort_keys=True,allow_nan=False)
    return copy.deepcopy(context)


def prepare_partial_worker_context(worker,data):
    local_error=None;context=None
    try:context=_validated_context(worker,data)
    except Exception as exc:local_error=repr(exc)
    records=_gather(dict(context=context,error=local_error))
    consistent=all(record['error'] is None and record['context']==records[0]['context'] for record in records)
    worker._partial_prepared_context=context if consistent else None
    if not consistent:
        raise PolicySafetyRefusal('Global partial worker context preparation refusal; local: '+str(local_error),matched_all_ranks=True)
    return context


def raw_partial_worker_snapshot(worker,identity,cuda):
    """Capture actual shards/gradients even if values are nonfinite; no gather."""
    if cuda:torch.cuda.synchronize(torch.cuda.current_device())
    model=worker.actor.actor_module
    parameters=dict(model.named_parameters())
    value=dict(version='tree_partial_worker_v1',identity=copy.deepcopy(identity),
        parameters={name:cpu_copy(parameter) for name,parameter in parameters.items()},
        local_shards={name:cpu_copy(parameter._local_shard) for name,parameter in parameters.items()
                      if hasattr(parameter,'_local_shard') and isinstance(parameter._local_shard,torch.Tensor)},
        gradients={name:None if parameter.grad is None else cpu_copy(parameter.grad) for name,parameter in parameters.items()},
        private_gradient_tensors={name:{attribute:cpu_copy(getattr(parameter,attribute))
            for attribute in ('_saved_grad_shard','_cpu_grad') if isinstance(getattr(parameter,attribute,None),torch.Tensor)}
            for name,parameter in parameters.items()},
        buffers={name:cpu_copy(value) for name,value in model.named_buffers()},
        optimizer=cpu_copy(worker.actor_optimizer.state_dict()),scheduler=cpu_copy(worker.actor_lr_scheduler.state_dict()),
        rng=rng_state(cuda),module_training_flags=[dict(name=name,training=module.training) for name,module in model.named_modules()],
        parameter_layout={name:dict(shape=list(parameter.shape),dtype=str(parameter.dtype),device=str(parameter.device),
            requires_grad=parameter.requires_grad,local_shard_shape=(list(parameter._local_shard.shape)
                if hasattr(parameter,'_local_shard') and isinstance(parameter._local_shard,torch.Tensor) else None))
            for name,parameter in parameters.items()},
        actual_param_offload_configured=worker._is_offload_param,actual_optimizer_offload_configured=worker._is_offload_optimizer,
        resume_permitted=False,automatic_rollback=False,outer_completed=False)
    state_finite=True;finite_error=None
    try:finite(value)
    except Exception as exc:state_finite=False;finite_error=repr(exc)
    value.update(state_finite=state_finite,state_finite_error=finite_error)
    # Gradients are retained verbatim but may be cleared by the next ordinary
    # zero_grad. Parameter/Adam contamination cannot be cleared this way.
    for key, fields in (('model_adam_finite', ('parameters','local_shards','buffers','optimizer','scheduler')),
                        ('gradients_finite', ('gradients','private_gradient_tensors'))):
        try:
            finite({field:value[field] for field in fields}); value[key] = True
        except Exception:
            value[key] = False
    return value


def persist_matched_partial_worker(worker,data,refusal,scheduler_before):
    """Return the same all-rank envelope after every rank attempts durable save.

    Unmatched errors cannot call any collective here. IO failures are included
    in the common result; no first Ray exception may hide another rank's save.
    """
    if not isinstance(refusal,PolicySafetyRefusal) or refusal.matched_all_ranks is not True:
        raise ValueError('Only matched all-rank PolicySafetyRefusal may save distributed partial state')
    started=time.monotonic();rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    context=None;context_error=None
    try:context=_validated_context(worker,data)
    except Exception as exc:context_error=repr(exc)
    contexts=_gather(dict(context=context,error=context_error))
    consistent=all(entry['error'] is None and entry['context']==contexts[0]['context'] for entry in contexts)
    canonical=next((entry['context'] for entry in contexts if entry['context'] is not None),None)
    driver_identity=canonical['identity'] if canonical else {}
    identity=dict(driver_identity,rank=rank,world_size=world)
    count=getattr(worker.actor,'_actual_committed_optimizer_calls',None)
    actual_after=worker.actor_lr_scheduler.last_epoch
    receipt=dict(rank=rank,world_size=world,identity=identity,
        attempted_outer=driver_identity.get('attempted_outer'),completed_outer=driver_identity.get('completed_outer'),
        committed_optimizer_calls=count if type(count) is int else None,
        scheduler_step_before_actual=scheduler_before,scheduler_step_after_actual=actual_after,
        status='PARTIAL_WORKER_STATE_INCOMPLETE',path=None,sha256=None,bytes=0,logical_state_sha256=None,
        state_finite=None,model_adam_finite=None,gradients_finite=None,error=None,
        refusal=repr(refusal),matched_all_ranks=True,safety_guard=refusal.safety_guard,
        optimizer_count_source='optimizer.step successful return before any audit/publish IO',
        resume_permitted=False,exact_same_world_reload_verified=False,automatic_rollback=False,outer_completed=False)
    try:
        if not consistent:raise ValueError('All-rank partial context invalid/mismatched: '+str(context_error))
        if type(count) is not int or not 0<=count<=20:raise ValueError('Actual partial optimizer return counter missing or invalid')
        directory=Path(canonical['directory'])
        state_path=directory/f'worker-{rank:02d}.pt';receipt_path=directory/f'worker-{rank:02d}.json'
        cuda=dist.is_initialized() and dist.get_backend()=='nccl'
        saved=raw_partial_worker_snapshot(worker,identity,cuda)
        saved.update(committed_optimizer_calls=count,scheduler_step_before_actual=scheduler_before,
                     scheduler_step_after_actual=actual_after,refusal=repr(refusal),
                     optimizer_opportunity_ledger=copy.deepcopy(getattr(worker.actor,'_fresh_opportunity_ledger',[])))
        receipt['logical_state_sha256']=digest(saved)
        receipt['state_finite']=saved['state_finite'];receipt['state_finite_error']=saved['state_finite_error']
        receipt['model_adam_finite']=saved['model_adam_finite'];receipt['gradients_finite']=saved['gradients_finite']
        receipt['total_committed_optimizer_calls']=getattr(worker.actor,'_fresh_total_committed_optimizer_calls',None)
        receipt['optimizer_opportunity_ledger']=saved['optimizer_opportunity_ledger']
        receipt['path']=str(state_path);receipt['sha256']=atomic_save(saved,state_path)
        receipt['bytes']=state_path.stat().st_size
        del saved
        receipt.update(status='PARTIAL_WORKER_STATE_SAVED',optimizer_state_entries=len(worker.actor_optimizer.state),
            scheduler_advanced_actual=(actual_after!=scheduler_before),snapshot_save_seconds=time.monotonic()-started,
            receipt_path=str(receipt_path))
        receipt['receipt_sha256']=atomic_save(receipt,receipt_path,json_format=True)
    except Exception as exc:
        receipt.update(status='PARTIAL_WORKER_STATE_INCOMPLETE',error=repr(exc),snapshot_save_seconds=time.monotonic()-started)
    workers=sorted(_gather(receipt),key=lambda record:record['rank'])
    all_saved=(consistent and len(workers)==world and [entry['rank'] for entry in workers]==list(range(world))
               and all(entry['status']=='PARTIAL_WORKER_STATE_SAVED' for entry in workers))
    ledger_complete = False
    if enabled(worker.config.actor) and all_saved:
        try:
            for entry in workers:
                validate_ledger(entry['optimizer_opportunity_ledger'], entry['committed_optimizer_calls'])
            ledger_complete = len({entry['committed_optimizer_calls'] for entry in workers}) == 1
        except ValueError:
            pass
    return dict(protocol=PROTOCOL,identity=driver_identity,identity_consistent=consistent,
        status='PARTIAL_WORKER_STATE_SAVED' if all_saved else 'PARTIAL_WORKER_STATE_INCOMPLETE',
        worker_states=workers,scheduler_advanced_actual=[
            entry['scheduler_step_after_actual']!=entry['scheduler_step_before_actual'] for entry in workers],
        resume_permitted=False,automatic_rollback=False,outer_completed=False,
        continuation_decision=('CONTINUE_FRESH_ROLLOUT' if enabled(worker.config.actor) and all_saved
            and ledger_complete
            and all(entry['safety_guard'] is True and entry['model_adam_finite'] is True
                    and entry['scheduler_step_before_actual']==driver_identity['completed_outer']
                    and entry['scheduler_step_after_actual']==driver_identity['completed_outer']
                        + int(entry['committed_optimizer_calls'] > 0) for entry in workers)
            else 'FAULT_STOP'),
        scope='Raw actual local worker state on matched refusal; no rollback, scheduler/outer credit, HF export or reload acceptance')


def completed_actor_result(worker, data, scheduler_before):
    """Gather before DP_COMPUTE_PROTO's rank-zero-meta concat; no exception inference."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    record = dict(rank=rank, attempt_id=data.meta_info['baseline_attempt_id'],
        accepted_outer=data.meta_info['baseline_outer_step'],
        completed_outer_before=data.meta_info['baseline_completed_outer'],
        optimizer_calls=worker.actor._actual_committed_optimizer_calls,
        total_optimizer_calls=worker.actor._fresh_total_committed_optimizer_calls,
        scheduler_before=scheduler_before,scheduler_after=worker.actor_lr_scheduler.last_epoch,
        direct_rank_record=copy.deepcopy(worker._fresh_direct_record),
        opportunities_processed=len(worker.actor._fresh_opportunity_ledger),
        opportunity_ledger=copy.deepcopy(worker.actor._fresh_opportunity_ledger),
        opportunities_skipped=sum(e['actual_writes']==0 for e in worker.actor._fresh_opportunity_ledger))
    ranks = sorted(_gather(record),key=lambda r:r['rank'])
    return dict(protocol=RESULT_PROTOCOL, decision='COMPLETED_OUTER',
        identity=copy.deepcopy(worker._partial_prepared_context['identity']),
        attempt_id=record['attempt_id'],accepted_outer=record['accepted_outer'],
        completed_outer_before=record['completed_outer_before'],ranks=ranks)


def reset_saved_guard_gradients(worker, envelope):
    """Clear only gradients after durable raw evidence; never reload model/Adam."""
    if envelope['continuation_decision'] != 'CONTINUE_FRESH_ROLLOUT':
        return envelope
    error = None
    try:
        worker.actor_optimizer.zero_grad()
        if any(parameter.grad is not None for parameter in worker.actor.actor_module.parameters()):
            raise ValueError('Normal zero_grad did not clear the saved refused gradients')
    except Exception as exc:
        error = repr(exc)
    rank = dist.get_rank() if dist.is_initialized() else 0
    gradients = [parameter.grad for parameter in worker.actor.actor_module.parameters() if parameter.grad is not None]
    gradients_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    private_retained = sorted({attribute for parameter in worker.actor.actor_module.parameters()
        for attribute in ('_saved_grad_shard', '_cpu_grad') if getattr(parameter,attribute,None) is not None})
    private_tensors = [getattr(parameter,attribute) for parameter in worker.actor.actor_module.parameters()
        for attribute in ('_saved_grad_shard','_cpu_grad') if isinstance(getattr(parameter,attribute,None),torch.Tensor)]
    private_finite = all(bool(torch.isfinite(value).all()) for value in private_tensors)
    if not private_finite:
        error = error or 'Nonfinite FSDP private gradient tensor remains after normal zero_grad; no manual diagnostic reset'
    resets = sorted(_gather(dict(rank=rank,error=error,normal_zero_grad=(error is None),
        public_gradients_after_reset_finite=gradients_finite,
        private_gradients_after_reset_finite=private_finite,
        private_gradient_attributes_still_present=private_retained,
        private_fsdp_gradient_reset_verified=False)),key=lambda r:r['rank'])
    result = copy.deepcopy(envelope)
    result['gradient_reset_after_raw_persistence'] = dict(ranks=resets,automatic_rollback=False,
        parameters_reloaded=False,adam_reloaded=False,raw_saved_gradients_unchanged=True)
    if any(record['error'] is not None for record in resets):
        result['continuation_decision'] = 'FAULT_STOP'
    return result
