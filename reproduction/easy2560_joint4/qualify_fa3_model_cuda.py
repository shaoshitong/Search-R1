"""Isolated native4 FA2/FA3 full-actor qualification on a bound CP>=9.

Only HF actor weights are loaded. Adam is empty and update APIs are forbidden;
saved optimizer/scheduler/RNG/data/grounding states are NOT restored. Production
must use a fresh process and its ordinary complete restoration afterwards.
Original source-pool contexts, masks and sampling LPs are reused without a new
rollout. Five-EM standardized advantages are diagnostic inputs, not production
grounding rewards. No automatic fallback or training continuation occurs here.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

sys.dont_write_bytecode = True

from reproduction.easy2560_joint4.qualify_sparse_cuda import prepare_micro_batch


def _need(condition, message):
    if not condition: raise ValueError('FA3_MODEL_DIAGNOSTIC_' + message)


def _within(path, parent):
    try: Path(path).relative_to(parent); return True
    except ValueError: return False


def _document(path):
    path = Path(path)
    _need(path.is_file() and path.stat().st_size <= 32*1024**2, 'SMALL_BINDING_DOCUMENT_REQUIRED')
    return json.loads(path.read_bytes())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def source_binding(base, raw_pool_file=None):
    """Route the successful source-step pool through its real update + audit.

    This reads only small metadata. Actual raw tensors remain on the source
    filesystem and are loaded on CPU by the independent diagnostic ranks.
    """
    base = Path(base).resolve(strict=True)
    run = _document(base/'logs/run-manifest.json'); binding = run.get('full_state_restore', {})
    step = binding.get('completed_outer')
    _need(binding.get('kind') == 'easy2560_native4' and type(step) is int and 9 <= step <= 200 and
          run.get('training_world_size') == 4, 'BOUND_NATIVE4_CP_AT_LEAST_NINE_REQUIRED')
    source = Path(binding['source_base']).resolve(strict=True)
    checkpoint = Path(binding['source_checkpoint']).resolve(strict=True)
    _need(source != base and _within(checkpoint, source), 'FOREIGN_CHECKPOINT_OR_SAME_BASE')
    receipt_path = Path(binding['source_receipt']).resolve(strict=True)
    _need(_sha(receipt_path) == binding['source_receipt_sha256'], 'BOUND_SOURCE_RECEIPT_SHA_DIFFERS')
    receipt = _document(receipt_path)
    _need(receipt.get('kind') == 'easy2560_native4' and
          receipt.get('status') == 'EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS' and
          receipt.get('step') == step and receipt.get('world_size') == 4 and
          receipt.get('run_id') == run['run_id'] and receipt.get('source_base') == str(source) and
          receipt.get('checkpoint') == str(checkpoint), 'BOUND_SOURCE_RECEIPT_IDENTITY_DIFFERS')
    source_run = _document(source/'logs/run-manifest.json')
    _need(source_run['run_id'] == run['run_id'] and source_run['source_commit'] == receipt['source_commit'],
          'SOURCE_RUN_IDENTITY_DIFFERS')
    complete = _document(checkpoint/'complete.json')
    hf_config = _document(checkpoint/'config.json')
    index = _document(checkpoint/'model.safetensors.index.json')
    tensor_count = len(index['weight_map'])
    _need(complete.get('step') == step and complete.get('run_id') == run['run_id'] and
          complete.get('finite_all_saved_tensors') is True and
          complete.get('validated_tensor_count') == receipt.get('hf_saved_tensor_count') == tensor_count == 339 and
          hf_config.get('model_type') == 'qwen2' and hf_config.get('num_hidden_layers') == 28 and
          hf_config.get('hidden_size') == 3584 and hf_config.get('num_attention_heads') == 28 and
          hf_config.get('num_key_value_heads') == 4, 'ACTUAL_COMPLETE_QWEN7B_CHECKPOINT_REQUIRED')
    updates = []
    # Ignore only a trailing in-progress JSONL record, never a completed row.
    with (source/'logs/actual-updates.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'): break
            row = json.loads(line)
            if row.get('step') == step and row.get('run_id') == run['run_id']: updates.append(row)
    _need(len(updates) == 1, 'UNIQUE_COMPLETED_SOURCE_STEP_REQUIRED')
    update = updates[0]; attempt = update.get('attempt_id', '')
    _need(re.fullmatch('[0-9a-f]{32}', attempt) is not None, 'ACTUAL_SOURCE_ATTEMPT_REQUIRED')
    audit_path = source/'artifacts/mixed-selection'/(attempt+'.json'); audit = _document(audit_path)
    snapshot = run['run_id']+':'+str(step)+':policy-'+attempt
    retry = audit.get('pool_retry')
    _need(type(retry) is int and retry >= 0 and audit.get('step') == step and
          audit.get('source_commit') == source_run['source_commit'] and
          audit.get('policy_snapshot_id') == snapshot and audit.get('fixed_candidate_budget') is True and
          audit.get('attempted_candidates') == 2560 and audit.get('selected_groups') == 512 and
          audit.get('responses_per_group') == 5, 'SUCCESSFUL_SOURCE_SELECTION_BINDING_DIFFERS')
    pool = source/'artifacts/mixed-candidate-pools'/(attempt+'-pool-'+str(retry))
    _need(pool.resolve(strict=True) == pool and audit.get('pool_path') == str(pool) and
          not (pool/'selection-failure.json').exists(), 'FAILED_OR_FOREIGN_SOURCE_POOL')
    if raw_pool_file is None:
        members = sorted((p for p in pool.glob('*.pt') if p.stem.isdigit()), key=lambda p:int(p.stem))
        _need(bool(members), 'ACTUAL_SOURCE_CANDIDATE_TENSORS_MISSING')
        raw_pool_file = members[0]
    raw_pool_file = Path(raw_pool_file).resolve(strict=True)
    _need(raw_pool_file.parent == pool and raw_pool_file.stem.isdigit() and raw_pool_file.suffix == '.pt' and
          raw_pool_file.is_file(), 'ORIGINAL_SOURCE_POOL_MEMBER_REQUIRED')
    proof = dict(source_base=str(source), source_step=step, source_checkpoint=str(checkpoint),
        source_receipt=str(receipt_path), source_receipt_sha256=binding['source_receipt_sha256'],
        source_commit=source_run['source_commit'], source_attempt_id=attempt,
        source_attempt_index=update.get('attempt_index'), source_policy_snapshot_id=snapshot,
        selection_audit=str(audit_path), selection_audit_sha256=_sha(audit_path),
        raw_pool_file=str(raw_pool_file), raw_pool_bytes=raw_pool_file.stat().st_size,
        model_layers=hf_config['num_hidden_layers'], hf_saved_tensor_count=tensor_count,
        original_context_mask_sampling_lp_preserved=True)
    return run, raw_pool_file, proof


def validate_raw_identity(raw, run, binding):
    identity = raw.get('identity', {})
    _need(identity.get('run_id') == run['run_id'] and
          identity.get('logical_update') == binding['source_step'] and
          identity.get('attempt_id') == binding['source_attempt_id'] and
          identity.get('attempt_index') == binding['source_attempt_index'] and
          identity.get('policy_snapshot_id') == binding['source_policy_snapshot_id'],
          'ORIGINAL_SOURCE_POOL_IDENTITY_DIFFERS')


def block_updates(worker):
    counters = dict(optimizer_step=0, scheduler_step=0)
    def block(name):
        def reject(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('FA3_MODEL_DIAGNOSTIC_UPDATE_FORBIDDEN: '+name)
        return reject
    worker.actor_optimizer.step = block('optimizer_step')
    scheduler = worker.actor_lr_scheduler
    scheduler.__class__ = type('FA3Diagnostic'+type(scheduler).__name__, (type(scheduler),),
                              {'step':block('scheduler_step')})
    return counters


def parameter_digest(parameters, *, chunk_elements=1048576):
    """Hash every local parameter byte in bounded chunks, without a model copy."""
    import torch
    digest = hashlib.sha256(); elements = count = 0
    for name, parameter in parameters:
        metadata = [name, list(parameter.shape), str(parameter.dtype), parameter.numel()]
        digest.update(json.dumps(metadata, separators=(',', ':')).encode()+b'\0')
        flat = parameter.detach().reshape(-1)
        for offset in range(0, flat.numel(), chunk_elements):
            chunk = flat[offset:offset+chunk_elements].contiguous().view(torch.uint8).cpu()
            digest.update(chunk.numpy().tobytes())
        elements += parameter.numel(); count += 1
    return dict(aggregate_sha256=digest.hexdigest(), local_parameter_elements=elements, local_parameter_shards=count)


def _rng_equal(before, after):
    import numpy as np
    import torch
    return (before['python'] == after['python'] and before['numpy'][0] == after['numpy'][0] and
        np.array_equal(before['numpy'][1], after['numpy'][1]) and before['numpy'][2:] == after['numpy'][2:] and
        torch.equal(before['torch_cpu'], after['torch_cpu']) and
        ((before['torch_cuda'] is None and after['torch_cuda'] is None) or
         (before['torch_cuda'] is not None and after['torch_cuda'] is not None and
          torch.equal(before['torch_cuda'], after['torch_cuda']))))


def qualify_flash3_policy(actor, micro_batch, temperature, *, adapter=None, attention_module=None,
                         group=None, expected_world_size=4):
    """Run FA2-1/2/3 then FA3-1; strict existing thresholds remain unchanged.

    expected_world_size=1 and an explicit fake adapter are only CPU mechanics
    fixtures. Production uses the actual installed 28-layer actor on four GPUs.
    The switch context covers BOTH forward and backward/checkpoint recompute.
    """
    import torch
    from search_g1 import sparse_policy_qualification as numerical
    from search_g1.logical_micro_loss import logical_policy_loss
    if adapter is None:
        from search_g1 import flash3_attention as adapter
    module = actor.actor_module
    attention_module = module if attention_module is None else attention_module
    group = getattr(module, 'process_group', None) if group is None else group
    environment = numerical._environment(module, expected_world_size, group)
    parameters, device, _, initialized, size = environment; config = actor.config
    _need(config.ppo_micro_batch_size == 8 and config.ppo_mini_batch_size == 32 and
          getattr(config, 'ppo_logical_micro_batch_size', None) == 4 and config.entropy_coeff == 0 and
          float(config.kl_loss_coef) == 0 and config.state_masking and config.policy_safety.enabled and
          not getattr(actor, '_selected_policy_logits', False), 'ORIGINAL_DENSE_MICRO8_LOGICAL4_RECIPE_REQUIRED')
    mask = micro_batch['loss_mask']; old = micro_batch['old_log_probs']; advantage = micro_batch['advantages']
    _need(micro_batch['responses'].shape[0] == 8 and mask.shape == old.shape == advantage.shape and
          bool(((mask == 0) | (mask == 1)).all()) and bool(mask.bool().any(dim=1).all()),
          'ORIGINAL_COMPLETE_PHYSICAL_EIGHT_POLICY_TARGETS_REQUIRED')
    _need(all(p.grad is None for _,p in parameters), 'PENDING_GRADIENTS_FORBIDDEN')
    audit = adapter.execution_report(attention_module, synchronize=device.type == 'cuda')
    if device.type == 'cuda':
        _need(audit.get('attention_layers') == 28, 'ALL28_REAL_ACTOR_ATTENTION_LAYERS_REQUIRED')
    compute_policy_loss = numerical._policy_loss_function()
    saved_rng = numerical._rng(device)
    attributes = {name:copy.deepcopy(getattr(actor,name)) if hasattr(actor,name) else numerical._MISSING
                  for name in numerical._LOG_ATTRIBUTES}
    modes = [(child, child.training) for child in module.modules()]
    before_parameters = parameter_digest(parameters)
    baseline_gradients = baseline_lp = baseline_loss = None
    new = loss = pg_loss = clipfrac = ppo_kl = None
    comparisons = {}; runs = []; started = time.time()
    try:
        for label in ('FA2-1','FA2-2','FA2-3','FA3-1'):
            numerical._barrier(initialized, group)
            numerical._clear_gradients(module); numerical._restore_attributes(actor, attributes)
            module.train(True); numerical._restore_rng(saved_rng, device)
            _need(_rng_equal(saved_rng, numerical._rng(device)), 'RUN_RNG_START_DIFFERS')
            enabled = label == 'FA3-1'
            prior = adapter.execution_report(attention_module, synchronize=device.type == 'cuda')
            with adapter.flash3_mode(attention_module, enabled=enabled):
                _, new = actor._forward_micro_batch(micro_batch, temperature=temperature, compute_entropy=False)
                pg_loss, clipfrac, ppo_kl = logical_policy_loss(compute_policy_loss,
                    logical_micro_batch_size=4, old_log_prob=old, log_prob=new, advantages=advantage,
                    eos_mask=mask, cliprange=config.clip_ratio, dual_clip=float(config.policy_safety.dual_clip))
                loss = pg_loss / 4
                loss.backward()
                after = adapter.execution_report(attention_module, synchronize=device.type == 'cuda')
                _need(after.get('flash3_enabled') is enabled, 'ATTENTION_SWITCH_STATE_DIFFERS')
                forward_delta = sum(after['forward_success_returns'].values())-sum(prior['forward_success_returns'].values())
                backward_delta = after['kernel_backward_success_returns']-prior['kernel_backward_success_returns']
                _need((forward_delta == 0 and backward_delta == 0) if not enabled else
                      (forward_delta >= audit['attention_layers'] and backward_delta >= audit['attention_layers']),
                      'ACTUAL_FA2_OR_FA3_FORWARD_BACKWARD_EXECUTION_NOT_PROVED')
            observed_lp = new.detach()[mask.bool()].to(device='cpu', copy=True)
            observed_loss = float(loss.detach().cpu())
            run = dict(run=label, flash3_enabled=enabled,
                local_loss=observed_loss if math.isfinite(observed_loss) else None,
                local_authored_tokens=observed_lp.numel(), actual_fa3_forward_returns_delta=forward_delta,
                actual_fa3_backward_returns_delta=backward_delta, execution_audit=after)
            if label == 'FA2-1':
                baseline_gradients = numerical._gradient_snapshot(parameters)
                baseline_lp, baseline_loss = observed_lp, observed_loss
                run['baseline_gradient_norm'] = numerical._gradient_comparison(baseline_gradients, parameters, environment, group)
            else:
                finite_loss = math.isfinite(baseline_loss) and math.isfinite(observed_loss)
                delta = abs(observed_loss-baseline_loss) if finite_loss else 0.0
                global_loss = numerical._reduce([delta,int(not finite_loss)], operation='max', environment=environment, group=group)
                comparison = dict(reference='FA2-1', actual=label,
                    log_probabilities=numerical._log_probability_comparison(baseline_lp,observed_lp,environment,group),
                    loss=dict(max_rank_absolute_error=global_loss[0] if global_loss[1] == 0 else None,
                              all_finite=global_loss[1] == 0),
                    gradients=numerical._gradient_comparison(baseline_gradients,parameters,environment,group))
                finite_lp = comparison['log_probabilities']['all_finite']
                moments = ([float(baseline_lp.double().square().sum()),
                            float((observed_lp.double()-baseline_lp.double()).square().sum())]
                           if finite_lp else [0.,0.])
                lp_reference, lp_difference = numerical._reduce(moments,operation='sum',environment=environment,group=group)
                comparison['log_probabilities']['relative_l2'] = (
                    math.sqrt(lp_difference/lp_reference) if finite_lp and lp_reference > 0 else None)
                relative_loss = numerical._reduce([
                    delta/abs(baseline_loss) if finite_loss and baseline_loss != 0 else 0.,
                    int(not finite_loss or baseline_loss == 0)],operation='max',environment=environment,group=group)
                comparison['loss']['max_rank_relative_error'] = relative_loss[0] if relative_loss[1] == 0 else None
                comparison.update(numerical._passing(comparison)); comparisons['FA2-1_'+label] = comparison
            runs.append(run); new = loss = pg_loss = clipfrac = ppo_kl = None
        numerical._barrier(initialized, group)
    finally:
        new = loss = pg_loss = clipfrac = ppo_kl = baseline_gradients = baseline_lp = None
        numerical._clear_gradients(module); numerical._restore_attributes(actor,attributes)
        for child, mode in modes: child.training = mode
        numerical._restore_rng(saved_rng,device)
    after_parameters = parameter_digest(parameters)
    _need(before_parameters == after_parameters, 'ACTUAL_PARAMETER_BYTES_CHANGED')
    _need(_rng_equal(saved_rng,numerical._rng(device)), 'CALLER_RNG_NOT_RESTORED')
    fa2_noise = {k:v for k,v in comparisons.items() if k != 'FA2-1_FA3-1'}
    return dict(format='actual_actor_FA2_repeat_FA3_full_policy_v1',
        accepted=all(c['accepted'] for c in comparisons.values()), thresholds=dict(numerical.THRESHOLDS),
        comparisons=comparisons, fa2_repeat_noise=fa2_noise,
        fa2_repeats_pass=all(c['accepted'] for c in fa2_noise.values()), local_runs=runs,
        world_size=size, physical_micro_per_rank=8, logical_micro_per_rank=4, gradient_accumulation=4,
        temperature=float(temperature), clip_ratio=float(config.clip_ratio), dual_clip=float(config.policy_safety.dual_clip),
        all_local_gradient_elements_compared=True, local_parameter_bytes_before=before_parameters,
        local_parameter_bytes_after=after_parameters, all_local_parameter_bytes_unchanged=True,
        same_rng_start_for_every_forward=True, caller_rng_restored=True,
        optimizer_calls=0, scheduler_calls=0, gradient_clipping_calls=0,
        normal_training_auto_call=False, automatic_fallback=False, started_unix=started, completed_unix=time.time(),
        scope='One original physical micro8 per rank; same full model and full contexts; strict FA2 noise and FA3 '
              'criteria both required, not whole-training equivalence or production-grounding qualification')


def _publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def _layer_shadow_function():
    from reproduction.easy2560_joint4.fa3_layer_shadow import run_layer_shadow
    return run_layer_shadow


def diagnose_actor(worker, micro, temperature, *, layer_shadow=False):
    """The optional shadow is a separate diagnostic, never a passing qualifier."""
    if not layer_shadow:
        return qualify_flash3_policy(worker.actor,micro,temperature,attention_module=worker.actor_module)
    from search_g1 import sparse_policy_qualification as numerical
    parameters=list(worker.actor.actor_module.named_parameters())
    _need(parameters and all(p.grad is None for _,p in parameters), 'SHADOW_PENDING_GRADIENTS_FORBIDDEN')
    device=next((p.device for _,p in parameters if p.device.type=='cuda'),parameters[0][1].device)
    before_rng=numerical._rng(device);before=parameter_digest(parameters)
    observed=_layer_shadow_function()(worker.actor,micro,temperature,attention_module=worker.actor_module)
    after=parameter_digest(parameters)
    _need(before==after and all(p.grad is None for _,p in parameters), 'SHADOW_PARAMETERS_OR_GRADIENTS_CHANGED')
    _need(_rng_equal(before_rng,numerical._rng(device)), 'SHADOW_CALLER_RNG_NOT_RESTORED')
    _need(observed.get('acceptance_status')=='DIAGNOSTIC_ONLY_NOT_QUALIFICATION' and
          observed.get('real_actor_forward_calls')==1 and observed.get('backward_calls')==0,
          'SHADOW_SINGLE_FORWARD_ONLY_CONTRACT_DIFFERS')
    return dict(format='actual_actor_FA2_single_forward_layer_shadow_v1',accepted=False,
        acceptance_status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',diagnostic_completed=True,diagnostic_only=True,
        layer_shadow=observed,all_local_parameter_bytes_unchanged=True,
        local_parameter_bytes_before=before,local_parameter_bytes_after=after,caller_rng_restored=True,
        real_actor_forward_calls=1,backward_calls=0,optimizer_calls=0,scheduler_calls=0,
        gradient_clipping_calls=0,automatic_fallback=False,normal_training_auto_call=False,
        original_full_model_qualification_replaced=False)


def run(base, output, raw_pool_file=None, *, layer_shadow=False):
    import torch
    import torch.distributed as dist
    from omegaconf import open_dict
    from reproduction.common8.configuration import resolved
    from search_g1.easy7b_infra import flags
    from verl.utils.weight_continuation import model_initialization_path
    base = Path(base).resolve(strict=True); output = Path(output).absolute()
    _need(_within(output.resolve(),base/'diagnostics') and not output.exists(), 'NEW_OWN_DIAGNOSTIC_OUTPUT_REQUIRED')
    run_manifest, raw_pool_file, binding = source_binding(base,raw_pool_file)
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    _need(int(os.environ['WORLD_SIZE']) == 4, 'TORCHRUN_FOUR_RANKS_REQUIRED')
    torch.cuda.set_device(local_rank)
    os.environ.update(SEARCH_G1_BASE=str(base), SEARCH_G1_PROFILE='common8', SEARCH_G1_TRAINING_WORLD_SIZE='4',
                      SEARCH_G1_SOURCE_COMMIT=run_manifest['source_commit'])
    for key in ('WG_PREFIX','WG_BACKEND','DISABLE_WORKER_INIT'): os.environ.pop(key,None)
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'): os.environ[key]='1'
    bound, config, _ = resolved(base)
    _need(bound == run_manifest and flags(bound) == dict(selected_policy_logits=False,
          exact_retrieval_cache=True,flash_attention_3_actor=True), 'EXPLICIT_FA3_DIAGNOSTIC_BASE_REQUIRED')
    checkpoint = Path(model_initialization_path(config.actor_rollout_ref.model,actor=True)).resolve()
    _need(str(checkpoint) == binding['source_checkpoint'], 'BOUND_ACTOR_MODEL_SOURCE_DIFFERS')
    with open_dict(config.actor_rollout_ref.actor.optim):
        config.actor_rollout_ref.actor.optim.total_training_steps = config.trainer.total_training_steps
        config.actor_rollout_ref.actor.optim.completed_updates = binding['source_step']
    raw = torch.load(raw_pool_file,map_location='cpu',weights_only=False)
    validate_raw_identity(raw,bound,binding)
    micro, selection = prepare_micro_batch(raw,rank); del raw; gc.collect()
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad
    started=time.time(); worker=ActorRolloutRefWorker(config.actor_rollout_ref,role='actor');worker.init_model()
    _need(worker._is_actor and not worker._is_rollout and not worker._is_ref and
          not hasattr(worker,'rollout') and not hasattr(worker,'ref_policy'), 'ACTOR_ONLY_INITIALIZATION_REQUIRED')
    _need(len(worker.actor_optimizer.state) == 0, 'TEMPORARY_ADAM_MUST_BE_EMPTY')
    scheduler_before=copy.deepcopy(worker.actor_lr_scheduler.state_dict()); calls=block_updates(worker)
    load_fsdp_param_and_grad(worker.actor_module_fsdp,device_id=torch.cuda.current_device(),load_grad=False)
    device=torch.device('cuda',torch.cuda.current_device());micro={k:v.to(device) for k,v in micro.items()}
    result=diagnose_actor(worker,micro,float(config.actor_rollout_ref.rollout.temperature),layer_shadow=layer_shadow)
    _need(calls == dict(optimizer_step=0,scheduler_step=0) and len(worker.actor_optimizer.state)==0 and
          worker.actor_lr_scheduler.state_dict()==scheduler_before,'OPTIMIZER_OR_SCHEDULER_CHANGED')
    local=dict(rank=rank,local_rank=local_rank,hostname=socket.gethostname(),gpu_name=torch.cuda.get_device_name(),
        diagnostic_slurm_job_id=os.environ.get('SLURM_JOB_ID'),diagnostic_pid=os.getpid(),
        selection=selection,update_calls=calls,fresh_adam_state_entries=0,scheduler_unchanged=True)
    local['layer_shadow_diagnostic' if layer_shadow else 'numerical_comparison']=result
    ranks=[None]*4;dist.all_gather_object(ranks,local)
    accepted=False if layer_shadow else all(r['numerical_comparison']['accepted'] for r in ranks)
    report=dict(format=('easy7b_dynamic_cp_layer_shadow_diagnostic_v1' if layer_shadow else
                        'easy7b_dynamic_cp_model_only_fa3_cuda_qualification_v1'),accepted=accepted,
        diagnostic_only=True,base=str(base),source_commit=bound['source_commit'],run_id=bound['run_id'],
        diagnostic_slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        source_binding=binding,model_source_completed_outer=binding['source_step'],world_size=4,ranks=ranks,
        actual_optimizer_state_restored=False,actual_scheduler_state_restored=False,saved_training_rng_restored=False,
        data_cursor_restored=False,formal_actual_restore_receipt_written=False,
        production_full_state_restore_still_required=True,rollout_or_reference_model_initialized=False,
        sampling_calls=0,optimizer_calls=0,scheduler_calls_after_construction=0,gradient_clipping_calls=0,
        normal_training_auto_call=False,automatic_fallback=False,physical_micro_per_rank=8,logical_micro_per_rank=4,
        started_unix=started,completed_unix=time.time(),
        scope='Actual bound CP>=9 full actor forward/backward only on32 original source-step mixed-EM trajectories; '
              'five-EM diagnostic advantages are NOT production grounding rewards; no fullstate restore/whole-training claim')
    if layer_shadow:
        report.update(acceptance_status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',diagnostic_completed=True,
            real_actor_forward_calls_per_rank=1,backward_calls=0,original_full_model_qualification_replaced=False,
            scope='One original FA2 actor forward per rank with exact-QKV FA3 shadows; no backward or parameter update; '
                  'not numerical qualification, fullstate restoration, or permission to switch production attention')
    if rank==0:
        _publish(output,report)
        print(('SEARCH_G1_FA3_LAYER_SHADOW=' if layer_shadow else 'SEARCH_G1_FA3_MODEL_DIAGNOSTIC=')+json.dumps(dict(accepted=accepted,output=str(output),
            source_step=binding['source_step'],model_only=True,production_full_state_restore_still_required=True)),flush=True)
    dist.barrier();del micro,worker;gc.collect();torch.cuda.empty_cache();dist.destroy_process_group()
    if not layer_shadow and not accepted:raise RuntimeError('FA3_MODEL_DIAGNOSTIC_NUMERICAL_THRESHOLDS_REJECTED; see '+str(output))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--raw-pool-file',help='Optional actual numeric .pt member of the bound source-step successful pool')
    parser.add_argument('--layer-shadow',action='store_true',
        help='One original FA2 forward with exact-QKV layer shadows; never a numerical qualification')
    args=parser.parse_args(argv)
    try:run(args.base,args.output,args.raw_pool_file,layer_shadow=args.layer_shadow)
    except BaseException as error:
        rank=os.environ.get('RANK','unknown');failure=Path(args.output).with_name(Path(args.output).stem+'-rank-'+rank+'-failure.json')
        try:
            if _within(failure.resolve(),Path(args.base).resolve()/'diagnostics') and not failure.exists():
                _publish(failure,dict(diagnostic_only=True,rank=rank,accepted=False,error_type=type(error).__name__,
                    error=str(error)[:2048],worker_reuse_permitted=False,automatic_fallback=False,
                    formal_actual_restore_receipt_written=False,checked_unix=time.time()))
        except Exception:pass
        raise


if __name__=='__main__':main()
