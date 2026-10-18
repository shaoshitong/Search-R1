"""Read-only probability diagnostics for legacy and explicit behavior-mu profiles."""
import math
from contextlib import contextmanager
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .training_state import atomic_save, cpu_copy, file_sha


def collective_device():
    return (torch.device('cuda', torch.cuda.current_device())
            if dist.is_initialized() and dist.get_backend() == 'nccl' else torch.device('cpu'))


def sync_prepare(error, device):
    flag = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item(): raise RuntimeError('Global sampling diagnostic preparation refusal; local: '+str(error))


def check_signature(values, device):
    signature = torch.tensor(values, device=device, dtype=torch.float64)
    lo, hi = signature.clone(), signature.clone()
    if dist.is_initialized():
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if not torch.equal(lo, hi): raise RuntimeError('Global sampling diagnostic rank signature mismatch')


def generation_preflight(worker, prompts):
    """Check matching opt-in/shape/recompute settings before FSDP/vLLM forwards."""
    device = collective_device()
    error = None
    try:
        config = worker.config.rollout.sampling_consistency
        enabled, enforce = bool(config.enabled), bool(config.enforce)
        do_sample = bool(prompts.meta_info.get('do_sample', True))
        recompute = bool(prompts.meta_info.get('recompute_log_prob', True))
        limit = float(config.max_abs_delta)
        temperature = float(worker.config.rollout.temperature)
        if not math.isfinite(limit) or limit < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid explicit probability diagnostic inputs')
        if enabled and do_sample and (not worker._is_actor or not recompute or worker.actor.use_ulysses_sp):
            raise ValueError('Raw sampling comparison requires same-turn actor recomputation and no sequence parallelism')
        rows = len(prompts)
    except Exception as exc:
        error = repr(exc)
    sync_prepare(error, device)
    check_signature([enabled,enforce,do_sample,recompute,limit,temperature,rows],device)


def reduce_pair_statistics(sampled, actor, mask, distribution, temperature, limit):
    """Compare identical emitted tokens before any decoding or rolling truncation."""
    device = collective_device()
    error = None
    try:
        if (not all(isinstance(value, torch.Tensor) for value in (sampled, actor, mask))
                or sampled.shape != actor.shape or mask.shape != actor.shape
                or any(value.device != device for value in (sampled, actor, mask))):
            raise ValueError('Paired probability/mask shapes and collective device must match exactly')
        if not math.isfinite(limit) or limit < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid probability diagnostic threshold or temperature')
        if not isinstance(distribution, dict): raise ValueError('Missing captured sampling distribution')
    except Exception as exc:
        error = repr(exc)
    sync_prepare(error, device)
    check_signature([limit, temperature], device)
    valid = mask.bool()
    local_error = 0
    expected = dict(temperature=temperature, top_p=1., top_k=-1, min_p=0.,
                    presence_penalty=0., frequency_penalty=0., repetition_penalty=1., n=1)
    local_error |= int(any(distribution.get(k) != v for k, v in expected.items()))
    # A filtered sampling distribution cannot be compared to plain actor softmax.
    local_error |= int(bool(distribution.get('logits_processors')))
    delta = torch.where(valid, actor, 0.).double() - torch.where(valid, sampled, 0.).double()
    finite = torch.isfinite(actor) & torch.isfinite(sampled) & torch.isfinite(delta)
    bad = (valid & ~finite).sum().double()
    safe = torch.where(valid & finite, delta, 0.)
    over_bound = (valid & finite & (safe > 30.)).sum().double()
    k3 = torch.expm1(torch.where(safe <= 30., safe, 0.)) - torch.where(safe <= 30., safe, 0.)
    maximum = safe.abs().max() if safe.numel() else torch.zeros((), device=device, dtype=torch.float64)
    if dist.is_initialized(): dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    scale = maximum.clamp_min(1.)
    sums = torch.stack([valid.sum().double(), bad, over_bound, safe.abs().sum(),
                        (safe/scale).square().sum(), k3.sum(), torch.tensor(float(local_error), device=device)])
    if dist.is_initialized():
        dist.all_reduce(sums)
    count, invalid, excessive, abs_sum, square_sum, kl_sum, metadata_invalid = sums.tolist()
    reasons = []
    if metadata_invalid: reasons.append('sampling_distribution_not_plain_actor_softmax_or_invalid_limit')
    if invalid: reasons.append('nonfinite_valid_sampling_or_actor_logprob')
    if excessive: reasons.append('sampling_difference_exceeds_numeric_bound')
    reduction_valid = all(math.isfinite(value) for value in (abs_sum, square_sum, kl_sum))
    if not reduction_valid: reasons.append('sampling_diagnostic_reduction_overflow')
    if not count: reasons.append('empty_sampling_token_set')
    if maximum.item() > limit: reasons.append('raw_turn_sampling_actor_max_abs_delta')
    return dict(allowed=not reasons, reasons=reasons, valid_tokens=int(count),
                invalid_tokens=int(invalid), numeric_bound_exceeded=int(excessive),
                max_abs_delta=maximum.item(), mean_abs_delta=abs_sum/count if count and math.isfinite(abs_sum) else None,
                rms_delta=scale.item()*math.sqrt(square_sum/count) if count and math.isfinite(square_sum) else None,
                sampled_k3_mean=kl_sum/count if count and not invalid and not excessive and reduction_valid else None,
                max_abs_delta_limit=limit,
                scope='All raw emitted response tokens in this distributed generation call; '
                      'same raw turn context including GPU-divisibility duplicate rows, not postprocessed final Tree leaves')


def record_raw_turn(worker, output):
    """Consume the temporary sampler field before it can reach Tree concatenation."""
    device = collective_device()
    error = None
    try:
        sampled = output.batch['__sampling_log_probs']
        distribution = output.meta_info['__sampling_distribution']
        config = worker.config.rollout.sampling_consistency
        actor = output.batch['old_log_probs']
        mask = output.batch['attention_mask'][:, -output.batch['responses'].shape[1]:]
        rank = dist.get_rank() if dist.is_initialized() else 0
        call = getattr(worker, '_sampling_consistency_calls', 0) + 1
        root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'sampling-consistency'
        saved_rows = len(output) if os.environ.get('TREE_PPO_CALIBRATION_ONLY') == '1' else min(4,len(output))
        record = dict(rank=rank, world_size=dist.get_world_size() if dist.is_initialized() else 1,
                  generation_call=call, scheduler_step=worker.actor_lr_scheduler.last_epoch,
                  distribution=distribution, enforced=bool(config.enforce),
                  code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                  config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                  data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                  job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(),run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
                  model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                  local_rows=len(output), saved_rows=saved_rows, optimizer_committed=0,
                  context_scope='Before response decode/retokenization/rolling-window truncation')
    except Exception as exc:
        error = repr(exc)
    sync_prepare(error, device)
    check_signature([bool(config.enforce),call],device)
    decision = reduce_pair_statistics(sampled, actor, mask, distribution,
                                      worker.config.rollout.temperature, float(config.max_abs_delta))
    worker._sampling_consistency_calls = call
    record['decision'] = decision
    failed = False
    try:
        snapshot = {key: cpu_copy(value[:saved_rows]) for key, value in output.batch.items()
                    if key != '__sampling_log_probs'}
        snapshot['sampled_log_probs'] = cpu_copy(sampled[:saved_rows])
        snapshot['distribution'] = distribution
        snapshot_path = root/f'rank-{rank:02d}-call-{call:05d}.pt'
        record['snapshot_path'] = str(snapshot_path)
        record['snapshot_sha256'] = atomic_save(snapshot, snapshot_path)
        path = root/f'rank-{rank:02d}-call-{call:05d}.json'
        record_sha = atomic_save(record, path, json_format=True)
    except Exception:
        failed = True
    flag = torch.tensor(int(failed), device=device)
    if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item(): raise RuntimeError('Global raw-turn sampling evidence-write failure')
    output.batch.pop('__sampling_log_probs')
    output.meta_info.pop('__sampling_distribution')
    worker._sampling_consistency_records = getattr(worker, '_sampling_consistency_records', []) + [str(path)]
    worker._sampling_consistency_entries = getattr(worker, '_sampling_consistency_entries', []) + [
        dict(path=str(path),sha256=record_sha,snapshot_path=record['snapshot_path'],snapshot_sha256=record['snapshot_sha256'])]
    if bool(config.enforce) and not decision['allowed']:
        raise RuntimeError('Global raw-turn sampling/FSDP consistency refusal: '+repr(decision['reasons']))


def calibration_monitor_snapshot(safety, data, corrected):
    """Bind the observed monitor, actual mu, full beta and frozen monitor beta."""
    snapshot = {key: cpu_copy(value) for key, value in safety.monitor.items()}
    if corrected:
        from .tree_behavior_mu import FIELD as BINDING_FIELD
        beta = safety.actor_monitor_beta
        old = safety.monitor['old_log_probs']
        if (not isinstance(beta, torch.Tensor) or not beta.is_floating_point()
                or beta.shape != old.shape
                or not torch.isfinite(beta[safety.effective_mask(safety.monitor)]).all()):
            raise ValueError('Missing/nonfinite frozen same-partition beta monitor')
        selected = safety.indices.detach().cpu().numpy()
        snapshot['actor_same_partition_beta_monitor'] = cpu_copy(beta)
        snapshot['behavior_mu_source_row_bindings'] = cpu_copy(data.non_tensor_batch[BINDING_FIELD][selected])
    return snapshot



@contextmanager
def calibration_actor_training_mode(worker, device, identity):
    """Matched train installation; restore every original module flag in finally.

    The outer RNG fork also covers custom train() implementations. Native
    scoring keeps its own existing RNG fork; no optimizer/scheduler is called.
    """
    model = worker.actor.actor_module
    saved = []
    mode = dict(requested_monitor_mode='train', optimizer_committed=0,
                scheduler_advanced=False, original_module_training_flags=[],
                monitor_module_training_flags=[], all_monitor_modules_training=False,
                original_module_modes_restored=False, body_returned=False,
                identity={key: identity[key] for key in
                          ('rank', 'world_size', 'code_commit', 'config_sha256', 'data_identity',
                           'model_identity', 'job_id', 'run_id')})
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        try:
            error = None
            try:
                saved = list(model.named_modules())
                original = [(name, module, module.training) for name, module in saved]
                if any(type(flag) is not bool for name, module, flag in original):
                    raise ValueError('Calibration requires actual boolean module.training flags')
                saved = original
                mode['original_module_training_flags'] = [dict(name=name, training=flag)
                                                         for name, module, flag in saved]
                model.train()
                current = list(model.named_modules())
                if [(name, id(module)) for name, module in current] != [(name, id(module)) for name, module, flag in saved]:
                    raise ValueError('Calibration train() changed module inventory')
                mode['monitor_module_training_flags'] = [dict(name=name, training=module.training)
                                                        for name, module in current]
                mode['all_monitor_modules_training'] = all(module.training is True for name, module in current)
                if not mode['all_monitor_modules_training']:
                    raise ValueError('Calibration must score in actual train mode on every module')
            except Exception as exc:
                error = repr(exc)
            sync_prepare(error, device)
            yield mode
            mode['body_returned'] = True
        finally:
            restore_error = None
            try:
                if saved and len(saved[0]) == 3:
                    for name, module, flag in saved:
                        # Calling train(flag) recursively would erase mixed original child modes.
                        module.training = flag
                    current = list(model.named_modules())
                    mode['restored_module_training_flags'] = [dict(name=name, training=module.training)
                                                              for name, module in current]
                    mode['original_module_modes_restored'] = (
                        [(name, id(module)) for name, module in current] == [(name, id(module)) for name, module, flag in saved]
                        and all(module.training is flag for name, module, flag in saved))
                if not mode['original_module_modes_restored']:
                    raise ValueError('Calibration module mode restoration incomplete')
            except Exception as exc:
                restore_error = repr(exc)
            mode['restore_error'] = restore_error
            worker._last_calibration_mode_evidence = mode
            io_error = None
            try:
                root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'probability-calibration'
                mode['mode_evidence_path'] = str(root/f'rank-{identity["rank"]:02d}-mode.json')
                mode['mode_evidence_sha256'] = atomic_save(mode, mode['mode_evidence_path'], json_format=True)
            except Exception as exc:
                io_error = repr(exc)
            sync_prepare(restore_error or io_error, device)


LEGACY_CALIBRATION_PROFILE = 'healthy_checkpoint_parent_v1'
FRESH_CALIBRATION_PROFILE = 'fresh_base_zero_v1'


def calibration_worker_state(worker, data):
    """Select explicit fresh-empty state or preserve the legacy restored parent."""
    profile = data.meta_info.get('calibration_profile', LEGACY_CALIBRATION_PROFILE)
    if profile not in (LEGACY_CALIBRATION_PROFILE, FRESH_CALIBRATION_PROFILE):
        raise ValueError('Unknown explicit calibration profile')
    state = worker.actor_optimizer.state
    scheduler = worker.actor_lr_scheduler.last_epoch
    if profile == FRESH_CALIBRATION_PROFILE:
        from .tree_fresh200 import refuse_recovery_environment
        refuse_recovery_environment()
        if getattr(worker.config.actor, 'fresh200_protocol', None) != 'tree_fresh200_five_leaf_v1':
            raise ValueError('Fresh zero calibration requires the explicit fresh200 actor protocol')
        provenance = data.meta_info['calibration_fresh_initialization']
        required = dict(calibration_profile=FRESH_CALIBRATION_PROFILE,
                        training_protocol='tree_fresh200_five_leaf_v1', initialized_global_steps=0,
                        driver_completed_outer=0, parent_completed_step=0, parent_cursor=0, checkpoint_loaded=False)
        if any(type(provenance.get(key)) is not type(value) or provenance.get(key) != value
               for key, value in required.items()):
            raise ValueError('Fresh initialization provenance is missing or differs from actual initial zero boundary')
        if (type(data.meta_info.get('baseline_outer_step')) is not int or data.meta_info['baseline_outer_step'] != 1
                or type(data.meta_info.get('calibration_parent_outer')) is not int
                or data.meta_info['calibration_parent_outer'] != 0):
            raise ValueError('Fresh calibration observes logical outer one before any completed outer')
        if len(state) != 0 or type(scheduler) is not int or scheduler != 0:
            raise ValueError('Fresh calibration requires entire empty Adam state and actual scheduler zero')
        steps = []
    else:
        if getattr(worker.config.actor, 'fresh200_protocol', None) == 'tree_fresh200_five_leaf_v1':
            raise ValueError('Fresh actor cannot silently use the legacy calibration profile')
        steps = sorted({int(value['step'].item()) if isinstance(value['step'], torch.Tensor) else int(value['step'])
                        for value in state.values() if 'step' in value})
        if steps != [int(os.environ['TREE_RECOVERY_PARENT_OPTIMIZER_STEPS'])]:
            raise ValueError('Actual Adam state differs from accepted parent')
    return dict(calibration_profile=profile, restored_optimizer_steps=steps,
                optimizer_state_entries=len(state), scheduler_step=scheduler)


def fresh_calibration_driver_provenance(trainer):
    """Validate the true initialized-zero/observed-one boundary, without recovery."""
    from .tree_fresh200 import refuse_recovery_environment
    refuse_recovery_environment()
    provenance = trainer.fresh_initialization_provenance
    required = dict(calibration_profile=FRESH_CALIBRATION_PROFILE,
                    training_protocol='tree_fresh200_five_leaf_v1', initialized_global_steps=0,
                    driver_completed_outer=0, parent_completed_step=0, parent_cursor=0, checkpoint_loaded=False)
    if any(type(provenance.get(key)) is not type(value) or provenance.get(key) != value
           for key, value in required.items()):
        raise ValueError('Fresh driver initialization proof differs from the actual zero boundary')
    if (type(trainer.global_steps) is not int or trainer.global_steps != 1
            or type(trainer.actual_outer_updates) is not int or trainer.actual_outer_updates != 0
            or trainer.checkpoint_sampler.cursor != 512):
        raise ValueError('Fresh calibration must observe outer one, actual zero updates, and exactly 512 consumed questions')
    expected = dict(code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                    config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                    data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                    model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                    job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(), run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'])
    workers = provenance['workers']
    if (len(workers) != 8 or {record['rank'] for record in workers} != set(range(8))
            or any(record['world_size'] != 8 or record['optimizer_state_entries'] != 0
                   or record['optimizer_step_values'] != [] or record['scheduler_step'] != 0
                   or record['optimizer_committed'] != 0 or record['scheduler_advanced'] is not False
                   or record['calibration_profile'] != FRESH_CALIBRATION_PROFILE
                   or record['checkpoint_recovery_environment_absent'] is not True
                   or Path(record['configured_model_path']).resolve() != Path(trainer.config.actor_rollout_ref.model.path).resolve()
                   or record['configured_model_path'] != record['expected_model_path']
                   or any(record.get(key) != value for key, value in expected.items()) for record in workers)
            or any(provenance.get(key) != value for key, value in expected.items())):
        raise ValueError('Fresh actual eight-rank initialization proof is incomplete or from another source/run')
    return provenance

def calibrate_actor(worker, data):
    """No optimizer call or scheduler advance; diagnostics only in a dedicated stage."""
    from .policy_safety_runtime import PolicySafetyRuntime
    device = collective_device()
    error = None
    try:
        if os.environ.get('TREE_PPO_CALIBRATION_ONLY') != '1':
            raise RuntimeError('Read-only calibration endpoint requires dedicated calibration stage')
        if not worker._is_actor or worker.actor.use_ulysses_sp:
            raise ValueError('Read-only calibration requires an actor without sequence parallelism')
        state = calibration_worker_state(worker, data)
        steps = state['restored_optimizer_steps']
        from .policy_safety_runtime import behavior_mu_correction_enabled
        corrected = behavior_mu_correction_enabled(worker.config.actor.policy_safety)
        rank = dist.get_rank()
        record = dict(rank=rank, world_size=dist.get_world_size(), optimizer_committed=0,
                  behavior_mu_corrected_profile=corrected,
                  probability_gate_scope=('actual_behavior_mu_binding_and_same_actor_monitor_repeat; original_sampler_actor_gate_not_passed'
                                          if corrected else 'original_raw_sampling_and_actor_initial_consistency'),
                  scheduler_advanced=False, parent_completed_outer=int(data.meta_info['calibration_parent_outer']),
                  observed_outer=int(data.meta_info['baseline_outer_step']), **state,
                  diagnostic_records=getattr(worker, '_sampling_consistency_records', []),
                  diagnostic_entries=getattr(worker, '_sampling_consistency_entries', []),
                  code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                  config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                  data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                  model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                  job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(),run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'])
        if ((not corrected and not record['diagnostic_records'])
                or len(record['diagnostic_entries']) != len(record['diagnostic_records'])
                or record['scheduler_step'] != record['parent_completed_outer']):
            raise ValueError('Missing sampling evidence or wrong restored scheduler')
    except Exception as exc:
        error = repr(exc)
    sync_prepare(error,device)
    fresh = record['calibration_profile'] == FRESH_CALIBRATION_PROFILE
    check_signature([0 if fresh else steps[0], record['scheduler_step'], record['observed_outer'],
                     len(record['diagnostic_entries']), corrected, fresh], device)
    if corrected:
        from .policy_safety_runtime import validate_actor_probability_profile
        validate_actor_probability_profile(worker.actor, data, worker.config.actor.policy_safety)
    with calibration_actor_training_mode(worker, device, record) as mode:
        safety = PolicySafetyRuntime(worker.actor, data.batch, data.meta_info, worker.config.actor.policy_safety,
                                     source_data=data)
        with torch.no_grad():
            decision = safety.measure('before_optimizer', publish=False)
        record.update(decision=decision,monitor_local_row_indices=safety.indices.tolist(),
                      additional_initial_actor_repeat_native_forward_calls=safety.initial_actor_repeat_native_forward_calls,
                      monitor_native_forward_calls=safety.monitor_native_forward_calls)
    record['actor_calibration_mode'] = mode
    if fresh:
        error = None
        try:
            after_state = calibration_worker_state(worker, data)
            if after_state != state:
                raise ValueError('Fresh calibration changed actual Adam/scheduler state')
            record['fresh_post_monitor_state'] = after_state
        except Exception as exc:
            error = repr(exc)
        sync_prepare(error, device)
    failed = False
    try:
        root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'probability-calibration'
        record['monitor_path'] = str(root/f'rank-{rank:02d}-monitor.pt')
        snapshot = calibration_monitor_snapshot(safety, data, corrected)
        if corrected:
            record['actual_behavior_mu_row_bindings_validated'] = True
            record['actual_same_partition_beta_monitor_saved'] = True
        record['monitor_sha256'] = atomic_save(snapshot, record['monitor_path'])
        atomic_save(record, root/f'rank-{rank:02d}.json', json_format=True)
    except Exception:
        failed = True
    flag = torch.tensor(int(failed), device=safety.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item(): raise RuntimeError('Global zero-write calibration evidence-write failure')
    return record


def calibration_boundary_matches(trainer, parent, corrected):
    """Original observation boundary or explicit corrected profile, no writes."""
    if type(corrected) is not bool:
        return False
    settings = trainer.config.actor_rollout_ref
    if (trainer.global_steps != parent + 1 or trainer.actual_outer_updates != parent
            or not settings.actor.policy_safety.enabled):
        return False
    consistency = settings.rollout.sampling_consistency
    if corrected:
        return consistency.enabled is False and consistency.enforce is True
    return bool(consistency.enabled) and not bool(consistency.enforce)


def finalize_calibration(trainer, batch):
    """Driver-only terminal observation, before the ordinary actor update path."""
    import json
    fresh = trainer.config.actor_rollout_ref.actor.get('fresh200_protocol') == 'tree_fresh200_five_leaf_v1'
    provenance = fresh_calibration_driver_provenance(trainer) if fresh else trainer.recovery_provenance
    parent = provenance['parent_completed_step']
    profile = FRESH_CALIBRATION_PROFILE if fresh else LEGACY_CALIBRATION_PROFILE
    from .policy_safety_runtime import behavior_mu_correction_enabled
    safety_config = trainer.config.actor_rollout_ref.actor.policy_safety
    corrected = behavior_mu_correction_enabled(safety_config)
    if corrected:
        from .controller import validate_behavior_mu_config
        validate_behavior_mu_config(trainer.config)
    if not calibration_boundary_matches(trainer, parent, corrected):
        raise RuntimeError('Dedicated zero-write calibration configuration/boundary changed')
    batch.meta_info['baseline_outer_step'] = trainer.global_steps
    batch.meta_info['calibration_parent_outer'] = parent
    batch.meta_info['calibration_profile'] = profile
    if fresh:
        batch.meta_info['calibration_fresh_initialization'] = {
            key: provenance[key] for key in ('calibration_profile', 'training_protocol', 'initialized_global_steps',
                'driver_completed_outer', 'parent_completed_step', 'parent_cursor', 'checkpoint_loaded')}
    # Exactly the existing update temperature, but no update_policy endpoint.
    batch.meta_info['temperature'] = trainer.config.actor_rollout_ref.rollout.temperature
    trainer.actor_rollout_wg.calibrate_policy_safety(batch)
    root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'probability-calibration'
    workers = [json.loads((root/f'rank-{rank:02d}.json').read_text()) for rank in range(8)]
    expected = dict(code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                    config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                    data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                    model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                    job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(),run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'])
    if any(record['rank'] != rank or record['world_size'] != 8
           or record['optimizer_committed'] != 0 or record['scheduler_advanced']
           or record['scheduler_step'] != parent or (not corrected and not record['diagnostic_records'])
           or record.get('calibration_profile') != profile
           or (fresh and (record['optimizer_state_entries'] != 0 or record['restored_optimizer_steps'] != []
                          or record.get('fresh_post_monitor_state') != dict(calibration_profile=profile,
                              restored_optimizer_steps=[], optimizer_state_entries=0, scheduler_step=0)))
           or record.get('behavior_mu_corrected_profile', False) != corrected
           or (corrected and record.get('actual_behavior_mu_row_bindings_validated') is not True)
           or (corrected and record.get('actual_same_partition_beta_monitor_saved') is not True)
           or record['parent_completed_outer'] != parent or record['observed_outer'] != trainer.global_steps
           or any(record.get(key) != value for key,value in expected.items())
           for rank, record in enumerate(workers)):
        raise RuntimeError('Zero-write calibration rank proof incomplete')
    for rank, record in enumerate(workers):
        mode = record['actor_calibration_mode']
        mode_path = Path(mode['mode_evidence_path'])
        if (mode_path != root/f'rank-{rank:02d}-mode.json'
                or file_sha(mode_path) != mode['mode_evidence_sha256']
                or mode['requested_monitor_mode'] != 'train'
                or mode['all_monitor_modules_training'] is not True
                or mode['original_module_modes_restored'] is not True or mode['body_returned'] is not True
                or mode['optimizer_committed'] != 0 or mode['scheduler_advanced'] is not False
                or mode['identity'] != dict(rank=rank, world_size=8, **expected)):
            raise RuntimeError('Calibration actual train-mode/restoration proof incomplete')
        saved_mode = json.loads(mode_path.read_text())
        if dict(saved_mode, mode_evidence_sha256=mode['mode_evidence_sha256']) != mode:
            raise RuntimeError('Calibration mode evidence differs from the final rank record')
        entries = record['diagnostic_entries']
        if [entry['path'] for entry in entries] != record['diagnostic_records']:
            raise RuntimeError('Calibration diagnostic records do not match pinned entries')
        for entry in entries:
            path = Path(entry['path'])
            if not path.resolve().is_relative_to(root.parent.resolve()) or file_sha(path) != entry['sha256']:
                raise RuntimeError('Calibration diagnostic identity or file hash changed')
            raw = json.loads(path.read_text())
            if (raw['rank'] != rank or raw['scheduler_step'] != parent
                    or any(raw.get(key) != value for key,value in expected.items())
                    or raw['snapshot_path'] != entry['snapshot_path']
                    or raw['snapshot_sha256'] != entry['snapshot_sha256']
                    or not Path(entry['snapshot_path']).resolve().is_relative_to(root.parent.resolve())
                    or file_sha(entry['snapshot_path']) != entry['snapshot_sha256']):
                raise RuntimeError('Raw-turn sampling snapshot binding changed')
        if (not Path(record['monitor_path']).resolve().is_relative_to(root.resolve())
                or file_sha(record['monitor_path']) != record['monitor_sha256']):
            raise RuntimeError('Calibration monitor snapshot binding changed')
    ready = dict(status=('FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES' if fresh and corrected
                         else 'FRESH_CALIBRATION_OBSERVED_ZERO_WRITES' if fresh
                         else 'BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES' if corrected
                         else 'CALIBRATION_OBSERVED_ZERO_WRITES'), workers=workers, calibration_profile=profile,
                 initialized_global_steps=provenance['initialized_global_steps'] if fresh else None,
                 behavior_mu_corrected_profile=corrected,
                 original_sampler_actor_probability_gate_accepted=False if corrected else None,
                 parent_completed_outer=parent, observed_outer=trainer.global_steps,
                 new_completed_outer=0, optimizer_committed=0, scheduler_advanced=False,
                 sampler_cursor_before_outer=provenance['parent_cursor'],
                 sampler_cursor_after_observation=trainer.checkpoint_sampler.cursor,
                 job_id=__import__('search_r1.baselines.execution_identity',fromlist=['execution_id']).execution_id(), code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                 config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                 data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                 model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                 rollout_reward=float(batch.batch['token_level_rewards'].sum(-1).mean()),
                 scope=('Fresh official base observation at logical outer one with empty Adam/scheduler zero, no optimizer, outer credit or final evaluation'
                        if fresh else 'Healthy checkpoint observation without optimizer, scheduler, outer credit or final evaluation'))
    atomic_save(ready, root/'ready.json', json_format=True)
    print('TREE_PPO_CALIBRATION_OBSERVED_ZERO_WRITES', flush=True)
    return ready
