"""One-shot D1/D2/D3/S1 actor numerical comparison in an isolated diagnostic job.

Never called by the training entry. The caller must supply a genuinely restored
actor and its original, already prepared physical-eight microbatch on every
rank. This helper performs forward/backward only: no Adam, clipping, scheduler,
sampling, data consumption, checkpoint write or retry. A failed/OOM/distributed
worker must be discarded with the independent diagnostic process.
"""
import copy
import math
import random
import time
from types import MappingProxyType


THRESHOLDS = MappingProxyType(dict(log_probability_max_abs=1e-3,
    log_probability_rms=1e-4, loss_abs=1e-5, gradient_relative_l2=1e-3,
    gradient_cosine_min=0.99999))
_LOG_ATTRIBUTES = ('_selected_projection_logged', '_actual_precision_logged', 'last_forward_precision')
_MISSING = object()


def _policy_loss_function():
    from verl.trainer.ppo import core_algos
    return core_algos.compute_policy_loss


def _rng(device):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=copy.deepcopy(np.random.get_state()),
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=(torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None))


def _restore_rng(saved, device):
    import numpy as np
    import torch
    random.setstate(saved['python']); np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch_cpu'])
    if saved['torch_cuda'] is not None:
        torch.cuda.set_rng_state(saved['torch_cuda'], device=device)


def _restore_attributes(actor, saved):
    for name, value in saved.items():
        if value is _MISSING:
            if hasattr(actor, name): delattr(actor, name)
        else:
            setattr(actor, name, copy.deepcopy(value))


def _clear_gradients(module):
    for parameter in module.parameters(): parameter.grad = None


def _environment(module, expected_world_size, group):
    import torch
    import torch.distributed as dist
    if type(expected_world_size) is not int or expected_world_size not in (1, 4):
        raise ValueError('Only actual four-rank diagnostics or the explicit single-rank CPU fixture are supported')
    parameters = list(module.named_parameters())
    if not parameters:
        raise ValueError('Qualification requires actual actor parameters')
    device = next((p.device for _, p in parameters if p.device.type == 'cuda'), parameters[0][1].device)
    initialized = dist.is_available() and dist.is_initialized()
    size = dist.get_world_size(group) if initialized else 1
    if size != expected_world_size:
        raise ValueError('Qualification must use the caller-bound actual rank count')
    if initialized and str(dist.get_backend(group)).lower() == 'nccl' and device.type != 'cuda':
        raise ValueError('Diagnostic caller must load actual FSDP actor parameter shards onto its CUDA device first')
    communication = device if initialized and str(dist.get_backend(group)).lower() == 'nccl' else torch.device('cpu')
    return parameters, device, communication, initialized, size


def _barrier(initialized, group):
    if initialized:
        import torch.distributed as dist
        dist.barrier(group=group)


def _reduce(values, *, operation, environment, group):
    import torch
    import torch.distributed as dist
    _, _, communication, initialized, _ = environment
    vector = torch.tensor(values, dtype=torch.float64, device=communication)
    if initialized:
        dist.all_reduce(vector, op=(dist.ReduceOp.SUM if operation == 'sum' else dist.ReduceOp.MAX), group=group)
    return vector.cpu().tolist()


def _gradient_snapshot(parameters):
    """Copy every actual local FSDP gradient shard once; never gather full model."""
    return [(name, tuple(parameter.shape), str(parameter.dtype),
             None if parameter.grad is None else parameter.grad.detach().to(device='cpu', copy=True))
            for name, parameter in parameters]


def _gradient_comparison(reference, parameters, environment, group, *, chunk_elements=1048576):
    import torch
    if len(reference) != len(parameters):
        raise ValueError('Actual parameter/gradient layout changed during qualification')
    ref_sq = got_sq = diff_sq = dot = maximum = 0.0
    elements = present = absent = bad = 0
    for (name, shape, dtype, saved), (current_name, parameter) in zip(reference, parameters):
        if (current_name, tuple(parameter.shape), str(parameter.dtype)) != (name, shape, dtype):
            raise ValueError('Actual parameter order/shape/dtype changed during qualification')
        current = parameter.grad
        if (saved is None) != (current is None):
            raise ValueError('Actual gradient presence changed during qualification: ' + name)
        if saved is None:
            absent += 1
            continue
        present += 1
        first = saved.reshape(-1); second = current.detach().reshape(-1)
        if first.shape != second.shape:
            raise ValueError('Actual local gradient shard shape changed')
        elements += first.numel()
        for begin in range(0, first.numel(), chunk_elements):
            end = begin + chunk_elements
            before = first[begin:end].to(dtype=torch.float64)
            after = second[begin:end].to(device='cpu', dtype=torch.float64)
            finite = bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all())
            if not finite:
                bad += 1
                continue
            difference = after - before
            ref_sq += float(before.square().sum()); got_sq += float(after.square().sum())
            diff_sq += float(difference.square().sum()); dot += float((before * after).sum())
            maximum = max(maximum, float(difference.abs().max()))
    sums = _reduce([ref_sq, got_sq, diff_sq, dot, elements, present, absent, bad],
                   operation='sum', environment=environment, group=group)
    maximum = _reduce([maximum], operation='max', environment=environment, group=group)[0]
    ref_sq, got_sq, diff_sq, dot, elements, present, absent, bad = sums
    finite = bad == 0 and all(math.isfinite(value) for value in sums)
    nonzero = finite and ref_sq > 0 and got_sq > 0
    return dict(reference_l2=math.sqrt(ref_sq) if finite else None,
                actual_l2=math.sqrt(got_sq) if finite else None,
                difference_l2=math.sqrt(diff_sq) if finite else None,
                relative_l2=(math.sqrt(diff_sq / ref_sq) if nonzero else None),
                cosine=(max(-1.0, min(1.0, dot / math.sqrt(ref_sq * got_sq))) if nonzero else None),
                max_absolute_error=maximum if finite else None,
                global_compared_gradient_elements=int(elements),
                global_parameter_shards_with_grad=int(present),
                global_parameter_shards_without_grad=int(absent),
                all_finite=finite, nonzero_reference_and_actual=nonzero,
                scope='All actual local FSDP gradient shards concatenated across all ranks; no sampled gradient subset')


def _log_probability_comparison(reference, current, environment, group):
    import torch
    import torch.distributed as dist
    before, after = reference.double(), current.double()
    if before.shape != after.shape:
        raise ValueError('Effective authored-token log-probability count changed')
    difference = (after - before).abs()
    finite = bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all())
    packet = dict(finite=finite, absolute_errors=difference.tolist() if finite else [])
    if environment[3]:
        packets = [None] * environment[4]
        dist.all_gather_object(packets, packet, group=group)
    else:
        packets = [packet]
    valid = all(value['finite'] for value in packets)
    all_errors = [value for packet in packets for value in packet['absolute_errors']]
    errors = torch.tensor(all_errors, dtype=torch.float64)
    valid = valid and errors.numel() > 0
    return dict(global_authored_tokens=int(errors.numel()) if valid else None,
                max_absolute_error=float(errors.max()) if valid else None,
                root_mean_square_error=float(errors.square().mean().sqrt()) if valid else None,
                p99_absolute_error=float(torch.quantile(errors, .99)) if valid else None,
                all_finite=valid, exact_global_p99=True,
                scope='Every authored loss-mask token across all ranks; padding/tool-only positions excluded')


def _passing(comparison):
    lp, gradient = comparison['log_probabilities'], comparison['gradients']
    checks = dict(
        log_probability_max_abs=lp['all_finite'] and lp['max_absolute_error'] <= THRESHOLDS['log_probability_max_abs'],
        log_probability_rms=lp['all_finite'] and lp['root_mean_square_error'] <= THRESHOLDS['log_probability_rms'],
        loss_abs=(comparison['loss']['all_finite'] and
                  comparison['loss']['max_rank_absolute_error'] <= THRESHOLDS['loss_abs']),
        gradient_relative_l2=(gradient['nonzero_reference_and_actual'] and
                              gradient['relative_l2'] <= THRESHOLDS['gradient_relative_l2']),
        gradient_cosine=(gradient['nonzero_reference_and_actual'] and
                         gradient['cosine'] >= THRESHOLDS['gradient_cosine_min']))
    return dict(accepted=all(checks.values()), checks=checks,
                failed_checks=[name for name, accepted in checks.items() if not accepted])


def qualify_sparse_policy(actor, micro_batch, temperature, *, group=None, expected_world_size=4):
    """Run actual actor D1/D2/D3/S1, returning only JSON-serializable scalars.

    The original RNG state is restored before every run and on all exits.
    Errors propagate; no failed worker is retried or used for real training.
    ``expected_world_size=1`` is for the tiny CPU fixture only. Production must
    use four restored FSDP actors and the actual original physical-eight batch.
    """
    import torch
    from search_g1.logical_micro_loss import logical_policy_loss
    module = actor.actor_module
    group = getattr(module, 'process_group', None) if group is None else group
    environment = _environment(module, expected_world_size, group)
    parameters, device, _, initialized, size = environment
    config = actor.config
    if (config.ppo_micro_batch_size != 8 or config.ppo_mini_batch_size != 32
            or getattr(config, 'ppo_logical_micro_batch_size', None) != 4
            or config.entropy_coeff != 0 or float(config.kl_loss_coef) != 0
            or not config.state_masking or not config.policy_safety.enabled):
        raise ValueError('Qualification requires unchanged micro8/logical4/accumulation4 zero-KL/entropy actor recipe')
    if micro_batch['responses'].shape[0] != 8:
        raise ValueError('Qualification requires the original complete eight-row physical microbatch')
    mask = micro_batch['loss_mask']; old = micro_batch['old_log_probs']; advantage = micro_batch['advantages']
    if mask.shape != old.shape or mask.shape != advantage.shape or not bool(mask.bool().any()):
        raise ValueError('Qualification requires original aligned nonempty policy targets')
    if any(parameter.grad is not None for _, parameter in parameters):
        raise ValueError('Restored diagnostic actor must start with no pending gradients')
    if device.type == 'cuda' and not getattr(actor, '_selected_policy_logits', False):
        # In the diagnostic namespace the optional hook is installed by its
        # explicit sparse manifest. D runs toggle only the per-actor dispatch.
        raise ValueError('Sparse diagnostic actor hook must already be installed explicitly')
    compute_policy_loss = _policy_loss_function()
    saved_rng = _rng(device)
    saved_attributes = {name: copy.deepcopy(getattr(actor, name)) if hasattr(actor, name) else _MISSING
                        for name in _LOG_ATTRIBUTES}
    selected_flag = getattr(actor, '_selected_policy_logits', _MISSING)
    modes = [(child, child.training) for child in module.modules()]
    baseline_gradients = baseline_lp = baseline_loss = None
    comparisons = {}; runs = []; new = loss = pg_loss = clipfrac = ppo_kl = None
    started = time.time()
    try:
        for label in ('D1', 'D2', 'D3', 'S1'):
            _barrier(initialized, group)
            _clear_gradients(module)
            _restore_attributes(actor, saved_attributes)
            module.train(True)
            _restore_rng(saved_rng, device)
            actor._selected_policy_logits = label == 'S1'
            _, new = actor._forward_micro_batch(micro_batch, temperature=temperature, compute_entropy=False)
            pg_loss, clipfrac, ppo_kl = logical_policy_loss(compute_policy_loss,
                logical_micro_batch_size=4, old_log_prob=old, log_prob=new,
                advantages=advantage, eos_mask=mask, cliprange=config.clip_ratio,
                dual_clip=float(config.policy_safety.dual_clip))
            loss = pg_loss / 4
            loss.backward()
            # Copy only actual authored LPs; gradient snapshots retain ALL local
            # shards rather than hidden/output samples or a gradient sketch.
            observed_lp = new.detach()[mask.bool()].to(device='cpu', copy=True)
            observed_loss = float(loss.detach().cpu())
            if label == 'D1':
                baseline_gradients = _gradient_snapshot(parameters)
                baseline_lp, baseline_loss = observed_lp, observed_loss
                baseline_norm = _gradient_comparison(baseline_gradients, parameters, environment, group)
                runs.append(dict(run=label, selected_policy_logits=False, local_loss=observed_loss,
                                 local_authored_tokens=observed_lp.numel(), baseline_gradient_norm=baseline_norm))
            else:
                finite_loss = math.isfinite(baseline_loss) and math.isfinite(observed_loss)
                delta = abs(observed_loss - baseline_loss) if finite_loss else 0.0
                global_loss = _reduce([delta, int(not finite_loss)], operation='max', environment=environment, group=group)
                comparison = dict(reference='D1', actual=label,
                    log_probabilities=_log_probability_comparison(baseline_lp, observed_lp, environment, group),
                    loss=dict(max_rank_absolute_error=global_loss[0] if global_loss[1] == 0 else None,
                              all_finite=global_loss[1] == 0),
                    gradients=_gradient_comparison(baseline_gradients, parameters, environment, group))
                comparison.update(_passing(comparison))
                comparisons['D1_' + label] = comparison
                runs.append(dict(run=label, selected_policy_logits=label == 'S1',
                                 local_loss=observed_loss, local_authored_tokens=observed_lp.numel()))
            new = loss = pg_loss = clipfrac = ppo_kl = None
        _barrier(initialized, group)
        result = dict(format='actual_actor_sparse_policy_D1_D2_D3_S1_v1',
            accepted=all(value['accepted'] for value in comparisons.values()),
            thresholds=dict(THRESHOLDS), comparisons=comparisons,
            dense_repeat_noise={key:value for key, value in comparisons.items() if key != 'D1_S1'},
            local_runs=runs, world_size=size,
            rank=(torch.distributed.get_rank(group) if initialized else 0),
            physical_micro_per_rank=8, logical_micro_per_rank=4, gradient_accumulation=4,
            temperature=float(temperature), clip_ratio=float(config.clip_ratio),
            dual_clip=float(config.policy_safety.dual_clip), all_local_gradient_elements_compared=True,
            optimizer_calls=0, scheduler_calls=0, gradient_clipping_calls=0,
            same_rng_start_for_every_forward=True, normal_training_auto_call=False,
            started_unix=started, completed_unix=time.time(),
            scope='One original restored-actor micro8 only; not proof for all training batches or whole-outer throughput')
        return result
    finally:
        # No collective recovery/continuation after an exception. torchrun's
        # caller must terminate the independent diagnostic allocation/process.
        new = loss = pg_loss = clipfrac = ppo_kl = baseline_gradients = baseline_lp = None
        _clear_gradients(module)
        if selected_flag is _MISSING:
            if hasattr(actor, '_selected_policy_logits'): delattr(actor, '_selected_policy_logits')
        else:
            actor._selected_policy_logits = selected_flag
        _restore_attributes(actor, saved_attributes)
        for child, mode in modes: child.training = mode
        _restore_rng(saved_rng, device)
