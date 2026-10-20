"""Isolated native two-pass updater for exact DP G10/G20 objectives.

No scheduler, default-runner patch, executable loader injection or automatic
GPU launch. The bounded native replay is explicit and retains inherited Adam,
EMA and RNG. A separate typed preparation binds any future authentic G20 run.
"""
import time
import torch
from . import probe_g10_train as native
from . import probe_ablation_train as u
from .exact_uniform_g5_loss import validate_source, weights
from .exact_uniform_g5_dp_loss import exact_uniform_g5_dp_loss

MODE = 'exact-uniform-all-real-G5-three-state-DP'


def bounded_exact_dp_update(model, ema, optimizer, reader, groups, device, *,
        loss_kwargs, combination_chunk=256, micro_trajectories=5, ema_decay=.99):
    size = validate_source(groups)
    weights(**loss_kwargs)
    if set(loss_kwargs) - {'trajectory_weight', 'three_event_weight'}:
        raise ValueError('Only explicit trajectory and three-event weights accepted')
    if type(micro_trajectories) is not int or micro_trajectories < 1 or not 0 <= float(ema_decay) < 1:
        raise ValueError('Positive micro and valid EMA required')
    if any(g['lr'] != 5e-6 or g['weight_decay'] != 0 for g in optimizer.param_groups):
        raise ValueError('Inherited Adam LR/weight decay changed')
    before, started = u.steps(optimizer), time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tids = [t for g in groups for t in g['trajectory_ids']]
    micros, parts, native_bytes = [], [], 0
    for start in range(0, len(tids), micro_trajectories):
        ids = tids[start:start + micro_trajectories]
        data = reader.get(ids, device)
        buffers = u.buffers(model)
        native_bytes += sum(v.numel() * v.element_size() for v in buffers.values())
        if native_bytes > 256 * 1024 ** 2:
            raise ValueError('Native replay snapshots exceed256MiB')
        micros.append((start, ids, u.rng_state(device), buffers))
        out = native._forward(model, data, device)
        if out.numel() != len(ids) or not bool(torch.isfinite(out).all()):
            raise ValueError('Incomplete/nonfinite actual trajectory logits')
        parts.append(out.detach().clone())
        del out, data
    after_rng, after_native = u.rng_state(device), u.buffers(model)
    logits = torch.cat(parts).reshape(len(groups), size).requires_grad_(True)
    del parts
    result = exact_uniform_g5_dp_loss(logits, groups,
        combination_chunk=combination_chunk, **loss_kwargs)
    derivative = torch.autograd.grad(result['loss'], logits)[0].detach().reshape(-1)
    expected = logits.detach().reshape(-1)
    if not bool(torch.isfinite(derivative).all()):
        raise ValueError('Nonfinite exact DP dL/dlogits')
    for start, ids, rng, buffers in micros:
        data = reader.get(ids, device)
        u.restore_rng(rng, device)
        u.restore_buffers(model, buffers)
        out = native._forward(model, data, device)
        if not torch.equal(out.detach(), expected[start:start + len(ids)]):
            raise ValueError('Native two-pass logits differ; no Adam update accepted')
        torch.autograd.backward(out, derivative[start:start + len(ids)].to(out.dtype))
        del out, data
    u.restore_rng(after_rng, device)
    u.restore_buffers(model, after_native)
    params = list(model.parameters())
    if not params or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError('Missing/nonfinite exact DP parameter gradient')
    optimizer.step()
    if u.steps(optimizer) != before + 1:
        raise ValueError('Exactly one original-question Adam update required')
    with torch.no_grad():
        for target, value in zip(ema.parameters(), params):
            target.mul_(float(ema_decay)).add_(value, alpha=1 - float(ema_decay))
    confusion = result['question_weighted_confusion']
    tp, fn, fp, tn = [confusion[k] for k in ('tp', 'fn', 'fp', 'tn')]
    tpr, tnr = (tp / (tp + fn) if tp + fn else None), (tn / (tn + fp) if tn + fp else None)
    return dict(loss=float(result['loss'].detach()),
        components={k: float(v) for k, v in result['components'].items()},
        grouping=dict(mode=MODE, source_group_size=size,
            combinations_per_question=result['combinations_per_question'],
            views_per_question=result['combinations_per_question'], view_weight=result['view_weight'],
            per_question_total_weight_exactly_one=True, all_combinations_exactly_once=True,
            weighted_mixed_fraction=result['weighted_Mixed_fraction'], source_groups_sha256=u.digest(groups)),
        original_questions=len(groups), unique_real_trajectories=len(tids),
        virtual_views=result['virtual_views'], view_trajectory_exposures=5 * result['virtual_views'],
        actual_forward_trajectory_exposures=2 * len(tids), forward_passes=2, backward_passes=1,
        micro_trajectories=micro_trajectories, maximum_retained_transformer_graph_microbatches=1,
        native_replay_snapshot_bytes=native_bytes, maximum_resident_pattern_rows=0,
        maximum_resident_G5_rows=result['maximum_resident_G5_rows'],
        maximum_resident_event_states=result['maximum_resident_event_states'],
        retained_loss_gradient_elements=result['retained_loss_gradient_elements'],
        combination_chunk=combination_chunk, event_evaluation=result['event_evaluation'],
        gradient_strategy='two_pass_same_micro_RNG_native_buffer_DP_chunked_first_order_logit_VJP',
        recomputed_logits_bit_identical=True, monolithic_bitexact_claim=False, first_order_only=True,
        inter_view_pairs_allocated=0, per_view_python_identity_metadata_allocated=0,
        denominators=result['denominators'], question_weighted_confusion=confusion,
        mixed_recall=tpr, nonmixed_recall=tnr,
        mixed_balanced_accuracy=(tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        seconds=time.monotonic() - started, actor_updates=0, supports_G20=size == 20,
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
