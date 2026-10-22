"""One real own-G5 per distinct question, including the balanced68 tail."""
import gc
import math
import time
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from . import mixed_group_checkpoint as ck, probe_ablation_train as u

MICRO_QUESTIONS = (128, 64, 32)


def selected_views(batch, lookup):
    qids, views = batch['source_qids'], batch['grouping']['groups']
    n = len(qids)
    if n not in (128, 68) or len(set(qids)) != n or len(views) != n:
        raise ValueError('One distinct own-G5 per question, B128 or finalB68 required')
    for qid, role, view in zip(qids, batch['roles'], views):
        group = lookup[qid]
        indices = view['source_indices']
        if (view['qid'] != qid or len(indices) != 5 or len(set(indices)) != 5
            or any(type(i) is not int or not 0 <= i < 10 for i in indices)
            or view['trajectory_ids'] != [group['trajectory_ids'][i] for i in indices]
            or view['labels'] != [group['labels'][i] for i in indices]
            or view['K'] != sum(view['labels'])
            or view['true_mixed'] != (0 < sum(view['labels']) < 5)
            or view['true_mixed'] != (role == 'mixed')
            or view['weight'] != 1. or view['weight_denominator'] != 1):
            raise ValueError('Selected G5 identities, real EM or question weight differs')
    if sum(view['true_mixed'] for view in views) != n // 2:
        raise ValueError('Actual selected real G5 classes must be balanced')
    return views


def _gradient_attempt(model, native, head, reader, lookup, batch, device, micro, frozen):
    qids = batch['source_qids']
    views = selected_views(batch, lookup)
    n = len(qids)
    model.train()
    native.zero_grad(set_to_none=False)
    head.zero_grad(set_to_none=False)
    losses_gpu = torch.empty(n, device=device, dtype=torch.float32)
    logits_gpu = torch.empty_like(losses_gpu)
    targets = torch.tensor([float(v['true_mixed']) for v in views], device=device)
    for start in range(0, n, micro):
        part = views[start:start+micro]
        ids = [tid for view in part for tid in view['trajectory_ids']]
        data = reader.get(ids, device)
        if (data.get('capture_layer', 26) != 26 or data['trajectory_ids'] != ids
            or data['qids'] != [view['qid'] for view in part for _ in range(5)]):
            raise ValueError('Actual selected own-G5/layer26 features differ')
        with (torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()):
            with (torch.no_grad() if frozen else nullcontext()):
                encoded = model.encode_trajectories(data)
            grouped = {key:value.reshape(len(part), 5, *value.shape[1:]) for key,value in encoded.items()}
            logits = model.group_logits(grouped).float()
        if logits.shape != (len(part),):
            raise ValueError('Exactly one direct Mixed logit per own-G5 required')
        per = F.binary_cross_entropy_with_logits(logits, targets[start:start+len(part)], reduction='none')
        (per.sum() / n).backward()
        losses_gpu[start:start+len(part)] = per.detach()
        logits_gpu[start:start+len(part)] = logits.detach()
        del data, encoded, grouped, logits, per
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in trainable):
        raise ValueError('Missing native/head gradient')
    native_params = [p for p in model.block.parameters() if p.requires_grad]
    head_params = [p for p in model.group_head.parameters() if p.requires_grad]
    native_l2 = torch.zeros((), device=device) if frozen else torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float()) for p in native_params]))
    head_l2 = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float()) for p in head_params]))
    finite = torch.stack([torch.isfinite(p.grad).all() for p in trainable]).all() & torch.isfinite(losses_gpu).all() & torch.isfinite(logits_gpu).all()
    clip = torch.nn.utils.clip_grad_norm_(trainable, 1.)
    predicted, truth = logits_gpu.ge(0), targets.bool()
    confusion = torch.stack(((predicted & truth).sum(), ((~predicted) & (~truth)).sum(),
                            (predicted & (~truth)).sum(), ((~predicted) & truth).sum())).float()
    packed = torch.cat((losses_gpu, torch.stack((native_l2, head_l2, clip, finite.float())), confusion)).detach().cpu().tolist()
    losses = packed[:n]
    nl2, hl2, clipped, isfinite = packed[n:n+4]
    if (isfinite != 1. or not math.isfinite(nl2) or (nl2 != 0 if frozen else nl2 <= 0)
        or not math.isfinite(hl2) or hl2 <= 0 or not math.isfinite(clipped)):
        raise ValueError('Nonfinite/zero native/head gradient or Mixed BCE')
    return losses, nl2, hl2, clipped, tuple(int(x) for x in packed[n+4:n+8])


def direct_update_unique(model, ema, native, head, reader, lookup, sampler, device,
                         ema_decay=.99, *, micro_questions=128, phase='phase2_joint_low_lr'):
    if micro_questions not in MICRO_QUESTIONS or phase not in ('phase2_joint_low_lr', 'phase3_head_only'):
        raise ValueError('Unknown physical microbatch or phase')
    frozen = phase == 'phase3_head_only'
    if any(p.requires_grad == frozen for p in model.block.parameters()):
        raise ValueError('Actual native trainability differs')
    if any(g['lr'] != 5e-7 for opt in (native, head) for g in opt.param_groups):
        raise ValueError('Both new epochs use LR5e-7')
    started = time.monotonic()
    state = sampler.state_dict()
    rng, buffers = u.rng_state(device), u.buffers(model)
    native_before, head_before = ck.optimizer_steps(native.state_dict(), True), ck.optimizer_steps(head.state_dict(), True)
    if native_before != (3898 if frozen else sampler.accepted_updates) or head_before != sampler.accepted_updates:
        raise ValueError('Actual sampler/twoAdam clocks differ')
    batch = sampler.next_batch()
    initial_batch = batch
    selected_views(batch, lookup)
    reader.accept_batch(batch)
    attempted, retries = [], []
    candidates = MICRO_QUESTIONS[MICRO_QUESTIONS.index(micro_questions):]
    for attempt, micro in enumerate(candidates):
        attempted.append(micro)
        try:
            losses, nl2, hl2, clipped, confusion = _gradient_attempt(model, native, head, reader, lookup, batch, device, micro, frozen)
        except torch.OutOfMemoryError as error:
            message = str(error)[:1000]
            error.__traceback__ = None
            native.zero_grad(set_to_none=False);head.zero_grad(set_to_none=False)
            u.restore_buffers(model, buffers);u.restore_rng(rng, device);sampler.load_state_dict(state)
            reader.reset_prefetch();gc.collect()
            if device.type == 'cuda':torch.cuda.empty_cache()
            if ck.optimizer_steps(native.state_dict(), True) != native_before or ck.optimizer_steps(head.state_dict(), True) != head_before:
                raise ValueError('Pre-Adam OOM unexpectedly mutated optimizer')
            retries.append(dict(micro_questions=micro, error=message, phase='before_either_Adam', identical_own_G5_retried=True))
            if attempt + 1 == len(candidates):raise
            reader.prepare(sampler);batch = sampler.next_batch()
            if batch != initial_batch:raise ValueError('OOM retry changed actual questions/G5')
            reader.accept_batch(batch)
            continue
        break
    # No retry after either optimizer can mutate: raw gradients first, then EMA.
    if not frozen:native.step()
    head.step()
    with torch.no_grad():
        for raw, average in zip(model.parameters(), ema.parameters()):
            average.mul_(ema_decay).add_(raw, alpha=1-ema_decay)
    sampler.observe(batch, losses)
    native_steps, head_steps = ck.optimizer_steps(native.state_dict(), True), ck.optimizer_steps(head.state_dict(), True)
    if native_steps != native_before + int(not frozen) or head_steps != head_before+1 or sampler.accepted_updates != head_steps:
        raise ValueError('Exactly one accepted raw/twoAdam/EMA/sampler update required')
    tp, tn, fp, fn = confusion
    n = len(batch['source_qids'])
    loss = sum(losses) / n
    return dict(binary_Mixed_BCE=loss, loss=loss, questions=n, effective_batch_questions=n,
        real_native_trajectories=5*n, views=n, mixed_views=n//2, nonmixed_views=n//2,
        per_question_view_total_weight=1., micro_questions=micro,
        native_forward_calls=math.ceil(n/micro), gradient_accumulation_microbatches=math.ceil(n/micro),
        native_gradient_L2=nl2, group_head_gradient_L2=hl2, gradient_norm_before_clip=clipped,
        native_steps=native_steps, head_steps=head_steps, native_frozen=frozen,
        native_Adam_updates_per_global_update=int(not frozen), head_Adam_updates_per_global_update=1,
        EMA_updates_per_global_update=1, native_lr=5e-7, head_lr=5e-7, phase=phase,
        confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn), mixed_recall=tp/(n/2),
        nonmixed_recall=tn/(n/2), mixed_balanced_accuracy=(tp+tn)/n,
        attempted_micro_questions=attempted, pre_Adam_OOM_retries=retries,
        seconds=time.monotonic()-started, batch_token=batch['token'],
        source_qids_sha256=u.digest(batch['source_qids']), sampler_epoch=batch['before']['epoch'],
        sampler_state_sha256=u.digest(sampler.state_dict()), trajectory_loss_computed=False,
        trajectory_accuracy_computed=False, updater_variant='one_unique_question_one_real_G5_v1'), batch
