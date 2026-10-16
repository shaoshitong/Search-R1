"""Fresh joint trajectory/G5 fitting on one retained question batch per epoch.

Only this new run's preceding complete checkpoint may be restored. Whole G5
questions stay on one rank in one update. The100 updates alternate255/260
trajectories, averaging the authorized global batch256 without duplicates.
"""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from reproduction.prefix20_online.features import REVISION, canonical, once, read, remote, sha
from reproduction.prefix20_online.launch import publish
from reproduction.prefix20_online.train_epoch import append, optimizer_learning_rates
from reproduction.prefix_only_correctness_ab.train_ddp import actual_adam_steps
from reproduction.prefix20_question_replay.features import FeatureCache
from reproduction.prefix20_question_replay.train_epoch import group_predictions, validate_datasets

BATCH_SEMANTICS = 'average255_260_complete_G5'
DATA_SCHEDULE = list(range(1, 71)) + list(range(1, 31))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def cosine_learning_rate(update_index, total_updates=10000, start=3e-5, end=3e-6):
    """Zero-based actual update clock, including both exact endpoint rates."""
    if (type(update_index) is not int or type(total_updates) is not int or total_updates < 2
            or not 0 <= update_index < total_updates or not 0 < end <= start
            or not all(math.isfinite(rate) for rate in (start, end))):
        raise ValueError('Invalid actual-update cosine schedule coordinate')
    if update_index == 0:
        return float(start)
    if update_index == total_updates - 1:
        return float(end)
    return float(end + .5 * (start - end) * (1 + math.cos(math.pi * update_index / (total_updates - 1))))


def question_groups(index, ids, expected_questions):
    """Build each real question's five aligned IDs/labels, never synthetic groups."""
    if len(ids) != expected_questions * 5 or len(set(ids)) != len(ids):
        raise ValueError('Require complete unique own-G5 trajectory identities')
    groups = defaultdict(list)
    for tid in ids:
        row = index[tid]
        if type(row['strict_em']) is not int or row['strict_em'] not in (0, 1):
            raise ValueError('Each trajectory requires its own strictEM binary label')
        groups[str(row['qid'])].append((tid, row['strict_em'], row['question_hash']))
    if len(groups) != expected_questions:
        raise ValueError('Question count differs from the complete retained split')
    result = []
    for qid, rows in sorted(groups.items()):
        rows.sort(key=lambda row: str(row[0]))
        if len(rows) != 5 or len({row[2] for row in rows}) != 1:
            raise ValueError('Incomplete G5 or conflicting question identity')
        result.append(dict(qid=qid, question_hash=rows[0][2],
                           trajectory_ids=[row[0] for row in rows], labels=[row[1] for row in rows]))
    return result


def natural_question_plan(question_count, seed, epoch, world=8):
    """100 complete-group updates, with every rank seeing640 groups exactly once."""
    if question_count != 5120 or world != 8 or type(epoch) is not int or epoch < 1:
        raise ValueError('Require5120 retained questions and eight ranks')
    shuffled = list(range(question_count))
    random.Random(seed + epoch).shuffle(shuffled)
    plan, offset, extras_start = [], 0, 0
    for step in range(100):
        size = 52 if (step + 1) % 5 == 0 else 51
        extras = size - world * 6
        counts = [6] * world
        for j in range(extras):
            counts[(extras_start + j) % world] += 1
        plan.append(dict(question_indices=shuffled[offset:offset + size],
                         global_questions=size, rank_questions=counts, extras_start=extras_start))
        extras_start = (extras_start + extras) % world
        offset += size
    if offset != question_count or any(sum(batch['rank_questions'][r] for batch in plan) != 640 for r in range(world)):
        raise RuntimeError('Complete-group plan did not cover the balanced natural epoch')
    return plan


def rank_question_slice(batch, rank, world=8):
    counts = batch['rank_questions']
    if (world != 8 or not 0 <= rank < world or len(counts) != world
            or any(count not in (6, 7) for count in counts)
            or sum(counts) != batch['global_questions']
            or len(batch['question_indices']) != batch['global_questions']):
        raise ValueError('Invalid complete-G5 rank partition')
    start = sum(counts[:rank])
    return batch['question_indices'][start:start + counts[rank]]


def ddp_mean_scale(local_questions, global_questions, world=8):
    """DDP averages gradients: weight local means by their true question count."""
    if world != 8 or local_questions not in (6, 7) or global_questions not in (51, 52):
        raise ValueError('Unexpected complete-group batch size')
    return world * local_questions / global_questions


def batch_to_device(batch, device):
    """Move tensor data only; authentic segment/trajectory metadata stays intact."""
    import torch
    return {key: value.detach().to(device).contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def evaluate_predictions(predictions, index, expected_questions, group_weight=1.0):
    """Post-update mixed accuracy and the same two stable training loss terms."""
    import torch
    from .loss import joint_loss
    rows = group_predictions(predictions, index, expected_questions)
    by_id = {tid: float(logit) for tid, logit, _label in predictions}
    logits = torch.tensor([[by_id[tid] for tid in row['trajectory_ids']] for row in rows], dtype=torch.float64)
    labels = torch.tensor([row['strict_em'] for row in rows], dtype=torch.float64)
    losses = joint_loss(logits, labels, group_weight=group_weight)
    tp = sum(row['mixed_label'] == 1 and row['predicted_mixed'] == 1 for row in rows)
    fp = sum(row['mixed_label'] == 0 and row['predicted_mixed'] == 1 for row in rows)
    fn = sum(row['mixed_label'] == 1 and row['predicted_mixed'] == 0 for row in rows)
    tn = len(rows) - tp - fp - fn
    positive_recall = tp / (tp + fn) if tp + fn else None
    negative_recall = tn / (tn + fp) if tn + fp else None
    metrics = dict(count=len(rows), mixed_count=tp + fn, nonmixed_count=tn + fp,
                   accuracy=(tp + tn) / len(rows), tp=tp, fp=fp, fn=fn, tn=tn,
                   balanced_accuracy=(positive_recall + negative_recall) / 2
                       if positive_recall is not None and negative_recall is not None else None,
                   mixed_recall=positive_recall, nonmixed_recall=negative_recall,
                   threshold=.5, formula='1-prod(p_j)-prod(1-p_j)', true_mixed='0<sum(own_strict_em)<5',
                   mean_q=math.fsum(row['mixed_probability'] for row in rows) / len(rows),
                   trajectory_bce=float(losses['trajectory_bce']), group_bce=float(losses['group_bce']),
                   total_loss=float(losses['loss']), group_weight=group_weight)
    return metrics, rows


def validate_saved_checkpoint(saved, receipt, binding, epoch):
    """No source-probe transition exists: restore only this new run's previous epoch."""
    expected = (epoch - 1) * 100
    if saved.get('epoch') != epoch - 1 or receipt.get('epoch') != epoch - 1:
        raise ValueError('Restore only the immediately preceding complete joint epoch')
    if (saved.get('binding') != binding or saved.get('binding_sha256') != digest(binding)
            or receipt.get('binding_sha256') != digest(binding)):
        raise ValueError('Joint run identity or scientific recipe differs')
    if (saved['actual_adam_steps_per_rank'] != [expected] * 8
            or receipt['actual_adam_steps_per_rank'] != [expected] * 8
            or saved.get('fresh_run_initial_adam_steps') != 0):
        raise ValueError('New joint run cannot inherit old probe Adam counters')
    schedule = saved['cosine_schedule']
    if schedule['completed_actual_updates_per_rank'] != expected:
        raise ValueError('Saved cosine clock differs from the actual Adam boundary')
    return expected


def schedule_state(completed, total, start, end):
    return dict(kind='cosine_actual_adam_update_no_warmup', total_actual_updates_per_rank=total,
                completed_actual_updates_per_rank=completed, learning_rate_start=start, learning_rate_end=end,
                last_update_learning_rate=cosine_learning_rate(completed - 1, total, start, end) if completed else None,
                next_update_learning_rate=cosine_learning_rate(completed, total, start, end) if completed < total else None)


def write_checkpoint(out, checkpoint, epoch, epochs, save_every):
    import torch
    partial, latest = out / 'latest.pt.partial', out / 'latest.pt'
    if partial.exists():
        raise FileExistsError('Uncommitted new-run checkpoint requires explicit recovery')
    torch.save(checkpoint, partial)
    with partial.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(partial, latest)
    checkpoint_sha = sha(latest)
    immutable = epoch % save_every == 0 or epoch == epochs
    if immutable:
        os.link(latest, out / ('head-epoch-%04d.pt' % epoch))
    receipt = dict(schema_version=1, epoch=epoch, source_data_epoch=checkpoint['source_data_epoch'],
                   binding_sha256=checkpoint['binding_sha256'], sha256=checkpoint_sha,
                   bytes=latest.stat().st_size, actual_adam_steps_per_rank=checkpoint['actual_adam_steps_per_rank'],
                   new_adam_steps_per_rank=100, fresh_run_initial_adam_steps=0,
                   actual_optimizer_learning_rates=checkpoint['actual_optimizer_learning_rates'],
                   actual_learning_rate=checkpoint['actual_optimizer_learning_rates'][0],
                   cosine_schedule=checkpoint['cosine_schedule'], immutable_saved=immutable, observed_utc=utc())
    temp = out / 'latest.json.partial'
    with temp.open('wb') as handle:
        handle.write(canonical(receipt) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, out / 'latest.json')
    directory = os.open(out, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return receipt


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from reproduction.prefix20_question_replay.model import QuestionPrefixCorrectnessProbe
    from .loss import joint_loss
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-inputs', required=True)
    parser.add_argument('--train-inputs-sha')
    parser.add_argument('--test-cache', required=True)
    parser.add_argument('--test-manifest-sha', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--source-data-epoch', required=True, type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--global-batch', type=int, default=256)
    parser.add_argument('--batch-semantics', default=BATCH_SEMANTICS)
    parser.add_argument('--lr-start', type=float, default=3e-5)
    parser.add_argument('--lr-end', type=float, default=3e-6)
    parser.add_argument('--group-weight', type=float, default=1.0)
    parser.add_argument('--total-optimizer-updates', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--encoder-micro-batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    epoch = args.epoch
    if (args.epochs != 100 or not 1 <= epoch <= 100 or args.source_data_epoch != DATA_SCHEDULE[epoch - 1]
            or args.global_batch != 256 or args.batch_semantics != BATCH_SEMANTICS
            or args.lr_start != 3e-5 or args.lr_end != 3e-6 or args.group_weight != 1.0
            or args.total_optimizer_updates != args.epochs * 100 or args.save_every != 10
            or args.encoder_micro_batch_size < 1):
        raise ValueError('Require the fresh100-epoch joint recipe and confirmed retained data schedule')
    rank, local, world = (int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError('Joint G5 fitting requires eight actual visible GPUs and ranks')
    out, model_dir = remote(args.output), remote(args.model_dir)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    claimed = False
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        test = FeatureCache(args.test_cache, args.test_manifest_sha)
        from .inline_encoder import InlineEncoder
        device = torch.device('cuda', local)
        train = InlineEncoder(args.train_inputs, str(model_dir),
            question_max_tokens=test.question_max_tokens, micro_batch_size=args.encoder_micro_batch_size,
            device=device, attention_implementation=test.spec.get('encoder_attention_implementation', 'sdpa'))
        if args.train_inputs_sha and train.manifest_sha != args.train_inputs_sha:
            raise ValueError('Actual short-input scan SHA differs from the retained source binding')
        if train.manifest.get('epoch') != args.source_data_epoch:
            raise ValueError('Feature cache does not correspond to the confirmed source data batch')
        train_ids, test_ids = validate_datasets(train, test)
        groups = question_groups(train.index, train_ids, 5120)
        plan = natural_question_plan(len(groups), args.seed, epoch, world)
        plan_sha = digest(plan)
        run_id = read(out.parent / 'preparation/launch-config.json')['run_id']
        model = QuestionPrefixCorrectnessProbe(model_dir, question_max_tokens=train.question_max_tokens)
        # The replay class normally prepares zero columns for migration. This
        # new experiment instead initializes a genuinely fresh standard Linear.
        model.output.reset_parameters()
        model = model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start, weight_decay=0)
        binding = dict(schema_version=1, role='fresh_question20_trajectory_and_mixed_joint100', run_id=run_id,
                       encoder_revision=REVISION, encoder_files_sha256=test.spec['encoder_files_sha256'],
                       feature_layer=27, feature_boundary='post_block', prefix_tokens=20, encoder_frozen=True,
                       actor_updates=0, question_hidden_passed_to_head=True,
                       question_max_tokens=train.question_max_tokens, head_max_tokens=model.max_tokens,
                       head_architecture='official_block35_then_flatten_Sx2048_to1',
                       initial_block='official_layer35_full_copy', initial_linear='fresh_nnLinear_reset_parameters',
                       old_probe_imported=False, old_optimizer_imported=False, fresh_run_initial_adam_steps=0,
                       loss='mean_trajectory_BCE_plus_group_weight_times_mean_G5_mixed_BCE',
                       group_weight=args.group_weight, class_weights=None, threshold=.5,
                       mixed_formula='1-prod(p_j)-prod(1-p_j)', true_mixed='0<sum(own_strict_em)<5',
                       sampling='natural_question_no_replacement_complete_G5_per_rank_per_update',
                       global_batch_average=256, actual_global_trajectory_batches=[255, 260],
                       actual_global_question_batches=[51, 52], batch_semantics=BATCH_SEMANTICS,
                       rank_question_batches=[6, 7], world_size=8, accumulation=1,
                       updates_per_epoch=100, rank_questions_per_epoch=640, rank_trajectories_per_epoch=3200,
                       ddp_loss_scale='world_size*local_question_count/global_question_count',
                       train_feature_extraction='inline_per_whole_G5_training_batch_frozen_layer27_postblock',
                       train_feature_storage='per_rank_cpu_ram_for_current_epoch_post_update_eval_only',
                       encoder_micro_batch_size=args.encoder_micro_batch_size,
                       encoder_attention_implementation=test.spec.get('encoder_attention_implementation', 'sdpa'),
                       lr_schedule='cosine_per_actual_adam_update_no_warmup',
                       learning_rate_start=args.lr_start, learning_rate_end=args.lr_end,
                       total_optimizer_updates_per_rank=args.total_optimizer_updates,
                       seed=args.seed, epochs=100, source_data_schedule=DATA_SCHEDULE, save_every=10,
                       test_manifest_sha256=test.manifest_sha, diagnostic_questions=512, diagnostic_trajectories=2560)
        binding_sha = digest(binding)
        rank_bindings = [None] * world
        dist.all_gather_object(rank_bindings, (binding_sha, train.manifest_sha, plan_sha))
        if len(set(rank_bindings)) != 1:
            raise RuntimeError('Rank data, whole-G5 plan, or joint recipe differs')
        epoch_dir = out / 'epochs' / ('epoch-%04d' % epoch)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            epoch_dir.mkdir(parents=True, exist_ok=False)
            if epoch == 1:
                if any((out / name).exists() for name in ('latest.pt', 'latest.json', 'run-claim.json')):
                    raise ValueError('Fresh joint epoch1 cannot import or overwrite an old probe state')
                once(out / 'run-claim.json', dict(binding=binding, binding_sha256=binding_sha, observed_utc=utc()))
            else:
                claim = read(out / 'run-claim.json')
                if claim['binding'] != binding or claim['binding_sha256'] != binding_sha:
                    raise ValueError('Joint epoch cannot change this new run identity or recipe')
        dist.barrier()
        claimed = True
        restored_step = 0
        if epoch > 1:
            receipt = read(out / 'latest.json')
            check = [sha(out / 'latest.pt') == receipt['sha256'] if rank == 0 else None]
            dist.broadcast_object_list(check, src=0)
            if check != [True]:
                raise ValueError('This new run complete checkpoint bytes differ')
            saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
            expected = validate_saved_checkpoint(saved, receipt, binding, epoch)
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor) and key != 'step':
                        state[key] = value.to(device)
            rng = saved['rank_rng'][rank]
            if rng['rank'] != rank:
                raise ValueError('Saved RNG rank identity differs')
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch_cpu'])
            torch.cuda.set_rng_state(rng['torch_cuda'], device)
            restored_step = actual_adam_steps(optimizer)
            if restored_step != expected:
                raise ValueError('Loaded new-run Adam counter differs from its complete boundary')
            last_rate = cosine_learning_rate(restored_step - 1, args.total_optimizer_updates, args.lr_start, args.lr_end)
            if any(rate != last_rate for rate in optimizer_learning_rates(optimizer)):
                raise ValueError('Loaded optimizer LR differs from its actual cosine update clock')
            del saved
        elif actual_adam_steps(optimizer) != 0 or optimizer.state:
            raise ValueError('Fresh initialization must have no inherited Adam moments or steps')
        if rank == 0:
            once(epoch_dir / 'initialization-or-restore.json', dict(epoch=epoch, run_id=run_id,
                 actual_adam_steps=restored_step, fresh_epoch1=epoch == 1, old_probe_imported=False,
                 old_optimizer_imported=False, model_copy_metadata=model.copy_metadata,
                 linear_initialization='fresh_nnLinear_reset_parameters' if epoch == 1 else 'this_run_previous_complete_checkpoint',
                 cosine_schedule=schedule_state(restored_step, args.total_optimizer_updates, args.lr_start, args.lr_end)))
        model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        # Aggregate local unscaled means by real question counts. G=5 is fixed,
        # so this gives the same natural-trajectory mean for the first BCE term.
        totals = [0.0, 0.0, 0.0, 0, 0]
        cached_batches = []
        model.train()
        for step, planned in enumerate(plan, start=1):
            local_groups = [groups[index] for index in rank_question_slice(planned, rank, world)]
            ids = [tid for group in local_groups for tid in group['trajectory_ids']]
            ys = torch.tensor([group['labels'] for group in local_groups], device=device, dtype=torch.float32)
            publish(epoch_dir / ('progress-rank-%d.json' % rank), dict(stage='INLINE_ENCODING_AND_JOINT_TRAINING',
                epoch=epoch, source_data_epoch=args.source_data_epoch, epoch_step=step,
                actual_adam_steps=actual_adam_steps(optimizer), local_trajectories=len(ids),
                global_trajectories=planned['global_questions'] * 5, observed_utc=utc()))
            batch = train.get_batch(ids, device=device)
            # Frozen encoder features are needed once for this training step.
            # Keep just this rank's consumed batches in CPU RAM for the later
            # post-update head diagnostic; no disk feature cache is created.
            cached_batches.append((list(ids), batch_to_device(batch, 'cpu')))
            rate = cosine_learning_rate(restored_step + step - 1, args.total_optimizer_updates, args.lr_start, args.lr_end)
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(batch).reshape(len(local_groups), 5)
            losses = joint_loss(logits.float(), ys, group_weight=args.group_weight)
            scale = ddp_mean_scale(len(local_groups), planned['global_questions'], world)
            backward_loss = losses['loss'] * scale
            finite = torch.isfinite(backward_loss).to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite.item()):
                raise FloatingPointError('Actual nonfinite joint head loss')
            backward_loss.backward()
            finite = torch.stack([torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                                  if parameter.grad is not None]).all().to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite.item()):
                raise FloatingPointError('Actual nonfinite joint head gradient')
            optimizer.step()
            actual = actual_adam_steps(optimizer)
            if actual != restored_step + step or any(value != rate for value in optimizer_learning_rates(optimizer)):
                raise RuntimeError('Actual Adam counter or cosine learning rate differs')
            local_n = len(local_groups)
            totals[0] += float(losses['trajectory_bce'].detach()) * local_n
            totals[1] += float(losses['group_bce'].detach()) * local_n
            totals[2] += float(losses['loss'].detach()) * local_n
            totals[3] += local_n
            totals[4] += len(ids)
            append(epoch_dir / ('rank-%d-updates.jsonl' % rank), dict(observed_utc=utc(), epoch=epoch,
                   source_data_epoch=args.source_data_epoch, epoch_step=step, actual_adam_steps=actual,
                   local_questions=local_n, local_trajectories=len(ids), global_questions=planned['global_questions'],
                   global_trajectories=planned['global_questions'] * 5, ddp_local_mean_scale=scale,
                   trajectory_bce=float(losses['trajectory_bce'].detach()), group_bce=float(losses['group_bce'].detach()),
                   total_loss=float(losses['loss'].detach()), backward_loss=float(backward_loss.detach()),
                   sampler_plan_sha256=plan_sha, actual_optimizer_learning_rates=optimizer_learning_rates(optimizer),
                   actual_learning_rate=rate))
            publish(epoch_dir / ('progress-rank-%d.json' % rank), dict(stage='JOINT_ADAM_UPDATE_COMPLETE',
                epoch=epoch, source_data_epoch=args.source_data_epoch, epoch_step=step, actual_adam_steps=actual,
                local_trajectories=len(ids), global_trajectories=planned['global_questions'] * 5, observed_utc=utc()))
        if totals[3:] != [640, 3200]:
            raise RuntimeError('Rank did not process exactly640 complete questions/3200 own trajectories')
        stats = torch.tensor(totals, device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        if [int(value) for value in stats[3:].tolist()] != [5120, 25600]:
            raise RuntimeError('Joint epoch did not naturally process every retained G5 exactly once')
        train_seconds = time.monotonic() - started
        model.eval()
        split_predictions, split_seconds = {}, {}
        with torch.no_grad():
            for split, cache, ids in (('train', train, train_ids), ('test', test, test_ids)):
                split_started = time.monotonic()
                predictions = []
                if split == 'train':
                    batches = ((batch_ids, batch_to_device(cpu_batch, device)) for batch_ids, cpu_batch in cached_batches)
                else:
                    assigned = ids[rank::world]
                    batches = ((assigned[start:start + 32], cache.get_batch(assigned[start:start + 32], device=device))
                               for start in range(0, len(assigned), 32))
                for batch_ids, batch in batches:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model.module(batch)
                    predictions.extend((tid, float(z), cache.index[tid]['strict_em'])
                                       for tid, z in zip(batch_ids, logits.cpu().tolist()))
                parts = [None] * world
                dist.all_gather_object(parts, predictions)
                combined = [row for part in parts for row in part]
                if (len(combined) != len(ids) or len({row[0] for row in combined}) != len(ids)
                        or {row[0] for row in combined} != set(ids)):
                    raise RuntimeError('Complete post-update ' + split + ' diagnostic coverage differs')
                split_predictions[split] = combined
                split_seconds[split] = time.monotonic() - split_started
        cached_batches.clear()
        counts = [None] * world
        dist.all_gather_object(counts, actual_adam_steps(optimizer))
        if counts != [epoch * 100] * world:
            raise RuntimeError('New joint epoch eight-rank actual Adam boundaries differ')
        rng = dict(rank=rank, python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                   torch_cuda=torch.cuda.get_rng_state(device), actual_adam_steps=actual_adam_steps(optimizer))
        all_rng = [None] * world
        dist.all_gather_object(all_rng, rng)
        if rank == 0:
            train_metrics, train_rows = evaluate_predictions(split_predictions['train'],
                {tid: train.index[tid] for tid in train_ids}, 5120, args.group_weight)
            test_metrics, test_rows = evaluate_predictions(split_predictions['test'],
                {tid: test.index[tid] for tid in test_ids}, 512, args.group_weight)
            schedule = schedule_state(counts[0], args.total_optimizer_updates, args.lr_start, args.lr_end)
            metrics = dict(epoch=epoch, source_data_epoch=args.source_data_epoch, run_id=run_id, observed_utc=utc(),
                train_mixed_group=train_metrics, test_mixed_group=test_metrics,
                train_mixed_group_accuracy=train_metrics['accuracy'], test_mixed_group_accuracy=test_metrics['accuracy'],
                mixed_group_accuracy=test_metrics['accuracy'],
                train_mixed_group_balanced_accuracy=train_metrics['balanced_accuracy'],
                test_mixed_group_balanced_accuracy=test_metrics['balanced_accuracy'],
                metric_semantics='same_post_update_head_complete_train_and_fixed_test_eval_no_grad',
                train_online_trajectory_bce=float(stats[0].item() / 5120),
                train_online_group_bce=float(stats[1].item() / 5120), train_online_total_loss=float(stats[2].item() / 5120),
                online_metric_semantics='natural_pre_own_batch_update_predictions_weighted_by_real_question_count',
                actual_adam_steps_per_rank=counts, new_adam_steps_per_rank=100, fresh_run_initial_adam_steps=0,
                train_questions=5120, train_trajectories=25600, train_unique_trajectories=25600,
                global_batch_average=256, batch_semantics=BATCH_SEMANTICS, accumulation=1,
                train_seconds=train_seconds, train_diagnostic_seconds=split_seconds['train'],
                test_diagnostic_seconds=split_seconds['test'], sampler_plan_sha256=plan_sha,
                train_inputs_scan_sha256=train.manifest_sha, test_manifest_sha256=test.manifest_sha,
                train_feature_extraction='inline_frozen_encoder_per_batch', train_disk_feature_cache_written=False,
                train_post_update_features='same_epoch_consumed_rank_batches_from_cpu_ram',
                actual_optimizer_learning_rates=optimizer_learning_rates(optimizer),
                actual_learning_rate=optimizer_learning_rates(optimizer)[0], cosine_schedule=schedule,
                cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
            once(epoch_dir / 'metrics.json', metrics)
            for split, rows in (('train', train_rows), ('test', test_rows)):
                prefix = 'train-' if split == 'train' else ''
                path = epoch_dir / (prefix + 'diagnostic-predictions.pt')
                torch.save(dict(epoch=epoch, split=split, predictions=split_predictions[split], binding_sha256=binding_sha), path)
                with path.open('rb') as handle:
                    os.fsync(handle.fileno())
                with (epoch_dir / (prefix + 'mixed-question-predictions.jsonl')).open('x', encoding='utf-8') as handle:
                    for row in rows:
                        handle.write(canonical(dict(epoch=epoch, split=split, **row)).decode() + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            checkpoint = dict(schema_version=1, epoch=epoch, source_data_epoch=args.source_data_epoch,
                run_id=run_id, binding=binding, binding_sha256=binding_sha, fresh_run_initial_adam_steps=0,
                model={key: value.detach().cpu() for key, value in model.module.state_dict().items()},
                optimizer=optimizer.state_dict(), rank_rng=all_rng, actual_adam_steps_per_rank=counts,
                train_inputs_scan_sha256=train.manifest_sha, test_manifest_sha256=test.manifest_sha,
                sampler_plan_sha256=plan_sha, actual_optimizer_learning_rates=optimizer_learning_rates(optimizer),
                cosine_schedule=schedule)
            receipt = write_checkpoint(out, checkpoint, epoch, args.epochs, args.save_every)
            once(epoch_dir / 'complete.json', dict(status='FRESH_QUESTION20_G5_JOINT_EPOCH_COMPLETE', **receipt,
                train_questions=5120, train_trajectories=25600, test_questions=512, test_trajectories=2560,
                train_mixed_group_accuracy=train_metrics['accuracy'], test_mixed_group_accuracy=test_metrics['accuracy'],
                mixed_group_accuracy=test_metrics['accuracy'], actor_updates=0, new_rollouts=0))
            append(out / 'metrics.jsonl', metrics)
            append(out / 'progress.jsonl', dict(stage='JOINT_EPOCH_COMPLETE', **receipt,
                train_mixed_group_accuracy=train_metrics['accuracy'], test_mixed_group_accuracy=test_metrics['accuracy'],
                mixed_group_accuracy=test_metrics['accuracy'], train_seconds=train_seconds))
            print(json.dumps(metrics, allow_nan=False), flush=True)
        dist.barrier()
    except BaseException as error:
        if claimed:
            path = out / 'epochs' / ('epoch-%04d' % epoch) / ('rank-%d-error.json' % rank)
            if not path.exists():
                once(path, dict(observed_utc=utc(), rank=rank, epoch=epoch,
                               error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
