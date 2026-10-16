"""Replay one retained G5 batch with the question-plus-twenty-token probe.

This is an isolated continuation of a completed prefix-only head checkpoint.
The old source epoch and the new replay iteration are separate coordinates.
It does not generate trajectories or update the frozen actor/encoder.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from reproduction.prefix20_online.features import REVISION, canonical, once, read, remote, sha
from reproduction.prefix20_online.train_epoch import (
    append, natural_plan, optimizer_learning_rates, rank_slice,
)
from reproduction.prefix_only_correctness_ab.train_ddp import actual_adam_steps
from reproduction.prefix20_online.mixed_diagnostic import mixed_probability

SOURCE_LEARNING_RATE = .00003
TARGET_LEARNING_RATE = .000003


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def binding_digest(binding):
    return hashlib.sha256(canonical(binding)).hexdigest()


def expected_adam_steps(source_steps, replay_iteration, updates_per_iteration=100):
    """Continue source Adam counters without treating replay as old online epochs."""
    if (len(source_steps) != 8 or any(type(v) is not int or v < 1 for v in source_steps)
            or len(set(source_steps)) != 1):
        raise ValueError('Require one complete synchronized eight-rank source Adam boundary')
    if type(replay_iteration) is not int or not 0 <= replay_iteration <= 70:
        raise ValueError('Replay iteration must be between zero and seventy')
    if updates_per_iteration != 100:
        raise ValueError('A retained 25600-trajectory batch requires exactly100 Adam updates')
    return [v + replay_iteration * updates_per_iteration for v in source_steps]


def validate_datasets(train, test):
    """Require the same complete G5 labels and fixed question-isolated diagnostic."""
    train_ids, test_ids = train.ids_by_split['train'], test.ids_by_split['test']
    question_sets = []
    for cache, ids, question_count in ((train, train_ids, 5120), (test, test_ids, 512)):
        counts = Counter(str(cache.index[tid]['qid']) for tid in ids)
        hashes = {cache.index[tid]['question_hash'] for tid in ids}
        if (len(ids) != question_count * 5 or len(set(ids)) != len(ids)
                or len(counts) != question_count or set(counts.values()) != {5}):
            raise ValueError('Require exact unique complete G5 questions in each split')
        if any(type(cache.index[tid]['strict_em']) is not int
               or cache.index[tid]['strict_em'] not in (0, 1) for tid in ids):
            raise ValueError('Each trajectory requires its own strict binary EM label')
        question_sets.append((set(counts), hashes))
    if set(train_ids) & set(test_ids) or any(question_sets[0][i] & question_sets[1][i] for i in (0, 1)):
        raise ValueError('Retained training questions overlap the fixed diagnostic')
    if train.spec['encoder_files_sha256'] != test.spec['encoder_files_sha256']:
        raise ValueError('Training and diagnostic use different official encoder bytes')
    if (train.max_tokens != test.max_tokens or train.question_max_tokens != test.question_max_tokens
            or train.max_tokens != train.question_max_tokens + 20):
        raise ValueError('All replay batches and the diagnostic require one fixed question-plus20 width')
    return train_ids, test_ids


def validate_source(saved, source_epoch, source_checkpoint_sha):
    if saved.get('epoch') != source_epoch or type(source_epoch) is not int or source_epoch < 1:
        raise ValueError('Source checkpoint does not match the actual completed online epoch')
    source_steps = saved['actual_adam_steps_per_rank']
    expected_adam_steps(source_steps, 0)
    if source_steps != [source_epoch * 100] * 8:
        raise ValueError('Source epoch and actual eight-rank Adam boundary differ')
    if (len(source_checkpoint_sha) != 64
            or any(c not in '0123456789abcdef' for c in source_checkpoint_sha)):
        raise ValueError('Source checkpoint must have independently pinned complete SHA256 bytes')
    if saved.get('binding_sha256') != binding_digest(saved['binding']):
        raise ValueError('Source checkpoint recipe digest differs')
    if (saved['binding'].get('prefix_tokens') != 20
            or saved['binding'].get('question_hidden_passed_to_head') is not False
            or saved['binding'].get('learning_rate') != SOURCE_LEARNING_RATE):
        raise ValueError('Source must be the authorized prefix20 LR3e-5 completed head')
    if len(saved['rank_rng']) != 8 or [r['rank'] for r in saved['rank_rng']] != list(range(8)):
        raise ValueError('Source checkpoint lacks all eight saved rank RNG states')
    return list(source_steps)


def validate_replay_checkpoint(saved, receipt, binding, iteration, source_steps):
    if saved.get('replay_iteration') != iteration - 1 or receipt.get('replay_iteration') != iteration - 1:
        raise ValueError('Latest replay checkpoint is not the immediately preceding complete iteration')
    binding_sha = binding_digest(binding)
    if (saved.get('binding') != binding or saved.get('binding_sha256') != binding_sha
            or receipt.get('binding_sha256') != binding_sha):
        raise ValueError('Replay checkpoint cannot change source, architecture, labels, or training recipe')
    expected = expected_adam_steps(source_steps, iteration - 1)
    if saved['actual_adam_steps_per_rank'] != expected or receipt['actual_adam_steps_per_rank'] != expected:
        raise ValueError('Replay checkpoint actual Adam counters differ from the source continuation')
    return expected


def group_predictions(predictions, index, expected_questions):
    """Preserve own-G5 identity for either a complete train or fixed test pass."""
    if expected_questions not in (512, 1024, 5120) or len(index) != expected_questions * 5:
        raise ValueError('Require a complete512,1024,or5120-question own-G5 identity index')
    if (len(predictions) != len(index) or len({r[0] for r in predictions}) != len(index)
            or {r[0] for r in predictions} != set(index)):
        raise ValueError('Duplicate, missing, or unexpected diagnostic trajectory IDs')
    groups = defaultdict(list)
    for tid, logit, label in predictions:
        row = index[tid]
        if type(label) is not int or label not in (0, 1) or label != row['strict_em']:
            raise ValueError('Prediction label differs from its own trajectory EM')
        if not math.isfinite(float(logit)):
            raise ValueError('Actual nonfinite diagnostic logit')
        groups[str(row['qid'])].append((str(tid), float(logit), label, row['question_hash']))
    if len(groups) != expected_questions:
        raise ValueError('Diagnostic question coverage differs from the requested complete split')
    result = []
    for qid, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row[0])
        if len(rows) != 5 or len({r[3] for r in rows}) != 1:
            raise ValueError('Incomplete own-G5 group or conflicting question identity')
        q, probabilities = mixed_probability([r[1] for r in rows])
        correct = sum(r[2] for r in rows)
        result.append(dict(qid=qid, question_hash=rows[0][3], trajectory_ids=[r[0] for r in rows],
                           probabilities=probabilities, strict_em=[r[2] for r in rows], correct_count=correct,
                           mixed_label=int(0 < correct < 5), mixed_probability=q,
                           predicted_mixed=int(q >= .5)))
    return result


def mixed_metrics(predictions, fixed_index, expected_questions=512):
    """Compute the requested G5 mixing decision at the unchanged probability threshold."""
    groups = group_predictions(predictions, fixed_index, expected_questions)
    tp = sum(r['mixed_label'] == 1 and r['predicted_mixed'] == 1 for r in groups)
    fp = sum(r['mixed_label'] == 0 and r['predicted_mixed'] == 1 for r in groups)
    fn = sum(r['mixed_label'] == 1 and r['predicted_mixed'] == 0 for r in groups)
    tn = len(groups) - tp - fp - fn
    metrics = dict(count=len(groups), accuracy=(tp + tn) / len(groups), tp=tp, fp=fp, fn=fn, tn=tn,
                   mixed_count=tp + fn, nonmixed_count=fp + tn, threshold=.5,
                   mean_q=math.fsum(r['mixed_probability'] for r in groups) / len(groups),
                   formula='1-prod(p_j)-prod(1-p_j)', true_mixed='0<sum(own_strict_em)<5')
    return metrics, groups


def write_checkpoint(out, checkpoint, iteration, iterations, save_every):
    """Commit state before publishing completion; never overwrite immutable boundaries."""
    import torch
    partial, latest = out / 'latest.pt.partial', out / 'latest.pt'
    if partial.exists():
        raise FileExistsError('Uncommitted replay checkpoint partial requires explicit recovery')
    torch.save(checkpoint, partial)
    with partial.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(partial, latest)
    checkpoint_sha = sha(latest)
    immutable = iteration % save_every == 0 or iteration == iterations
    if immutable:
        os.link(latest, out / ('head-replay-%04d.pt' % iteration))
    receipt = dict(schema_version=1, replay_iteration=iteration, source_epoch=checkpoint['source_epoch'],
                   source_data_epoch=checkpoint['source_data_epoch'], binding_sha256=checkpoint['binding_sha256'],
                   sha256=checkpoint_sha, bytes=latest.stat().st_size,
                   actual_adam_steps_per_rank=checkpoint['actual_adam_steps_per_rank'],
                   new_adam_steps_per_rank=100, source_adam_steps_per_rank=checkpoint['source_adam_steps_per_rank'],
                   actual_optimizer_learning_rates=checkpoint['actual_optimizer_learning_rates'],
                   actual_learning_rate=checkpoint['actual_optimizer_learning_rates'][0],
                   source_learning_rate=SOURCE_LEARNING_RATE, target_learning_rate=TARGET_LEARNING_RATE,
                   immutable_saved=immutable, observed_utc=utc())
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
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from .features import FeatureCache
    from .model import QuestionPrefixCorrectnessProbe, migrate_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-manifest-sha')
    parser.add_argument('--test-cache', required=True)
    parser.add_argument('--test-manifest-sha', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--replay-iteration', required=True, type=int)
    parser.add_argument('--source-data-epoch', required=True, type=int)
    parser.add_argument('--source-epoch', required=True, type=int)
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--source-checkpoint-sha', required=True)
    parser.add_argument('--iterations', type=int, default=70)
    parser.add_argument('--global-batch', type=int, default=256)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=TARGET_LEARNING_RATE)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    iteration = args.replay_iteration
    if (args.iterations != 70 or not 1 <= iteration <= 70 or args.source_data_epoch != iteration
            or args.global_batch != 256 or args.save_every != 10 or args.learning_rate != TARGET_LEARNING_RATE):
        raise ValueError('Replay requires ordered retainedE1..70, BS256, every10 save, and LR3e-6')
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError('Probe replay requires eight actual visible GPUs and ranks')
    out, model_dir, source_path = remote(args.output), remote(args.model_dir), remote(args.source_checkpoint)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    claimed = False
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        train = FeatureCache(args.train_cache, args.train_manifest_sha)
        test = FeatureCache(args.test_cache, args.test_manifest_sha)
        if train.manifest.get('epoch') != args.source_data_epoch:
            raise ValueError('Encoded retained batch does not match the ordered source data epoch')
        train_ids, test_ids = validate_datasets(train, test)
        labels = [train.index[tid]['strict_em'] for tid in train_ids]
        plan = natural_plan(len(train_ids), args.global_batch, world, args.seed, iteration)
        if len(plan) != 100:
            raise ValueError('The complete retained batch must give exactly100 global minibatches')
        plan_sha = binding_digest(plan)
        device = torch.device('cuda', local)
        model = QuestionPrefixCorrectnessProbe(model_dir, question_max_tokens=train.question_max_tokens)
        model = model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        # Source state is immutable and pinned by the stop/restore controller.
        # Hash it only for the initial transition; later iterations load their own committed latest.
        if iteration == 1:
            check = [sha(source_path) == args.source_checkpoint_sha if rank == 0 else None]
            dist.broadcast_object_list(check, src=0)
            if check != [True]:
                raise ValueError('Actual complete source checkpoint bytes differ')
            saved = torch.load(source_path, map_location='cpu', weights_only=False)
            source_steps = validate_source(saved, args.source_epoch, args.source_checkpoint_sha)
        else:
            claim = read(out / 'run-claim.json')
            source_steps = claim['binding']['source_adam_steps_per_rank']
            expected_adam_steps(source_steps, iteration - 1)
            receipt = read(out / 'latest.json')
            check = [sha(out / 'latest.pt') == receipt['sha256'] if rank == 0 else None]
            dist.broadcast_object_list(check, src=0)
            if check != [True]:
                raise ValueError('Actual complete replay checkpoint bytes differ')
            saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        binding = dict(schema_version=1, role='question_plus20_retained70_replay_correctness_head',
                       source_epoch=args.source_epoch, source_checkpoint_sha256=args.source_checkpoint_sha,
                       source_adam_steps_per_rank=source_steps,
                       encoder_revision=REVISION, encoder_files_sha256=test.spec['encoder_files_sha256'],
                       feature_layer=27, feature_boundary='post_block', prefix_tokens=20,
                       encoder_frozen=True, actor_updates=0, question_hidden_passed_to_head=True,
                       question_max_tokens=train.question_max_tokens, head_max_tokens=model.max_tokens,
                       head_architecture='one_full_official_last_block35_flatten_Sx2048_to1',
                       loss='single_mean_BCEWithLogits', class_weights=None,
                       sampling='natural_no_replacement_every_trajectory_once',
                       global_batch=256, local_batch=32, world_size=8, accumulation=1,
                       learning_rate=args.learning_rate, source_learning_rate=SOURCE_LEARNING_RATE,
                       target_learning_rate=TARGET_LEARNING_RATE, seed=args.seed, replay_iterations=70,
                       source_data_epochs=list(range(1, 71)), save_every=10,
                       test_manifest_sha256=test.manifest_sha, diagnostic_questions=512,
                       diagnostic_trajectories=2560, mixed_threshold=.5,
                       mixed_formula='1-prod(p_j)-prod(1-p_j)')
        binding_sha = binding_digest(binding)
        rank_bindings = [None] * world
        dist.all_gather_object(rank_bindings, (binding_sha, train.manifest_sha, plan_sha))
        if len(set(rank_bindings)) != 1:
            raise RuntimeError('Rank feature data or replay recipe differs')
        iteration_dir = out / 'iterations' / ('iteration-%04d' % iteration)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            iteration_dir.mkdir(parents=True, exist_ok=False)
            if iteration == 1:
                once(out / 'run-claim.json', dict(binding=binding, binding_sha256=binding_sha, observed_utc=utc()))
            elif claim['binding'] != binding or claim['binding_sha256'] != binding_sha:
                raise ValueError('Replay iteration cannot change the source or scientific recipe')
        dist.barrier()
        claimed = True
        if iteration == 1:
            saved, migration = migrate_checkpoint(saved, model, learning_rate=args.learning_rate)
        else:
            validate_replay_checkpoint(saved, receipt, binding, iteration, source_steps)
            migration = saved['question_representation_transition']
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        if optimizer_learning_rates(optimizer) != [args.learning_rate] * len(optimizer.param_groups):
            raise ValueError('Actual restored optimizer LR differs from the authorized LR3e-6')
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
        if restored_step != expected_adam_steps(source_steps, iteration - 1)[rank]:
            raise ValueError('Actual restored head Adam step does not continue the completed source')
        if rank == 0:
            once(iteration_dir / 'restore.json', dict(observed_utc=utc(), replay_iteration=iteration,
                 source_epoch=args.source_epoch, actual_adam_steps=restored_step,
                 source_checkpoint_sha256=args.source_checkpoint_sha,
                 source_learning_rate=SOURCE_LEARNING_RATE, target_learning_rate=TARGET_LEARNING_RATE,
                 actual_optimizer_learning_rates=optimizer_learning_rates(optimizer),
                 question_representation_transition=migration))
        del saved
        model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        total_loss, total_count = 0.0, 0
        model.train()
        for step, batch_indices in enumerate(plan, start=1):
            indices = rank_slice(batch_indices, rank, world)
            ids = [train_ids[index] for index in indices]
            ys = torch.tensor([labels[index] for index in indices], device=device, dtype=torch.float32)
            batch = train.get_batch(ids, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits.float(), ys, reduction='mean')
            finite = torch.isfinite(loss).to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite.item()):
                raise FloatingPointError('Actual nonfinite head BCE')
            loss.backward()
            finite = torch.stack([torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                                  if parameter.grad is not None]).all().to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite.item()):
                raise FloatingPointError('Actual nonfinite head gradient')
            optimizer.step()
            actual = actual_adam_steps(optimizer)
            actual_rates = optimizer_learning_rates(optimizer)
            if actual != restored_step + step or any(rate != args.learning_rate for rate in actual_rates):
                raise RuntimeError('Actual head Adam step or bound LR did not continue correctly')
            count = len(ids)
            total_loss += float(loss.detach()) * count
            total_count += count
            append(iteration_dir / ('rank-%d-updates.jsonl' % rank), dict(observed_utc=utc(),
                   source_epoch=args.source_epoch, replay_iteration=iteration,
                   source_data_epoch=args.source_data_epoch, iteration_step=step,
                   actual_adam_steps=actual, local_trajectories=count,
                   loss=float(loss.detach()), sampler_plan_sha256=plan_sha,
                   actual_optimizer_learning_rates=actual_rates, actual_learning_rate=actual_rates[0]))
        stats = torch.tensor([total_loss, total_count], device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        if int(stats[1].item()) != 25600:
            raise RuntimeError('Replay did not process every retained trajectory exactly once')
        train_seconds = time.monotonic() - started
        model.eval()
        diagnostic_started = time.monotonic()
        split_predictions, split_diagnostic_seconds = {}, {}
        # Both complete passes use the same frozen, post-100th-update head.
        # These are not the online predictions made before each batch update.
        with torch.no_grad():
            for split, cache, all_ids in (('train', train, train_ids), ('test', test, test_ids)):
                split_started = time.monotonic()
                predictions = []
                assigned = all_ids[rank::world]
                for start in range(0, len(assigned), 32):
                    ids = assigned[start:start + 32]
                    batch = cache.get_batch(ids, device=device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model.module(batch)
                    predictions.extend((tid, float(z), cache.index[tid]['strict_em'])
                                       for tid, z in zip(ids, logits.cpu().tolist()))
                parts = [None] * world
                dist.all_gather_object(parts, predictions)
                combined = [item for part in parts for item in part]
                if (len(combined) != len(all_ids) or len({v[0] for v in combined}) != len(all_ids)
                        or {v[0] for v in combined} != set(all_ids)):
                    raise RuntimeError('Complete post-update ' + split + ' trajectory coverage differs')
                split_predictions[split] = combined
                split_diagnostic_seconds[split] = time.monotonic() - split_started
        counts = [None] * world
        dist.all_gather_object(counts, actual_adam_steps(optimizer))
        if counts != expected_adam_steps(source_steps, iteration):
            raise RuntimeError('Actual eight-rank Adam replay boundaries differ')
        diagnostic_seconds = time.monotonic() - diagnostic_started
        rank_rng = dict(rank=rank, python=random.getstate(), numpy=np.random.get_state(),
                        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device),
                        actual_adam_steps=actual_adam_steps(optimizer))
        all_rng = [None] * world
        dist.all_gather_object(all_rng, rank_rng)
        if rank == 0:
            train_mixed, train_groups = mixed_metrics(split_predictions['train'],
                {tid: train.index[tid] for tid in train_ids}, expected_questions=5120)
            test_mixed, test_groups = mixed_metrics(split_predictions['test'],
                {tid: test.index[tid] for tid in test_ids}, expected_questions=512)
            metrics = dict(replay_iteration=iteration, source_epoch=args.source_epoch,
                           source_data_epoch=args.source_data_epoch, observed_utc=utc(),
                           train_mixed_group=train_mixed, test_mixed_group=test_mixed,
                           train_mixed_group_accuracy=train_mixed['accuracy'],
                           test_mixed_group_accuracy=test_mixed['accuracy'],
                           mixed_group=test_mixed, mixed_group_accuracy=test_mixed['accuracy'],
                           mixed_metric_semantics='complete train and fixed test eval/no_grad passes of the same post-update head',
                           actual_adam_steps_per_rank=counts, new_adam_steps_per_rank=100,
                           train_trajectories=25600, train_unique_trajectories=25600,
                           train_seconds=train_seconds, diagnostic_seconds=diagnostic_seconds,
                           train_post_update_diagnostic_seconds=split_diagnostic_seconds['train'],
                           test_diagnostic_seconds=split_diagnostic_seconds['test'],
                           train_manifest_sha256=train.manifest_sha, test_manifest_sha256=test.manifest_sha,
                           sampler_plan_sha256=plan_sha, sampling='natural_no_replacement',
                           global_batch=256, local_batch=32, accumulation=1,
                           actual_optimizer_learning_rates=optimizer_learning_rates(optimizer),
                           actual_learning_rate=optimizer_learning_rates(optimizer)[0],
                           source_learning_rate=SOURCE_LEARNING_RATE, target_learning_rate=TARGET_LEARNING_RATE,
                           cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                           cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
            once(iteration_dir / 'metrics.json', metrics)
            for split, groups in (('train', train_groups), ('test', test_groups)):
                prefix = 'train-' if split == 'train' else ''
                path = iteration_dir / (prefix + 'diagnostic-predictions.pt')
                torch.save(dict(replay_iteration=iteration, source_epoch=args.source_epoch,
                                split=split, predictions=split_predictions[split], binding_sha256=binding_sha,
                                metric_semantics='complete_post_update_eval_no_grad'), path)
                with path.open('rb') as handle:
                    os.fsync(handle.fileno())
                with (iteration_dir / (prefix + 'mixed-question-predictions.jsonl')).open('x', encoding='utf-8') as handle:
                    for row in groups:
                        handle.write(canonical(dict(replay_iteration=iteration, split=split, **row)).decode() + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            checkpoint = dict(schema_version=1, replay_iteration=iteration, source_epoch=args.source_epoch,
                              source_data_epoch=args.source_data_epoch, binding=binding, binding_sha256=binding_sha,
                              source_adam_steps_per_rank=source_steps,
                              model={key: value.detach().cpu() for key, value in model.module.state_dict().items()},
                              optimizer=optimizer.state_dict(), rank_rng=all_rng,
                              actual_adam_steps_per_rank=counts,
                              train_manifest_sha256=train.manifest_sha, test_manifest_sha256=test.manifest_sha,
                              actual_optimizer_learning_rates=optimizer_learning_rates(optimizer),
                              question_representation_transition=migration)
            receipt = write_checkpoint(out, checkpoint, iteration, args.iterations, args.save_every)
            once(iteration_dir / 'complete.json', dict(status='QUESTION_PLUS20_REPLAY_ITERATION_COMPLETE',
                 **receipt, train_trajectories=25600, test_trajectories=2560,
                 train_mixed_group_accuracy=train_mixed['accuracy'],
                 test_mixed_group_accuracy=test_mixed['accuracy'],
                 mixed_group_accuracy=test_mixed['accuracy'], actor_updates=0, new_rollouts=0))
            append(out / 'metrics.jsonl', metrics)
            append(out / 'progress.jsonl', dict(stage='REPLAY_ITERATION_COMPLETE', **receipt,
                   train_mixed_group_accuracy=train_mixed['accuracy'],
                   test_mixed_group_accuracy=test_mixed['accuracy'],
                   mixed_group_accuracy=test_mixed['accuracy'], train_seconds=train_seconds,
                   diagnostic_seconds=diagnostic_seconds))
            print(json.dumps(metrics, allow_nan=False), flush=True)
        dist.barrier()
    except BaseException as error:
        if claimed:
            path = out / 'iterations' / ('iteration-%04d' % iteration) / ('rank-%d-error.json' % rank)
            if not path.exists():
                once(path, dict(observed_utc=utc(), rank=rank, replay_iteration=iteration,
                     error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
