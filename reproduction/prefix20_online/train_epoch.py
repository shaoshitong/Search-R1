"""One no-replacement fresh epoch of eight-rank twenty-token head fitting.

Sequential rollout/encoding/training phases release all eight GPUs between them.
The latest checkpoint bridges processes; every tenth boundary is immutable.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from reproduction.prefix_only_correctness_ab.train_ddp import binary_metrics, actual_adam_steps
from .features import FeatureCache, REVISION, canonical, once, read, remote, sha


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(canonical(value).decode() + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def natural_plan(size, global_batch, world, seed, epoch):
    """Exactly one occurrence of every trajectory; no padded sampler duplicates."""
    if size < world or size % world or global_batch < world or global_batch % world:
        raise ValueError('Dataset and global batch must permit equal nonempty rank slices')
    ids = list(range(size))
    random.Random(seed + epoch).shuffle(ids)
    return [ids[start:start + global_batch] for start in range(0, size, global_batch)]


def rank_slice(batch, rank, world):
    if len(batch) % world or not 0 <= rank < world:
        raise ValueError('Global batch cannot be split equally across DDP ranks')
    local = len(batch) // world
    return batch[rank * local:(rank + 1) * local]


def validate_datasets(train, test):
    from collections import Counter
    from .model import expected_policy_positions
    train_ids, test_ids = train.ids_by_split['train'], test.ids_by_split['test']
    question_sets = []
    for cache, ids, expected_q in ((train, train_ids, 5120), (test, test_ids, 512)):
        counts = Counter(str(cache.index[tid]['qid']) for tid in ids)
        hashes = {cache.index[tid]['question_hash'] for tid in ids}
        if len(ids) != expected_q * 5 or len(counts) != expected_q or set(counts.values()) != {5}:
            raise ValueError('Require exact complete G5 questions in each split')
        for tid in ids:
            row = cache.index[tid]
            if len(expected_policy_positions(row['segment_bounds'])) != row['length']:
                raise ValueError('Policy prefix segment coverage differs')
        question_sets.append((set(counts), hashes))
    if set(train_ids) & set(test_ids) or any(question_sets[0][i] & question_sets[1][i] for i in (0, 1)):
        raise ValueError('Fresh training questions overlap the fixed diagnostic')
    if train.spec['encoder_files_sha256'] != test.spec['encoder_files_sha256']:
        raise ValueError('Training and fixed diagnostic use different official encoder bytes')
    return train_ids, test_ids


def auroc(logits, labels):
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    ordered = sorted(zip(logits, labels))
    rank_sum, start = 0.0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum(y for _, y in ordered[start:end])
        start = end
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from .model import PrefixCorrectnessProbe
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-manifest-sha')
    parser.add_argument('--test-cache', required=True)
    parser.add_argument('--test-manifest-sha', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--global-batch', type=int, default=25600)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError('Probe fitting requires eight actual visible GPUs and ranks')
    if not 1 <= args.epoch <= args.epochs or args.save_every < 1:
        raise ValueError('Invalid epoch boundary')
    out, model_dir = remote(args.output), remote(args.model_dir)
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
        train_ids, test_ids = validate_datasets(train, test)
        labels = [train.index[tid]['strict_em'] for tid in train_ids]
        plan = natural_plan(len(train_ids), args.global_batch, world, args.seed, args.epoch)
        plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
        binding = dict(schema_version=1, role='online20_fresh_trajectory_correctness_head',
                       encoder_revision=REVISION, encoder_files_sha256=test.spec['encoder_files_sha256'],
                       feature_layer=27, feature_boundary='post_block', prefix_tokens=20,
                       encoder_frozen=True, question_hidden_passed_to_head=False,
                       head_architecture='one_full_official_last_block35_flatten_linear_width20',
                       loss='single_mean_BCEWithLogits', class_weights=None,
                       sampling='natural_no_replacement_every_trajectory_once',
                       global_batch=args.global_batch, world_size=world, accumulation=1,
                       learning_rate=args.learning_rate, seed=args.seed, epochs=args.epochs,
                       save_every=args.save_every, test_manifest_sha256=test.manifest_sha,
                       diagnostic_questions=512, diagnostic_trajectories=2560)
        binding_sha = hashlib.sha256(canonical(binding)).hexdigest()
        rank_bindings = [None] * world
        dist.all_gather_object(rank_bindings, (binding_sha, train.manifest_sha, plan_sha))
        if len(set(rank_bindings)) != 1:
            raise RuntimeError('Rank dataset or sampling binding differs')
        epoch_dir = out / 'epochs' / ('epoch-%04d' % args.epoch)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            epoch_dir.mkdir(parents=True, exist_ok=False)
            if args.epoch == 1:
                once(out / 'run-claim.json', dict(binding=binding, binding_sha256=binding_sha, observed_utc=utc()))
            else:
                if read(out / 'run-claim.json')['binding_sha256'] != binding_sha:
                    raise ValueError('Epoch cannot change fixed training or diagnostic binding')
        dist.barrier()
        claimed = True
        device = torch.device('cuda', local)
        model = PrefixCorrectnessProbe(model_dir, max_tokens=20).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        restored_step = 0
        if args.epoch > 1:
            receipt = read(out / 'latest.json')
            latest = out / 'latest.pt'
            if receipt['epoch'] != args.epoch - 1 or receipt['binding_sha256'] != binding_sha:
                raise ValueError('Latest checkpoint is not the immediately preceding complete epoch')
            # One rank checks checkpoint bytes; every rank loads the same committed file.
            check = [sha(latest) == receipt['sha256'] if rank == 0 else None]
            dist.broadcast_object_list(check, src=0)
            if check != [True]:
                raise ValueError('Latest head checkpoint bytes differ')
            saved = torch.load(latest, map_location='cpu', weights_only=False)
            if saved['epoch'] != args.epoch - 1 or saved['binding_sha256'] != binding_sha:
                raise ValueError('Checkpoint epoch/binding differs')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor) and key != 'step':
                        state[key] = value.to(device)
            state = saved['rank_rng'][rank]
            if state['rank'] != rank:
                raise ValueError('Checkpoint RNG rank identity differs')
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch_cpu'])
            torch.cuda.set_rng_state(state['torch_cuda'], device)
            restored_step = actual_adam_steps(optimizer)
            if restored_step != saved['actual_adam_steps_per_rank'][rank]:
                raise ValueError('Checkpoint actual Adam step differs')
            del saved
        model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        total_loss, total_correct, total_positive, total_count = 0.0, 0, 0, 0
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
            if actual != restored_step + step:
                raise RuntimeError('Actual head Adam counter did not advance exactly once')
            count = len(ids)
            total_loss += float(loss.detach()) * count
            total_correct += int(((logits.detach() >= 0) == ys.bool()).sum().item())
            total_positive += int(ys.sum().item())
            total_count += count
            append(epoch_dir / ('rank-%d-updates.jsonl' % rank), dict(observed_utc=utc(), epoch=args.epoch,
                   epoch_step=step, actual_adam_steps=actual, local_trajectories=count,
                   loss=float(loss.detach()), sampler_plan_sha256=plan_sha))
        stats = torch.tensor([total_loss, total_correct, total_positive, total_count], device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        if int(stats[3].item()) != 25600:
            raise RuntimeError('Epoch did not process every fresh trajectory exactly once')
        train_seconds = time.monotonic() - started
        model.eval()
        diagnostic_started = time.monotonic()
        predictions = []
        assigned = test_ids[rank::world]
        local_batch = args.global_batch // world
        with torch.no_grad():
            for start in range(0, len(assigned), local_batch):
                ids = assigned[start:start + local_batch]
                batch = test.get_batch(ids, device=device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model.module(batch)
                predictions.extend((tid, float(z), test.index[tid]['strict_em']) for tid, z in zip(ids, logits.cpu().tolist()))
        parts, counts = [None] * world, [None] * world
        dist.all_gather_object(parts, predictions)
        dist.all_gather_object(counts, actual_adam_steps(optimizer))
        if counts != [restored_step + len(plan)] * world:
            raise RuntimeError('Actual Adam boundaries differ between ranks')
        all_predictions = [item for part in parts for item in part]
        if len(all_predictions) != 2560 or {v[0] for v in all_predictions} != set(test_ids):
            raise RuntimeError('Fixed diagnostic coverage differs')
        diagnostic_seconds = time.monotonic() - diagnostic_started
        rank_rng = dict(rank=rank, python=random.getstate(), numpy=np.random.get_state(),
                        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device),
                        actual_adam_steps=actual_adam_steps(optimizer))
        all_rng = [None] * world
        dist.all_gather_object(all_rng, rank_rng)
        if rank == 0:
            logits, ys = [v[1] for v in all_predictions], [v[2] for v in all_predictions]
            metrics = dict(epoch=args.epoch, observed_utc=utc(), actual_adam_steps_per_rank=counts,
                           train_trajectories=25600, train_unique_trajectories=25600,
                           train_positive=int(stats[2].item()),
                           train_online_bce=float(stats[0].item() / 25600),
                           train_online_accuracy=float(stats[1].item() / 25600),
                           train_metric_semantics='one-pass online predictions before each own batch update',
                           train_seconds=train_seconds, diagnostic_seconds=diagnostic_seconds,
                           raw=binary_metrics(logits, ys), auroc=auroc(logits, ys),
                           test_manifest_sha256=test.manifest_sha, train_manifest_sha256=train.manifest_sha,
                           sampler_plan_sha256=plan_sha, sampling='natural_no_replacement',
                           global_batch=args.global_batch, last_global_batch=len(plan[-1]),
                           cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                           cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
            once(epoch_dir / 'metrics.json', metrics)
            torch.save(dict(epoch=args.epoch, predictions=all_predictions, binding_sha256=binding_sha),
                       epoch_dir / 'diagnostic-predictions.pt')
            checkpoint = dict(schema_version=1, epoch=args.epoch, binding=binding, binding_sha256=binding_sha,
                              model={key: value.detach().cpu() for key, value in model.module.state_dict().items()},
                              optimizer=optimizer.state_dict(), rank_rng=all_rng,
                              actual_adam_steps_per_rank=counts,
                              train_manifest_sha256=train.manifest_sha, test_manifest_sha256=test.manifest_sha)
            partial, latest = out / 'latest.pt.partial', out / 'latest.pt'
            if partial.exists():
                raise FileExistsError('Uncommitted checkpoint partial requires explicit recovery')
            torch.save(checkpoint, partial)
            with partial.open('rb') as handle:
                os.fsync(handle.fileno())
            os.replace(partial, latest)
            checkpoint_sha = sha(latest)
            if args.epoch % args.save_every == 0 or args.epoch == args.epochs:
                target = out / ('head-epoch-%04d.pt' % args.epoch)
                os.link(latest, target)
            receipt = dict(epoch=args.epoch, binding_sha256=binding_sha, sha256=checkpoint_sha,
                           bytes=latest.stat().st_size, actual_adam_steps_per_rank=counts,
                           immutable_saved=args.epoch % args.save_every == 0 or args.epoch == args.epochs,
                           observed_utc=utc())
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
            once(epoch_dir / 'complete.json', dict(status='FRESH_ONLINE20_EPOCH_COMPLETE', **receipt,
                 train_trajectories=25600, test_trajectories=2560, actor_updates=0))
            append(out / 'metrics.jsonl', metrics)
            append(out / 'progress.jsonl', dict(stage='EPOCH_COMPLETE', **receipt,
                   train_seconds=train_seconds, diagnostic_seconds=diagnostic_seconds))
            print(json.dumps(metrics, allow_nan=False), flush=True)
        dist.barrier()
    except BaseException as error:
        if claimed:
            path = out / 'epochs' / ('epoch-%04d' % args.epoch) / ('rank-%d-error.json' % rank)
            if not path.exists():
                once(path, dict(observed_utc=utc(), rank=rank, error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
