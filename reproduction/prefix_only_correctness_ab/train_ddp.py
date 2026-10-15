"""Offline two-rank prefix-only correctness fitting; no actor updates."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

EPOCHS = 20
STEPS_PER_EPOCH = 100
GLOBAL_BATCH = 256
WORLD = 2
LOCAL_BATCH = 128


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def append(path, value):
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def write_once(path, value):
    with open(path, 'x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def batch_plan(labels, sampling, rng):
    """Return identical global plans on all ranks, before contiguous rank slicing."""
    if len(labels) != 25600 or set(labels) != {0, 1}:
        raise ValueError('Expected 25600 binary training trajectories with both classes')
    if sampling == 'natural':
        ids = list(range(len(labels)))
        rng.shuffle(ids)
        return [ids[i:i + GLOBAL_BATCH] for i in range(0, len(ids), GLOBAL_BATCH)]
    positive = [i for i, label in enumerate(labels) if label == 1]
    negative = [i for i, label in enumerate(labels) if label == 0]
    batches = []
    for _ in range(STEPS_PER_EPOCH):
        batch = [rng.choice(positive) for _ in range(128)] + [rng.choice(negative) for _ in range(128)]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def binary_metrics(logits, labels, shift=0.0):
    if len(logits) != len(labels) or not logits:
        raise ValueError('Nonempty aligned logits and labels required')
    pairs = [(float(z) + shift, int(y)) for z, y in zip(logits, labels)]
    if any(not math.isfinite(z) or y not in (0, 1) for z, y in pairs):
        raise ValueError('Invalid binary predictions')
    probs = [1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z)) for z, _ in pairs]
    tp = sum(z >= 0 and y == 1 for z, y in pairs)
    tn = sum(z < 0 and y == 0 for z, y in pairs)
    fp = sum(z >= 0 and y == 0 for z, y in pairs)
    fn = sum(z < 0 and y == 1 for z, y in pairs)
    npos, nneg = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / npos if npos else 0.0
    ordered = sorted(probs)
    return dict(count=len(pairs), positive=npos, negative=nneg, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=(tp + tn) / len(pairs),
                balanced_accuracy=(tp / npos + tn / nneg) / 2 if npos and nneg else None,
                precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                bce=sum(max(z, 0) - z * y + math.log1p(math.exp(-abs(z))) for z, y in pairs) / len(pairs),
                brier=sum((p - y) ** 2 for p, (_, y) in zip(probs, pairs)) / len(pairs),
                majority_accuracy=max(npos, nneg) / len(pairs), probability_mean=sum(probs) / len(probs),
                probability_quantiles={str(q): ordered[round(q * (len(ordered) - 1))] for q in (0, .01, .1, .5, .9, .99, 1)},
                saturation_below_0_01=sum(p < .01 for p in probs) / len(probs),
                saturation_above_0_99=sum(p > .99 for p in probs) / len(probs))


def validate_cache(cache, prefix):
    train, test = cache.ids_by_split['train'], cache.ids_by_split['test']
    if len(train) != 25600 or len(test) != 2560 or len(set(train + test)) != 28160:
        raise ValueError('Require unique 25600 train and 2560 test trajectories')
    groups = {}
    for split, ids, nq in [('train', train, 5120), ('test', test, 512)]:
        qids = collections.Counter(str(cache.index[tid]['qid']) for tid in ids)
        hashes = {str(cache.index[tid]['question_hash']) for tid in ids}
        if len(qids) != nq or set(qids.values()) != {5}:
            raise ValueError('Require exactly five trajectories per question')
        for tid in ids:
            row = cache.index[tid]
            if type(row['strict_em']) is not int or row['strict_em'] not in (0, 1):
                raise ValueError('Require strict trajectory EM labels')
            if not 0 < int(row['length']) == int(row['prefix_length']) <= prefix:
                raise ValueError('Only actual nonempty prefix positions may enter the head')
            if int(row['prompt_length']) < 1 or int(row['prompt_length']) + int(row['length']) > 32768:
                raise ValueError('Invalid original absolute prefix position span')
        groups[split] = (set(qids), hashes)
    if groups['train'][0] & groups['test'][0] or groups['train'][1] & groups['test'][1]:
        raise ValueError('Question leakage between fitting and diagnosis')
    if cache.max_tokens != 80:
        raise ValueError('All four heads require the same maximum prefix width80')
    if cache.feature_dim != 2048:
        raise ValueError('Unexpected feature width')
    if cache.manifest.get('representation') != 'prefix_only_first_request_layer27_v1':
        raise ValueError('Wrong feature representation')
    spec = cache.spec
    if int(spec['prefix_length']) != prefix:
        raise ValueError('Wrong feature prefix arm')
    return train, test


def actual_adam_steps(optimizer):
    values = [int(state['step'].item()) if hasattr(state['step'], 'item') else int(state['step'])
              for state in optimizer.state.values() if 'step' in state]
    if not values:
        return 0
    if len(values) != sum(len(group['params']) for group in optimizer.param_groups) or len(set(values)) != 1:
        raise RuntimeError('Adam state missing or inconsistent between head parameters')
    return values[0]


def main():
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from .features import PrefixFeatureCache
    from .model import PrefixCorrectnessProbe
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--cache-manifest-sha')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--prefix-tokens', required=True, type=int, choices=(10, 20, 40, 80))
    parser.add_argument('--sampling', choices=('balanced', 'natural'), default='balanced')
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    output = Path(args.output)
    initialized = False
    claimed = False
    try:
        if world != WORLD or torch.cuda.device_count() != WORLD:
            raise ValueError('Exactly two visible GPUs and two DDP ranks required')
        if not output.is_absolute() or str(output).startswith(('/Users/', '/tmp/', '/dev/shm/')):
            raise ValueError('Full training outputs require persistent remote absolute storage')
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
        initialized = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cache = PrefixFeatureCache(args.cache, expected_manifest_sha=args.cache_manifest_sha)
        train_ids, test_ids = validate_cache(cache, args.prefix_tokens)
        if cache.spec['encoder_revision'] != 'aa8e72537993ba99e69dfaafa59ed015b17504d1' or Path(cache.spec['encoder_model_dir']).resolve() != Path(args.model_dir).resolve():
            raise ValueError('Head initialization must use the same official encoder source as features')
        labels = [cache.index[tid]['strict_em'] for tid in train_ids]
        prior = sum(labels) / len(labels)
        if not 0 < prior < 1:
            raise ValueError('Both training classes required')
        prior_shift = math.log(prior / (1 - prior)) if args.sampling == 'balanced' else 0.0
        binding = dict(schema_version=1, role='offline_prefix_only_correctness_head', prefix_tokens=args.prefix_tokens,
                       seed=args.seed, sampling=args.sampling, balanced_global_positive=128 if args.sampling == 'balanced' else None,
                       global_batch=GLOBAL_BATCH, local_batch=LOCAL_BATCH, world_size=WORLD, accumulation=1,
                       epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, planned_adam_steps_per_rank=2000,
                       learning_rate=.0003, weight_decay=0, loss='single_mean_BCEWithLogits',
                       model_dir=str(Path(args.model_dir).resolve()), cache=str(Path(args.cache).resolve()),
                       cache_manifest=cache.manifest, shared_max_tokens=cache.max_tokens,
                       feature_layer=27, feature_boundary='post_block', encoder_trainable=False,
                       head_input='prefix_positions_only', head_max_prefix=80,
                       encoder_context='complete_actual_prompt_plus_actual_raw_prefix',
                       head_rope_positions='original_prompt_length_plus_prefix_offset',
                       question_hidden_passed_to_head=False,
                       natural_training_positive_prior=prior, corrected_logit_shift=prior_shift,
                       correction_assumption='Label-conditional sampling stability; calibration is not guaranteed',
                       diagnostic_trajectories=2560, diagnostic_questions=512)
        binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, binding_hash)
        if len(set(gathered)) != 1:
            raise RuntimeError('Rank cache or binding disagreement')
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise FileExistsError('Training namespace already occupied; no implicit resume or overwrite')
            write_once(output / 'run-claim.json', dict(binding=binding, binding_sha256=binding_hash, observed_utc=utc()))
        dist.barrier()
        claimed = True
        device = torch.device('cuda', local_rank)
        model = PrefixCorrectnessProbe(args.model_dir, cache.max_tokens).to(device=device, dtype=torch.float32)
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003, weight_decay=0)
        sampler = random.Random(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        if rank == 0:
            append(output / 'progress.jsonl', dict(stage='INITIALIZED', observed_utc=utc(), epoch=0,
                                                 actual_adam_steps_per_rank=[0] * WORLD,
                                                 block_copy=model.module.copy_metadata, binding_sha256=binding_hash))

        def evaluate(epoch):
            before = actual_adam_steps(optimizer)
            model.eval()
            assigned = test_ids[rank::WORLD]
            predictions = []
            started = time.monotonic()
            with torch.no_grad():
                for start in range(0, len(assigned), LOCAL_BATCH):
                    ids = assigned[start:start + LOCAL_BATCH]
                    batch = cache.get_batch(ids, device=device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model.module(batch)
                    predictions.extend((tid, float(z), cache.index[tid]['strict_em']) for tid, z in zip(ids, logits.cpu().tolist()))
            parts = [None] * WORLD
            dist.all_gather_object(parts, predictions)
            all_predictions = [item for part in parts for item in part]
            if len(all_predictions) != 2560 or {v[0] for v in all_predictions} != set(test_ids):
                raise RuntimeError('Diagnostic predictions do not cover exact test trajectories')
            if actual_adam_steps(optimizer) != before:
                raise RuntimeError('Diagnostic evaluation changed Adam counter')
            rank_counts = [None] * WORLD
            dist.all_gather_object(rank_counts, before)
            if rank_counts != [epoch * STEPS_PER_EPOCH] * WORLD:
                raise RuntimeError('Actual diagnostic boundary rank counters differ')
            if rank == 0:
                logits = [v[1] for v in all_predictions]
                ys = [v[2] for v in all_predictions]
                record = dict(epoch=epoch, observed_utc=utc(), split='new_512_question_diagnostic',
                              actual_adam_steps_per_rank=rank_counts, elapsed_seconds=time.monotonic() - started,
                              raw=binary_metrics(logits, ys), prior_corrected=binary_metrics(logits, ys, prior_shift),
                              natural_training_positive_prior=prior, corrected_logit_shift=prior_shift)
                append(output / 'metrics.jsonl', record)
                # Full diagnostic predictions remain on remote storage, with trajectory identities.
                torch.save(dict(epoch=epoch, predictions=all_predictions, binding_sha256=binding_hash),
                           output / ('diagnostic-epoch-%02d.pt' % epoch))
            model.train()

        def save(epoch):
            rank_rng = dict(rank=rank, python=random.getstate(), numpy=np.random.get_state(),
                            torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device),
                            sampler=sampler.getstate(), actual_adam_steps=actual_adam_steps(optimizer))
            all_rng = [None] * WORLD
            dist.all_gather_object(all_rng, rank_rng)
            if len({v['actual_adam_steps'] for v in all_rng}) != 1:
                raise RuntimeError('Actual rank Adam counts differ')
            if rank == 0:
                path = output / ('head-epoch-%02d.pt' % epoch)
                temp = output / (path.name + '.partial')
                if path.exists() or temp.exists():
                    raise FileExistsError('Checkpoint exists')
                torch.save(dict(schema_version=1, epoch=epoch, binding=binding, binding_sha256=binding_hash,
                                model={k: v.detach().cpu() for k, v in model.module.state_dict().items()},
                                optimizer=optimizer.state_dict(), rank_rng=all_rng,
                                actual_adam_steps_per_rank=[v['actual_adam_steps'] for v in all_rng]), temp)
                os.replace(temp, path)
            dist.barrier()

        evaluate(0)
        save(0)
        start_time = time.monotonic()
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.monotonic()
            plan = batch_plan(labels, args.sampling, sampler)
            plan_hash = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
            plans = [None] * WORLD
            dist.all_gather_object(plans, plan_hash)
            if len(set(plans)) != 1 or len(plan) != STEPS_PER_EPOCH:
                raise RuntimeError('Global sampling plans disagree')
            total_loss = total_correct = total_positive = 0.0
            for step, indices in enumerate(plan, start=1):
                local = indices[rank * LOCAL_BATCH:(rank + 1) * LOCAL_BATCH]
                ids = [train_ids[i] for i in local]
                ys = torch.tensor([labels[i] for i in local], device=device, dtype=torch.float32)
                batch = cache.get_batch(ids, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits.float(), ys, reduction='mean')
                finite = torch.tensor(int(bool(torch.isfinite(loss))), device=device)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not int(finite.item()):
                    raise FloatingPointError('Nonfinite BCE loss')
                loss.backward()
                gradient_checks = [torch.isfinite(p.grad).all() if p.grad is not None else torch.tensor(False, device=device)
                                   for p in model.parameters()]
                finite = torch.stack(gradient_checks).all().to(dtype=torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not int(finite.item()):
                    raise FloatingPointError('Missing or nonfinite head gradient')
                expected = (epoch - 1) * STEPS_PER_EPOCH + step
                if actual_adam_steps(optimizer) != expected - 1:
                    raise RuntimeError('Unexpected Adam count before update')
                optimizer.step()
                actual = actual_adam_steps(optimizer)
                if expected == 1 and any(state[key].dtype != torch.float32 for state in optimizer.state.values() for key in ('exp_avg', 'exp_avg_sq')):
                    raise RuntimeError('Adam moments must be FP32')
                if actual != expected or any(p.dtype != torch.float32 for p in model.parameters()):
                    raise RuntimeError('Adam count or FP32 master parameter invariant failed')
                append(output / ('rank-%d-updates.jsonl' % rank), dict(observed_utc=utc(), epoch=epoch,
                       epoch_step=step, actual_adam_steps=actual, local_trajectories=LOCAL_BATCH,
                       loss=float(loss.detach()), sampler_plan_sha256=plan_hash))
                total_loss += float(loss.detach()) * LOCAL_BATCH
                total_correct += float(((logits.detach() >= 0) == ys.bool()).sum())
                total_positive += float(ys.sum())
            stats = torch.tensor([total_loss, total_correct, total_positive], device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            evaluate(epoch)
            save(epoch)
            if rank == 0:
                append(output / 'progress.jsonl', dict(stage='EPOCH_COMPLETE', observed_utc=utc(), epoch=epoch,
                    actual_adam_steps_per_rank=[epoch * STEPS_PER_EPOCH] * WORLD,
                    sampler_plan_sha256=plan_hash, train_draw_count=25600,
                    train_draw_positive=int(stats[2].item()), sampled_train_bce=float(stats[0].item() / 25600),
                    sampled_train_accuracy=float(stats[1].item() / 25600),
                    elapsed_seconds=time.monotonic() - epoch_start,
                    cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device))))
        counts = [None] * WORLD
        dist.all_gather_object(counts, actual_adam_steps(optimizer))
        if counts != [2000] * WORLD:
            raise RuntimeError('Final actual Adam budget incomplete')
        if rank == 0:
            result = dict(status='OFFLINE_PREFIX_ONLY_HEAD_20_EPOCHS_COMPLETE', observed_utc=utc(), epoch=20,
                          actual_adam_steps_per_rank=counts, elapsed_seconds=time.monotonic() - start_time,
                          binding_sha256=binding_hash, diagnostic_evaluations=21,
                          actor_updates=0, formal_training_started=False)
            write_once(output / 'RESULT.json', result)
            append(output / 'progress.jsonl', dict(stage='COMPLETE', **result))
    except BaseException as exc:
        if claimed and output.exists():
            path = output / ('ERROR.json' if rank == 0 else 'rank-%d-error.json' % rank)
            if not path.exists():
                write_once(path, dict(observed_utc=utc(), rank=rank, error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
