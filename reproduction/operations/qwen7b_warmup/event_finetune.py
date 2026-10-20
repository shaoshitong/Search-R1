"""Isolated, config-bound continuation of an audited balanced Prefix20 probe.

This entry point never submits jobs, changes a source checkpoint, or replaces
the probe block/head.  A new phase starts only from a sealed, independently
audited balanced checkpoint.  Every restart restores raw/EMA/Adam/four-rank RNG;
the first available next update is replayed bit-exactly before acceptance.

The optimized group term is a three-event NLL.  Binary ``group_bce`` remains a
separate diagnostic and is never relabelled as the optimized objective.
"""
import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# FeatureReader imports native_capture's fixed tensor geometry.  Bind it before
# those imports, rather than changing QMAX only after module initialization.
os.environ['QWEN7B_QUESTION_MAX_TOKENS'] = '273'
os.environ['SEARCH_G1_B_PREFIX_TOKENS'] = '20'
from .balanced_finetune import restore, selection
from .balanced_sampling import plan, proof
from .download_model import atomic, fingerprints
from .event_candidate_loss import event_candidate_loss
from .model import Probe, native_buffers
from .packed_features import FeatureReader
from .train import actual_steps, model_hash, optimizer_hash

ROLE = 'BALANCED_PREFIX20_THREE_EVENT_CONTINUATION'
SOURCE_ROLE = 'EPOCH1_BALANCED_RANK1_CONTINUATION'
SOURCE_AUDIT = 'BALANCED_PREFIX20_INDEPENDENT_CPU_SHA_RAW_EMA_ADAM_FOUR_RANK_RNG_VERIFIED'
OBJECTIVE = 'trajectory_mean_bce+three_event_group_nll+mixed_pairwise_rank1'
STOP = False
REQUIRED_CODE = (
    'reproduction/operations/qwen7b_warmup/event_finetune.py',
    'reproduction/operations/qwen7b_warmup/event_candidate_loss.py',
    'reproduction/operations/qwen7b_warmup/download_model.py',
    'reproduction/operations/qwen7b_warmup/balanced_loss.py',
    'reproduction/operations/qwen7b_warmup/balanced_sampling.py',
    'reproduction/operations/qwen7b_warmup/balanced_finetune.py',
    'reproduction/operations/qwen7b_warmup/train.py',
    'reproduction/operations/qwen7b_warmup/model.py',
    'reproduction/operations/qwen7b_warmup/packed_features.py',
    'reproduction/operations/qwen7b_warmup/native_capture.py',
    'reproduction/operations/qwen7b_warmup/replay_diagnostics.py',
    'reproduction/operations/qwen7b_warmup/heldout_selection.py',
    'reproduction/prefix20_group_joint100/loss.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def acquire_writer_lock(base, rank):
    """A second direct entry point cannot become a writer, even without sbatch."""
    handle = None
    accepted = [True]
    if rank == 0:
        (base/'probe').mkdir(parents=True, exist_ok=True)
        handle = (base/'probe/event-fit.lock').open('a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            handle = None
            accepted[0] = False
        if handle is not None:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(dict(pid=os.getpid(), job_id=os.environ.get('SLURM_JOB_ID'),
                                         code_sha256=fingerprints(__file__)['sha256'], epoch_time=time.time())))
            handle.flush()
    dist.broadcast_object_list(accepted, src=0)
    assert accepted[0], 'This phase already has a live event-probe writer'
    return handle


def stop(_signal, _frame):
    global STOP
    STOP = True


def validate_config(config):
    """Fail closed on a recipe change or an ambiguous epoch/count lineage."""
    for key in ('source_epoch', 'start_epoch', 'fit_epochs', 'source_steps',
                'seed', 'fit_epoch_groups'):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError('%s must be an integer' % key)
    assert config['source_epoch'] >= 1 and config['source_steps'] > 0
    assert config['start_epoch'] == config['source_epoch'] + 1
    assert config['fit_epochs'] >= config['start_epoch']
    assert config['fit_world'] == 4 and config['prefix_tokens'] == 20
    assert config['question_max_tokens'] == 273
    assert config['fit_epoch_groups'] == 67198
    assert config['pair_weight'] == 1. and config['pair_margin'] == 1.
    assert config['lr'] == 5e-6 and config['weight_decay'] == 0.
    assert config['ema_decay'] == .99
    assert config['objective'] == OBJECTIVE
    assert config['source_checkpoint']['epoch'] == config['source_epoch']
    assert config['source_checkpoint']['steps'] == config['source_steps']
    assert config['source_checkpoint']['completed_epoch'] is True
    assert config['source_checkpoint']['cursor'] == 0
    assert len(config['source_commit']) == 40
    assert set(REQUIRED_CODE).issubset(config['source_file_manifest'])


def verify_runtime(config):
    root = Path(__file__).resolve().parents[3]
    for relative, expected in config['source_file_manifest'].items():
        path = Path(relative)
        assert not path.is_absolute() and '..' not in path.parts
        assert fingerprints(root / path)['sha256'] == expected, relative
    assert config['fit_file_sha256'] == fingerprints(__file__)['sha256']


def expected_steps(config, groups, epoch, cursor=0, completed=False):
    """Count actual deterministic epoch plans, never infer from an old CP1."""
    assert config['start_epoch'] <= epoch <= config['fit_epochs']
    own_batches = plan(groups, config['seed'], epoch, config['fit_epoch_groups'])
    assert isinstance(cursor, int) and not isinstance(cursor, bool)
    assert 0 <= cursor <= len(own_batches)
    assert not completed or cursor == 0
    previous = sum(len(plan(groups, config['seed'], e, config['fit_epoch_groups']))
                   for e in range(config['start_epoch'], epoch))
    return config['source_steps'] + previous + (len(own_batches) if completed else cursor)


def position(saved, config, groups):
    """Return restart position; a pre-evaluation cursor stays in that epoch."""
    epoch, cursor = saved['epoch'], saved['cursor']
    completed = saved['completed_epoch']
    assert isinstance(completed, bool)
    assert saved['steps'] == expected_steps(config, groups, epoch, cursor, completed)
    assert saved['stage'] == ('event_epoch_completed' if completed else
                            'event_evaluation_pending' if cursor == len(plan(
                                groups, config['seed'], epoch, config['fit_epoch_groups']))
                            else 'event_training')
    if completed:
        assert saved['metrics'] is not None, 'A completed epoch must seal its evaluation'
    return epoch + int(completed), cursor, saved['steps']


def retain_first_update(root, receipt, config):
    """Recover an interrupted metadata publication without repeating an update."""
    path = root/'first-event-update.pt'
    descriptor = root/'first-event-update.json'
    if receipt['steps'] == config['source_steps']+1:
        assert receipt['epoch'] == config['start_epoch'] and receipt['cursor'] == 1
        if not path.exists():
            os.link(receipt['path'], path)
        assert os.path.samefile(path, receipt['path'])
        if descriptor.exists():
            assert read(descriptor)['sha256'] == receipt['sha256']
        else:
            atomic(descriptor, dict(receipt, path=str(path)))
    else:
        assert path.is_file() and descriptor.is_file(), 'Missing accepted first-update artifact'
        assert read(descriptor)['steps'] == config['source_steps']+1


def verify_source(config, saved, verified_fingerprint=None):
    """Bind the immutable source, its real CPU audit, and its original recipe."""
    receipt = config['source_checkpoint']
    actual = verified_fingerprint or fingerprints(receipt['path'])
    assert actual['sha256'] == receipt['sha256'] and actual['bytes'] == receipt['bytes']
    audit_descriptor = config['source_audit']
    assert fingerprints(audit_descriptor['path'])['sha256'] == audit_descriptor['sha256']
    audit = read(audit_descriptor['path'])
    assert audit['status'] == SOURCE_AUDIT and audit['checked_ranks'] == 4
    assert audit['pending_evaluation'] is False
    assert audit['checkpoint']['sha256'] == receipt['sha256']
    assert audit['checkpoint']['bytes'] == receipt['bytes']
    assert audit['checkpoint']['completed_epoch'] is True
    assert Path(audit['immutable_checkpoint']).resolve() == Path(receipt['path']).resolve()
    assert saved['role'] == SOURCE_ROLE
    assert saved['completed_epoch'] is True and saved['cursor'] == 0
    assert saved['epoch'] == config['source_epoch']
    assert saved['steps'] == config['source_steps'] and saved['actor_updates'] == 0
    assert saved['metrics'] is not None
    source = Path(config['source_phase'])
    assert saved['config_sha256'] == config['source_config_sha256'] == fingerprints(source / 'config.json')['sha256']
    assert saved['bank_sha256'] == config['mixed_bank_sha256'] == fingerprints(source / 'bank/groups.json')['sha256']
    assert saved['fit_code_sha256'] == config['source_fit_file_sha256']
    old_config = read(source / 'config.json')
    publication = read(source/'preparation/source-publication.json')
    assert publication['commit'] == saved['source_commit']
    old_fit = Path(publication['code_root'])/'reproduction/operations/qwen7b_warmup/balanced_finetune.py'
    assert fingerprints(old_fit)['sha256'] == config['source_fit_file_sha256']
    assert old_config['prefix_tokens'] == 20 and old_config['pair_weight'] == 1.
    assert old_config['seed'] == config['seed']
    assert old_config['fit_epoch_groups'] == config['fit_epoch_groups']
    for key, field in (('raw', 'model_sha256'), ('ema', 'ema_sha256'),
                       ('adam', 'optimizer_sha256')):
        assert all(row[field] == audit['hashes'][key] for row in saved['rank_states'])
    assert [r['rank'] for r in saved['rank_states']] == list(range(4))
    assert all(r['steps'] == config['source_steps'] for r in saved['rank_states'])
    return audit


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def rng_hash(rng):
    digest = hashlib.sha256(repr(rng['python']).encode())
    name, values, index, gaussian, cached = rng['numpy']
    digest.update(repr((name, index, gaussian, cached)).encode())
    digest.update(values.tobytes())
    for key in ('torch', 'cuda'):
        digest.update(rng[key].cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def buffer_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(native_buffers(model).items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def update(ddp, model, ema, optimizer, reader, groups, indices, rank, device,
           config, commit=True):
    own = indices[rank::4]
    optimizer.zero_grad(set_to_none=True)
    before = actual_steps(optimizer)
    fraction = sum(0 < sum(groups[i]['labels']) < 5 for i in indices) / len(indices)
    assert 0 < fraction < 1
    stats = torch.zeros(12, dtype=torch.float64, device=device)
    started = time.monotonic()
    for offset in range(0, len(own), 2):
        part = [groups[i] for i in own[offset:offset + 2]]
        n = len(part)
        data = reader.get([tid for g in part for tid in g['trajectory_ids']], device)
        labels = torch.tensor([g['labels'] for g in part], dtype=torch.float32, device=device)
        masks = torch.tensor([g['pair_mask'] for g in part], dtype=torch.bool, device=device)
        with (ddp.no_sync() if offset + 2 < len(own) else nullcontext()):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = ddp(data).reshape(n, 5)
            loss = event_candidate_loss(z, labels, masks, config['pair_weight'],
                                        config['pair_margin'], fraction)
            (loss['loss'] * (4 * n / len(indices))).backward()
        stats += torch.tensor([
            n, n * float(loss['trajectory_bce'].detach()),
            n * float(loss['group_bce'].detach()),
            n * float(loss['group_event_nll'].detach()),
            n * float(loss['event_correction'].detach()),
            n * float(loss['pairwise_loss'].detach()),
            n * float(loss['loss'].detach()), float(loss['eligible_pairs']),
            float(loss['pairwise_correct']), float(loss['eligible_groups']),
            n * float(loss['mean_p_range'].detach()), 0.],
            dtype=torch.float64, device=device)
    finite = torch.tensor(int(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                  for p in model.parameters())), device=device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    assert finite.item() == 1, 'Nonfinite event-probe gradient'
    if commit:
        optimizer.step()
        assert actual_steps(optimizer) == before + 1
        with torch.no_grad():
            for target, value in zip(ema.parameters(), model.parameters()):
                target.mul_(config['ema_decay']).add_(value, alpha=1-config['ema_decay'])
    else:
        assert actual_steps(optimizer) == before
    stats[11] = time.monotonic() - started
    dist.all_reduce(stats)
    n = float(stats[0])
    return dict(trajectory_bce=float(stats[1]/n), group_bce=float(stats[2]/n),
                group_event_nll=float(stats[3]/n), event_correction=float(stats[4]/n),
                pairwise_loss=float(stats[5]/n), optimized_total_loss=float(stats[6]/n),
                pairwise_accuracy=float(stats[8]/stats[7]) if stats[7] else None,
                eligible_pairs=int(stats[7]), eligible_groups=int(stats[9]),
                mean_p_range=float(stats[10]/n), mean_rank_step_seconds=float(stats[11]/4),
                objective=OBJECTIVE)


def evaluation_stats(logits, labels, masks, config, mixed_fraction):
    """One pass over z yields both existing binary metrics and optimized terms."""
    n = len(labels)
    loss = event_candidate_loss(logits, labels, masks, config['pair_weight'],
                                config['pair_margin'], mixed_fraction)
    target = loss['mixed_label'].bool()
    predicted = loss['mixed_probability'] >= .5
    predicted_count = (logits >= 0).sum(1)
    true_count = labels.sum(1)
    classes = lambda k: torch.where(k == 0, 0, torch.where(k == 5, 2, 1))
    confusion = torch.bincount(3*classes(true_count)+classes(predicted_count), minlength=9)
    probabilities = logits.sigmoid()
    ranges = probabilities.amax(1)-probabilities.amin(1)
    values = [n, n*float(loss['trajectory_bce']), n*float(loss['group_bce']),
              int((predicted & target).sum()), int((predicted & ~target).sum()),
              int((~predicted & target).sum()), int((~predicted & ~target).sum()),
              int(((logits >= 0) == labels.bool()).sum()), labels.numel(),
              float(loss['mixed_probability'].sum()), int(target.sum())]
    values += confusion.tolist()
    values += [n*float(loss['group_event_nll']), n*float(loss['event_correction']),
               n*float(loss['pairwise_loss']), n*float(loss['loss']),
               float(loss['eligible_pairs']), float(loss['pairwise_correct']),
               float(loss['eligible_groups']), float((ranges <= .01).sum()),
               float(((probabilities >= .4) & (probabilities <= .6)).all(1).sum()),
               float(ranges.sum())]
    return torch.tensor(values, dtype=torch.float64, device=logits.device), loss


def summarized_stats(values):
    n, tb, gb, tp, fp, fn, tn, correct, trajectories, q, mixed = values[:11].tolist()
    assert n > 0
    return dict(groups=int(n), questions=int(n), trajectories=int(trajectories), trajectory_bce=tb/n,
                group_bce=gb/n, total_loss=(tb+gb)/n, binary_bce_sum=(tb+gb)/n,
                mixed_accuracy=(tp+tn)/n,
                mixed_balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None,
                trajectory_accuracy=correct/trajectories, mean_predicted_mixed=q/n,
                true_mixed_fraction=mixed/n, confusion=dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)),
                three_class_order=['all_wrong', 'mixed', 'all_correct'],
                three_class_confusion=values[11:20].reshape(3, 3).tolist(),
                group_event_nll=float(values[20]/n), event_correction=float(values[21]/n),
                pairwise_loss=float(values[22]/n), optimized_total_loss=float(values[23]/n),
                eligible_pairs=int(values[24]),
                pairwise_accuracy=float(values[25]/values[24]) if values[24] else None,
                eligible_groups=int(values[26]), range_le001_fraction=float(values[27]/n),
                all_p_in04_06_fraction=float(values[28]/n), mean_p_range=float(values[29]/n),
                objective=OBJECTIVE, weight_kind='EMA0.99', resampled=False)


@torch.no_grad()
def evaluate_event(model, reader, groups, rank, device, output, epoch, split,
                   config, mixed_subset=False):
    """Read/infer each split once; Mixed-only separation shares diagnostic z."""
    model.eval()
    values = torch.zeros(30, dtype=torch.float64, device=device)
    mixed_values = torch.zeros_like(values)
    own = groups[rank::4]
    fraction = sum(0 < sum(g['labels']) < 5 for g in groups) / len(groups)
    assert 0 < fraction <= 1
    rounds = math.ceil(math.ceil(len(groups)/4)/8)
    destination = output/('epoch-%04d-%s-rank%d.jsonl.gz' % (epoch, split, rank))
    temporary = destination.with_suffix('.tmp')
    io_seconds = compute_seconds = 0.
    started = time.monotonic()
    with gzip.open(temporary, 'wt') as stream:
        for iteration in range(rounds):
            offset = 8*iteration
            part = own[offset:offset+8]
            if part:
                tick = time.monotonic()
                data = reader.get([tid for g in part for tid in g['trajectory_ids']], device)
                y = torch.tensor([g['labels'] for g in part], dtype=torch.float32, device=device)
                mask = torch.tensor([g['pair_mask'] for g in part], dtype=torch.bool, device=device)
                io_seconds += time.monotonic()-tick
                tick = time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(data).reshape(len(part), 5)
                row, loss = evaluation_stats(logits, y, mask, config, fraction)
                values += row
                if mixed_subset:
                    keep = loss['mixed_label'].bool()
                    if bool(keep.any()):
                        row, _ = evaluation_stats(logits[keep], y[keep], mask[keep], config, 1.)
                        mixed_values += row
                for group, p, q in zip(part, logits.sigmoid().float().cpu().tolist(),
                                       loss['mixed_probability'].float().cpu().tolist()):
                    stream.write(json.dumps(dict(qid=group['qid'], trajectory_ids=group['trajectory_ids'],
                        p_j=p, predicted_mixed=q, strict_em=group['labels'],
                        true_mixed=0 < sum(group['labels']) < 5, epoch=epoch, split=split,
                        weight_kind='EMA0.99'))+'\n')
                compute_seconds += time.monotonic()-tick
            if (iteration+1) % 32 == 0 or iteration+1 == rounds:
                stream.flush()
                atomic(output/('eval-%s-rank%d-progress.json' % (split, rank)),
                       dict(epoch=epoch, split=split, rank=rank,
                            completed_local_groups=min(offset+8, len(own)),
                            expected_local_groups=len(own), io_seconds=io_seconds,
                            compute_seconds=compute_seconds, elapsed_seconds=time.monotonic()-started,
                            epoch_time=time.time()))
                heartbeat = torch.tensor([min(offset+8, len(own))], dtype=torch.long, device=device)
                dist.all_reduce(heartbeat)
    temporary.replace(destination)
    dist.all_reduce(values)
    assert int(values[0]) == len(groups)
    result = summarized_stats(values)
    result['prediction_file_pattern'] = str(output/('epoch-%04d-%s-rank*.jsonl.gz' % (epoch, split)))
    if mixed_subset:
        dist.all_reduce(mixed_values)
        assert int(mixed_values[0]) == sum(0 < sum(g['labels']) < 5 for g in groups)
        result['mixed_only_separation'] = summarized_stats(mixed_values)
    return result


def save(base, model, ema, optimizer, epoch, cursor, steps, metrics, rank, config,
         immutable_name=None):
    batches = plan(read(base/'bank/groups.json')['train'], config['seed'], epoch,
                   config['fit_epoch_groups'])
    completed = cursor == 0
    assert steps == expected_steps(config, read(base/'bank/groups.json')['train'], epoch, cursor, completed)
    assert not completed or metrics is not None
    own = dict(rank=rank, model_sha256=model_hash(model), ema_sha256=model_hash(ema),
               optimizer_sha256=optimizer_hash(optimizer), steps=actual_steps(optimizer), rng=capture_rng())
    states = [None]*4
    dist.all_gather_object(states, own)
    for key in ('model_sha256', 'ema_sha256', 'optimizer_sha256'):
        assert len({s[key] for s in states}) == 1
    assert all(s['steps'] == steps for s in states)
    if rank == 0:
        root = base/'probe'
        dest = root/('epoch-%04d.pt' % epoch if completed else 'inflight.pt')
        stage = 'event_epoch_completed' if completed else 'event_evaluation_pending' if cursor == len(batches) else 'event_training'
        state = dict(role=ROLE, objective=OBJECTIVE, stage=stage, epoch=epoch, cursor=cursor,
                     completed_epoch=completed, steps=steps, source_steps=config['source_steps'],
                     source_epoch=config['source_epoch'], start_epoch=config['start_epoch'],
                     raw=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                     raw_buffers=native_buffers(model), ema_buffers=native_buffers(ema),
                     rank_states=states, metrics=metrics, actor_updates=0,
                     config_sha256=fingerprints(base/'config.json')['sha256'],
                     bank_sha256=config['mixed_bank_sha256'], fit_code_sha256=config['fit_file_sha256'],
                     source_commit=config['source_commit'], source_checkpoint=config['source_checkpoint'],
                     source_audit=config['source_audit'], sampling_plan=proof(read(base/'bank/groups.json')['train'], batches),
                     epoch_time=time.time())
        temporary = dest.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(dest)
        receipt = dict(path=str(dest), **fingerprints(dest), epoch=epoch, cursor=cursor,
                       completed_epoch=completed, stage=stage, steps=steps,
                       source_steps=config['source_steps'], source_epoch=config['source_epoch'],
                       actor_updates=0, source_commit=config['source_commit'], objective=OBJECTIVE, metrics=metrics)
        loaded = torch.load(dest, map_location='cpu', mmap=True, weights_only=False)
        assert loaded['steps'] == steps and len(loaded['rank_states']) == 4
        assert all(bool(torch.isfinite(v).all()) for key in ('raw', 'ema') for v in loaded[key].values())
        assert {int(item['step']) for item in loaded['optimizer']['state'].values()} == {steps}
        assert all(bool(torch.isfinite(item[key]).all()) for item in loaded['optimizer']['state'].values()
                   for key in ('exp_avg', 'exp_avg_sq'))
        assert all(set(row['rng']) == {'python', 'numpy', 'torch', 'cuda'} for row in loaded['rank_states'])
        del loaded
        receipt['independent_cpu_reload'] = True
        atomic(root/'latest.json', receipt)
        if immutable_name is not None:
            immutable = root/(immutable_name+'.pt')
            assert not immutable.exists(), 'Existing qualification must be inspected, never overwritten'
            os.link(dest, immutable)
            immutable_receipt = dict(receipt, path=str(immutable))
            atomic(root/(immutable_name+'.json'), immutable_receipt)
        if completed:
            atomic(root/('epoch-%04d.json' % epoch), receipt)
    dist.barrier()


def replay_next_update(saved, ddp, model, ema, optimizer, reader, groups,
                       indices, rank, device, config, root, epoch, batch, steps):
    """Accept only the second bit-exact execution; preserve checkpoint RAM too."""
    from .replay_diagnostics import snapshot, compare
    # Evaluation may have updated nonpersistent RoPE buffers or consumed RNG.
    # Capture the actual boundary while reusing unchanged checkpoint tensors.
    boundary = dict(saved)
    assert model_hash(model) == saved['rank_states'][rank]['model_sha256']
    assert model_hash(ema) == saved['rank_states'][rank]['ema_sha256']
    assert optimizer_hash(optimizer) == saved['rank_states'][rank]['optimizer_sha256']
    boundary['raw_buffers'] = native_buffers(model)
    boundary['ema_buffers'] = native_buffers(ema)
    boundary['rank_states'] = [dict(row) for row in saved['rank_states']]
    boundary['rank_states'][rank]['rng'] = capture_rng()
    model.train()
    update(ddp, model, ema, optimizer, reader, groups, indices, rank, device, config)
    expected = (model_hash(model), model_hash(ema), optimizer_hash(optimizer),
                buffer_hash(model), buffer_hash(ema), rng_hash(capture_rng()))
    reference = snapshot(model, ema, optimizer)
    restore(boundary, model, ema, optimizer, rank, device)
    model.train()
    metrics = update(ddp, model, ema, optimizer, reader, groups, indices, rank, device, config)
    actual = (model_hash(model), model_hash(ema), optimizer_hash(optimizer),
              buffer_hash(model), buffer_hash(ema), rng_hash(capture_rng()))
    if actual != expected:
        atomic(root/('next-update-replay-diagnostics-rank%d.json' % rank),
               dict(expected_hashes=expected, actual_hashes=actual,
                    differences=compare(reference, snapshot(model, ema, optimizer))))
    del reference
    exact = torch.tensor(int(actual == expected), device=device)
    dist.all_reduce(exact, op=dist.ReduceOp.MIN)
    assert exact.item() == 1, 'Event objective next-update replay mismatch'
    assert actual_steps(optimizer) == steps + 1
    ranks = [None]*4
    dist.all_gather_object(ranks, dict(rank=rank, raw=actual[0], ema=actual[1], adam=actual[2],
                                     raw_buffers=actual[3], ema_buffers=actual[4], rng=actual[5], bit_exact=True))
    if rank == 0:
        atomic(root/('next-update-proof-step-%08d.json' % (steps+1)),
               dict(status='ACTUAL_FOUR_RANK_THREE_EVENT_NEXT_UPDATE_BIT_EXACT',
                    source_steps=config['source_steps'], previous_steps=steps, steps=steps+1,
                    epoch=epoch, batch=batch, rank_proofs=ranks, objective=OBJECTIVE,
                    source_checkpoint=config['source_checkpoint'], epoch_time=time.time()))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    config = read(base/'config.json')
    validate_config(config)
    verify_runtime(config)
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    assert os.environ.get('FLASH_ATTENTION_DETERMINISTIC') == '1'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    # Separate from a launcher-owned fit.lock, so either invocation is safe.
    writer_lock = acquire_writer_lock(base, rank)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop)
    os.environ['QWEN7B_QUESTION_MAX_TOKENS'] = '273'
    random.seed(config['seed']+rank)
    np.random.seed(config['seed']+rank)
    torch.manual_seed(config['seed']+rank)
    groups = read(base/'bank/groups.json')
    assert len(groups['train']) == 169028
    assert len(groups['diagnostic']) == 512
    assert {g['qid'] for g in groups['train']}.isdisjoint(g['qid'] for g in groups['diagnostic'])
    assert fingerprints(base/'bank/groups.json')['sha256'] == config['mixed_bank_sha256']
    reader = FeatureReader(read(Path(config['training_base'])/'bank/features.json'))
    heldout = Path(config['selection_base'])
    held_groups = read(heldout/'bank/groups.json')['selection_validation']
    assert len(held_groups) == 2048
    fit_ids = {g['qid'] for g in groups['train']}
    assert fit_ids.isdisjoint(g['qid'] for g in held_groups)
    held_reader = FeatureReader(read(heldout/'bank/features.json'))
    model = Probe(config['model_dir'], 273, 20).to(device)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    root = base/'probe'
    predictions = root/'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    resumed = (root/'latest.json').exists()
    receipt = read(root/'latest.json') if resumed else config['source_checkpoint']
    actual_fingerprint = [fingerprints(receipt['path']) if rank == 0 else None]
    dist.broadcast_object_list(actual_fingerprint, src=0)
    assert actual_fingerprint[0]['sha256'] == receipt['sha256']
    assert actual_fingerprint[0]['bytes'] == receipt['bytes']
    saved = torch.load(receipt['path'], map_location='cpu', mmap=True, weights_only=False)
    if resumed:
        assert saved['role'] == ROLE and saved['objective'] == OBJECTIVE
        assert saved['source_checkpoint'] == config['source_checkpoint']
        assert saved['source_audit'] == config['source_audit']
        assert saved['source_steps'] == config['source_steps'] and saved['source_epoch'] == config['source_epoch']
        assert saved['config_sha256'] == fingerprints(base/'config.json')['sha256']
        assert saved['fit_code_sha256'] == config['fit_file_sha256']
        assert saved['bank_sha256'] == config['mixed_bank_sha256']
        assert saved['sampling_plan'] == proof(groups['train'], plan(groups['train'], config['seed'], saved['epoch'], config['fit_epoch_groups']))
        epoch, cursor, steps = position(saved, config, groups['train'])
        if rank == 0:
            retain_first_update(root, receipt, config)
        dist.barrier()
    else:
        verify_source(config, saved, actual_fingerprint[0])
        epoch, cursor, steps = config['start_epoch'], 0, config['source_steps']
    restore(saved, model, ema, optimizer, rank, device)
    dist.barrier()
    if rank == 0:
        atomic(root/'restore-acceptance.json', dict(status='ACTUAL_EVENT_FOUR_RANK_RAW_EMA_ADAM_RNG_RESTORED',
               checkpoint=receipt, source_checkpoint=config['source_checkpoint'], epoch=epoch, cursor=cursor,
               steps=steps, actor_updates=0, prefix_tokens=20, objective=OBJECTIVE, epoch_time=time.time()))
    if epoch <= config['fit_epochs']:
        batches = plan(groups['train'], config['seed'], epoch, config['fit_epoch_groups'])
        warm_batch = batches[cursor if cursor < len(batches) else 0]
        model.train()
        update(ddp, model, ema, optimizer, reader, groups['train'], warm_batch, rank, device, config, commit=False)
        restore(saved, model, ema, optimizer, rank, device)
        if rank == 0:
            atomic(root/'deterministic-warmup.json', dict(status='DDP_GRADIENT_BUCKET_WARMUP_NO_OPTIMIZER_UPDATE',
                   steps=steps, epoch=epoch, cursor=cursor, objective=OBJECTIVE, epoch_time=time.time()))
    needs_replay = True
    for e in range(epoch, config['fit_epochs']+1):
        batches = plan(groups['train'], config['seed'], e, config['fit_epoch_groups'])
        epoch_start_steps = steps - cursor if e == epoch else steps
        model.train()
        if rank == 0:
            atomic(root/('sampling-epoch-%04d.json' % e), proof(groups['train'], batches))
        for index, indices in enumerate(batches):
            if e == epoch and index < cursor:
                continue
            if needs_replay:
                metrics = replay_next_update(saved, ddp, model, ema, optimizer, reader, groups['train'],
                                             indices, rank, device, config, root, e, index+1, steps)
                needs_replay = False
                qualification = 'first-event-update' if not resumed else 'resumed-update-step-%08d' % (steps+1)
            else:
                metrics = update(ddp, model, ema, optimizer, reader, groups['train'], indices, rank, device, config)
                qualification = None
            steps += 1
            assert steps == epoch_start_steps+index+1
            if rank == 0:
                row = dict(epoch=e, batch=index+1, batches=len(batches), steps=steps,
                           source_steps=config['source_steps'], actor_updates=0, epoch_time=time.time(), **metrics)
                with (root/'curve.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                atomic(root/'progress.json', row)
            flag = torch.tensor(int(STOP), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if qualification or (index+1) % 200 == 0 or flag.item():
                save(base, model, ema, optimizer, e, index+1, steps, None, rank, config, qualification)
            if qualification:
                del saved
            if flag.item():
                dist.destroy_process_group()
                return
        save(base, model, ema, optimizer, e, len(batches), steps, None, rank, config)
        if rank == 0:
            atomic(root/'phase.json', dict(epoch=e, phase='event_evaluation_pending', steps=steps, epoch_time=time.time()))
        sampled = [groups['train'][i] for batch in batches for i in batch]
        metrics = dict(train=evaluate_event(ema, reader, sampled, rank, device, predictions, e, 'train', config),
                       diagnostic=evaluate_event(ema, reader, groups['diagnostic'], rank, device, predictions,
                                                 e, 'diagnostic', config, mixed_subset=True))
        metrics['sampling_plan'] = proof(groups['train'], batches)
        metrics['diagnostic_mixed_separation'] = metrics['diagnostic']['mixed_only_separation']
        metrics['selection2048'] = selection(ema, held_reader, held_groups, rank, device,
                                           predictions/('epoch-%04d-unseen2048.jsonl.gz' % e))
        save(base, model, ema, optimizer, e, 0, steps, metrics, rank, config)
        if rank == 0:
            atomic(root/'metrics-latest.json', dict(epoch=e, steps=steps, metrics=metrics, objective=OBJECTIVE, epoch_time=time.time()))
    if rank == 0:
        final = read(root/'latest.json')
        assert final['epoch'] == config['fit_epochs'] and final['completed_epoch'] and final['metrics'] is not None
        atomic(root/'complete.json', dict(status='BALANCED_PREFIX20_THREE_EVENT_FINETUNE_COMPLETE',
               source_epoch=config['source_epoch'], start_epoch=config['start_epoch'], end_epoch=config['fit_epochs'],
               additional_epochs=config['fit_epochs']-config['source_epoch'], source_steps=config['source_steps'],
               steps=steps, latest=final, objective=OBJECTIVE, actor_updates=0, epoch_time=time.time(),
               selection_quality_acceptance_claimed=False))
    dist.destroy_process_group()
    if writer_lock is not None:
        writer_lock.close()


if __name__ == '__main__':
    main()
