"""One-H100 adapter integration on eight authentic, already viewed own-G5s.

This changes only a private probe copy, never the formal-training checkpoint.
It is not a four-rank/Ray acceptance, benchmark or checkpoint-selection run.
"""
import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from .group_probe_adapter import (GroupProbeAdapter, WARMUP_SHA256, file_sha256,
    probe_rng, rng_state, target4variance)
from .probe_update import atomic_json
from .recipe import MODEL, WARMUP

FIXTURE = ('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/'
    'algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2/'
    'fixture/fixture-complete.json')
FIXTURE_SHA256 = '745bdef27a1098603142a3e9f5506b4bd2777f8b699933923dbc1a208f44f1f8'
REPRESENTATIVES = ('block.self_attn.q_proj.weight',
                   'group_head.output.weight', 'group_head.output.bias')


def checked_json(reference):
    path = Path(reference['path'])
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Input manifest SHA differs: ' + str(path))
    return json.loads(path.read_text())


def load_real_groups(fixture_path, fixture_sha256, questions=8):
    if file_sha256(fixture_path) != fixture_sha256:
        raise ValueError('Authentic compact warmup fixture SHA differs')
    fixture = json.loads(Path(fixture_path).read_text())
    validation = checked_json(fixture['validation_groups'])
    # These are previously viewed diagnostic questions, not final51713 or a
    # fresh checkpoint-selection pool. Identities are selected before labels.
    selected = [validation[q] for q in sorted(validation)[:questions]]
    wanted = {tid for group in selected for tid in group['trajectory_ids']}
    records, used_shards = {}, []
    shards = [ref for ref in fixture['token_shards']
              if Path(ref['path']).name.startswith('validation-rank-')]
    if len(shards) != 2:
        raise ValueError('Expected the two compact validation-token shards')
    for reference in shards:
        if file_sha256(reference['path']) != reference['sha256']:
            raise ValueError('Compact validation shard SHA differs')
        used_shards.append(reference)
        with gzip.open(reference['path'], 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['trajectory_id'] in wanted:
                    records[row['trajectory_id']] = row
                if len(records) == len(wanted):
                    break
    if set(records) != wanted:
        raise ValueError('The eight real own-G5 groups are incomplete')
    groups = []
    for group in selected:
        rows = [records[tid] for tid in group['trajectory_ids']]
        prompt = rows[0]['prompt']
        if any(row['prompt'] != prompt or row['qid'] != group['qid'] for row in rows):
            raise ValueError('Own-G5 prompt/question alignment differs')
        groups.append(dict(qid=group['qid'], trajectory_ids=group['trajectory_ids'],
            prompt_token_ids=prompt, policy_token_ids=[row['policy_token_ids'] for row in rows],
            strict_em=group['labels']))
    evidence = dict(fixture=dict(path=str(fixture_path), sha256=fixture_sha256),
        validation_groups=fixture['validation_groups'], token_shards=used_shards,
        questions=len(groups), trajectories=len(wanted),
        qid_sha256=hashlib.sha256(json.dumps([g['qid'] for g in groups]).encode()).hexdigest(),
        source='already_viewed_warmup_diagnostic_own_G5', final51713_read=False)
    return groups, evidence


def step(adapter, groups):
    adapter.activate(training=True)
    adapter.probe.train()
    started = time.monotonic()
    with probe_rng(adapter):
        adapter.optimizer.zero_grad(set_to_none=True)
        target = torch.tensor([target4variance(g['strict_em']) for g in groups],
                              dtype=torch.float32, device=adapter.device)
        data = adapter.capture(groups, 80)
        with adapter.autocast():
            logits = adapter.probe(data)
        loss = (torch.sigmoid(logits).float() - target).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.probe.parameters(), 1.,
                                             error_if_nonfinite=True)
        adapter.optimizer.step()
        with torch.no_grad():
            for raw, average in zip(adapter.probe.parameters(), adapter.ema.parameters()):
                average.mul_(.9995).add_(raw, alpha=.0005)
        adapter.online_updates += 1
        adapter.EMA_updates += 1
    torch.cuda.synchronize(adapter.device)
    return dict(loss=float(loss.detach()), gradient_norm_before_clip=float(norm),
        seconds=time.monotonic() - started, online_updates=adapter.online_updates,
        actual_Adam_updates=adapter.warmup_cursor + adapter.online_updates,
        EMA_updates=adapter.EMA_updates,
        encoder_has_gradients=any(p.grad is not None for p in adapter.encoder.parameters()))


def representative_state(adapter):
    named = dict(adapter.probe.named_parameters())
    ema = dict(adapter.ema.named_parameters())
    out = {}
    for name in REPRESENTATIVES:
        parameter = named[name]
        out[name] = dict(raw=parameter.detach().cpu().clone(),
                         EMA=ema[name].detach().cpu().clone(), Adam={})
        for key, value in adapter.optimizer.state[parameter].items():
            out[name]['Adam'][key] = (value.detach().cpu().clone()
                                      if torch.is_tensor(value) else copy.deepcopy(value))
    return out


def rng_equal(left, right):
    import numpy as np
    return (left['python'] == right['python']
        and left['numpy'][0] == right['numpy'][0]
        and np.array_equal(left['numpy'][1], right['numpy'][1])
        and left['numpy'][2:] == right['numpy'][2:]
        and torch.equal(left['torch'], right['torch'])
        and torch.equal(left['cuda'], right['cuda']))


def compare_representatives(expected, actual):
    rows = {}
    for name in REPRESENTATIVES:
        row = {}
        for field in ('raw', 'EMA'):
            left, right = expected[name][field], actual[name][field]
            row[field] = dict(max_abs_difference=float((left - right).abs().max()),
                bitwise_equal=torch.equal(left, right),
                within_tolerance=torch.allclose(left, right, atol=1e-7, rtol=1e-5))
        row['Adam'] = {}
        for key, left in expected[name]['Adam'].items():
            right = actual[name]['Adam'][key]
            if torch.is_tensor(left):
                equal = (torch.equal(left, right) if key == 'step' else
                         torch.allclose(left, right, atol=1e-6, rtol=1e-4))
                row['Adam'][key] = dict(max_abs_difference=float((left - right).abs().max()),
                    bitwise_equal=torch.equal(left, right), within_tolerance=equal)
            else:
                row['Adam'][key] = dict(within_tolerance=left == right)
        rows[name] = row
    return rows


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Run this actual adapter integration on one allocated H100')
    output = Path(args.base) / 'preparation/probe-native-integration'
    if not str(output.resolve()).startswith('/data/user/'):
        raise ValueError('All integration tensors/checkpoints must remain in remote shared storage')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'writer.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    complete = output / 'complete.json'
    if complete.exists():
        print(complete.read_text(), flush=True)
        return
    started = time.monotonic()
    progress = lambda stage, **kw: atomic_json(output / 'progress.json', dict(
        stage=stage, pid=os.getpid(), job_id=os.environ.get('SLURM_JOB_ID'),
        elapsed_seconds=time.monotonic() - started, time=time.time(), **kw))
    progress('READ_AUTHENTIC_SMALL_INPUT')
    groups, evidence = load_real_groups(args.fixture, args.fixture_sha256, args.questions)
    progress('LOAD_INHERITED_E9_ADAPTER', input=evidence)
    adapter = GroupProbeAdapter.from_warmup(args.model_dir, args.checkpoint_path,
        checkpoint_sha256=args.checkpoint_sha256, device='cuda', rank=0)
    progress('ACTUAL_FROZEN_ENCODER_AND_PROBE_FORWARD')
    scores = adapter.score_groups(groups, prefix=80, batch_size=8)
    if not all(math.isfinite(s) for s in scores):
        raise FloatingPointError('Nonfinite native probe score')
    progress('PRIVATE_ADAM_UPDATE1', score_range=[min(scores), max(scores)])
    update1 = step(adapter, groups)
    adapter.offload()
    saved_representatives = representative_state(adapter)
    saved_probe_rng = copy.deepcopy(adapter.rng)
    private_checkpoint = output / 'private-state-after-update1.pt'
    temporary = private_checkpoint.with_suffix('.tmp')
    torch.save(adapter.state_dict([adapter.rng]), temporary)
    temporary.replace(private_checkpoint)
    checkpoint_ref = dict(path=str(private_checkpoint), bytes=private_checkpoint.stat().st_size,
                          sha256=file_sha256(private_checkpoint))
    progress('PRIVATE_NEXT_ADAM_UPDATE2', checkpoint=checkpoint_ref, update1=update1)
    update2 = step(adapter, groups)
    expected = representative_state(adapter)
    expected_rng = copy.deepcopy(adapter.rng)
    adapter.offload()
    progress('RESTORE_UPDATE1_AND_REPEAT_NEXT_ADAM_STEP', update2=update2)
    state = torch.load(private_checkpoint, map_location='cpu', weights_only=False)
    adapter.load_state_dict(state)
    del state
    loaded_representatives = representative_state(adapter)
    loading_comparison = compare_representatives(saved_representatives, loaded_representatives)
    loaded_state_bitwise = (all(value['bitwise_equal'] for row in loading_comparison.values()
        for key, value in row.items() if key != 'Adam') and
        all(value['bitwise_equal'] for row in loading_comparison.values()
            for value in row['Adam'].values()))
    loaded_rng_bitwise = rng_equal(saved_probe_rng, adapter.rng)
    loaded_counter_matches = adapter.online_updates == 1
    repeated2 = step(adapter, groups)
    actual = representative_state(adapter)
    comparison = compare_representatives(expected, actual)
    actual_rng_equal = rng_equal(expected_rng, adapter.rng)
    adapter.offload()
    losses_equal = math.isclose(update2['loss'], repeated2['loss'], abs_tol=1e-7, rel_tol=1e-6)
    all_parameters_ok = all(value['within_tolerance'] for row in comparison.values()
        for key, value in row.items() if key != 'Adam')
    all_Adam_ok = all(value['within_tolerance'] for row in comparison.values()
                     for value in row['Adam'].values())
    accepted = (loaded_state_bitwise and loaded_rng_bitwise and loaded_counter_matches
        and losses_equal and all_parameters_ok and all_Adam_ok and actual_rng_equal
        and update2['actual_Adam_updates'] == repeated2['actual_Adam_updates'] == adapter.warmup_cursor + 2
        and not any(row['encoder_has_gradients'] for row in (update1, update2, repeated2)))
    report = dict(status='ONE_GPU_NATIVE_PROBE_ADAPTER_INTEGRATION_PASS' if accepted else
                  'ONE_GPU_NATIVE_PROBE_ADAPTER_INTEGRATION_MISMATCH', input=evidence,
        checkpoint_source=dict(path=args.checkpoint_path, sha256=args.checkpoint_sha256),
        model_dir=args.model_dir, GPU=torch.cuda.get_device_name(), GPU_count=1,
        source_checkpoint_read_only=True, formal_probe_or_actor_modified=False,
        scope='actual official-frozen15 encoder + inherited raw probe FW/BW/Adam and representative next-step restore',
        score_semantics='4*predicted_population_EM_variance', prefix=80,
        score_range=[min(scores), max(scores)], update1=update1, update2=update2,
        restored_update2=repeated2, next_loss_matches=losses_equal,
        saved_vs_loaded_representative_comparison=loading_comparison,
        loaded_representative_raw_EMA_Adam_bitwise_equal=loaded_state_bitwise,
        loaded_probe_RNG_bitwise_equal=loaded_rng_bitwise,
        loaded_counter_matches=loaded_counter_matches,
        representative_comparison=comparison, probe_RNG_matches=actual_rng_equal,
        next_step_tolerances=dict(raw_EMA_atol=1e-7, raw_EMA_rtol=1e-5,
                                  Adam_moment_atol=1e-6, Adam_moment_rtol=1e-4),
        private_checkpoint=checkpoint_ref, actual_optimizer_executions=3,
        private_model_updates_after_restore=2, seconds=time.monotonic() - started,
        not_claimed=['four-rank NCCL/Ray integration', 'all-parameter numerical equivalence',
                     'formal training progress', 'selection quality or speed benchmark'])
    atomic_json(output / 'report.json', report)
    if not accepted:
        raise RuntimeError('Native probe representative resume comparison failed; see remote report')
    private_checkpoint.unlink()
    report['private_checkpoint_removed_after_acceptance'] = not private_checkpoint.exists()
    atomic_json(output / 'report.json', report)
    atomic_json(complete, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--model-dir', default=MODEL)
    parser.add_argument('--checkpoint-path', default=WARMUP)
    parser.add_argument('--checkpoint-sha256', default=WARMUP_SHA256)
    parser.add_argument('--fixture', default=FIXTURE)
    parser.add_argument('--fixture-sha256', default=FIXTURE_SHA256)
    parser.add_argument('--questions', type=int, choices=(8,), default=8)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        output = Path(args.base) / 'preparation/probe-native-integration'
        if str(output.resolve()).startswith('/data/user/'):
            atomic_json(output / 'failure.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
