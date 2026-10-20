"""Bounded remote-GPU E1/E3 EMA downstream drift, never backend acceptance.

Use the same SHA-bound fixture/native references as replay qualification.  No
labels enter this tool, no threshold is tuned, and no raw tensor is serialized
or returned.  Only aggregate p_j/q_i drift, fixed0.5 threshold changes and
fixture-ranking changes are written as a small remote report.

Checkpoint JSON: {"checkpoints":[{"name":"E1","path":...,"bytes":...,
"sha256":...,"epoch":1,"steps":10464}, {"name":"E3",...,"epoch":3,...}]}.
The JSON itself must match --checkpoints-sha256; both files are freshly hashed
on the allocated GPU node before loading each EMA.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from .replay_feature_reader import ReplayFeatureReader
from .replay_probe_layers import (DIM, PREFIX, QMAX, REVISION, atomic, checked_file,
                                  fingerprints, read, remote)


def differences(before, after):
    if len(before) != len(after) or not before:
        raise ValueError('Aligned nonempty probability vectors required')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in list(before)+list(after)):
        raise ValueError('Finite probabilities required')
    delta = [b-a for a,b in zip(before, after)]
    flips_up = sum(a < .5 <= b for a,b in zip(before, after))
    flips_down = sum(b < .5 <= a for a,b in zip(before, after))
    return dict(count=len(delta), mean_absolute_difference=sum(abs(v) for v in delta)/len(delta),
                max_absolute_difference=max(map(abs, delta)),
                rms_difference=math.sqrt(sum(v*v for v in delta)/len(delta)),
                mean_signed_replay_minus_native=sum(delta)/len(delta),
                fixed_threshold=.5, threshold_changes=flips_up+flips_down,
                native_below_to_replay_at_or_above=flips_up,
                native_at_or_above_to_replay_below=flips_down)


def average_ranks(scores):
    result = [None]*len(scores)
    order = sorted(range(len(scores)), key=lambda i:-scores[i])
    start = 0
    while start < len(order):
        stop = start+1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        for i in order[start:stop]:
            result[i] = (start+stop-1)/2
        start = stop
    return result


def ranking(before, after, qids):
    if len(before) != len(after) or len(before) != len(qids) or len(set(qids)) != len(qids):
        raise ValueError('Aligned unique question ranking required')
    left, right = average_ranks(before), average_ranks(after)
    center_l, center_r = sum(left)/len(left), sum(right)/len(right)
    numerator = sum((a-center_l)*(b-center_r) for a,b in zip(left, right))
    denominator = math.sqrt(sum((a-center_l)**2 for a in left)*sum((b-center_r)**2 for b in right))
    old = sorted(range(len(qids)), key=lambda i:(-before[i], qids[i]))
    new = sorted(range(len(qids)), key=lambda i:(-after[i], qids[i]))
    old_rank, new_rank = {qids[i]:j for j,i in enumerate(old)}, {qids[i]:j for j,i in enumerate(new)}
    concordant = discordant = native_only_ties = replay_only_ties = both_ties = 0
    for i in range(len(qids)):
        for j in range(i):
            a, b = before[i]-before[j], after[i]-after[j]
            if a == 0 and b == 0:
                both_ties += 1
            elif a == 0:
                native_only_ties += 1
            elif b == 0:
                replay_only_ties += 1
            elif a*b > 0:
                concordant += 1
            else:
                discordant += 1
    tau_denominator = math.sqrt((concordant+discordant+native_only_ties)*
                                (concordant+discordant+replay_only_ties))
    budgets = sorted(set((1, min(5, len(qids)), min(512, len(qids)))))
    return dict(questions=len(qids), ranking_scope='declared_fixture_only_not_Mixed@512',
                spearman_with_average_ties=numerator/denominator if denominator else None,
                kendall_tau_b=(concordant-discordant)/tau_denominator if tau_denominator else None,
                strict_pair_order_inversions=discordant, concordant_pairs=concordant,
                native_tie_became_strict=native_only_ties, replay_tie_became_strict=replay_only_ties,
                unchanged_tied_pairs=both_ties,
                questions_changing_deterministic_rank=sum(old_rank[q] != new_rank[q] for q in qids),
                max_absolute_rank_shift=max(abs(old_rank[q]-new_rank[q]) for q in qids),
                top1_changed=qids[old[0]] != qids[new[0]],
                top_k_overlap={str(k):dict(overlap=len(set(old[:k]) & set(new[:k])), k=k)
                               for k in budgets})


def comparison(native, replay, ordered):
    """Pure aggregate math over probabilities; never uses correctness labels."""
    ids = [row['trajectory_id'] for row in ordered]
    if set(native) != set(ids) or set(replay) != set(ids) or len(ids) != len(set(ids)):
        raise ValueError('Probability results must cover exactly the requested own trajectories')
    groups = defaultdict(list)
    for row in ordered:
        groups[str(row['qid'])].append(row['trajectory_id'])
    if any(len(own) != 5 for own in groups.values()):
        raise ValueError('Downstream q_i comparison requires authentic complete G5 groups')
    before_p, after_p = [native[tid] for tid in ids], [replay[tid] for tid in ids]
    def mixed(own, values):
        p = [values[tid] for tid in own]
        return max(0., min(1., 1-math.prod(p)-math.prod(1-v for v in p)))
    qids = list(groups)
    before_q = [mixed(groups[qid], native) for qid in qids]
    after_q = [mixed(groups[qid], replay) for qid in qids]
    return dict(trajectory_probabilities=differences(before_p, after_p),
                mixed_probabilities=differences(before_q, after_q),
                ranking=ranking(before_q, after_q, qids),
                labels_read=False, threshold_fitted=False, threshold=.5,
                accepted_for_training=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification-root', type=Path, required=True)
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--checkpoints-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--micro-trajectories', type=int, default=4)
    args = parser.parse_args()
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['QWEN7B_QUESTION_MAX_TOKENS'] = str(QMAX)
    os.environ['SEARCH_G1_B_PREFIX_TOKENS'] = str(PREFIX)
    import torch
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        raise RuntimeError('Remote Linux allocated CUDA execution required; no local model loading')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Use an authorized existing Slurm GPU allocation')
    device = torch.device(args.device)
    if device.type != 'cuda' or not 1 <= args.micro_trajectories <= 16:
        raise ValueError('Allocated CUDA device and bounded trajectory micro batch required')
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = remote(args.qualification_root)
    qualified_file = fingerprints(root/'complete.json')
    if qualified_file['sha256'] != args.qualification_sha256:
        raise ValueError('Qualification completion SHA differs')
    qualified = read(root/'complete.json')
    if (qualified['status'] != 'BOUNDED_HF_REPLAY_CAPTURE_AND_NATIVE_COMPARISON_COMPLETE' or
            qualified['model_revision'] != REVISION or qualified['bit_exact_native_vllm_claimed'] is not False):
        raise ValueError('Wrong qualification lineage/model/backend claim')
    selection_file = checked_file(qualified['source_selection'])
    selection = read(selection_file['path'])
    ordered = selection['trajectories']
    if not 1 <= len(ordered) <= 128 or len(ordered) != qualified['trajectories']:
        raise ValueError('Compare only the declared bounded replay fixture')
    qids = defaultdict(list)
    for row in ordered:
        qids[str(row['qid'])].append(row['trajectory_id'])
    if any(len(tids) != 5 for tids in qids.values()):
        raise ValueError('Fixture must contain complete real G5 questions')
    for descriptor in qualified['source_native_files']:
        checked_file(descriptor)
    for descriptor in qualified['source_prefix_files']:
        checked_file(descriptor)
    replay_reader = ReplayFeatureReader([qualified['feature_manifests']['26']], capture_layer=26)
    from .packed_features import FeatureReader
    from .model import Probe, native_buffers, restore_buffers
    from .train import model_hash
    native_reader = FeatureReader([descriptor['path'] for descriptor in selection['native_layer26_manifests']])
    for row in ordered:
        tid = row['trajectory_id']
        if tid not in native_reader.records or tid not in replay_reader.records:
            raise ValueError('Native/replay banks lack an own declared trajectory')
        if str(native_reader.records[tid][1]['qid']) != str(row['qid']) or str(replay_reader.records[tid][1]['qid']) != str(row['qid']):
            raise ValueError('Own-trajectory question identity differs between backends')
    checkpoints_path = remote(args.checkpoints)
    checkpoint_spec_file = fingerprints(checkpoints_path)
    if checkpoint_spec_file['sha256'] != args.checkpoints_sha256:
        raise ValueError('Fixed checkpoint specification SHA differs')
    checkpoints = read(checkpoints_path)['checkpoints']
    if len(checkpoints) != 2 or {row['name'] for row in checkpoints} != {'E1', 'E3'}:
        raise ValueError('This fixed comparison requires both E1 and E3, without checkpoint selection')
    if {row['name']:row['epoch'] for row in checkpoints} != {'E1':1, 'E3':3}:
        raise ValueError('Checkpoint epoch identity differs')
    model_config_file = checked_file(qualified['model_files']['config.json'])
    model_dir = Path(model_config_file['path']).parent
    checked_file(qualified['model_files']['model.safetensors.index.json'])
    output = remote(args.output)
    output.mkdir(parents=True, exist_ok=False)
    atomic(output/'intent.json', dict(status='FIXED_E1_E3_DOWNSTREAM_DRIFT_COMPARISON',
        qualification=qualified_file, checkpoint_spec=checkpoint_spec_file,
        trajectories=len(ordered), questions=len(qids), labels_read=False,
        threshold_fitted=False, accepted_for_training=False))
    model = Probe(str(model_dir), QMAX, PREFIX).eval().requires_grad_(False).to(device)
    results = {}
    ids = [row['trajectory_id'] for row in ordered]
    try:
        for descriptor in checkpoints:
            checkpoint_file = checked_file(descriptor)
            state = torch.load(checkpoint_file['path'], map_location='cpu', mmap=True, weights_only=False)
            if (not state['completed_epoch'] or state['cursor'] != 0 or state['epoch'] != descriptor['epoch'] or
                    state['steps'] != descriptor['steps'] or state['actor_updates'] != 0):
                raise ValueError('Only sealed, completed and exactly identified probe checkpoints qualify')
            if len(state['rank_states']) != 4:
                raise ValueError('Missing original checkpoint rank provenance')
            model.load_state_dict(state['ema'], strict=True)
            restore_buffers(model, state['ema_buffers'])
            ema_sha = model_hash(model)
            if any(row['ema_sha256'] != ema_sha for row in state['rank_states']):
                raise ValueError('Actual loaded EMA parameter hash differs from checkpoint provenance')
            native, replay = {}, {}
            started = time.monotonic()
            with torch.inference_mode():
                for start in range(0, len(ids), args.micro_trajectories):
                    own = ids[start:start+args.micro_trajectories]
                    left, right = native_reader.get(own, device), replay_reader.get(own, device)
                    if (not torch.equal(left['attention_mask'], right['attention_mask']) or
                            not torch.equal(left['position_ids'], right['position_ids'])):
                        raise ValueError('Compare identical retained token slots and native positions only')
                    restore_buffers(model, state['ema_buffers'])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        p = model(left).sigmoid().float().cpu().tolist()
                    restore_buffers(model, state['ema_buffers'])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        q = model(right).sigmoid().float().cpu().tolist()
                    native.update(zip(own, p))
                    replay.update(zip(own, q))
                    atomic(output/'progress.json', dict(checkpoint=descriptor['name'],
                        completed_trajectories=start+len(own), expected_trajectories=len(ids),
                        elapsed_seconds=time.monotonic()-started, actor_optimizer_updates=0,
                        probe_optimizer_updates=0, epoch_time=time.time()))
            if model_hash(model) != ema_sha:
                raise ValueError('Frozen EMA parameters changed during backend comparison')
            restore_buffers(model, state['ema_buffers'])
            current_buffers = native_buffers(model)
            if set(current_buffers) != set(state['ema_buffers']) or any(
                    not torch.equal(value, state['ema_buffers'][name]) for name,value in current_buffers.items()):
                raise ValueError('EMA native RoPE buffers did not restore exactly')
            results[descriptor['name']] = dict(checkpoint=checkpoint_file, epoch=state['epoch'],
                steps=state['steps'], weight_kind='EMA0.99', ema_sha256=ema_sha,
                frozen_parameters_unchanged=True, native_buffers_restored_exactly=True,
                metrics=comparison(native, replay, ordered), elapsed_seconds=time.monotonic()-started)
            del state
    finally:
        native_reader.close()
        replay_reader.close()
    report = dict(status='FIXED_E1_E3_NATIVE_VS_HF_REPLAY_DOWNSTREAM_COMPARISON_COMPLETE',
        qualification=qualified_file, checkpoint_spec=checkpoint_spec_file,
        replay_reader_proof=replay_reader.proof, checkpoints=results,
        model_revision=REVISION, feature_layer=26, trajectories=len(ids), questions=len(qids),
        labels_read=False, thresholds_tuned=False, fixed_threshold=.5,
        actor_optimizer_updates=0, probe_optimizer_updates=0,
        accepted_for_training=False, full_bank_backend_equivalence_claimed=False,
        code_files={name:fingerprints(Path(__file__).with_name(name)) for name in
            ('compare_replay_probe_outputs.py', 'replay_feature_reader.py', 'model.py', 'packed_features.py', 'train.py')},
        epoch_time=time.time())
    atomic(output/'complete.json', report)
    print(json.dumps(dict(status=report['status'], output=str(output),
        checkpoints={name:value['metrics'] for name,value in results.items()},
        accepted_for_training=False)), flush=True)


if __name__ == '__main__':
    main()
