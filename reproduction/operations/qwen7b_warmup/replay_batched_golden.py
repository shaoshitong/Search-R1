"""Bounded label-blind scalar/batched HF replay and frozen E4 EMA comparison.

Only already sealed training prefixes enter this qualification.  No rollout,
training, threshold fitting, native-vLLM equivalence, or final-pool read occurs.
Features stay in remote RAM; durable outputs contain small aggregate evidence.
"""
import argparse
import fcntl
import gc
import gzip
import json
import os
from pathlib import Path
import time

from . import replay_probe_layers as r
from .compare_replay_probe_outputs import comparison


def probabilities(model, features, source_buffers, device):
    import torch
    from .model import restore_buffers
    output = []
    with torch.inference_mode():
        for start in range(0, len(features), 8):
            own = features[start:start+8]
            batch = {key:torch.stack([row[26][key] for row in own]).to(device)
                     for key in ('features', 'attention_mask', 'position_ids')}
            restore_buffers(model, source_buffers)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output.extend(model(batch).sigmoid().float().cpu().tolist())
    return output


def run(config_path, config_sha, output):
    import torch
    from transformers import AutoModel
    from .replay_batched_context import BatchedContextCapture, compare_replay_features
    from .model import Probe, restore_buffers
    from .train import model_hash
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Owned remote CUDA Slurm allocation required')
    config_ref = r.fingerprints(r.remote(config_path))
    if config_ref['sha256'] != config_sha:
        raise ValueError('Golden config SHA differs')
    config = r.read(config_path)
    base = r.remote(output)
    base.mkdir(parents=True, exist_ok=True)
    guard = (base/'run.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (base/'complete.json').exists():
        raise ValueError('Existing golden evidence is immutable')
    source = r.checked_file(config['fixture'])
    selection = r.read(source['path'])
    if selection['questions'] != 8 or len(selection['trajectories']) != 40 or selection['labels_read_for_selection']:
        raise ValueError('Exact label-blind eight-question authentic G5 fixture required')
    wanted = {v['trajectory_id']:v['qid'] for v in selection['trajectories']}
    rows = {}
    for item in selection['prefix_files']:
        r.checked_file(item)
        with gzip.open(item['path'], 'rt') as stream:
            for line in stream:
                value = json.loads(line)
                tid = value['trajectory_id']
                if tid in rows or tid not in wanted:
                    raise ValueError('Repeated/undeclared fixture trajectory')
                rows[tid] = r.normalize_prefix(value, tid, wanted[tid], True)
    ordered = [rows[item['trajectory_id']] for item in selection['trajectories']]
    if len(rows) != 40:
        raise ValueError('Real fixture prefix coverage incomplete')
    counts = {}
    for row in ordered:
        counts[row['qid']] = counts.get(row['qid'], 0)+1
    if len(counts) != 8 or set(counts.values()) != {5}:
        raise ValueError('Real G5 question coverage differs')
    model_dir = selection['source_model_binding']['official_view']
    proofs, model_config = r.verify_model(model_dir, selection)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', local_files_only=True,
        trust_remote_code=False, low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    def causal(context, positions):
        if positions != list(range(len(context))):
            raise ValueError('Original causal positions differ')
        with torch.inference_mode():
            ids = torch.tensor([context], dtype=torch.long, device=device)
            value = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.tensor([positions], dtype=torch.long, device=device),
                output_hidden_states=True, use_cache=False, return_dict=True)
            return {layer:value.hidden_states[layer+1][0].detach().to('cpu', dtype=torch.bfloat16, copy=True)
                    for layer in r.LAYERS}
    started = time.monotonic()
    scalar = []
    for index, row in enumerate(ordered):
        scalar.append(r.replay_features(row, causal))
        r.atomic(base/'progress.json', dict(stage='SCALAR_CAUSAL_REFERENCE', trajectories=index+1,
            expected_trajectories=40, elapsed_seconds=time.monotonic()-started, time=time.time()))
    scalar_seconds = time.monotonic()-started
    scalar_peak = torch.cuda.max_memory_allocated(device)
    candidates = {}
    feature_metrics = {}
    timings = {}
    for batch in (16, 32):
        torch.cuda.reset_peak_memory_stats(device)
        own_started = time.monotonic()
        capture = BatchedContextCapture(model, batch_size=batch, max_context_tokens=16384,
            device=device, expected_hidden_size=r.DIM)
        values = capture.replay_many(ordered)
        candidates[batch] = values
        feature_metrics[str(batch)] = compare_replay_features(scalar, values)
        timings[str(batch)] = dict(seconds=time.monotonic()-own_started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
            context_batch=batch, padded_token_budget=16384)
        r.atomic(base/'progress.json', dict(stage='BATCHED_CAUSAL_REFERENCE', context_batch=batch,
            trajectories=40, scalar_seconds=scalar_seconds, batch_seconds=timings[str(batch)]['seconds'], time=time.time()))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = r.checked_file(config['checkpoint'])
    state = torch.load(checkpoint['path'], map_location='cpu', mmap=True, weights_only=False)
    if not state['completed_epoch'] or state['epoch'] != 4 or state['cursor'] != 0 or state['actor_updates'] != 0:
        raise ValueError('Actual sealed E4 probe required')
    probe = Probe(model_dir, r.QMAX, r.PREFIX).eval().requires_grad_(False).to(device)
    probe.load_state_dict(state['ema'], strict=True)
    restore_buffers(probe, state['ema_buffers'])
    initial_hash = model_hash(probe)
    if any(v['ema_sha256'] != initial_hash for v in state['rank_states']):
        raise ValueError('Loaded E4 EMA does not match source provenance')
    scalar_p = probabilities(probe, scalar, state['ema_buffers'], device)
    scalar_map = {row['trajectory_id']:p for row,p in zip(ordered, scalar_p)}
    downstream = {}
    for batch, values in candidates.items():
        predicted = probabilities(probe, values, state['ema_buffers'], device)
        own = comparison(scalar_map,
            {row['trajectory_id']:p for row,p in zip(ordered, predicted)}, ordered)
        def scores(probabilities_):
            result = {}
            for row,p in zip(ordered, probabilities_):
                result.setdefault(row['qid'], []).append(p)
            import math
            return {qid:1-math.prod(group)-math.prod(1-p for p in group) for qid,group in result.items()}
        left, right = scores(scalar_p), scores(predicted)
        left_order = sorted(left, key=lambda q:(-left[q], q))
        right_order = sorted(right, key=lambda q:(-right[q], q))
        own.update(complete_fixture_question_order_equal=left_order == right_order,
            top4_identical=left_order[:4] == right_order[:4], top4_overlap=len(set(left_order[:4])&set(right_order[:4])))
        downstream[str(batch)] = own
    if model_hash(probe) != initial_hash:
        raise ValueError('Frozen E4 probe parameters changed')
    result = dict(status='REAL_SCALAR_BATCHED_HF_REPLAY_AND_E4_GOLDEN_COMPARISON_COMPLETE',
        config=config_ref, fixture=source, checkpoint=checkpoint, ema_model_sha256=initial_hash,
        model_files=proofs, trajectories=40, questions=8,
        source_runtime='same_gpu_huggingface_flash_attention_2_frozen_causal_prefill',
        model_revision=r.REVISION, scalar=dict(seconds=scalar_seconds, peak_cuda_bytes=scalar_peak),
        batched=timings, feature_comparison=feature_metrics, downstream=downstream,
        labels_read=False, threshold_fitted=False, probe_updates=0, actor_updates=0,
        final_pool_read=False, native_vllm_bitexact_claim=False,
        qualification_policy='Report actual drift, fixed0.5 changes, complete rank and top4; parent accepts before production',
        auto_accepted_for_production=False, time=time.time())
    r.atomic(base/'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = run(args.config, args.config_sha256, args.output)
    except BaseException as exc:
        base = Path(args.output)
        if base.exists():
            r.atomic(base/'failure.json', dict(error_type=type(exc).__name__, error=str(exc)[:1000], time=time.time()))
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
