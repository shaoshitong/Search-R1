#!/usr/bin/env python3
"""Read one trusted REMOTE candidate chunk and profile saved-turn CPU decoding.

Run on an HPC CPU allocation with the existing training environment. No model,
CUDA, network, Slurm commands, source edits or training process interaction.
Only scalar aggregates, hashes and timing summaries are printed/saved. The raw
candidate file stays in its existing remote location. Never run data loading on
Mac or download a candidate chunk to use this script.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def summary(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    def q(f):
        return values[min(len(values)-1, int((len(values)-1)*f))]
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values),
                minimum=values[0], p50=q(.5), p90=q(.9), p99=q(.99), maximum=values[-1])


def prefix_sample(turn, tokenizer, prefix_function):
    stats = dict(calls=0, tokens_passed=0, decoder_seconds=0.,
                 visible_calls=0, executor_calls=0)
    def decode(skip_special):
        def run(tokens):
            began = time.perf_counter()
            try:
                return tokenizer.decode(tokens, skip_special_tokens=skip_special,
                                        clean_up_tokenization_spaces=False)
            finally:
                stats['calls'] += 1
                stats['tokens_passed'] += len(tokens)
                stats['decoder_seconds'] += time.perf_counter() - began
                stats['executor_calls' if skip_special else 'visible_calls'] += 1
        return run
    stored = turn['prefix_cut']
    kwargs = dict(eos_token_id=turn['eos_token_id'], pad_token_id=turn['pad_token_id'],
                  special_token_ids=turn['special_token_ids'])
    if stored and 'dual_decode_projection_role' in stored:
        kwargs.update(executor_decode=decode(True), effective_stop_token_ids=
                      turn['effective_stop_provenance']['effective_all_stop_token_ids'])
    began = time.perf_counter()
    cut = prefix_function(turn['raw_generated_token_ids'], decode(False), **kwargs)
    elapsed = time.perf_counter() - began
    return dict(seconds=elapsed, **stats,
        raw_tokens=turn['raw_generated_token_count'], kept_tokens=turn['kept_token_count'],
        request_tokens=len(turn['actual_request_token_ids']),
        exact_cut_equal=cut == stored,
        stored_cut_sha256=digest(stored), actual_cut_sha256=digest(cut),
        kept_count_equal=cut['kept_token_count'] == turn['kept_token_count'],
        kept_token_prefix_equal=cut['kept_token_ids'] ==
            turn['raw_generated_token_ids'][:turn['kept_token_count']],
        actual_decode_projection_role=cut.get('dual_decode_projection_role', cut.get('decode_projection_role')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--candidate-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='CP52 tokenizer files; model weights are never loaded')
    parser.add_argument('--max-turns', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--outer-actions', type=int, default=37907)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.platform == 'darwin':
        raise SystemExit('Remote CPU only: do not load/download candidate tensors on Mac.')
    if args.max_turns < 1 or args.repeats < 1:
        raise SystemExit('Use a positive bounded CPU sample and repeat count.')
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import torch
    from transformers import AutoTokenizer
    from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag
    began = time.perf_counter()
    # Owned training output is trusted pickle; map_location prevents CUDA loads.
    record = torch.load(args.candidate_file, map_location='cpu', weights_only=False)
    load_seconds = time.perf_counter() - began
    trajectory_turns = record['output'].non_tensor_batch['behavior_turns']
    turns = [turn for trajectory in trajectory_turns for turn in trajectory]
    raw = [int(t['raw_generated_token_count']) for t in turns]
    kept = [int(t['kept_token_count']) for t in turns]
    requests = [len(t['actual_request_token_ids']) for t in turns]
    aggregate = dict(trajectories=len(trajectory_turns), actions=len(turns),
        raw_generated_tokens=summary(raw), retained_policy_tokens=summary(kept),
        request_prompt_tokens=summary(requests),
        actual_raw_minus_retained_tokens=sum(raw)-sum(kept),
        raw_retained_ratio=sum(raw)/sum(kept) if sum(kept) else None,
        turns_per_trajectory=summary([len(t) for t in trajectory_turns]),
        turn_index_counts=dict(Counter(str(t['turn_index']) for t in turns)),
        finish_reason_counts=dict(Counter(t['finish_reason'] for t in turns)),
        stop_reason_counts=dict(Counter(str(t['stop_reason']) for t in turns)),
        raw_500_token_actions=sum(n == 500 for n in raw),
        closing_tag_counts=dict(Counter(str((t.get('prefix_cut') or {}).get('closing_tag')) for t in turns)),
        declared_raw_count_matches_token_ids=all(len(t['raw_generated_token_ids']) == n
                                                for t, n in zip(turns, raw)),
        retained_mask_tokens=int(record['output'].batch['sampling_mask'].sum().item()))
    sample_size = min(args.max_turns, len(turns))
    indices = ([round(i*(len(turns)-1)/(sample_size-1)) for i in range(sample_size)]
               if sample_size > 1 else [0])
    began = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer_load_seconds = time.perf_counter() - began
    # Warm the CPU tokenizer once. This is not part of measured prefix work.
    tokenizer.decode(turns[indices[0]]['raw_generated_token_ids'], skip_special_tokens=False,
                     clean_up_tokenization_spaces=False)
    runs = []
    for _ in range(args.repeats):
        rows = [prefix_sample(turns[i], tokenizer, raw_prefix_to_first_closing_tag) for i in indices]
        total = sum(r['seconds'] for r in rows)
        runs.append(dict(sampled_turns=len(rows), total_prefix_cpu_seconds=total,
            per_action_ms=total*1000/len(rows),
            tokenizer_decode_calls=sum(r['calls'] for r in rows),
            tokenizer_decode_token_work=sum(r['tokens_passed'] for r in rows),
            tokenizer_decode_cpu_seconds=sum(r['decoder_seconds'] for r in rows),
            per_action_decode_calls=statistics.mean(r['calls'] for r in rows),
            all_exact_cut_equal=all(r['exact_cut_equal'] for r in rows),
            all_kept_count_equal=all(r['kept_count_equal'] for r in rows),
            all_kept_token_prefix_equal=all(r['kept_token_prefix_equal'] for r in rows),
            cut_mismatch_count=sum(not r['exact_cut_equal'] for r in rows),
            stored_cut_hashes_sha256=digest([r['stored_cut_sha256'] for r in rows]),
            replay_cut_hashes_sha256=digest([r['actual_cut_sha256'] for r in rows]),
            sample_raw_tokens=summary([r['raw_tokens'] for r in rows]),
            sample_retained_tokens=summary([r['kept_tokens'] for r in rows]),
            sample_request_tokens=summary([r['request_tokens'] for r in rows])))
    median_ms = statistics.median(r['per_action_ms'] for r in runs)
    result = dict(format='saved_actual_rollout_cpu_prefix_cost_v1', epoch=time.time(),
        host=platform.node(), python=sys.version, repo=str(repo),
        candidate_file=str(args.candidate_file.resolve()), candidate_bytes=args.candidate_file.stat().st_size,
        candidate_load_seconds=load_seconds, tokenizer_path=str(args.checkpoint.resolve()),
        tokenizer_class=type(tokenizer).__name__, tokenizer_fast=bool(getattr(tokenizer, 'is_fast', False)),
        tokenizer_load_seconds=tokenizer_load_seconds,
        aggregate=aggregate, sample_indices_sha256=digest(indices),
        sample_selection='equally spaced complete-action indices in saved chunk', runs=runs,
        median_per_action_ms=median_ms,
        projected_single_driver_cpu_seconds=median_ms*args.outer_actions/1000,
        projected_actions=args.outer_actions,
        cpu_projection_scope='sample arithmetic only, not production wall fraction or a measured upper bound',
        production_prefix_function_executed=True, production_runtime_modified=False,
        model_loaded=False, cuda_allocated=False, network_called=False,
        scalar_or_hash_outputs_only=True,
        limitations=['One existing chunk, not the complete outer; evenly spaced sample is not a full distribution guarantee.',
                    'CPU prefix phase only; excludes GPU, Ray, retrieval, GroupCapture and other driver work.',
                    'Tokenizer and filesystem hardware affect CPU timing; projected serial driver cost is not total wall time.'])
    text = json.dumps(result, sort_keys=True, indent=2)
    if args.output:
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
