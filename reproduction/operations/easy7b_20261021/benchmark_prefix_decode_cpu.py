#!/usr/bin/env python3
"""Independent remote CPU comparison of HF scalar, native scalar and batch decoding.

A diagnostic copy of _dual_raw_prefix changes only its prefix decoding loop.
The real Qwen2TokenizerFast backend decodes blocks of 16 original prefixes.
Every prefix is still searched in increasing token order: there is no binary
search, tag-token shortcut, Unicode repair, or change to stop/cut semantics.
Existing production runtime and saved candidate tensors remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def batch_dual_raw_prefix(raw_generated_token_ids, decode, executor_decode,
                          batch_executor_decode, *, chunk_size, eos_token_id,
                          pad_token_id, special_token_ids, effective_stop_token_ids):
    """Diagnostic copy of production _dual_raw_prefix, with block prefix decoding."""
    from reproduction.common8.behavior_trace import (
        _tokens, _require, BehaviorTraceError, document_sha256,
        DECODE_PROJECTION_ROLE, DUAL_PROJECTION_ROLE)
    raw = _tokens(raw_generated_token_ids, 'raw_generated_token_ids')
    _require(type(eos_token_id) is int and type(pad_token_id) is int, 'missing_actual_eos_pad_identity')
    special = set(_tokens(special_token_ids, 'special_token_ids')) | {eos_token_id, pad_token_id}
    _require(effective_stop_token_ids is not None, 'missing_effective_stop_provenance')
    stops = sorted(set(_tokens(effective_stop_token_ids, 'effective_stop_token_ids')))
    _require(eos_token_id in stops, 'tokenizer_eos_not_actual_stop')
    trailing = bool(raw and raw[-1] == eos_token_id)
    trailing_actual_stop = bool(raw and raw[-1] != eos_token_id and raw[-1] in stops)
    eligible = raw[:-1] if trailing or trailing_actual_stop else raw
    visible, projected = decode(eligible), executor_decode(eligible)
    _require(isinstance(visible, str) and isinstance(projected, str), 'invalid_decoder_result')
    closing = '</search>' if '</search>' in projected else '</answer>' if '</answer>' in projected else None
    count = len(eligible)
    retained_visible, retained_projected, execution, suffix = visible, projected, projected, ''
    if closing is not None:
        found = False
        for block_start in range(1, len(eligible) + 1, chunk_size):
            counts = list(range(block_start, min(block_start + chunk_size, len(eligible) + 1)))
            candidates = batch_executor_decode([eligible[:n] for n in counts])
            for count, candidate in zip(counts, candidates):
                _require(isinstance(candidate, str), 'invalid_decoder_result')
                if closing in candidate:
                    retained_projected, retained_visible = candidate, decode(eligible[:count])
                    _require(isinstance(retained_visible, str) and visible.startswith(retained_visible) and
                             projected.startswith(candidate), 'ambiguous_decoding_boundary')
                    end = candidate.find(closing) + len(closing)
                    execution, suffix = candidate[:end], candidate[end:]
                    found = True
                    break
            if found:
                break
        if not found:
            raise BehaviorTraceError('closing_tag_without_raw_prefix_boundary')
    kept = eligible[:count]
    _require(not (set(stops) | {eos_token_id, pad_token_id}).intersection(kept),
             'included_actual_stop_or_pad_token')
    positions = [index for index, token in enumerate(kept) if token in special]
    classification = 'none' if not suffix else 'whitespace' if suffix.isspace() else 'nonwhitespace'
    previous = executor_decode(eligible[:max(0, count - 1)])
    _require(isinstance(previous, str), 'invalid_decoder_result')
    cut = dict(kept_token_count=count, kept_token_ids=kept, closing_tag=closing, execution_text=execution,
        decoded_retained_prefix=retained_visible, retained_token_boundary_suffix=suffix,
        token_boundary_suffix_classification=classification,
        retained_trailing_whitespace=suffix if classification == 'whitespace' else '',
        token_boundary_whitespace_retained=classification == 'whitespace',
        trailing_eos_excluded=trailing, raw_generated_token_count=len(raw),
        raw_tokens_sha256=document_sha256(raw), retained_tokens_sha256=document_sha256(kept),
        decode_projection_role=DECODE_PROJECTION_ROLE, decoded_full_generation=visible,
        full_decode_replacement_count=visible.count('\ufffd'),
        retained_decode_replacement_count=retained_visible.count('\ufffd'),
        dual_decode_projection_role=DUAL_PROJECTION_ROLE,
        decoded_executor_full_generation=projected, decoded_executor_retained_prefix=retained_projected,
        decoded_executor_previous_prefix=previous,
        executor_full_decode_replacement_count=projected.count('\ufffd'),
        executor_retained_decode_replacement_count=retained_projected.count('\ufffd'),
        included_nonterminal_special_positions=positions,
        effective_stop_token_ids_sha256=document_sha256(stops))
    if trailing_actual_stop:
        cut.update(trailing_actual_stop_excluded=True, excluded_trailing_stop_token_id=raw[-1])
    return cut


def sample(turn, tokenizer, *, chunk_size=None, native_scalar=False):
    from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag
    from reproduction.operations.easy7b_20261021.profile_saved_rollout_cpu import digest
    stats = dict(scalar_decode_calls=0, native_scalar_decode_calls=0,
                 batch_decode_calls=0, prefixes_in_batches=0,
                 decoded_token_work=0, decoder_cpu_seconds=0.)
    def decoder(skip):
        def run(tokens):
            began = time.perf_counter()
            try:
                if native_scalar:
                    return tokenizer.backend_tokenizer.decode(tokens, skip_special_tokens=skip)
                return tokenizer.decode(tokens, skip_special_tokens=skip,
                                        clean_up_tokenization_spaces=False)
            finally:
                stats['scalar_decode_calls'] += 1
                stats['native_scalar_decode_calls'] += int(native_scalar)
                stats['decoded_token_work'] += len(tokens)
                stats['decoder_cpu_seconds'] += time.perf_counter() - began
        return run
    def batch_decoder(prefixes):
        began = time.perf_counter()
        try:
            values = tokenizer.backend_tokenizer.decode_batch(prefixes, skip_special_tokens=True)
            if len(values) != len(prefixes):
                raise RuntimeError('Independent batch decoder returned incomplete prefixes')
            return values
        finally:
            stats['batch_decode_calls'] += 1
            stats['prefixes_in_batches'] += len(prefixes)
            stats['decoded_token_work'] += sum(map(len, prefixes))
            stats['decoder_cpu_seconds'] += time.perf_counter() - began
    kwargs = dict(eos_token_id=turn['eos_token_id'], pad_token_id=turn['pad_token_id'],
                  special_token_ids=turn['special_token_ids'],
                  effective_stop_token_ids=turn['effective_stop_provenance']['effective_all_stop_token_ids'])
    began = time.perf_counter()
    if chunk_size is None:
        cut = raw_prefix_to_first_closing_tag(turn['raw_generated_token_ids'], decoder(False),
                                              executor_decode=decoder(True), **kwargs)
    else:
        cut = batch_dual_raw_prefix(turn['raw_generated_token_ids'], decoder(False), decoder(True),
                                    batch_decoder, chunk_size=chunk_size, **kwargs)
    elapsed = time.perf_counter() - began
    return dict(seconds=elapsed, **stats, exact_cut_equal=cut == turn['prefix_cut'],
        kept_count_equal=cut['kept_token_count'] == turn['kept_token_count'],
        kept_ids_equal=cut['kept_token_ids'] == turn['raw_generated_token_ids'][:turn['kept_token_count']],
        cut_sha256=digest(cut), stored_cut_sha256=digest(turn['prefix_cut']))


def aggregate(rows):
    from reproduction.operations.easy7b_20261021.profile_saved_rollout_cpu import digest
    return dict(actions=len(rows), total_cpu_seconds=sum(r['seconds'] for r in rows),
        per_action_ms=statistics.mean(r['seconds'] for r in rows)*1000,
        scalar_decode_calls=sum(r['scalar_decode_calls'] for r in rows),
        native_scalar_decode_calls=sum(r['native_scalar_decode_calls'] for r in rows),
        batch_decode_calls=sum(r['batch_decode_calls'] for r in rows),
        prefixes_in_batches=sum(r['prefixes_in_batches'] for r in rows),
        decoded_token_work=sum(r['decoded_token_work'] for r in rows),
        decoder_cpu_seconds=sum(r['decoder_cpu_seconds'] for r in rows),
        all_exact_cut_equal=all(r['exact_cut_equal'] for r in rows),
        all_kept_count_equal=all(r['kept_count_equal'] for r in rows),
        all_kept_ids_equal=all(r['kept_ids_equal'] for r in rows),
        cut_mismatch_count=sum(not r['exact_cut_equal'] for r in rows),
        cut_hashes_sha256=digest([r['cut_sha256'] for r in rows]),
        stored_cut_hashes_sha256=digest([r['stored_cut_sha256'] for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--candidate-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--max-turns', type=int, default=256)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--outer-actions', type=int, default=37907)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.platform == 'darwin':
        raise SystemExit('Remote CPU only: do not load/download candidate tensors on Mac.')
    if min(args.max_turns, args.chunk_size, args.repeats) < 1:
        raise SystemExit('Positive CPU sample, chunk size and repeats required.')
    repo = args.repo.resolve(); sys.path.insert(0, str(repo))
    import torch
    from transformers import AutoTokenizer
    from reproduction.operations.easy7b_20261021.profile_saved_rollout_cpu import digest, summary
    began = time.perf_counter()
    record = torch.load(args.candidate_file, map_location='cpu', weights_only=False)
    load_seconds = time.perf_counter() - began
    all_turns = [turn for trajectory in record['output'].non_tensor_batch['behavior_turns'] for turn in trajectory]
    turns = [turn for turn in all_turns if 'dual_decode_projection_role' in (turn.get('prefix_cut') or {})]
    n = min(args.max_turns, len(turns))
    indices = [round(i*(len(turns)-1)/(n-1)) for i in range(n)] if n > 1 else [0]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if type(tokenizer).__name__ != 'Qwen2TokenizerFast' or not hasattr(tokenizer.backend_tokenizer, 'decode_batch'):
        raise RuntimeError('This independent experiment qualifies Qwen2TokenizerFast backend decode_batch only')
    # Warm both exact APIs before timing, without importing/initializing a model.
    sample(turns[indices[0]], tokenizer)
    sample(turns[indices[0]], tokenizer, native_scalar=True)
    sample(turns[indices[0]], tokenizer, chunk_size=args.chunk_size)
    runs = []
    for repeat in range(args.repeats):
        measured = {}
        # Rotate first position across all three modes to reduce order bias.
        modes = ('scalar', 'native_scalar', 'batch')
        offset = repeat % len(modes)
        order = modes[offset:] + modes[:offset]
        for name in order:
            measured[name] = aggregate([sample(turns[i], tokenizer,
                chunk_size=args.chunk_size if name == 'batch' else None,
                native_scalar=name == 'native_scalar') for i in indices])
        measured['equal_scalar_and_batch_cut_hashes'] = measured['scalar']['cut_hashes_sha256'] == measured['batch']['cut_hashes_sha256']
        measured['equal_all_mode_cut_hashes'] = len({measured[name]['cut_hashes_sha256'] for name in modes}) == 1
        measured['speedup'] = measured['scalar']['total_cpu_seconds']/measured['batch']['total_cpu_seconds']
        measured['native_scalar_speedup'] = measured['scalar']['total_cpu_seconds']/measured['native_scalar']['total_cpu_seconds']
        measured['order'] = list(order)
        runs.append(measured)
    scalar_ms = statistics.median(r['scalar']['per_action_ms'] for r in runs)
    native_ms = statistics.median(r['native_scalar']['per_action_ms'] for r in runs)
    batch_ms = statistics.median(r['batch']['per_action_ms'] for r in runs)
    result = dict(format='independent_qwen2_prefix_decode_batch_cpu_v1', epoch=time.time(),
        host=platform.node(), repo=str(repo), candidate_file=str(args.candidate_file.resolve()),
        candidate_bytes=args.candidate_file.stat().st_size, candidate_load_seconds=load_seconds,
        checkpoint_tokenizer=str(args.checkpoint.resolve()), tokenizer_class=type(tokenizer).__name__,
        tokenizers_parallelism=os.environ.get('TOKENIZERS_PARALLELISM'),
        source_actions=len(all_turns), dual_actions=len(turns), sampled_actions=n,
        sample_indices_sha256=digest(indices), sample_policy='equally spaced dual action indices',
        sample_raw_tokens=summary([turns[i]['raw_generated_token_count'] for i in indices]),
        sample_kept_tokens=summary([turns[i]['kept_token_count'] for i in indices]),
        chunk_size=args.chunk_size, runs=runs, scalar_median_per_action_ms=scalar_ms,
        native_scalar_median_per_action_ms=native_ms,
        native_scalar_median_cpu_speedup=scalar_ms/native_ms,
        batch_median_per_action_ms=batch_ms, median_cpu_speedup=scalar_ms/batch_ms,
        all_sampled_full_cuts_identical=all(r['equal_all_mode_cut_hashes'] and
            all(r[name]['all_exact_cut_equal'] and r[name]['all_kept_count_equal'] and
                r[name]['all_kept_ids_equal'] for name in ('scalar', 'native_scalar', 'batch')) for r in runs),
        projected_single_driver_cpu_seconds=dict(actions=args.outer_actions,
            scalar=scalar_ms*args.outer_actions/1000, native_scalar=native_ms*args.outer_actions/1000,
            batch=batch_ms*args.outer_actions/1000,
            scope='sample arithmetic only, not production wall time or a measured upper bound'),
        production_runtime_modified=False, model_loaded=False, cuda_allocated=False,
        network_called=False, scalar_or_hash_outputs_only=True,
        limitations=['Sampled saved CPU actions only; no end-to-end training speedup proof.',
                    'Native scalar uses the unchanged production prefix function with direct backend decode callbacks.',
                    'Diagnostic copied loop; original production runtime remains unchanged.',
                    'Backend decode_batch may decode up to chunk_size-1 unused future prefixes.',
                    'No Unicode monotonicity assumption: every original prefix is inspected in increasing order.'])
    text = json.dumps(result, sort_keys=True, indent=2)
    if args.output:
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
