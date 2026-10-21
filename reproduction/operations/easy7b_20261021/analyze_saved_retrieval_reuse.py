"""CPU-only, read-only query-reuse audit of one completed Easy2560 pool.

Example (run on the source remote; never copy raw tensors to the Mac)::

  python -B analyze_saved_retrieval_reuse.py --base /remote/original-run \
      --outer 145 --output /remote/separate-audits/outer145-query-reuse.json

The five original full-candidate raw files are loaded serially on CPU.  Only
counts/provenance are written, never query strings, tokens, or trajectories.
Persisted trajectory/turn order is NOT actual asynchronous HTTP order.  The
unlimited-cache duplicate fraction is an empty-cache potential upper bound;
the bounded LRU calculation is an explicitly labelled order simulation.
No model, tokenizer, GPU, training, or retrieval service is initialized.
"""
import argparse
from collections import Counter, OrderedDict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import uuid


class InsufficientEvidence(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise InsufficientEvidence(reason)


def metadata(path):
    value = path.lstat()
    need(stat.S_ISREG(value.st_mode), 'source_is_not_a_regular_file')
    return dict(bytes=value.st_size, device=value.st_dev, inode=value.st_ino,
                mtime_ns=value.st_mtime_ns)


def read_small(path, limit=32 * 1024 * 1024):
    before = metadata(path)
    need(before['bytes'] <= limit, 'small_metadata_exceeds_read_limit')
    raw = path.read_bytes()
    need(metadata(path) == before and len(raw) == before['bytes'], 'metadata_changed_during_read')
    return raw, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), **before)


def read_json(path):
    raw, proof = read_small(path)
    return json.loads(raw), proof


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def bind(base, outer):
    run, run_proof = read_json(base / 'logs/run-manifest.json')
    need(run.get('easy2560_selection') is True and run.get('question_probe') is None and
         run.get('data_selection_mode') == 'mixed_first' and
         run.get('selection_candidate_questions') == 2560 and
         run.get('selection_selected_questions') == 512,
         'not_the_full_candidate_easy2560_experiment')
    raw, actual_proof = read_small(base / 'logs/actual-updates.jsonl')
    lines = raw.splitlines()
    records = [json.loads(line) for line in lines if line]
    matches = [row for row in records if row.get('step') == outer]
    need(len(matches) == 1, 'no_unique_completed_actual_outer')
    actual = matches[0]
    need(actual.get('run_id') == run.get('run_id') and
         actual.get('full_budget_complete') is True and
         actual.get('opportunities_processed') == 20 and
         actual.get('actual_optimizer_writes') == 20 and
         actual.get('update_status') == 'complete',
         'outer_is_not_a_successful_full_budget_completion')
    attempt = actual.get('attempt_id')
    need(type(attempt) is str and re.fullmatch('[0-9a-f]{32}', attempt) is not None,
         'missing_complete_attempt_identity')
    audit, audit_proof = read_json(base / 'artifacts/mixed-selection' / (attempt + '.json'))
    need(audit.get('step') == outer and audit.get('fixed_candidate_budget') is True and
         audit.get('attempted_candidates') == 2560 and audit.get('responses_per_group') == 5 and
         audit.get('selected_groups') == 512 and
         audit.get('policy_snapshot_id') == run['run_id'] + ':' + str(outer) + ':policy-' + attempt,
         'selected_pool_binding_or_complete_dimensions_differ')
    groups = audit.get('groups')
    need(type(groups) is list and len(groups) == 2560 and
         [group.get('ordinal') for group in groups] == list(range(2560)),
         'full_candidate_group_ordinals_missing')
    all_ids = [tid for group in groups for tid in group.get('trajectory_ids', [])]
    need(len(all_ids) == len(set(all_ids)) == 12800 and
         all(type(tid) is str and tid for tid in all_ids), 'full_candidate_trajectory_identity_differs')
    pool = Path(audit.get('pool_path', ''))
    parent = base / 'artifacts/mixed-candidate-pools'
    need(pool.is_absolute() and pool.resolve() == pool and pool.parent == parent and
         pool.name == attempt + '-pool-' + str(audit.get('pool_retry')),
         'successful_pool_path_is_not_the_bound_original_pool')
    discarded = audit.get('discarded_insufficient_pools', [])
    need(type(discarded) is list and type(audit.get('pool_retry')) is int and
         len(discarded) == audit['pool_retry'], 'discarded_pool_accounting_differs')
    discarded_calls = sum(item['raw_actual_retrieval_calls'] for item in discarded)
    expected_calls = audit['raw_actual_retrieval_calls'] - discarded_calls
    need(type(expected_calls) is int and expected_calls >= 0, 'missing_current_pool_actual_retrieval_count')
    config, config_proof = read_json(base / 'logs/resolved-training-config.json')
    retriever = config['resolved_config']['retriever']
    need(type(retriever.get('topk')) is int and retriever['topk'] > 0, 'missing_bound_retrieval_topk')
    admission, admission_proof = read_json(base / 'logs/retrieval-admission.json')
    need(admission.get('run_id') == run['run_id'] and admission.get('accepted') is True and
         all(admission.get(key) for key in ('index_sha256', 'corpus_sha256', 'encoder_assets')),
         'missing_same_run_retriever_asset_binding')
    asset_key = hashlib.sha256(json.dumps(
        {key: admission[key] for key in ('index_sha256', 'corpus_sha256', 'encoder_assets')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(run=run, actual=actual, audit=audit, groups=groups, pool=pool,
                expected_calls=expected_calls, excluded_discarded_pools=len(discarded),
                topk=retriever['topk'], asset_key=asset_key,
                metadata_sources=[run_proof, actual_proof, audit_proof, config_proof, admission_proof])


def queries_from_output(output, expected_ids):
    """Use saved executor text, not the stricter QA/real-EM answer parser."""
    batch_ids = output.non_tensor_batch['behavior_trajectory_id'].tolist()
    turns = output.non_tensor_batch['behavior_turns'].tolist()
    actual = output.meta_info['actual_retrieval_stats']
    need(batch_ids == expected_ids and len(turns) == len(actual) == 2560,
         'chunk_original_trajectory_order_or_retrieval_counts_differ')
    queries = []
    counts = []
    for tid, trajectory, expected_count in zip(batch_ids, turns, actual):
        need(type(trajectory) is list and 1 <= len(trajectory) <= 5,
             'missing_or_excess_native_turns')
        row_queries = []
        for index, turn in enumerate(trajectory):
            need(type(turn) is dict and turn.get('trajectory_id') == tid and
                 turn.get('turn_index') == index and type(turn.get('final_turn')) is bool,
                 'native_turn_identity_or_order_differs')
            cut = turn.get('prefix_cut')
            need(type(cut) is dict and type(cut.get('execution_text')) is str and
                 cut.get('kept_token_count') == turn.get('kept_token_count'),
                 'actual_saved_executor_projection_missing')
            # Exact LLMGenerationManager.postprocess_predictions implementation:
            # first regex action; group(2).strip(); no lowercasing/QA strictness.
            match = re.search(r'<(search|answer)>(.*?)</\1>', cut['execution_text'], re.DOTALL)
            if match is not None and match.group(1) == 'search' and not turn['final_turn']:
                row_queries.append(match.group(2).strip())
        need(type(expected_count) is int and len(row_queries) == expected_count,
             'reconstructed_logical_retrieval_count_differs_from_actual_row')
        queries.extend(row_queries); counts.append(expected_count)
    return queries, counts


def audit_pool(binding, outer, output_path, capacity):
    # CUDA is hidden before importing torch or unpickling native DataProto.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    import torch
    need(not torch.cuda.is_initialized(), 'cuda_was_initialized_in_cpu_audit_process')
    result = dict(schema_version=1, role='saved_full_candidate_exact_retrieval_reuse_audit',
        status='ANALYSIS_IN_PROGRESS', checked_unix=time.time(), outer=outer,
        base=str(binding['pool'].parent.parent.parent), run_id=binding['run']['run_id'],
        attempt_id=binding['actual']['attempt_id'], pool_path=str(binding['pool']),
        source_successful_full_budget_outer=True, full_candidate_pool=True,
        selected_only_journal_used=False, query_text_or_raw_tokens_written=False,
        models_or_tokenizers_loaded=False, retrieval_or_generation_performed=False,
        cuda_initialized=False, expected_chunks=5, completed_chunks=0,
        expected_candidate_questions=2560, expected_candidate_trajectories=12800,
        expected_current_successful_pool_actual_calls=binding['expected_calls'],
        excluded_discarded_insufficient_pools=binding['excluded_discarded_pools'],
        cache_key_scope=dict(exact_query_string=True, topk=binding['topk'], return_scores=True,
                             frozen_retrieval_asset_fingerprint=binding['asset_key']),
        order_warning='NOT_ACTUAL_ASYNC_HTTP_ORDER: saved chunk, original trajectory, then turn order only',
        unlimited_cache_label='EMPTY_CACHE_POTENTIAL_DUPLICATE_UPPER_BOUND_NOT_MEASURED_CACHE_HIT_RATE',
        lru_label='PERSISTED_ORDER_SIMULATION_ONLY_NOT_MEASURED_ASYNC_CACHE_HIT_RATE',
        lru_capacity=capacity, metadata_sources=binding['metadata_sources'], chunks=[])
    cumulative, lru = Counter(), OrderedDict()
    lru_hits = lru_misses = 0
    publish(output_path, result)
    for offset in range(0, 2560, 512):
        path = binding['pool'] / (str(offset) + '.pt')
        before = metadata(path)
        need(path.resolve() == path, 'raw_chunk_symlink_or_noncanonical_path')
        with path.open('rb') as stream:
            saved = torch.load(stream, map_location='cpu', weights_only=False)
        need(metadata(path) == before, 'raw_chunk_changed_during_cpu_load')
        need(saved.get('ordinals') == list(range(offset, offset + 512)),
             'raw_chunk_question_ordinals_differ')
        identity = saved['identity']
        need(identity.get('run_id') == binding['run']['run_id'] and
             identity.get('logical_update') == outer and
             identity.get('attempt_id') == binding['actual']['attempt_id'] and
             identity.get('policy_snapshot_id') == binding['audit']['policy_snapshot_id'],
             'raw_chunk_successful_attempt_identity_differs')
        groups = binding['groups'][offset:offset + 512]
        expected_ids = [tid for group in groups for tid in group['trajectory_ids']]
        need(identity.get('trajectory_ids') == expected_ids and
             saved.get('strict_correctness') == [group['strict_correctness'] for group in groups],
             'raw_chunk_original_trajectory_or_selection_EM_binding_differs')
        queries, counts = queries_from_output(saved['output'], expected_ids)
        local = Counter(queries)
        hits_before, misses_before = lru_hits, lru_misses
        for query in queries:
            # Other cache-key components are fixed and bound for this pool.
            if query in lru:
                lru_hits += 1; lru.move_to_end(query)
            else:
                lru_misses += 1; lru[query] = None
                if len(lru) > capacity:
                    lru.popitem(last=False)
        cumulative.update(local)
        calls, unique = len(queries), len(local)
        total_calls, total_unique = sum(cumulative.values()), len(cumulative)
        result['chunks'].append(dict(offset=offset, candidate_questions=512, trajectories=2560,
            actual_calls=sum(counts), unique_exact_query_strings=unique,
            empty_cache_unlimited_duplicate_calls=calls - unique,
            empty_cache_unlimited_duplicate_ratio=(calls - unique) / calls if calls else 0.0,
            cumulative_actual_calls=total_calls, cumulative_unique_exact_query_strings=total_unique,
            cumulative_empty_cache_duplicate_upper_bound_calls=total_calls - total_unique,
            cumulative_empty_cache_duplicate_upper_bound_ratio=(total_calls - total_unique) / total_calls if total_calls else 0.0,
            persisted_order_simulated_lru_hits=lru_hits - hits_before,
            persisted_order_simulated_lru_misses=lru_misses - misses_before,
            source_raw=dict(path=str(path), **before), source_raw_SHA_rehashed=False))
        result.update(completed_chunks=len(result['chunks']), checked_unix=time.time(),
                      cumulative_actual_calls=total_calls,
                      cumulative_unique_exact_query_strings=total_unique,
                      persisted_order_simulated_lru_hits=lru_hits,
                      persisted_order_simulated_lru_misses=lru_misses)
        publish(output_path, result)
        print(json.dumps(dict(status=result['status'], outer=outer,
              completed_chunks=len(result['chunks']), cumulative_actual_calls=total_calls,
              actual_async_cache_hit_rate_measured=False)), flush=True)
        del saved, queries, counts, local
        gc.collect()
    need(result['cumulative_actual_calls'] == binding['expected_calls'],
         'five_chunk_retrieval_total_differs_from_successful_pool_actual_audit')
    need(not torch.cuda.is_initialized(), 'cuda_was_initialized_during_cpu_audit')
    total, unique = result['cumulative_actual_calls'], result['cumulative_unique_exact_query_strings']
    result.update(status='COMPLETE', checked_unix=time.time(), analyzed_trajectories=12800,
        empty_cache_unlimited_duplicate_upper_bound_calls=total - unique,
        empty_cache_unlimited_duplicate_upper_bound_ratio=(total - unique) / total if total else 0.0,
        persisted_order_simulated_lru_hit_ratio=lru_hits / total if total else 0.0,
        actual_async_cache_hit_rate_measured=False)
    publish(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lru-capacity', type=int, default=4096)
    args = parser.parse_args()
    base, output = args.base, args.output
    need(base.is_absolute() and base.resolve() == base and base.is_dir(), 'explicit_canonical_source_base_required')
    need(output.is_absolute() and not output.resolve().is_relative_to(base) and
         output.resolve() == output, 'audit_output_must_be_outside_original_training_directory')
    need(args.outer > 0 and args.lru_capacity > 0, 'positive_outer_and_lru_capacity_required')
    try:
        result = audit_pool(bind(base, args.outer), args.outer, output, args.lru_capacity)
    except Exception as error:
        reason = str(error) if isinstance(error, InsufficientEvidence) else type(error).__name__
        failure = dict(status='INSUFFICIENT_EVIDENCE', outer=args.outer, base=str(base),
                       checked_unix=time.time(), reason=reason,
                       actual_async_cache_hit_rate_measured=False,
                       query_text_or_raw_tokens_written=False, source_training_modified=False)
        if output.exists():
            previous = json.loads(output.read_text())
            failure['completed_chunks_before_failure'] = previous.get('completed_chunks', 0)
            failure['partial_progress_before_failure'] = previous
        publish(output, failure)
        print(json.dumps(failure), flush=True)
        return 2
    print(json.dumps(dict(status=result['status'], output=str(output),
          completed_chunks=result['completed_chunks'], actual_async_cache_hit_rate_measured=False)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
