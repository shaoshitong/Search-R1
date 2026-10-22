"""Independent saved-query E5/FAISS batching timing; no corpus or HTTP service.

All timings and numerical differences are diagnostic observations, never an
acceptance decision. Only small scalar/hash/ID reports are written; query text,
embeddings, index bytes, actor weights and rollout tensors are not output.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
BATCHES = (1, 2, 4, 8, 16, 32)
QUERY_COUNT = 128


def schedule():
    """Same query order in every run; reverse batch order on the second round."""
    return [(round_index + 1, size) for round_index, sizes in
            enumerate((BATCHES, BATCHES[::-1])) for size in sizes]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def compare(reference_ids, reference_scores, ids, scores):
    """Compare every ordered top3 slot; relative error uses the singleton score."""
    if not (len(reference_ids) == len(reference_scores) == len(ids) == len(scores)):
        raise ValueError('Retrieval comparison row count differs')
    mismatch_rows = mismatch_slots = 0
    maximum_absolute = maximum_relative = 0.0
    for expected_ids, expected_scores, actual_ids, actual_scores in zip(
            reference_ids, reference_scores, ids, scores):
        if not all(len(row) == 3 for row in
                   (expected_ids, expected_scores, actual_ids, actual_scores)):
            raise ValueError('Retrieval comparison requires ordered top3')
        mismatches = sum(a != b for a, b in zip(expected_ids, actual_ids))
        mismatch_rows += bool(mismatches); mismatch_slots += mismatches
        for expected, actual in zip(expected_scores, actual_scores):
            difference = abs(float(actual) - float(expected))
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, difference / max(abs(float(expected)), 1e-12))
    return dict(compared_queries=len(ids), compared_ordered_slots=3 * len(ids),
        ordered_top3_ids_equal=mismatch_slots == 0, id_mismatch_queries=mismatch_rows,
        id_mismatch_slots=mismatch_slots, score_max_abs=maximum_absolute,
        score_max_relative=maximum_relative, relative_denominator_floor=1e-12,
        score_comparison_scope='same ordered slots, including any ID mismatches',
        numerical_acceptance_decided=False)


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(path, report, stage, **values):
    report.update(stage=stage, checked_unix=time.time(), **values)
    publish(path, report)
    print(json.dumps(dict(stage=stage, checked_unix=report['checked_unix'],
        **{key: value for key, value in values.items() if key in
           ('round', 'batch_size', 'completed_runs', 'query_count', 'seconds', 'error_type')})), flush=True)


def synchronize(torch):
    for device in range(2):
        torch.cuda.synchronize(device)


def measure(torch, encoder, index, queries, batch_size, *, capture_embeddings=False):
    ids, scores = [], []
    embeddings = []
    encoder_seconds = faiss_seconds = cleanup_seconds = 0.0
    synchronize(torch); started = time.perf_counter()
    for offset in range(0, len(queries), batch_size):
        synchronize(torch); began = time.perf_counter()
        vectors = encoder.encode(queries[offset:offset + batch_size])
        synchronize(torch); encoder_seconds += time.perf_counter() - began
        began = time.perf_counter()
        current_scores, current_ids = index.search(vectors, 3)
        synchronize(torch); faiss_seconds += time.perf_counter() - began
        ids.extend(current_ids.tolist()); scores.extend(current_scores.tolist())
        if capture_embeddings:
            embeddings.append(vectors.copy())
        del vectors, current_scores, current_ids
        # DenseRetriever.batch_search also calls this after its document lookup.
        began = time.perf_counter(); torch.cuda.empty_cache()
        synchronize(torch); cleanup_seconds += time.perf_counter() - began
    synchronize(torch); seconds = time.perf_counter() - started
    return dict(query_count=len(queries), batch_size=batch_size,
        seconds=seconds, queries_per_second=len(queries) / seconds,
        encoder_seconds=encoder_seconds, faiss_seconds=faiss_seconds,
        post_search_cleanup_seconds=cleanup_seconds,
        ordered_ids_sha256=digest(ids), ordered_scores_sha256=digest(scores),
        first_two_top3_ids=ids[:2]), ids, scores, embeddings


def measure_faiss_only(torch, index, embeddings, batch_size):
    """Reuse singleton encoder vectors held only in RAM, preserving query order."""
    ids, scores = [], []
    faiss_seconds = cleanup_seconds = 0.0
    synchronize(torch); started = time.perf_counter()
    for offset in range(0, len(embeddings), batch_size):
        synchronize(torch); began = time.perf_counter()
        current_scores, current_ids = index.search(embeddings[offset:offset + batch_size], 3)
        synchronize(torch); faiss_seconds += time.perf_counter() - began
        ids.extend(current_ids.tolist()); scores.extend(current_scores.tolist())
        del current_scores, current_ids
        began = time.perf_counter(); torch.cuda.empty_cache()
        synchronize(torch); cleanup_seconds += time.perf_counter() - began
    synchronize(torch); seconds = time.perf_counter() - started
    return dict(query_count=len(embeddings), batch_size=batch_size, seconds=seconds,
        queries_per_second=len(embeddings) / seconds, faiss_seconds=faiss_seconds,
        post_search_cleanup_seconds=cleanup_seconds, ordered_ids_sha256=digest(ids),
        ordered_scores_sha256=digest(scores), first_two_top3_ids=ids[:2],
        embedding_source='first measured singleton encoder outputs, RAM only'), ids, scores


def run(args, report):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import torch
    import faiss
    import numpy as np
    from reproduction.operations.easy7b_20261021.compare_saved_observations import choose_cases
    from reproduction.easy2560_joint4.retrieval_service import make_four_shard_cloner, original_config
    from search_r1.search import retrieval_server as upstream

    event(args.output, report, 'READING_SAVED_CPU_CANDIDATES')
    saved = torch.load(args.raw_pool_file, map_location='cpu', weights_only=False)
    cases, selection = choose_cases(saved['output'], QUERY_COUNT)
    queries = [case['query'] for case in cases]
    if len(queries) != QUERY_COUNT or len(set(queries)) != QUERY_COUNT:
        raise ValueError('Need128 distinct actual saved queries')
    report.update(query_count=len(queries), ordered_queries_sha256=digest(queries),
        selection=selection, selected_case_identity_sha256=digest([
            {key: case[key] for key in ('trajectory_id_sha256', 'turn_index', 'row_index')}
            for case in cases]))
    del saved, cases; gc.collect()
    if torch.cuda.device_count() != 2 or faiss.get_num_gpus() != 2:
        raise ValueError('Independent benchmark requires exactly two visible GPUs')
    torch.cuda.set_device(0)
    report.update(physical_gpu_count=2, gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
        logical_shards=4, logical_to_physical_gpu=[0, 1, 0, 1],
        torch_version=torch.__version__, faiss_module=str(Path(faiss.__file__).resolve()),
        query_prefix='query: ', pooling='mean', query_max_length=256,
        encoder_dtype='float16', index_shards_float16=True,
        timing_scope='encoder including original CPU copy/empty_cache + FAISS top3 + original post-search empty_cache; no corpus lookup/HTTP/queue',
        schedule=[dict(round=r, batch_size=b) for r, b in schedule()])
    event(args.output, report, 'READING_CPU_FLAT_INDEX', query_count=len(queries))
    cpu_index = faiss.read_index(str(args.index))
    report.update(index_ntotal=int(cpu_index.ntotal), index_dimension=int(cpu_index.d),
        index_metric=int(cpu_index.metric_type))
    resources = [faiss.StandardGpuResources() for _ in range(4)]
    options = faiss.GpuMultipleClonerOptions(); options.useFloat16 = True; options.shard = True
    event(args.output, report, 'CLONING_FOUR_LOGICAL_SHARDS')
    index = make_four_shard_cloner(faiss, resources, [0, 1, 0, 1])(cpu_index, co=options)
    report['actual_index_shards'] = int(index.count())
    del cpu_index; gc.collect()
    config = original_config(upstream, args.index, '/dev/null', args.encoder)
    event(args.output, report, 'LOADING_ORIGINAL_E5_ENCODER')
    encoder = upstream.Encoder(model_name=config.retrieval_method,
        model_path=config.retrieval_model_path, pooling_method=config.retrieval_pooling_method,
        max_length=config.retrieval_query_max_length, use_fp16=config.retrieval_use_fp16)
    for size in BATCHES:
        event(args.output, report, 'WARMUP', batch_size=size)
        measure(torch, encoder, index, queries[:size], size)
    reference_ids = reference_scores = singleton_embeddings = None
    for round_index, size in schedule():
        event(args.output, report, 'MEASURING', round=round_index, batch_size=size)
        row, ids, scores, embedding_parts = measure(torch, encoder, index, queries, size,
            capture_embeddings=reference_ids is None)
        if reference_ids is None:
            reference_ids, reference_scores = ids, scores
            singleton_embeddings = np.concatenate(embedding_parts, axis=0)
            report.update(singleton_embeddings_shape=list(singleton_embeddings.shape),
                singleton_embeddings_dtype=str(singleton_embeddings.dtype),
                singleton_embeddings_sha256=hashlib.sha256(singleton_embeddings.tobytes()).hexdigest(),
                singleton_embeddings_saved=False)
            row['singleton_reference'] = True
        del embedding_parts
        row.update(round=round_index,
            versus_singleton=compare(reference_ids, reference_scores, ids, scores))
        event(args.output, report, 'MEASURING_FAISS_ONLY', round=round_index, batch_size=size)
        faiss_row, faiss_ids, faiss_scores = measure_faiss_only(torch, index, singleton_embeddings, size)
        faiss_row['versus_singleton'] = compare(reference_ids, reference_scores, faiss_ids, faiss_scores)
        row['faiss_only'] = faiss_row
        report['runs'].append(row)
        event(args.output, report, 'RUN_COMPLETE', round=round_index, batch_size=size,
              completed_runs=len(report['runs']), seconds=row['seconds'])
    summary = []
    singleton_seconds = statistics.median(row['seconds'] for row in report['runs'] if row['batch_size'] == 1)
    singleton_faiss_seconds = statistics.median(row['faiss_only']['seconds'] for row in report['runs'] if row['batch_size'] == 1)
    for size in BATCHES:
        rows = [row for row in report['runs'] if row['batch_size'] == size]
        seconds = statistics.median(row['seconds'] for row in rows)
        summary.append(dict(batch_size=size, repeats=len(rows), seconds_median=seconds,
            encoder_seconds_median=statistics.median(row['encoder_seconds'] for row in rows),
            faiss_seconds_median=statistics.median(row['faiss_seconds'] for row in rows),
            queries_per_second=len(queries) / seconds, singleton_over_batch_speedup=singleton_seconds / seconds,
            all_ordered_top3_ids_equal=all(row['versus_singleton']['ordered_top3_ids_equal'] for row in rows),
            score_max_abs=max(row['versus_singleton']['score_max_abs'] for row in rows),
            score_max_relative=max(row['versus_singleton']['score_max_relative'] for row in rows),
            faiss_only=dict(seconds_median=statistics.median(row['faiss_only']['seconds'] for row in rows),
                singleton_over_batch_speedup=singleton_faiss_seconds / statistics.median(row['faiss_only']['seconds'] for row in rows),
                all_ordered_top3_ids_equal=all(row['faiss_only']['versus_singleton']['ordered_top3_ids_equal'] for row in rows),
                score_max_abs=max(row['faiss_only']['versus_singleton']['score_max_abs'] for row in rows),
                score_max_relative=max(row['faiss_only']['versus_singleton']['score_max_relative'] for row in rows))))
    event(args.output, report, 'COMPLETED_TIMING_NUMERICS_PENDING_JUDGMENT', summary=summary,
          completed_runs=len(report['runs']), numerical_acceptance_decided=False,
          production_batching_qualified=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-pool-file', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema='saved_query_e5_four_shard_batching_diagnostic_v1',
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), hostname=os.uname().nodename,
        raw_pool_file=str(args.raw_pool_file), index_path=str(args.index), encoder_path=str(args.encoder),
        corpus_loaded=False, actor_loaded=False, production_service_called=False,
        production_modified=False, query_text_written=False, tensor_results_saved=False,
        numerical_acceptance_decided=False, production_batching_qualified=False, runs=[])
    try:
        run(args, report)
    except Exception as error:
        traceback.print_exc()
        event(args.output, report, 'FAILED_DIAGNOSTIC', error_type=type(error).__name__,
              failed_stage=report.get('stage'), production_batching_qualified=False)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
