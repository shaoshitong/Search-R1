"""Original FP16 E5/Flat retrieval, with the whole Flat index on five GPUs.

Only device placement and bounded upload/query batching differ from the frozen
retriever. Corpus order, all original vectors, global IDs and top3 are retained.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shard_counts(weights):
    require(len(weights) == 5 and all(math.isfinite(x) and x > 0 for x in weights), 'Five positive retrieval weights required')
    shares = [128 * x / sum(weights) for x in weights]
    counts = [math.floor(x) for x in shares]
    for index in sorted(range(5), key=lambda i: (shares[i] - counts[i], -i), reverse=True)[:128 - sum(counts)]:
        counts[index] += 1
    require(sum(counts) == 128 and min(counts) > 0, 'Weights must place all 128 shards across five GPUs')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'corpus', 'encoder', 'cache', 'data-cache', 'run-id', 'allocation-file'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--weights', default='17,22,22,33,34')
    parser.add_argument('--port', type=int, default=28831)
    args = parser.parse_args()
    allocation = json.loads(Path(args.allocation_file).read_text())
    indices, expected = allocation['retrieval_indices'], allocation['retrieval_cuda_uuids']
    require(len(indices) == 5 and os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(map(str, indices)), 'Retriever numeric CVD differs from admitted five devices')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    from reproduction.common8.native_allocation import cuda_uuids
    require(cuda_uuids(5) == expected, 'Retriever actual CUDA UUIDs differ from allocation')
    counts = shard_counts([float(x) for x in args.weights.split(',')])
    import datasets
    import faiss
    import numpy as np
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from search_r1.search import retrieval_server as upstream
    from reproduction.operations.qwen7b_warmup import retriever as original
    require(torch.cuda.device_count() == faiss.get_num_gpus() == 5, 'Retriever requires exactly its five admitted GPUs')
    cache, data_cache = Path(args.cache), Path(args.data_cache)
    cache.mkdir(parents=True, exist_ok=True)
    corpus_path, extraction = original.prepare_corpus(args.corpus, data_cache)
    upstream.load_corpus = lambda _: datasets.load_dataset('json', data_files=str(corpus_path),
        split='train', num_proc=4, cache_dir=str(data_cache / 'arrow'))
    if (data_cache / 'prebuilt-arrow.json').exists():
        from reproduction.operations.b_replica_retained18.prebuilt_corpus import load as load_prebuilt
        upstream.load_corpus = lambda _: load_prebuilt(data_cache)
    config = upstream.Config(retrieval_method='e5', retrieval_topk=3, index_path=args.index,
        corpus_path=args.corpus, faiss_gpu=True, retrieval_model_path=args.encoder,
        retrieval_pooling_method='mean', retrieval_query_max_length=256,
        retrieval_use_fp16=True, retrieval_batch_size=32)
    started = time.monotonic()
    resources, flat_shards, placements = [], [], []

    def streamed_flat(cpu, co=None, **kwargs):
        require(isinstance(cpu, faiss.IndexFlat) and co.useFloat16
                and cpu.metric_type in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2), 'Expected original full FP32 CPU Flat input for FP16 GPU clone')
        require(cpu.d == 768 and cpu.ntotal >= 128, 'Original index shape differs')
        vectors = faiss.rev_swig_ptr(cpu.get_xb(), cpu.ntotal * cpu.d).reshape(cpu.ntotal, cpu.d)
        # One resource per physical GPU. Uploads stay on this initialization
        # thread; no threaded FAISS search uses these shared CUDA resources.
        for _ in range(5):
            resource = faiss.StandardGpuResources()
            resource.setTempMemory(128 * 1024 ** 2)
            resources.append(resource)
        merged = faiss.IndexShards(cpu.d, False, True)
        cls = faiss.GpuIndexFlatIP if cpu.metric_type == faiss.METRIC_INNER_PRODUCT else faiss.GpuIndexFlatL2
        ordinal = added = 0
        checked, by_device = [], [0] * 5
        for device, number in enumerate(counts):
            for _ in range(number):
                begin, end = cpu.ntotal * ordinal // 128, cpu.ntotal * (ordinal + 1) // 128
                require((end - begin) * cpu.d * 2 < 256 * 1024 ** 2, 'Logical FP16 shard exceeds the bounded reallocation size')
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device, cfg.useFloat16 = device, True
                gpu = cls(resources[device], cpu.d, cfg)
                flat_shards.append(gpu)
                for offset in range(begin, end, 16384):
                    gpu.add(vectors[offset:min(offset + 16384, end)])
                for offset in (0, (end - begin) // 2, end - begin - 1):
                    want = vectors[begin + offset:begin + offset + 1].astype(np.float16).astype(np.float32)
                    require(np.array_equal(gpu.reconstruct_n(offset, 1), want), 'Sampled original FP16 vector storage/order differs')
                    checked.append(begin + offset)
                require(gpu.ntotal == end - begin, 'GPU logical shard lost vectors')
                merged.add_shard(gpu)  # successive_ids=True adds original contiguous global offsets.
                placements.append(dict(logical_shard=ordinal, device=device, physical_gpu=indices[device],
                    gpu_uuid=expected[device], begin=begin, end=end, fp16_vector_bytes=(end - begin) * cpu.d * 2))
                ordinal += 1
                added += end - begin
                by_device[device] += (end - begin) * cpu.d * 2
                progress = dict(status='STREAMING_FULL_ORIGINAL_FLAT_FP16_ACROSS_FIVE_GPUS', added=added,
                    total=int(cpu.ntotal), dimension=int(cpu.d), physical_gpus=5, logical_flat_shards=ordinal,
                    shard_counts=counts, fp16_vector_bytes_by_device=by_device, scratch_bytes_per_device=128 * 1024 ** 2,
                    upload_vectors=16384, original_global_ids=True, elapsed_seconds=time.monotonic() - started)
                original.publish(cache / 'gpu-index-load-progress.json', progress)
                print('GPU_INDEX_LOAD=' + json.dumps(progress), flush=True)
        require(ordinal == merged.count() == 128 and added == merged.ntotal == cpu.ntotal
                and merged.metric_type == cpu.metric_type, 'Full original index membership/metric differs')
        original.publish(cache / 'gpu-index-load-verified.json', dict(progress,
            status='FULL_ORIGINAL_FLAT_FP16_ALL_VECTORS_PLACED_AND_SAMPLED_STORAGE_VERIFIED',
            placements=placements, sampled_offsets=checked, sampled_storage_exact_FP16=True,
            source_cpu_index_dtype_unchanged=True, metric=int(cpu.metric_type)))
        torch.cuda.set_device(0)  # Original E5 .cuda() and query .cuda() use first retrieval GPU.
        return merged

    upstream.faiss.index_cpu_to_all_gpus = streamed_flat
    try:
        retriever = upstream.DenseRetriever(config)
    except torch.cuda.OutOfMemoryError as error:
        original.publish(cache / 'initialization-typed-oom.json', dict(status='ACTUAL_TYPED_RETRIEVER_INITIALIZATION_OOM',
            error_type=type(error).__name__, error=str(error), original_full_index_not_reduced=True,
            retry_performed=False, epoch=time.time()))
        raise
    gc.collect()
    torch.cuda.set_device(0)
    require(retriever.index.d == 768 and retriever.index.ntotal == len(retriever.corpus)
            and retriever.index.count() == 128 and cuda_uuids(5) == expected, 'Original corpus/index/UUID binding differs')
    ready = dict(run_id=args.run_id, pid=os.getpid(), ready=True, gpu_shards=5,
        logical_flat_shards=128, dimension=int(retriever.index.d), documents=int(retriever.index.ntotal),
        index=str(Path(args.index).resolve()), corpus_archive=str(Path(args.corpus).resolve()),
        encoder=str(Path(args.encoder).resolve()), corpus=extraction, encoder_dtype='float16', index_dtype='float16',
        pooling='mean', query_prefix='query: ', topk=3, batch_queries=32, encoder_physical_gpu=indices[0],
        encoder_cuda_uuid=expected[0], actual_cuda_uuids=expected, numeric_cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        shard_counts=counts, original_global_ids=True, endpoint='http://127.0.0.1:%d/retrieve' % args.port,
        initialization_seconds=time.monotonic() - started, cpu_full_index_retained=False, synthetic_queries=0)
    original.publish(cache / 'ready.json', ready)
    print('SHARED_VRAM_RETRIEVER_READY=' + json.dumps(ready), flush=True)
    batching = original.BatchedRequests(retriever, batch_queries=32)
    app = FastAPI()

    @app.get('/health')
    async def health():
        return dict(ok=True, **ready, actual_http_requests=batching.requests, actual_queries=batching.queries,
                    actual_query_batches=batching.batches, actual_retrieval_seconds=batching.retrieval_seconds)

    @app.post('/retrieve')
    async def retrieve(request: upstream.QueryRequest):
        if request.topk not in (None, 0, 3):
            raise HTTPException(status_code=400, detail='This experiment uses original top3')
        return await batching.submit(request.queries, request.return_scores)

    uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
