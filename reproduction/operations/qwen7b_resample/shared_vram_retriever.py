"""Original FP16 E5/Flat retrieval, with the whole Flat index on four/five GPUs.

Only device placement and bounded upload/query batching differ from the frozen
retriever. Corpus order, all original vectors, global IDs and top3 are retained.
"""
import argparse
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import threading
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shard_counts(weights, logical_shards=128):
    require(logical_shards in (64, 128), 'Expected 64 or 128 logical retrieval shards')
    require(len(weights) in (4, 5) and all(math.isfinite(x) and x > 0 for x in weights), 'Four or five positive retrieval weights required')
    shares = [logical_shards * x / sum(weights) for x in weights]
    counts = [math.floor(x) for x in shares]
    for index in sorted(range(len(weights)), key=lambda i: (shares[i] - counts[i], -i), reverse=True)[:logical_shards - sum(counts)]:
        counts[index] += 1
    require(sum(counts) == logical_shards and min(counts) > 0, 'Weights must place every logical shard across five GPUs')
    return counts


class OwnerThreadRetriever:
    """Initialize, search and release one backend on one persistent thread."""

    def __init__(self, factory):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='q7b-faiss-owner')
        self.thread_id = None
        self._backend = self._cleanup = None
        self._closing = self._released = False
        try:
            self.ready = self.executor.submit(self._initialize, factory).result()
        except BaseException:
            self.executor.shutdown(wait=True)
            raise

    def _initialize(self, factory):
        self.thread_id = threading.get_ident()
        self._backend, ready, self._cleanup = factory()
        return ready

    def _search(self, queries):
        require(threading.get_ident() == self.thread_id, 'FAISS search left its owner thread')
        started = time.monotonic()
        results, scores = self._backend.batch_search(query_list=queries, num=3, return_score=True)
        return results, scores, time.monotonic() - started

    async def search(self, queries):
        require(not self._closing, 'Retriever owner is shutting down')
        return await asyncio.get_running_loop().run_in_executor(self.executor, self._search, list(queries))

    def _release(self):
        require(threading.get_ident() == self.thread_id, 'FAISS cleanup left its owner thread')
        cleanup, self._cleanup = self._cleanup, None
        self._backend = None
        try:
            if cleanup is not None:
                cleanup()
        finally:
            self._released = True

    async def aclose(self):
        if self._released:
            return
        self._closing = True
        release = asyncio.get_running_loop().run_in_executor(self.executor, self._release)
        cancelled = False
        try:
            # A cancelled HTTP shutdown must not cancel cleanup queued behind
            # the last search. Keep the loop responsive until the owner exits.
            while not release.done():
                try:
                    await asyncio.shield(release)
                except asyncio.CancelledError:
                    cancelled = True
            release.result()
        finally:
            self.executor.shutdown(wait=True)
        if cancelled:
            raise asyncio.CancelledError

    def close(self):
        if self._released:
            return
        self._closing = True
        try:
            self.executor.submit(self._release).result()
        finally:
            self.executor.shutdown(wait=True)


class AsyncBatchedRequests:
    """HTTP-loop FIFO batching; the only CUDA caller is the backend owner."""

    def __init__(self, owner, *, batch_queries=32, delay_seconds=.002):
        require(batch_queries > 0 and delay_seconds >= 0, 'Invalid HTTP microbatch bounds')
        self.owner = owner
        self.batch_queries, self.delay_seconds = batch_queries, delay_seconds
        self.pending = deque()
        self.task = None
        self.closed = False
        self.requests = self.queries = self.batches = 0
        self.retrieval_seconds = 0.0

    async def submit(self, queries, return_scores):
        require(not self.closed, 'HTTP retrieval batcher is shutting down')
        if not queries:
            return {'result': []}
        future = asyncio.get_running_loop().create_future()
        self.pending.append((list(queries), bool(return_scores), future))
        if self.task is None:
            self.task = asyncio.create_task(self.drain())
        return await future

    @staticmethod
    def fail(items, error):
        for _, _, future in items:
            if not future.done():
                future.set_exception(error)

    async def drain(self):
        chosen = []
        try:
            while self.pending:
                await asyncio.sleep(self.delay_seconds)
                chosen, queries = [], []
                while self.pending:
                    item = self.pending[0]
                    if chosen and len(queries) + len(item[0]) > self.batch_queries:
                        break
                    chosen.append(self.pending.popleft())
                    queries.extend(item[0])
                    if len(queries) >= self.batch_queries:
                        break
                try:
                    results, scores, elapsed = await self.owner.search(queries)
                    require(len(results) == len(queries) and len(scores) == len(queries), 'Original retriever returned incomplete batch')
                    self.requests += len(chosen)
                    self.queries += len(queries)
                    self.batches += 1
                    self.retrieval_seconds += elapsed
                    responses, offset = [], 0
                    for items, with_scores, future in chosen:
                        end = offset + len(items)
                        payload = [[{'document': doc, 'score': float(score)} for doc, score in zip(docs, values)]
                                   for docs, values in zip(results[offset:end], scores[offset:end])]
                        responses.append((future, {'result': payload if with_scores else results[offset:end]}))
                        offset = end
                    # Construct every response first: a formatting failure must
                    # fail the whole chosen group, not leave partial successes.
                    for future, response in responses:
                        if not future.done():
                            future.set_result(response)
                except Exception as error:
                    self.fail(chosen, error)
                chosen = []
        except BaseException as error:
            self.fail(chosen, error)
            self.fail(self.pending, error)
            self.pending.clear()
            raise
        finally:
            self.task = None

    async def aclose(self):
        self.closed = True
        if self.task is not None:
            await asyncio.shield(self.task)


def build_backend(args, allocation, counts):
    """Called only by OwnerThreadRetriever, including all CUDA imports/checks."""
    indices, expected = allocation['retrieval_indices'], allocation['retrieval_cuda_uuids']
    devices = len(indices)
    from reproduction.common8.native_allocation import cuda_uuids
    require(cuda_uuids(devices) == expected, 'Retriever actual CUDA UUIDs differ from allocation')
    import datasets
    import faiss
    import numpy as np
    import torch
    from search_r1.search import retrieval_server as upstream
    from reproduction.operations.qwen7b_warmup import retriever as original
    require(torch.cuda.device_count() == faiss.get_num_gpus() == devices, 'Retriever device count differs from its admitted GPUs')
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
    retriever = None

    def cleanup():
        nonlocal retriever
        errors = []
        for device in range(devices):
            try:
                torch.cuda.synchronize(device)
            except Exception as error:
                errors.append(error)
        retriever = None
        flat_shards.clear()
        resources.clear()
        gc.collect()
        for device in range(devices):
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def streamed_flat(cpu, co=None, **kwargs):
        require(isinstance(cpu, faiss.IndexFlat) and co.useFloat16
                and cpu.metric_type in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2), 'Expected original full FP32 CPU Flat input for FP16 GPU clone')
        require(cpu.d == 768 and cpu.ntotal >= args.logical_shards, 'Original index shape differs')
        vectors = faiss.rev_swig_ptr(cpu.get_xb(), cpu.ntotal * cpu.d).reshape(cpu.ntotal, cpu.d)
        # One resource per physical GPU. Uploads stay on this initialization
        # thread; no threaded FAISS search uses these shared CUDA resources.
        for _ in range(devices):
            resource = faiss.StandardGpuResources()
            resource.setTempMemory(128 * 1024 ** 2)
            resources.append(resource)
        merged = faiss.IndexShards(cpu.d, False, True)
        cls = faiss.GpuIndexFlatIP if cpu.metric_type == faiss.METRIC_INNER_PRODUCT else faiss.GpuIndexFlatL2
        ordinal = added = 0
        checked, by_device = [], [0] * devices
        for device, number in enumerate(counts):
            for _ in range(number):
                begin, end = cpu.ntotal * ordinal // args.logical_shards, cpu.ntotal * (ordinal + 1) // args.logical_shards
                limit = (512 if args.logical_shards == 64 else 256) * 1024 ** 2
                require((end - begin) * cpu.d * 2 < limit, 'Logical FP16 shard exceeds the bounded reallocation size')
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
                progress = dict(status='STREAMING_FULL_ORIGINAL_FLAT_FP16_ACROSS_ADMITTED_GPUS', added=added,
                    total=int(cpu.ntotal), dimension=int(cpu.d), physical_gpus=devices, logical_flat_shards=ordinal,
                    shard_counts=counts, fp16_vector_bytes_by_device=by_device, scratch_bytes_per_device=128 * 1024 ** 2,
                    upload_vectors=16384, original_global_ids=True, elapsed_seconds=time.monotonic() - started)
                original.publish(cache / 'gpu-index-load-progress.json', progress)
                print('GPU_INDEX_LOAD=' + json.dumps(progress), flush=True)
        require(ordinal == merged.count() == args.logical_shards and added == merged.ntotal == cpu.ntotal
                and merged.metric_type == cpu.metric_type, 'Full original index membership/metric differs')
        original.publish(cache / 'gpu-index-load-verified.json', dict(progress,
            status='FULL_ORIGINAL_FLAT_FP16_ALL_VECTORS_PLACED_AND_SAMPLED_STORAGE_VERIFIED',
            placements=placements, sampled_offsets=checked, sampled_storage_exact_FP16=True,
            source_cpu_index_dtype_unchanged=True, metric=int(cpu.metric_type)))
        torch.cuda.set_device(0)  # Original E5 .cuda() and query .cuda() use first retrieval GPU.
        return merged

    original_cloner = upstream.faiss.index_cpu_to_all_gpus
    try:
        upstream.faiss.index_cpu_to_all_gpus = streamed_flat
        try:
            retriever = upstream.DenseRetriever(config)
        finally:
            upstream.faiss.index_cpu_to_all_gpus = original_cloner
        gc.collect()
        torch.cuda.set_device(0)
        require(retriever.index.d == 768 and retriever.index.ntotal == len(retriever.corpus)
                and retriever.index.count() == args.logical_shards and cuda_uuids(devices) == expected, 'Original corpus/index/UUID binding differs')
        ready = dict(run_id=args.run_id, pid=os.getpid(), ready=True, gpu_shards=devices,
            logical_flat_shards=args.logical_shards, dimension=int(retriever.index.d), documents=int(retriever.index.ntotal),
            index=str(Path(args.index).resolve()), corpus_archive=str(Path(args.corpus).resolve()),
            encoder=str(Path(args.encoder).resolve()), corpus=extraction, encoder_dtype='float16', index_dtype='float16',
            pooling='mean', query_prefix='query: ', topk=3, batch_queries=32, encoder_physical_gpu=indices[0],
            encoder_cuda_uuid=expected[0], actual_cuda_uuids=expected, numeric_cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            shard_counts=counts, original_global_ids=True, endpoint='http://127.0.0.1:%d/retrieve' % args.port,
            initialization_seconds=time.monotonic() - started, cpu_full_index_retained=False, synthetic_queries=0,
            cuda_owner_thread_id=threading.get_ident())
        original.publish(cache / 'ready.json', ready)
        print('SHARED_VRAM_RETRIEVER_READY=' + json.dumps(ready), flush=True)
        return retriever, ready, cleanup
    except BaseException as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            original.publish(cache / 'initialization-typed-oom.json', dict(status='ACTUAL_TYPED_RETRIEVER_INITIALIZATION_OOM',
                error_type=type(error).__name__, error=str(error), original_full_index_not_reduced=True,
                retry_performed=False, epoch=time.time()))
        try:
            cleanup()
        except BaseException:
            pass  # Preserve the original initialization failure.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'corpus', 'encoder', 'cache', 'data-cache', 'run-id', 'allocation-file'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--weights', default='17,22,22,33,34')
    parser.add_argument('--logical-shards', type=int, choices=(64, 128), default=128)
    parser.add_argument('--port', type=int, default=28831)
    args = parser.parse_args()
    allocation = json.loads(Path(args.allocation_file).read_text())
    indices = allocation['retrieval_indices']
    require(len(indices) in (4, 5) and os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(map(str, indices)), 'Retriever numeric CVD differs from admitted devices')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    counts = shard_counts([float(x) for x in args.weights.split(',')], args.logical_shards)
    require(len(counts) == len(indices), 'Retrieval weight/device counts differ')
    import uvicorn
    from fastapi import FastAPI, HTTPException
    owner = OwnerThreadRetriever(lambda: build_backend(args, allocation, counts))
    try:
        # The module (and its CUDA libraries) was first imported by the owner.
        from search_r1.search.retrieval_server import QueryRequest
        ready = owner.ready
        batching = AsyncBatchedRequests(owner, batch_queries=32)
        app = FastAPI()

        @app.get('/health')
        async def health():
            return dict(ok=True, **ready, actual_http_requests=batching.requests, actual_queries=batching.queries,
                        actual_query_batches=batching.batches, actual_retrieval_seconds=batching.retrieval_seconds)

        @app.post('/retrieve')
        async def retrieve(request: QueryRequest):
            if request.topk not in (None, 0, 3):
                raise HTTPException(status_code=400, detail='This experiment uses original top3')
            return await batching.submit(request.queries, request.return_scores)

        @app.on_event('shutdown')
        async def shutdown():
            try:
                await batching.aclose()
            finally:
                await owner.aclose()

        uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1)
    finally:
        owner.close()


if __name__ == '__main__':
    main()
