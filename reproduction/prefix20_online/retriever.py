"""Own loopback E5/Flat retrieval service for frozen-policy probe rollouts.

The upstream DenseRetriever supplies its original FP16 E5 mean pooling,
eight-way FP16 Flat sharding, document order and JSON result payload. Only
concurrent HTTP requests are coalesced; all CUDA work stays on the thread that
initialized FAISS. This wrapper does not enter the old actor/PPO admission path.
"""
import argparse
import asyncio
from collections import deque
import gc
import hashlib
import json
import os
from pathlib import Path
import tarfile
import threading
import time


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def prepare_corpus(archive, cache):
    """Extract the one upstream corpus member into this run's private cache.

    ``wiki-18.jsonl`` is actually a tar archive. Do not pass its archive bytes to
    the JSON reader and do not restore archived ownership or path names.
    """
    from reproduction.common8.temporary_cache import CORPUS_MEMBER
    archive, cache = Path(archive), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target, receipt = cache/'wiki_dump.jsonl', cache/'corpus.json'
    if target.exists() or receipt.exists():
        saved = json.loads(receipt.read_text())
        stat = archive.stat()
        if (saved['source'], saved['source_size'], saved['source_mtime_ns']) != (
                str(archive.resolve()), stat.st_size, stat.st_mtime_ns):
            raise ValueError('Retained corpus cache belongs to different input')
        if target.stat().st_size != saved['content_bytes']:
            raise ValueError('Retained corpus extraction is incomplete')
        return target, saved
    with archive.open('rb') as source, tarfile.open(fileobj=source, mode='r:') as tar:
        members = tar.getmembers()
        if len(members) != 1 or not members[0].isfile():
            raise ValueError('Original corpus archive requires one regular member')
        member = members[0]
        if member.name != CORPUS_MEMBER['name'] or member.size != CORPUS_MEMBER['size']:
            raise ValueError('Original corpus member or byte count differs')
        partial = cache/'wiki_dump.jsonl.partial'
        hasher, count = hashlib.sha256(), 0
        with partial.open('xb') as output, tar.extractfile(member) as content:
            for chunk in iter(lambda: content.read(8*1024**2), b''):
                output.write(chunk)
                hasher.update(chunk)
                count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if count != member.size:
            raise ValueError('Incomplete corpus extraction')
        os.replace(partial, target)
        stat = os.fstat(source.fileno())
    saved = dict(source=str(archive.resolve()), source_size=stat.st_size,
                 source_mtime_ns=stat.st_mtime_ns, content_bytes=count,
                 content_sha256=hasher.hexdigest(), member=member.name)
    publish(receipt, saved)
    return target, saved


class BatchedRequests:
    """Coalesce HTTP requests without using CUDA from a worker thread."""
    def __init__(self, retriever, *, batch_queries=512, delay_seconds=.002):
        self.retriever = retriever
        self.batch_queries, self.delay_seconds = batch_queries, delay_seconds
        self.pending = deque()
        self.task = None
        self.thread = threading.get_ident()
        self.requests = self.queries = self.batches = 0
        self.retrieval_seconds = 0.0

    async def submit(self, queries, return_scores):
        if not queries:
            return {'result': []}
        future = asyncio.get_running_loop().create_future()
        self.pending.append((list(queries), bool(return_scores), future))
        if self.task is None:
            self.task = asyncio.create_task(self.drain())
        return await future

    async def drain(self):
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
                started = time.monotonic()
                try:
                    if threading.get_ident() != self.thread:
                        raise RuntimeError('FAISS requests left their initialization thread')
                    results, scores = self.retriever.batch_search(
                        query_list=queries, num=3, return_score=True)
                    if len(results) != len(queries) or len(scores) != len(queries):
                        raise RuntimeError('Original retriever returned incomplete batch')
                    self.requests += len(chosen)
                    self.queries += len(queries)
                    self.batches += 1
                    self.retrieval_seconds += time.monotonic() - started
                    offset = 0
                    for items, with_scores, future in chosen:
                        end = offset + len(items)
                        payload = [[{'document': doc, 'score': float(score)}
                                    for doc, score in zip(docs, values)]
                                   for docs, values in zip(results[offset:end], scores[offset:end])]
                        if not with_scores:
                            payload = results[offset:end]
                        if not future.done():
                            future.set_result({'result': payload})
                        offset = end
                except Exception as error:
                    for _, _, future in chosen:
                        if not future.done():
                            future.set_exception(error)
        finally:
            self.task = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--data-cache', help='Reuse this run\'s already complete corpus/Arrow cache')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--port', type=int, default=18010)
    parser.add_argument('--batch-queries', type=int, default=512)
    args = parser.parse_args()
    if args.batch_queries < 1 or not 1 <= args.port <= 65535:
        parser.error('Positive query batch and valid port required')
    import datasets
    import faiss
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from search_r1.search import retrieval_server as upstream
    if torch.cuda.device_count() != 8 or faiss.get_num_gpus() != 8:
        raise ValueError('Probe rollout retrieval requires its assigned eight GPUs')
    cache = Path(args.cache)
    data_cache = Path(args.data_cache) if args.data_cache else cache
    cache.mkdir(parents=True, exist_ok=True)
    corpus_path, extraction = prepare_corpus(args.corpus, data_cache)
    upstream.load_corpus = lambda _: datasets.load_dataset(
        'json', data_files=str(corpus_path), split='train', num_proc=4,
        cache_dir=str(data_cache/'arrow'))
    config = upstream.Config(retrieval_method='e5', retrieval_topk=3,
        index_path=args.index, corpus_path=args.corpus, faiss_gpu=True,
        retrieval_model_path=args.encoder, retrieval_pooling_method='mean',
        retrieval_query_max_length=256, retrieval_use_fp16=True,
        retrieval_batch_size=args.batch_queries)
    started = time.monotonic()
    # DenseRetriever overwrites its CPU Index reference with the cloned GPU
    # IndexShards; do not retain an extra copy of the 64 GB CPU index.
    retriever = upstream.DenseRetriever(config)
    gc.collect()
    if retriever.index.count() != 8 or retriever.index.d != 768:
        raise ValueError('Original Flat index did not initialize all eight shards')
    if retriever.index.ntotal != len(retriever.corpus):
        raise ValueError('Original index and corpus document counts differ')
    ready = dict(run_id=args.run_id, pid=os.getpid(), ready=True,
        gpu_shards=8, dimension=int(retriever.index.d), documents=int(retriever.index.ntotal),
        index=str(Path(args.index).resolve()), corpus_archive=str(Path(args.corpus).resolve()),
        encoder=str(Path(args.encoder).resolve()), corpus=extraction,
        encoder_dtype='float16', index_dtype='float16', pooling='mean',
        query_prefix='query: ', topk=3, batch_queries=args.batch_queries,
        endpoint='http://127.0.0.1:'+str(args.port)+'/retrieve',
        initialization_seconds=time.monotonic()-started,
        cpu_full_index_retained=False, synthetic_queries=0)
    publish(cache/'ready.json', ready)
    print('PREFIX20_RETRIEVER_READY='+json.dumps(ready), flush=True)
    batching = BatchedRequests(retriever, batch_queries=args.batch_queries)
    app = FastAPI()

    @app.get('/health')
    async def health():
        return dict(ok=True, **ready, actual_http_requests=batching.requests,
                    actual_queries=batching.queries, actual_query_batches=batching.batches,
                    actual_retrieval_seconds=batching.retrieval_seconds)

    @app.post('/retrieve')
    async def retrieve(request: upstream.QueryRequest):
        if request.topk not in (None, 0, 3):
            raise HTTPException(status_code=400, detail='This experiment uses original top3')
        return await batching.submit(request.queries, request.return_scores)

    uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
