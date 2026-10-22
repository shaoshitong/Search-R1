"""Standalone two-physical-GPU service using the original four logical shards.

Uses the existing E5 encoder, original FP16 Flat cloner, original corpus and
the unchanged upstream HTTP handler.  Four quarter-index shards retain their
order and are mapped [0, 1, 0, 1] onto two allocated GPUs.  This is a separate
Slurm service, not an actor/trainer process.  Cache is disabled.

Large source assets are opened read-only.  The TAR's original JSONL member and
Arrow cache live only in a newly owned remote cache directory (default tmpfs).
The default has no rebatching. An explicitly enabled private app batches only
FAISS vectors after unchanged singleton E5 calls; no background FAISS thread.
"""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tarfile
import threading
import time


REPO = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def rss_bytes():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def atomic(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_record(path, expected_bytes=None):
    path = Path(path).resolve(strict=True)
    value = path.stat()
    if not path.is_file() or (expected_bytes is not None and value.st_size != expected_bytes):
        raise ValueError('Original retrieval asset file/size differs: ' + str(path))
    return dict(path=str(path), bytes=value.st_size, device=value.st_dev,
                inode=value.st_ino, mtime_ns=value.st_mtime_ns)


def original_config(upstream, index, corpus, encoder):
    return upstream.Config(retrieval_method='e5', retrieval_topk=3,
        index_path=str(index), corpus_path=str(corpus), faiss_gpu=True,
        retrieval_model_path=str(encoder), retrieval_pooling_method='mean',
        retrieval_query_max_length=256, retrieval_use_fp16=True, retrieval_batch_size=512)


def make_four_shard_cloner(faiss, resources, mapping):
    """Keep the original four-shard order; require the actual multiple_py API."""
    if mapping != [0, 1, 0, 1] or len(resources) != 4:
        raise ValueError('Expected four original logical quarter shards on GPU mapping [0,1,0,1]')
    multiple = getattr(faiss, 'index_cpu_to_gpu_multiple_py', None)
    if not callable(multiple):
        raise RuntimeError('Runtime FAISS has no index_cpu_to_gpu_multiple_py; no two-shard fallback')

    def clone(index, co=None, ngpu=-1):
        if co is None or co.useFloat16 is not True or co.shard is not True:
            raise ValueError('Original FP16 sharded cloner options are required')
        # gpus is explicit: the Python helper's implicit range(4) would try to
        # access two unallocated GPUs.  The parent control confirms this API.
        return multiple(resources, index, co=co, gpus=mapping)
    return clone


def track_batch_search(backend, statistics, owner_thread):
    def search(*args, **kwargs):
        if threading.get_ident() != owner_thread:
            raise RuntimeError('FAISS service call moved off its initialization thread')
        queries = kwargs.get('query_list', args[0] if args else [])
        count = 1 if isinstance(queries, str) else len(queries)
        statistics['logical_requests'] += 1
        statistics['logical_queries'] += count
        start = time.monotonic()
        try:
            value = backend(*args, **kwargs)
        except BaseException:
            statistics['failed_requests'] += 1
            raise
        else:
            statistics['completed_requests'] += 1
            statistics['completed_queries'] += count
            statistics['last_completed_unix'] = time.time()
            return value
        finally:
            statistics['backend_seconds'] += time.monotonic() - start
    return search


def singleton_faiss_app(upstream, retriever, identity, statistics, owner_thread,
                        batch_size=16, max_wait_ms=2):
    """Optional private app; preserve singleton E5 math and original fallback."""
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from search_g1.singleton_faiss_batcher import SingletonFaissBatcher

    identity.update(request_rebatching_enabled=True, singleton_e5=True,
        faiss_batch_size=batch_size, faiss_max_wait_ms=max_wait_ms,
        backend_seconds_scope='singleton batches: owner-thread encoder plus FAISS, excluding queue/documents; fallback: original batch_search wall time')
    statistics.update(batch_attempts=0, batch_input_queries=0, max_batch_input_queries=0)

    class TimedBatcher(SingletonFaissBatcher):
        def _execute(self, batch):
            active = sum(not future.cancelled() for _, future in batch)
            if not active:
                return
            began = time.monotonic()
            statistics['batch_attempts'] += 1
            statistics['batch_input_queries'] += active
            statistics['max_batch_input_queries'] = max(statistics['max_batch_input_queries'], active)
            try:
                return super()._execute(batch)
            finally:
                statistics['backend_seconds'] += time.monotonic() - began

    @asynccontextmanager
    async def lifespan(app):
        if threading.get_ident() != owner_thread:
            raise RuntimeError('Private FAISS batching app moved off its initialization thread')
        app.state.batcher = TimedBatcher(retriever.encoder, retriever.index,
            max_batch_size=batch_size, max_wait_ms=max_wait_ms)
        try:
            yield
        finally:
            await app.state.batcher.close()

    app = FastAPI(lifespan=lifespan)

    @app.post('/retrieve-original')
    async def retrieve_original(request: upstream.QueryRequest):
        return await upstream.retrieve_endpoint(request)

    @app.post('/retrieve')
    async def retrieve(request: upstream.QueryRequest):
        topk = request.topk or upstream.config.retrieval_topk
        if len(request.queries) != 1 or topk != 3:
            return await upstream.retrieve_endpoint(request)
        statistics['logical_requests'] += 1
        statistics['logical_queries'] += 1
        try:
            scores, ids = await app.state.batcher.submit(request.queries[0])
            documents = upstream.load_docs(retriever.corpus, ids.tolist())
            result = ([dict(document=doc, score=score) for doc, score in
                       zip(documents, scores.tolist())] if request.return_scores else documents)
        except BaseException:
            statistics['failed_requests'] += 1
            raise
        statistics['completed_requests'] += 1
        statistics['completed_queries'] += 1
        statistics['last_completed_unix'] = time.time()
        return dict(result=[result])

    @app.get('/health')
    async def health():
        return dict(identity, ok=True, statistics=dict(statistics), rss_bytes=rss_bytes(),
                    observed_unix=time.time())

    return app


def extract_original_member(archive, target, status):
    from reproduction.common8.temporary_cache import CORPUS_MEMBER
    with tarfile.open(archive, mode='r:') as source:
        members = source.getmembers()
        regular = [member for member in members if member.isfile()]
        if (len(regular) != 1 or regular[0].name != CORPUS_MEMBER['name'] or
                regular[0].size != CORPUS_MEMBER['size']):
            raise ValueError('Original corpus TAR member differs')
        member = source.extractfile(regular[0])
        if member is None:
            raise ValueError('Original corpus JSONL member is unavailable')
        total = last_report = 0
        started = time.monotonic()
        with member, target.open('xb') as output:
            while chunk := member.read(8 * 1024 * 1024):
                output.write(chunk); total += len(chunk)
                if total - last_report >= 512 * 1024 * 1024:
                    status('EXTRACTING_ORIGINAL_CORPUS', extracted_bytes=total,
                           total_bytes=regular[0].size, elapsed_seconds=time.monotonic() - started)
                    last_report = total
            output.flush(); os.fsync(output.fileno())
        if total != CORPUS_MEMBER['size']:
            raise ValueError('Original corpus extraction was incomplete')
    status('ORIGINAL_CORPUS_EXTRACTED', extracted_bytes=total,
           total_bytes=total, elapsed_seconds=time.monotonic() - started)
    return target


def stage64(retriever, parquet, identity, path):
    import pandas as pd
    from reproduction.common8.verify_data import probe_question_id
    frame = pd.read_parquet(parquet)
    if len(frame) != 64:
        raise ValueError('Optional stage64 source does not contain exactly 64 rows')
    queries = []
    for chat in frame['prompt']:
        content = chat[0]['content']
        marks = list(re.finditer(r'Question:\s*', content))
        if not marks:
            raise ValueError('Optional stage64 original question marker is missing')
        queries.append(content[marks[-1].end():].strip())
    vectors = retriever.encoder.encode(queries)
    scores, ids = retriever.index.search(vectors, 3)
    documents = [[hashlib.sha256(json.dumps(retriever.corpus[int(index)],
        sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                  for index in row] for row in ids]
    value = dict(identity, role='actual_standalone_service_original_batch64',
        accepted=True, acceptance_scope='actual same-batch64 service GPU producer; four-physical-GPU equality remains pending',
        query_count=64, physical_gpu_count=2, logical_shards=4,
        gpu_document_ids=ids.tolist(), gpu_document_scores=scores.tolist(),
        gpu_document_sha256=documents, stage_sha256=digest(parquet),
        question_ids=[probe_question_id(chat) for chat in frame['prompt']],
        query_strings_written=False, four_gpu_numerical_comparison_pending=True,
        checked_unix=time.time())
    atomic(path, value)
    return dict(path=str(path), sha256=digest(path), query_count=64)


def serve(args):
    from reproduction.common8.stage_retrieval import ASSETS
    import datasets
    import faiss
    import torch
    import uvicorn
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    (base / 'logs').mkdir(exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdecimal():
        raise ValueError('An explicit Slurm allocation is required for this standalone service')
    if torch.cuda.device_count() != 2 or faiss.get_num_gpus() != 2:
        raise ValueError('Standalone retrieval service requires exactly two allocated visible GPUs')
    cache = (args.cache_root.resolve() if args.cache_root else
             Path('/dev/shm') / ('easy2560-retrieval-service-' + str(os.getuid()) + '-' + job))
    initialized_thread = threading.get_ident()
    assets = json.loads((REPO / 'reproduction/common8/encoder-assets.json').read_text())
    index_record = source_record(args.index, ASSETS['e5_Flat.index']['size'])
    corpus_record = source_record(args.corpus, ASSETS['wiki-18.jsonl']['size'])
    encoder = args.encoder.resolve(strict=True)
    encoder_files = {name: source_record(encoder / name, entry['size'])
                     for name, entry in assets['files'].items()}
    host = socket.gethostname()
    identity = dict(service_run_id=args.run_id, run_id=args.run_id,
        service_job_id=job, slurm_job_id=job, source_commit=args.source_commit,
        training_run_id=args.training_run_id, hostname=host, host=host,
        endpoint='http://' + host + ':' + str(args.port) + '/retrieve',
        physical_gpu_count=2, physical_gpus=2,
        physical_gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
        logical_shards=4, logical_to_physical_gpu=[0, 1, 0, 1],
        gpu_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_code_sha256={name: digest(REPO / name) for name in
            ('reproduction/easy2560_joint4/retrieval_service.py',
             'search_r1/search/retrieval_server.py', 'reproduction/common8/encoder-assets.json')},
        index_sha256=ASSETS['e5_Flat.index']['sha256'],
        corpus_sha256=ASSETS['wiki-18.jsonl']['sha256'],
        encoder_model_id=assets['model_id'], encoder_revision=assets['revision'],
        encoder_manifest_sha256=digest(REPO / 'reproduction/common8/encoder-assets.json'),
        index_file=index_record, corpus_file=corpus_record, encoder_files=encoder_files,
        asset_sha256_semantics='canonical previously verified asset identities; this service checks file sizes/metadata',
        fresh_full_asset_SHA_rehashed=False, use_float16=True,
        query_prefix='query: ', pooling='mean', query_max_length=256,
        encoder_dtype='float16', default_topk=3, retrieval_batch_size=512,
        exact_query_cache_enabled=False, request_rebatching_enabled=False,
        faiss_calls_on_initialization_thread=True, bind_host=args.host, port=args.port,
        owned_cache=str(cache), omp_num_threads=os.environ.get('OMP_NUM_THREADS'))
    statistics = dict(logical_requests=0, logical_queries=0, completed_requests=0,
        completed_queries=0, failed_requests=0, backend_seconds=0.0,
        last_completed_unix=None)

    def status(phase, **extra):
        value = dict(identity, phase=phase, checked_unix=time.time(), rss_bytes=rss_bytes(), **extra)
        atomic(base / 'status.json', value)
        print('RETRIEVAL_SERVICE_STATUS=' + json.dumps(
            dict(phase=phase, unix=value['checked_unix'], rss_bytes=value['rss_bytes'], **extra)), flush=True)

    # Reserve after checking read-only source metadata, so a missing source
    # cannot leave an unused large-cache namespace behind.
    cache.mkdir(mode=0o700, parents=False, exist_ok=False)
    cache_identity = cache.stat()
    try:
        status('STARTING_ORIGINAL_BACKEND')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ['HF_DATASETS_CACHE'] = str(cache / 'datasets')
        spec = importlib.util.spec_from_file_location('standalone_original_retrieval_backend',
            REPO / 'search_r1/search/retrieval_server.py')
        upstream = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = upstream; spec.loader.exec_module(upstream)
        resources = [faiss.StandardGpuResources() for _ in range(4)]
        mapping = [0, 1, 0, 1]
        original_four_shard_clone = make_four_shard_cloner(faiss, resources, mapping)
        def clone(index, co=None, ngpu=-1):
            started = time.monotonic()
            status('CLONING_ORIGINAL_FOUR_LOGICAL_SHARDS', logical_to_physical_gpu=mapping)
            value = original_four_shard_clone(index, co=co, ngpu=ngpu)
            status('ORIGINAL_FOUR_LOGICAL_SHARDS_READY', actual_logical_shards=int(value.count()),
                   clone_elapsed_seconds=time.monotonic() - started,
                   actual_free_gpu_bytes=[int(torch.cuda.mem_get_info(i)[0]) for i in range(2)])
            return value
        identity['faiss_module'] = str(Path(faiss.__file__).resolve())
        try:
            identity['faiss_multiple_py_signature'] = str(inspect.signature(faiss.index_cpu_to_gpu_multiple_py))
        except (ValueError, TypeError):
            identity['faiss_multiple_py_signature'] = 'runtime Python/SWIG signature unavailable'
        original_read, original_clone = faiss.read_index, faiss.index_cpu_to_all_gpus

        def read_index(path, *positional, **keyword):
            if Path(path) != Path(index_record['path']):
                raise ValueError('Unexpected original index path')
            status('READING_ORIGINAL_CPU_INDEX', source_bytes=index_record['bytes'])
            index = original_read(path, *positional, **keyword)
            status('ORIGINAL_CPU_INDEX_LOADED', ntotal=int(index.ntotal), dimension=int(index.d),
                   metric_type=int(index.metric_type))
            return index

        def corpus(path):
            if Path(path) != Path(corpus_record['path']):
                raise ValueError('Unexpected original corpus archive path')
            text = extract_original_member(Path(path), cache / 'original-corpus.jsonl', status)
            status('BUILDING_ORIGINAL_CORPUS_ARROW')
            value = datasets.load_dataset('json', data_files=str(text), split='train',
                num_proc=4, cache_dir=str(cache / 'datasets'))
            status('ORIGINAL_CORPUS_ARROW_READY', documents=len(value))
            return value

        upstream.load_corpus = corpus
        upstream.config = original_config(upstream, index_record['path'], corpus_record['path'], encoder)
        faiss.read_index, faiss.index_cpu_to_all_gpus = read_index, clone
        try:
            upstream.retriever = upstream.get_retriever(upstream.config)
        finally:
            faiss.read_index, faiss.index_cpu_to_all_gpus = original_read, original_clone
        retriever = upstream.retriever
        gc.collect()
        if retriever.index.count() != 4 or retriever.index.ntotal != len(retriever.corpus):
            raise ValueError('Original four logical shard/full corpus coverage differs')
        identity.update(ntotal=int(retriever.index.ntotal), dimension=int(retriever.index.d),
                        metric_type=int(retriever.index.metric_type))
        retriever.batch_search = track_batch_search(retriever.batch_search, statistics, initialized_thread)
        app = upstream.app
        if args.faiss_batch_size > 0:
            app = singleton_faiss_app(upstream, retriever, identity, statistics,
                initialized_thread, args.faiss_batch_size, args.faiss_max_wait_ms)
            identity['source_code_sha256']['search_g1/singleton_faiss_batcher.py'] = digest(
                REPO / 'search_g1/singleton_faiss_batcher.py')
        proof = stage64(retriever, args.stage_parquet, identity,
                        base / 'logs/stage64-admission.json') if args.stage_parquet else None
        identity['optional_stage64_proof'] = proof
        identity['admission_sha256'] = proof['sha256'] if proof else None
        status('READY', numerical_comparison_with_four_physical_gpus_pending=True)
        atomic(base / 'logs/service-ready.json', dict(identity, ready=True, ready_unix=time.time()))

        if args.faiss_batch_size == 0:
            @upstream.app.get('/health')
            async def health():
                return dict(identity, ok=True, statistics=dict(statistics), rss_bytes=rss_bytes(),
                            observed_unix=time.time())

        # The original async handler performs encoder/FAISS work directly on
        # this very same main event-loop thread.  Do not add worker processes.
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
        status('STOPPED', statistics=dict(statistics))
    except BaseException as error:
        status('FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        current = cache.stat() if cache.exists() else None
        if (current is not None and cache.resolve() == cache and current.st_uid == os.getuid() and
                (current.st_dev, current.st_ino) == (cache_identity.st_dev, cache_identity.st_ino)):
            shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--training-run-id')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--encoder', required=True, type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--stage-parquet', type=Path)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=18020)
    parser.add_argument('--logical-shards', type=int, choices=(4,), default=4)
    parser.add_argument('--faiss-batch-size', type=int, default=0,
                        help='Private singleton-E5/batched-FAISS app; 0 preserves the original service')
    parser.add_argument('--faiss-max-wait-ms', type=float, default=2)
    args = parser.parse_args()
    if (not args.base.is_absolute() or not re.fullmatch('[0-9a-f]{32}', args.run_id) or
            not re.fullmatch('[0-9a-f]{40}', args.source_commit) or not 1 <= args.port <= 65535):
        parser.error('Require an absolute service base, run UUID hex, exact source commit, and valid port')
    if args.faiss_batch_size < 0:
        parser.error('--faiss-batch-size must be nonnegative')
    serve(args)


if __name__ == '__main__':
    main()
