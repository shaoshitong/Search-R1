"""CPU localhost forwarding for one explicitly bound external GPU retriever.

The training recipe keeps http://127.0.0.1:18010/retrieve.  This process loads no
encoder, index, corpus, tokenizer, or CUDA library.  The remote service produces
the real stage64 document/score evidence; this process only binds that evidence
to the training job and forwards complete original JSON batches.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


REPO = Path(__file__).resolve().parents[2]
RECEIPT_KEYS = {'service_ready_path', 'service_ready_sha256', 'source_commit',
                'service_job_id', 'endpoint'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_document(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('External retriever evidence must have an absolute canonical path')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024**2:
        raise ValueError('External retriever evidence must be a small regular JSON file')
    raw = path.read_bytes()
    after = path.lstat()
    key = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if key(before) != key(after) or len(raw) != before.st_size:
        raise ValueError('External retriever evidence changed while being read')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('External retriever evidence must contain a JSON object')
    return value, hashlib.sha256(raw).hexdigest()


def endpoint_urls(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError('External retrieval endpoint must be explicit')
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.path != '/retrieve' or parsed.query or parsed.fragment or
            any(character.isspace() for character in endpoint)):
        raise ValueError('External retrieval endpoint must name the original /retrieve handler')
    return endpoint, urlunsplit((parsed.scheme, parsed.netloc, '/health', '', ''))


def external_binding(receipt_path):
    """Freeze real service-ready/stage64 bytes before the single training submit."""
    if receipt_path is None:
        return None
    from reproduction.common8.stage_retrieval import ASSETS
    receipt_path = Path(receipt_path)
    receipt, receipt_sha = read_document(receipt_path)
    if not RECEIPT_KEYS.issubset(receipt):
        raise ValueError('External retriever receipt is missing its explicit service binding')
    for key, length in (('service_ready_sha256', 64), ('source_commit', 40)):
        if not isinstance(receipt[key], str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', receipt[key]):
            raise ValueError('External retriever receipt has an invalid ' + key)
    if type(receipt['service_job_id']) is not str or not receipt['service_job_id'].isdecimal():
        raise ValueError('External retriever receipt must name the actual service Slurm job')
    endpoint_urls(receipt['endpoint'])
    ready, ready_sha = read_document(receipt['service_ready_path'])
    if (ready_sha != receipt['service_ready_sha256'] or ready.get('ready') is not True or
            any(ready.get(key) != receipt[key] for key in ('source_commit', 'service_job_id', 'endpoint')) or
            ready.get('slurm_job_id') != receipt['service_job_id'] or
            ready.get('physical_gpu_count') != 2 or ready.get('logical_shards') != 4 or
            ready.get('logical_to_physical_gpu') != [0, 1, 0, 1] or
            ready.get('exact_query_cache_enabled') is not False or
            ready.get('request_rebatching_enabled') is not False):
        raise ValueError('External retriever ready evidence differs from the submitted service')
    for name, key in (('e5_Flat.index', 'index_sha256'), ('wiki-18.jsonl', 'corpus_sha256')):
        if ready.get(key) != ASSETS[name]['sha256']:
            raise ValueError('External retriever uses different original retrieval assets')
    encoder_sha = digest(REPO / 'reproduction/common8/encoder-assets.json')
    if ready.get('encoder_manifest_sha256') != encoder_sha:
        raise ValueError('External retriever encoder asset version differs')
    staged = ready.get('optional_stage64_proof') or {}
    stage, stage_sha = read_document(staged.get('path', ''))
    if (stage_sha != staged.get('sha256') or staged.get('query_count') != 64 or
            stage.get('role') != 'actual_standalone_service_original_batch64' or
            stage.get('accepted') is not True or stage.get('query_count') != 64 or
            any(stage.get(key) != ready.get(key) for key in (
                'service_run_id', 'service_job_id', 'source_commit', 'endpoint',
                'index_sha256', 'corpus_sha256', 'encoder_manifest_sha256',
                'physical_gpu_count', 'logical_shards', 'logical_to_physical_gpu'))):
        raise ValueError('External retriever actual stage64 producer proof differs')
    if (len(stage.get('question_ids', [])) != 64 or
            any(len(stage.get(key, [])) != 64 for key in
                ('gpu_document_ids', 'gpu_document_scores', 'gpu_document_sha256'))):
        raise ValueError('External retriever stage64 evidence has incomplete rows')
    for ids, scores, hashes in zip(stage['gpu_document_ids'], stage['gpu_document_scores'],
                                  stage['gpu_document_sha256']):
        if (len(ids) != 3 or len(scores) != 3 or len(hashes) != 3 or
                any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
            raise ValueError('External retriever stage64 evidence has incomplete document/score rows')
    return dict(receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
                receipt=receipt, service_ready_path=receipt['service_ready_path'],
                service_ready_sha256=ready_sha, service_ready=ready,
                stage64_path=staged['path'], stage64_sha256=stage_sha, stage64_admission=stage,
                source_commit=receipt['source_commit'], service_job_id=receipt['service_job_id'],
                endpoint=receipt['endpoint'], physical_gpu_count=2, logical_shards=4,
                binding_scope='external GPU producer evidence and CPU localhost forwarding; not four-physical-GPU numerical equality')


def admission_for_run(run, external, training_job_id):
    """Bind the existing GPU result to this consumer without claiming local GPU work."""
    from reproduction.common8.model_identity import SEVEN, identity
    if identity(run) != SEVEN or run.get('external_retriever') != external:
        raise ValueError('External retriever forwarding is scoped to the explicitly bound native4 Easy7B run')
    if str(training_job_id) == external['service_job_id']:
        raise ValueError('External retriever and actor training must have different Slurm jobs')
    ready, stage = external['service_ready'], external['stage64_admission']
    if ready.get('training_run_id') not in (None, run['run_id']):
        raise ValueError('External retriever is reserved for a different training run')
    if (stage['stage_sha256'] != run['input_sha256']['stage.parquet'] or
            ready.get('source_code_sha256', {}).get('search_r1/search/retrieval_server.py') !=
            run['source_sha256']['search_r1/search/retrieval_server.py']):
        raise ValueError('External retriever stage data or original handler code differs')
    encoder = json.loads((REPO / 'reproduction/common8/encoder-assets.json').read_text())
    value = dict(deepcopy(stage), run_id=run['run_id'], slurm_job_id=str(training_job_id),
                 source_commit=run['source_commit'], service_source_commit=external['source_commit'],
                 service_run_id=ready['service_run_id'], service_job_id=external['service_job_id'],
                 service_stage64_sha256=external['stage64_sha256'],
                 service_ready_sha256=external['service_ready_sha256'],
                 external_receipt_sha256=external['receipt_sha256'],
                 encoder_assets=encoder, shard_count=4, local_proxy_gpu_count=0,
                 local_proxy_only=True, gpu_evidence_producer='separate_standalone_service_job',
                 scope='actual external two-physical-GPU/four-logical-shard stage64 results bound to this CPU localhost proxy; no local GPU retrieval or four-physical-GPU equality claim',
                 checked_unix=time.time())
    return value


class UpstreamHTTPError(RuntimeError):
    def __init__(self, status, body):
        self.status, self.body = status, body
        super().__init__('External retrieval HTTP status ' + str(status))


class HTTPBackend:
    def __init__(self, endpoint, timeout=300):
        self.endpoint, self.health_endpoint = endpoint_urls(endpoint)
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def request(self, payload=None, *, health=False):
        request = Request(self.health_endpoint if health else self.endpoint,
                          data=None if health else json.dumps(payload, ensure_ascii=False).encode(),
                          headers={} if health else {'Content-Type': 'application/json'},
                          method='GET' if health else 'POST')
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            raw = error.read()
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeError):
                body = dict(detail='External retrieval request failed')
            raise UpstreamHTTPError(error.code, body) from None

    def batch_search(self, query_list, num=None, return_score=False):
        # One cache key is the *complete* original HTTP batch. The bound backend
        # receives that original batch, not the singleton used for memoization.
        if len(query_list) != 1 or return_score:
            raise ValueError('CPU proxy cache expects one complete original payload')
        return [self.request(json.loads(query_list[0]))]


class ProxyForwarder:
    def __init__(self, backend, external, *, cache_enabled=False):
        from search_g1.exact_query_retrieval_cache import ExactQueryRetrievalCache, RetrievalAssetVersions
        ready = external['service_ready']
        self.backend = backend
        self.cache = (ExactQueryRetrievalCache(backend.batch_search,
            versions=RetrievalAssetVersions(ready['encoder_manifest_sha256'], ready['index_sha256'],
                                            ready['corpus_sha256']),
            default_topk=3, capacity=4096, enabled=True) if cache_enabled else None)
        self.logical_requests = self.logical_queries = 0

    def retrieve(self, payload):
        self.logical_requests += 1
        queries = payload.get('queries') if isinstance(payload, dict) else None
        self.logical_queries += len(queries) if isinstance(queries, list) else 0
        if self.cache is None:
            return self.backend.request(payload)
        key = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        return self.cache.batch_search([key])[0]

    def statistics(self):
        return dict(logical_requests=self.logical_requests, logical_queries=self.logical_queries,
                    exact_query_cache_enabled=self.cache is not None,
                    cache_key_unit='complete_original_HTTP_payload_including_original_batch_order',
                    cache=self.cache.statistics() if self.cache is not None else None)


def validate_live_health(health, external):
    ready = external['service_ready']
    if (health.get('ok') is not True or any(health.get(key) != ready.get(key) for key in (
            'service_run_id', 'service_job_id', 'source_commit', 'endpoint',
            'physical_gpu_count', 'logical_shards', 'logical_to_physical_gpu',
            'index_sha256', 'corpus_sha256', 'encoder_manifest_sha256')) or
            health.get('admission_sha256') != external['stage64_sha256']):
        raise ValueError('External retrieval live service identity differs from its frozen receipt')


def serve(base):
    # Imported only inside the CPU server entry, never during binding/tests.
    import uvicorn
    from fastapi import FastAPI, Request as FastAPIRequest
    from fastapi.responses import JSONResponse
    from reproduction.a800.owned_processes import publish
    from reproduction.common8.topology import execution_job_id
    from search_g1.easy7b_infra import flags
    base = Path(base)
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    external = run['external_retriever']
    if external_binding(external['receipt_path']) != external:
        raise ValueError('External retrieval evidence changed after training preparation')
    backend = HTTPBackend(external['endpoint'])
    validate_live_health(backend.request(health=True), external)
    job = execution_job_id(run)
    proof = admission_for_run(run, external, job)
    target = base / 'logs/retrieval-admission.json'
    if target.exists():
        raise ValueError('Local retrieval admission already exists; inspect the existing proxy')
    publish(target, proof)
    forwarder = ProxyForwarder(backend, external, cache_enabled=flags(run)['exact_retrieval_cache'])
    app = FastAPI()

    @app.post('/retrieve')
    async def retrieve(request: FastAPIRequest):
        try:
            return JSONResponse(forwarder.retrieve(await request.json()))
        except UpstreamHTTPError as error:
            return JSONResponse(error.body, status_code=error.status)

    @app.get('/health')
    async def health():
        actual = backend.request(health=True)
        validate_live_health(actual, external)
        return dict(ok=True, run_id=run['run_id'], slurm_job_id=job,
                    admission_sha256=digest(target), ntotal=proof['ntotal'],
                    shard_count=4, shard_count_scope='external_service_logical_shards',
                    service_job_id=external['service_job_id'], service_source_commit=external['source_commit'],
                    physical_gpu_count=2, logical_shards=4, local_proxy_gpu_count=0,
                    endpoint=external['endpoint'], forwarding_only=True,
                    upstream_statistics=actual.get('statistics'), **forwarder.statistics())

    uvicorn.run(app, host='127.0.0.1', port=18010, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=os.environ.get('SEARCH_G1_BASE'))
    args = parser.parse_args()
    if args.base is None:
        parser.error('Explicit training base is required')
    serve(args.base)


if __name__ == '__main__':
    main()
