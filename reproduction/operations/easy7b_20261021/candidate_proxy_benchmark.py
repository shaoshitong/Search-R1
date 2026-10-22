"""Cold-cache CPU proxies for a private candidate-generation A/B experiment.

Legacy uses the original ProxyForwarder. Async offloads only an independent
urllib HTTP call; completed-cache lookup/insert stays on its owner loop. Neither
mode modifies production proxy contracts or merges in-flight requests.
"""
import argparse
import asyncio
from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from reproduction.easy2560_joint4.retrieval_proxy import HTTPBackend, ProxyForwarder, UpstreamHTTPError
from search_g1.exact_query_retrieval_cache import RetrievalAssetVersions

CAPACITY = 4096


def private_urls(endpoint):
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.path not in ('/retrieve', '/retrieve-original') or parsed.query or parsed.fragment):
        raise ValueError('Require an explicit private /retrieve or /retrieve-original endpoint')
    return endpoint, urlunsplit((parsed.scheme, parsed.netloc, '/health', '', ''))


class PrivateHTTPBackend(HTTPBackend):
    """Keep original payload handling, with a new independent opener per call."""
    def __init__(self, endpoint, timeout=300):
        self.endpoint, self.health_endpoint = private_urls(endpoint)
        self.timeout = timeout

    def request(self, payload=None, *, health=False):
        request = Request(self.health_endpoint if health else self.endpoint,
            data=None if health else json.dumps(payload, ensure_ascii=False).encode(),
            headers={} if health else {'Content-Type': 'application/json'},
            method='GET' if health else 'POST')
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            try:
                body = json.loads(error.read())
            except (ValueError, UnicodeError):
                body = dict(detail='Private retrieval request failed')
            raise UpstreamHTTPError(error.code, body) from None


class AsyncProxyForwarder:
    def __init__(self, backend, ready):
        self.backend = backend
        self.versions = RetrievalAssetVersions(ready['encoder_manifest_sha256'],
            ready['index_sha256'], ready['corpus_sha256'])
        self.owner_thread, self.owner_loop = threading.get_ident(), None
        self.entries = OrderedDict()
        self.logical_requests = self.logical_queries = 0
        self.counts = dict(logical_calls=0, logical_queries=0, backend_calls=0,
            backend_queries=0, served_hit_queries=0, lookup_hit_queries=0,
            lookup_miss_queries=0, backend_exceptions=0, uncacheable_calls=0, evictions=0)

    def _owner(self):
        loop = asyncio.get_running_loop()
        if self.owner_loop is None:
            self.owner_loop = loop
        if threading.get_ident() != self.owner_thread or loop is not self.owner_loop:
            raise RuntimeError('Completed retrieval cache must stay on its owner loop')

    async def retrieve(self, payload):
        self._owner()
        self.logical_requests += 1
        queries = payload.get('queries') if isinstance(payload, dict) else None
        self.logical_queries += len(queries) if isinstance(queries, list) else 0
        # Exactly the original proxy's complete-payload serialization; list
        # order, topk, score mode, and any other payload field remain in the key.
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False)
        key = (self.versions, serialized)
        self.counts['logical_calls'] += 1; self.counts['logical_queries'] += 1
        if key in self.entries:
            self.entries.move_to_end(key)
            self.counts['lookup_hit_queries'] += 1; self.counts['served_hit_queries'] += 1
            return deepcopy(self.entries[key])
        self.counts['lookup_miss_queries'] += 1
        self.counts['backend_calls'] += 1; self.counts['backend_queries'] += 1
        try:
            # Only this call runs on a worker thread. No Session/opener, LRU,
            # in-flight promise, encoder, or FAISS object is shared with it.
            result = await asyncio.to_thread(self.backend.request, payload)
        except BaseException:
            self.counts['backend_exceptions'] += 1
            raise
        self._owner()
        self.entries[key] = deepcopy(result)
        self.entries.move_to_end(key)
        while len(self.entries) > CAPACITY:
            self.entries.popitem(last=False); self.counts['evictions'] += 1
        return deepcopy(result)

    def statistics(self):
        cache = dict(self.counts, enabled=True, entries=len(self.entries), capacity=CAPACITY,
            versions=dict(encoder=self.versions.encoder, index=self.versions.index, corpus=self.versions.corpus))
        total = self.counts['logical_queries']
        cache['served_hit_rate'] = self.counts['served_hit_queries'] / total if total else 0.0
        return dict(logical_requests=self.logical_requests, logical_queries=self.logical_queries,
            exact_query_cache_enabled=True,
            cache_key_unit='complete_original_HTTP_payload_including_original_batch_order', cache=cache)


def make_forwarder(mode, backend, ready):
    if mode == 'legacy':
        return ProxyForwarder(backend, dict(service_ready=ready), cache_enabled=True)
    if mode == 'async':
        return AsyncProxyForwarder(backend, ready)
    raise ValueError('Unknown private candidate proxy mode')


async def dispatch(mode, forwarder, payload):
    return (await forwarder.retrieve(payload)) if mode == 'async' else forwarder.retrieve(payload)


def make_app(endpoint, ready, mode):
    from fastapi import FastAPI, Request as FastAPIRequest
    from fastapi.responses import JSONResponse
    forwarder = make_forwarder(mode, PrivateHTTPBackend(endpoint), ready)
    app = FastAPI(); app.state.forwarder = forwarder

    @app.post('/retrieve')
    async def retrieve(request: FastAPIRequest):
        try:
            return JSONResponse(await dispatch(mode, forwarder, await request.json()))
        except UpstreamHTTPError as error:
            return JSONResponse(error.body, status_code=error.status)

    @app.get('/health')
    async def health():
        result = forwarder.statistics(); cache = result['cache']
        return dict(result, ok=True, mode=mode, endpoint=endpoint,
            cold_cache_at_start=True, completed_cache_only=True, inflight_requests_merged=False,
            backend_requests=cache['backend_calls'], cache_hits=cache['served_hit_queries'],
            failed_requests=cache['backend_exceptions'], production_proxy_modified=False,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), cpu_only=True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--service-ready', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--mode', choices=('legacy', 'async'), required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ready = json.loads(args.service_ready.read_text())
    import uvicorn
    uvicorn.run(make_app(args.endpoint, ready, args.mode), host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
