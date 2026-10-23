"""Cold-cache CPU proxies for a private candidate-generation A/B experiment.

Legacy uses the original ProxyForwarder. Async offloads only an independent
urllib HTTP call; completed-cache lookup/insert stays on its owner loop. Neither
mode modifies production proxy contracts or merges in-flight requests.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from reproduction.easy2560_joint4.retrieval_proxy import ProxyForwarder, UpstreamHTTPError
from reproduction.easy2560_joint4.async_retrieval_proxy import (
    AsyncProxyForwarder, PrivateHTTPBackend, private_urls,
)


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
