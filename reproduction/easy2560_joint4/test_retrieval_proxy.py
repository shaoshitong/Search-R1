"""CPU mock-HTTP coverage; no real service, encoder, checkpoint, or GPU access."""
from copy import deepcopy
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import retrieval_proxy as proxy


def fixture(directory, endpoint='http://retrieval-node:18020/retrieve'):
    from reproduction.common8.stage_retrieval import ASSETS
    root = Path(directory).resolve()
    ready_path, stage_path, receipt_path = [root / name for name in
        ('service-ready.json', 'stage64-admission.json', 'external-receipt.json')]
    ready = dict(ready=True, service_run_id='f' * 32, run_id='f' * 32,
        service_job_id='1234', slurm_job_id='1234', source_commit='b' * 40,
        endpoint=endpoint, physical_gpu_count=2,
        logical_shards=4, logical_to_physical_gpu=[0, 1, 0, 1],
        exact_query_cache_enabled=False, request_rebatching_enabled=False,
        index_sha256=ASSETS['e5_Flat.index']['sha256'],
        corpus_sha256=ASSETS['wiki-18.jsonl']['sha256'],
        encoder_manifest_sha256=proxy.digest(proxy.REPO / 'reproduction/common8/encoder-assets.json'),
        source_code_sha256={'search_r1/search/retrieval_server.py': proxy.digest(
            proxy.REPO / 'search_r1/search/retrieval_server.py')}, ntotal=21015324,
        training_run_id=None)
    stage = dict(ready, role='actual_standalone_service_original_batch64', accepted=True,
        query_count=64, question_ids=['question-' + str(i) for i in range(64)],
        gpu_document_ids=[[i, i+1, i+2] for i in range(64)],
        gpu_document_scores=[[.875, .75, .625] for _ in range(64)],
        gpu_document_sha256=[['d'*64, 'e'*64, 'f'*64] for _ in range(64)],
        stage_sha256='c' * 64, four_gpu_numerical_comparison_pending=True)
    stage_path.write_text(json.dumps(stage))
    ready['optional_stage64_proof'] = dict(path=str(stage_path), sha256=proxy.digest(stage_path), query_count=64)
    ready_path.write_text(json.dumps(ready))
    receipt = dict(service_ready_path=str(ready_path), service_ready_sha256=proxy.digest(ready_path),
                   source_commit=ready['source_commit'], service_job_id='1234', endpoint=ready['endpoint'])
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path, ready, stage


class BindingTests(unittest.TestCase):
    def test_small_frozen_ready_and_gpu_producer_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path, ready, stage = fixture(directory)
            external = proxy.external_binding(path)
            self.assertEqual(external['receipt_sha256'], proxy.digest(path))
            self.assertEqual(external['service_ready'], ready)
            self.assertEqual(external['stage64_admission'], stage)
            from reproduction.common8.model_identity import SEVEN
            run = dict(profile='common8', execution_site='hpc3', execution_backend='slurm',
                training_world_size=4, easy2560_selection=True, data_selection_mode='mixed_first',
                actor_logical_micro_batch_per_rank=4, run_id='a'*32, source_commit='a'*40,
                model_revision=SEVEN[1], budget=dict(model_id=SEVEN[0], model_revision=SEVEN[1]),
                input_sha256={'stage.parquet': 'c'*64}, source_sha256=ready['source_code_sha256'],
                external_retriever=external)
            result = proxy.admission_for_run(run, external, '5678')
            self.assertEqual(result['run_id'], run['run_id'])
            self.assertEqual(result['slurm_job_id'], '5678')
            self.assertEqual(result['service_job_id'], '1234')
            self.assertEqual(result['physical_gpu_count'], 2)
            self.assertEqual(result['local_proxy_gpu_count'], 0)
            self.assertTrue(result['four_gpu_numerical_comparison_pending'])
            for key in ('gpu_document_ids', 'gpu_document_scores', 'gpu_document_sha256', 'question_ids'):
                self.assertEqual(result[key], stage[key])
            with self.assertRaisesRegex(ValueError, 'different Slurm jobs'):
                proxy.admission_for_run(run, external, '1234')

    def test_changed_or_missing_ready_proof_is_not_admitted(self):
        self.assertIsNone(proxy.external_binding(None))
        with tempfile.TemporaryDirectory() as directory:
            path, ready, _ = fixture(directory)
            ready_path = Path(directory) / 'service-ready.json'
            ready['physical_gpu_count'] = 4
            ready_path.write_text(json.dumps(ready))
            with self.assertRaisesRegex(ValueError, 'ready evidence differs'):
                proxy.external_binding(path)

    def test_live_service_health_is_bound_to_service_not_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path, ready, _ = fixture(directory)
            external = proxy.external_binding(path)
            health = dict(ready, ok=True, admission_sha256=external['stage64_sha256'])
            proxy.validate_live_health(health, external)
            with self.assertRaisesRegex(ValueError, 'live service identity differs'):
                proxy.validate_live_health(dict(health, service_job_id='9999'), external)

    def test_baseline_submission_argv_unchanged_and_external_position_explicit(self):
        from reproduction.easy2560_joint4.control import BASE, launch_command
        baseline = launch_command(BASE, 'a'*40, 8)
        self.assertEqual(baseline[-4:], [BASE, 'a'*40, '8', ''])
        external = launch_command(BASE, 'a'*40, 8, external_retriever_receipt='/remote/external.json')
        self.assertEqual(external[:-2], baseline)
        self.assertEqual(external[-2:], ['baseline', '/remote/external.json'])
        candidate = launch_command(BASE, 'a'*40, 8, '/remote/restore.json',
            'sparse-exact-v1', '/remote/external.json')
        self.assertEqual(candidate[-3:], ['/remote/restore.json', 'sparse-exact-v1', '/remote/external.json'])


class MockHTTPTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        self.health_response = {}
        health_response = self.health_response
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                raw = json.dumps(health_response).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(payload)
                if payload.get('fail'):
                    value, status = dict(detail='mock original backend unavailable'), 503
                else:
                    result = [[dict(document=dict(contents=query, nested=dict(order=[i, j])),
                                    score=.875-i/16-j/32) for j in range(payload.get('topk', 3))]
                              for i, query in enumerate(payload['queries'])]
                    if not payload.get('return_scores', False):
                        result = [[item['document'] for item in row] for row in result]
                    value, status = dict(result=result, extra=dict(original_payload=payload)), 200
                raw = json.dumps(value).encode()
                self.send_response(status); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.backend = proxy.HTTPBackend('http://127.0.0.1:' + str(self.server.server_port) + '/retrieve')
        self.external = dict(service_ready=dict(encoder_manifest_sha256='e', index_sha256='i', corpus_sha256='c'))

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_disabled_default_forwards_exact_payload_and_complete_json_each_time(self):
        forwarder = proxy.ProxyForwarder(self.backend, self.external)
        payload = dict(queries=['重复 Q?', 'Different Q', '重复 Q?'], topk=2, return_scores=True)
        first, second = forwarder.retrieve(payload), forwarder.retrieve(payload)
        self.assertEqual(self.calls, [payload, payload])
        self.assertEqual(first, second)
        self.assertEqual(first['extra']['original_payload'], payload)
        self.assertNotEqual(first['result'][0][0]['score'], first['result'][2][0]['score'])
        self.assertFalse(forwarder.statistics()['exact_query_cache_enabled'])

    def test_cache_reuses_complete_original_batch_without_losing_scores_or_other_json(self):
        forwarder = proxy.ProxyForwarder(self.backend, self.external, cache_enabled=True)
        payload = dict(queries=['A', 'B', 'A'], topk=3, return_scores=True)
        first = forwarder.retrieve(payload)
        expected = deepcopy(first)
        first['result'][0].clear(); first['extra']['original_payload']['queries'].append('caller edit')
        self.assertEqual(forwarder.retrieve(payload), expected)
        self.assertEqual(self.calls, [payload])
        permutation = dict(payload, queries=['B', 'A', 'A'])
        forwarder.retrieve(permutation)
        self.assertEqual(self.calls[-1], permutation)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(forwarder.statistics()['logical_requests'], 3)
        self.assertEqual(forwarder.statistics()['logical_queries'], 9)

    def test_score_mode_missing_defaults_and_original_errors_remain_distinct(self):
        forwarder = proxy.ProxyForwarder(self.backend, self.external, cache_enabled=True)
        payloads = [dict(queries=['Q'], topk=2, return_scores=False),
                    dict(queries=['Q'], topk=2, return_scores=True), dict(queries=['Q'], topk=2)]
        for payload in payloads:
            actual = forwarder.retrieve(payload)
            self.assertEqual(actual['extra']['original_payload'], payload)
            self.assertEqual(forwarder.retrieve(payload), actual)
        self.assertEqual(self.calls, payloads)
        failing = dict(queries=['Q'], topk=2, return_scores=True, fail=True)
        for _ in range(2):
            with self.assertRaises(proxy.UpstreamHTTPError) as caught:
                forwarder.retrieve(failing)
            self.assertEqual(caught.exception.status, 503)
            self.assertEqual(caught.exception.body, dict(detail='mock original backend unavailable'))
        self.assertEqual(len(self.calls), 5)

    def test_real_fastapi_handlers_forward_json_and_publish_service_bound_health(self):
        # The real ASGI request/response handlers execute on one original owner
        # thread; the mock backend is a separate loopback HTTP server only.
        from starlette.requests import Request
        from reproduction.common8.model_identity import SEVEN
        with tempfile.TemporaryDirectory() as directory:
            receipt, ready, _ = fixture(directory, self.backend.endpoint)
            external = proxy.external_binding(receipt)
            self.health_response.update(ready, ok=True, admission_sha256=external['stage64_sha256'])
            base = Path(directory) / 'training'
            (base / 'logs').mkdir(parents=True)
            run = dict(profile='common8', execution_site='hpc3', execution_backend='slurm',
                training_world_size=4, easy2560_selection=True, data_selection_mode='mixed_first',
                actor_logical_micro_batch_per_rank=4, run_id='a'*32, source_commit='a'*40,
                model_revision=SEVEN[1], budget=dict(model_id=SEVEN[0], model_revision=SEVEN[1]),
                input_sha256={'stage.parquet': 'c'*64}, source_sha256=ready['source_code_sha256'],
                external_retriever=external,
                infra_optimizations=dict(selected_policy_logits=True, exact_retrieval_cache=True))
            (base / 'logs/run-manifest.json').write_text(json.dumps(run))
            payload = dict(queries=['A', 'B', 'A'], topk=3, return_scores=True)
            def inspect_app(app, **kwargs):
                self.assertEqual(kwargs, dict(host='127.0.0.1', port=18010, workers=1))
                retrieve = next(route.endpoint for route in app.routes if route.path == '/retrieve')
                health = next(route.endpoint for route in app.routes if route.path == '/health')
                async def invoke():
                    async def receive():
                        return dict(type='http.request', body=json.dumps(payload).encode(), more_body=False)
                    first = await retrieve(Request(dict(type='http', method='POST', path='/retrieve', headers=[]), receive))
                    second = await retrieve(Request(dict(type='http', method='POST', path='/retrieve', headers=[]), receive))
                    self.assertEqual(first.status_code, 200)
                    self.assertEqual(first.body, second.body)
                    self.assertEqual(json.loads(first.body)['extra']['original_payload'], payload)
                    live = await health()
                    self.assertEqual(live['slurm_job_id'], '5678')
                    self.assertEqual(live['service_job_id'], '1234')
                    self.assertEqual((live['physical_gpu_count'], live['logical_shards'], live['local_proxy_gpu_count']), (2, 4, 0))
                    self.assertEqual(live['logical_requests'], 2)
                asyncio.run(invoke())
            with patch('uvicorn.run', side_effect=inspect_app), patch(
                    'reproduction.common8.topology.execution_job_id', return_value='5678'):
                proxy.serve(base)
            self.assertEqual(self.calls, [payload])
            proof = json.loads((base / 'logs/retrieval-admission.json').read_text())
            self.assertTrue(proof['local_proxy_only'])
            self.assertEqual(proof['service_stage64_sha256'], external['stage64_sha256'])


if __name__ == '__main__':
    unittest.main()
