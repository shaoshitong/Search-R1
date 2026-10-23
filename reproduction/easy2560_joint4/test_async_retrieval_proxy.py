"""CPU coverage for explicit production async forwarding and frozen service mode."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import retrieval_proxy as proxy
from reproduction.easy2560_joint4.async_retrieval_proxy import AsyncProxyForwarder, PrivateHTTPBackend
from reproduction.easy2560_joint4 import test_retrieval_proxy as legacy_tests
from search_g1.easy7b_infra import flags, profile_flags, LEAN_PROFILES, FA3_PROFILES

PROFILE = 'fa3-lean-batched-search-stop-v1'
MODE = dict(request_rebatching_enabled=True, singleton_e5=True,
            faiss_batch_size=16, faiss_max_wait_ms=2)


def batched_fixture(directory, endpoint='http://retrieval-node:18020/retrieve'):
    receipt, ready, stage = legacy_tests.fixture(directory, endpoint)
    ready.update(MODE); stage.update(MODE)
    root = Path(directory)
    (root / 'stage64-admission.json').write_text(json.dumps(stage))
    ready['optional_stage64_proof']['sha256'] = proxy.digest(root / 'stage64-admission.json')
    (root / 'service-ready.json').write_text(json.dumps(ready))
    value = json.loads(receipt.read_text())
    value['service_ready_sha256'] = proxy.digest(root / 'service-ready.json')
    receipt.write_text(json.dumps(value))
    return receipt, ready, stage


class AsyncProfileTests(unittest.TestCase):
    def test_only_new_proxy_flag_and_same_actor_recipe_four_gpu_200_outer(self):
        from reproduction.common8.configuration import arguments
        from reproduction.easy2560_joint4.control import BASE, launch_command
        from reproduction.easy2560_joint4.prepare_training import selection_settings
        from search_g1.test_search_action_stop_wiring import SearchStopWiringTests
        old = profile_flags('fa3-lean-search-stop-v1'); new = profile_flags(PROFILE)
        self.assertTrue(new.pop('async_retrieval_proxy')); self.assertEqual(new, old)
        self.assertNotIn('async_retrieval_proxy', old)
        self.assertIn(PROFILE, LEAN_PROFILES); self.assertIn(PROFILE, FA3_PROFILES)
        with tempfile.TemporaryDirectory() as td:
            base, run = SearchStopWiringTests().fixture(td, 'fa3-lean-search-stop-v1')
            _, old_args = arguments(base)
            run.update(selection_settings('/remote/predecessor', 8, infra_profile=PROFILE))
            (base / 'logs/run-manifest.json').write_text(json.dumps(run))
            _, new_args = arguments(base)
            self.assertEqual(new_args, old_args)
            self.assertEqual(run['target_updates'], 200)
            self.assertTrue(flags(run)['async_retrieval_proxy'])
        old_launch = launch_command(BASE, 'a'*40, 8, infra_profile='fa3-lean-search-stop-v1')
        new_launch = launch_command(BASE, 'a'*40, 8, infra_profile=PROFILE)
        self.assertEqual(old_launch[:-1], new_launch[:-1]); self.assertEqual(new_launch[-1], PROFILE)
        self.assertIn('--gres=gpu:4', new_launch)

    def test_production_and_benchmark_use_same_async_classes_and_old_default(self):
        from reproduction.operations.easy7b_20261021 import candidate_proxy_benchmark as benchmark
        external = dict(endpoint='http://private/retrieve', service_ready=dict(
            encoder_manifest_sha256='e', index_sha256='i', corpus_sha256='c'))
        old_backend, old = proxy.forwarding_components(external, dict(exact_retrieval_cache=True))
        self.assertIs(type(old_backend), proxy.HTTPBackend); self.assertIs(type(old), proxy.ProxyForwarder)
        new_backend, new = proxy.forwarding_components(external, profile_flags(PROFILE))
        self.assertIs(type(new_backend), benchmark.PrivateHTTPBackend)
        self.assertIs(type(new), benchmark.AsyncProxyForwarder)
        self.assertIs(type(new), AsyncProxyForwarder)


class BatchedBindingTests(unittest.TestCase):
    def test_measured_mode_binds_real_ready_stage_and_live_health(self):
        with tempfile.TemporaryDirectory() as td:
            receipt, ready, _ = batched_fixture(td)
            external = proxy.external_binding(receipt)
            self.assertEqual({key: external['service_ready'][key] for key in MODE}, MODE)
            live = dict(ready, ok=True, admission_sha256=external['stage64_sha256'])
            proxy.validate_live_health(live, external)
            for change in (dict(request_rebatching_enabled=False), dict(singleton_e5=False),
                           dict(faiss_batch_size=8), dict(faiss_max_wait_ms=16)):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'live service identity'):
                    proxy.validate_live_health(dict(live, **change), external)

    def test_unmeasured_or_stage_mismatched_mode_is_rejected(self):
        for change in (dict(singleton_e5=False), dict(faiss_batch_size=8),
                       dict(faiss_batch_size=True), dict(faiss_max_wait_ms=16), dict(faiss_max_wait_ms=True)):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                receipt, ready, _ = batched_fixture(td)
                ready.update(change)
                Path(td, 'service-ready.json').write_text(json.dumps(ready))
                value = json.loads(receipt.read_text()); value['service_ready_sha256'] = proxy.digest(Path(td, 'service-ready.json'))
                receipt.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'measured retrieval modes'):
                    proxy.external_binding(receipt)
        with tempfile.TemporaryDirectory() as td:
            receipt, ready, stage = batched_fixture(td)
            stage['faiss_batch_size'] = 8
            Path(td, 'stage64-admission.json').write_text(json.dumps(stage))
            ready['optional_stage64_proof']['sha256'] = proxy.digest(Path(td, 'stage64-admission.json'))
            Path(td, 'service-ready.json').write_text(json.dumps(ready))
            value = json.loads(receipt.read_text()); value['service_ready_sha256'] = proxy.digest(Path(td, 'service-ready.json'))
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'stage64 producer proof'):
                proxy.external_binding(receipt)


class AsyncProductionHTTPTests(unittest.TestCase):
    setUp = legacy_tests.MockHTTPTests.setUp
    tearDown = legacy_tests.MockHTTPTests.tearDown

    def test_real_handlers_keep_cache_on_owner_loop_offload_only_http_and_preserve_health(self):
        from starlette.requests import Request
        from search_g1.test_search_action_stop_wiring import SearchStopWiringTests
        with tempfile.TemporaryDirectory() as td:
            receipt, ready, _ = batched_fixture(td, self.backend.endpoint)
            external = proxy.external_binding(receipt)
            self.health_response.update(ready, ok=True, admission_sha256=external['stage64_sha256'],
                                        statistics=dict(batch_attempts=7, batch_input_queries=11))
            base, run = SearchStopWiringTests().fixture(Path(td)/'training', PROFILE)
            run.update(source_commit='a'*40, external_retriever=external, input_sha256={'stage.parquet': 'c'*64},
                       source_sha256=ready['source_code_sha256'])
            (base / 'logs/run-manifest.json').write_text(json.dumps(run))
            request_threads = []; original_request = PrivateHTTPBackend.request
            forwarders = []; original_components = proxy.forwarding_components

            def tracked_request(backend, *args, **kwargs):
                request_threads.append(threading.get_ident())
                return original_request(backend, *args, **kwargs)

            def components(*args):
                result = original_components(*args); forwarders.append(result[1]); return result

            def inspect_app(app, **kwargs):
                self.assertEqual(kwargs, dict(host='127.0.0.1', port=18010, workers=1))
                retrieve = next(route.endpoint for route in app.routes if route.path == '/retrieve')
                health = next(route.endpoint for route in app.routes if route.path == '/health')
                async def post(payload):
                    async def receive():
                        return dict(type='http.request', body=json.dumps(payload).encode(), more_body=False)
                    return await retrieve(Request(dict(type='http', method='POST', path='/retrieve', headers=[]), receive))
                async def invoke():
                    payload = dict(queries=['A', 'B', 'A'], topk=3, return_scores=True)
                    first = await post(payload); second = await post(payload)
                    self.assertEqual(first.body, second.body)
                    self.assertEqual(json.loads(first.body)['extra']['original_payload'], payload)
                    for _ in range(2):
                        error = await post(dict(queries=['bad'], fail=True))
                        self.assertEqual(error.status_code, 503)
                        self.assertEqual(json.loads(error.body), dict(detail='mock original backend unavailable'))
                    live = await health()
                    self.assertEqual(live['upstream_statistics'], self.health_response['statistics'])
                    self.assertEqual((live['physical_gpu_count'], live['logical_shards'], live['local_proxy_gpu_count']), (2,4,0))
                    self.assertEqual(live['cache']['served_hit_queries'], 1)
                    self.assertEqual(live['cache']['backend_exceptions'], 2)
                    self.assertIs(forwarders[0].owner_loop, asyncio.get_running_loop())
                    self.assertEqual(forwarders[0].owner_thread, threading.get_ident())
                asyncio.run(invoke())
            with patch('uvicorn.run', side_effect=inspect_app), patch(
                    'reproduction.common8.topology.execution_job_id', return_value='5678'), patch.object(
                    proxy, 'forwarding_components', side_effect=components), patch.object(
                    PrivateHTTPBackend, 'request', tracked_request):
                proxy.serve(base)
            self.assertEqual(request_threads[0], threading.get_ident())  # startup identity check
            self.assertTrue(all(ident != threading.get_ident() for ident in request_threads[1:]))
            self.assertEqual(len(self.calls), 3)  # one successful miss and two uncached errors
            self.assertTrue(json.loads((base/'logs/retrieval-admission.json').read_text())['local_proxy_only'])


if __name__ == '__main__':
    unittest.main()
