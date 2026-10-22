import asyncio
import threading
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4.retrieval_service import (
    make_four_shard_cloner, original_config, track_batch_search, singleton_faiss_app,
)


class RetrievalServiceTests(unittest.TestCase):
    def test_original_backend_settings_and_no_four_training_gpu_dependency(self):
        config = original_config(SimpleNamespace(Config=lambda **kw: kw), '/i', '/c', '/e')
        self.assertEqual(config, dict(retrieval_method='e5', retrieval_topk=3,
            index_path='/i', corpus_path='/c', faiss_gpu=True, retrieval_model_path='/e',
            retrieval_pooling_method='mean', retrieval_query_max_length=256,
            retrieval_use_fp16=True, retrieval_batch_size=512))

    def test_four_logical_shards_keep_original_options_and_explicit_physical_map(self):
        calls = []
        def multiple(resources, index, *, co, gpus):
            calls.append((resources, index, co, gpus)); return 'four-original-shards'
        resources = [object() for _ in range(4)]
        faiss = SimpleNamespace(index_cpu_to_gpu_multiple_py=multiple)
        cloner = make_four_shard_cloner(faiss, resources, [0, 1, 0, 1])
        options = SimpleNamespace(useFloat16=True, shard=True)
        index = object()
        self.assertEqual(cloner(index, co=options), 'four-original-shards')
        self.assertIs(calls[0][0], resources)
        self.assertIs(calls[0][1], index)
        self.assertIs(calls[0][2], options)
        self.assertEqual(calls[0][3], [0, 1, 0, 1])

    def test_unknown_api_or_changed_sharding_never_silently_falls_back(self):
        with self.assertRaises(RuntimeError):
            make_four_shard_cloner(SimpleNamespace(), [None] * 4, [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            make_four_shard_cloner(SimpleNamespace(), [None] * 2, [0, 1])
        cloner = make_four_shard_cloner(SimpleNamespace(index_cpu_to_gpu_multiple_py=lambda *a, **k: None),
                                      [None] * 4, [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            cloner(object(), co=SimpleNamespace(useFloat16=False, shard=True))

    def test_original_query_batch_argument_identity_scores_and_result_are_preserved(self):
        calls = []
        result = ([[dict(contents='doc')]], [[0.75]])
        def original(*args, **kwargs):
            calls.append((args, kwargs, threading.get_ident())); return result
        stats = dict(logical_requests=0, logical_queries=0, completed_requests=0,
            completed_queries=0, failed_requests=0, backend_seconds=0.0, last_completed_unix=None)
        wrapped = track_batch_search(original, stats, threading.get_ident())
        queries = ['Exact query?', 'Other query']
        self.assertIs(wrapped(query_list=queries, num=3, return_score=True), result)
        self.assertIs(calls[0][1]['query_list'], queries)
        self.assertEqual(calls[0][1], dict(query_list=queries, num=3, return_score=True))
        self.assertEqual(stats['completed_queries'], 2)
        self.assertEqual(stats['logical_requests'], 1)


class PrivateBatchingAPITests(unittest.IsolatedAsyncioTestCase):
    def make_app(self):
        from pydantic import BaseModel
        from search_g1.test_singleton_faiss_batcher import FakeEncoder, FakeIndex

        class QueryRequest(BaseModel):
            queries: list[str]
            topk: Optional[int] = None
            return_scores: bool = False

        encoder, index = FakeEncoder(), FakeIndex()
        retriever = SimpleNamespace(encoder=encoder, index=index,
            corpus={i: dict(contents='document-%d' % i) for i in range(100)})
        fallback = []

        async def original(request):
            fallback.append((list(request.queries), request.topk, request.return_scores,
                             threading.get_ident()))
            return dict(result=[['original-fallback'] for _ in request.queries])

        upstream = SimpleNamespace(QueryRequest=QueryRequest, config=SimpleNamespace(retrieval_topk=3),
            retrieve_endpoint=original, load_docs=lambda corpus, ids: [corpus[int(i)] for i in ids])
        identity = dict(service_job_id='private-fake', request_rebatching_enabled=False)
        stats = dict(logical_requests=0, logical_queries=0, completed_requests=0,
            completed_queries=0, failed_requests=0, backend_seconds=0.0, last_completed_unix=None)
        app = singleton_faiss_app(upstream, retriever, identity, stats,
            threading.get_ident(), batch_size=2, max_wait_ms=50)
        return app, encoder, index, fallback

    async def test_singletons_keep_original_json_document_order_and_owner_thread(self):
        import httpx
        app, encoder, index, fallback = self.make_app()
        with patch('reproduction.easy2560_joint4.retrieval_service.rss_bytes', return_value=128):
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://private') as client:
                    plain, scored = await asyncio.gather(
                        client.post('/retrieve', json=dict(queries=['2'])),
                        client.post('/retrieve', json=dict(queries=['3'], topk=3, return_scores=True)))
                    self.assertEqual(plain.json(), dict(result=[[dict(contents='document-%d' % i)
                        for i in (20, 21, 22)]]))
                    rows = scored.json()['result'][0]
                    self.assertEqual([row['document']['contents'] for row in rows],
                                     ['document-30', 'document-31', 'document-32'])
                    self.assertAlmostEqual(rows[0]['score'], 3.3, places=6)
                    health = (await client.get('/health')).json()
        self.assertEqual(encoder.calls, [['2'], ['3']])
        self.assertEqual(index.batch_sizes, [2])
        self.assertEqual(set(encoder.threads + index.threads), {threading.get_ident()})
        self.assertEqual(fallback, [])
        self.assertTrue(health['request_rebatching_enabled'] and health['singleton_e5'])
        self.assertEqual((health['faiss_batch_size'], health['faiss_max_wait_ms']), (2, 50))
        self.assertEqual(health['statistics']['completed_queries'], 2)
        self.assertEqual(health['statistics']['batch_attempts'], 1)
        self.assertTrue(app.state.batcher.worker.done())

    async def test_multiquery_and_non3_topk_use_original_endpoint(self):
        import httpx
        app, encoder, index, fallback = self.make_app()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://private') as client:
                multi = await client.post('/retrieve', json=dict(queries=['2', '3'], topk=3, return_scores=True))
                other_topk = await client.post('/retrieve', json=dict(queries=['4'], topk=5))
                original = await client.post('/retrieve-original', json=dict(queries=['5'], topk=3))
        self.assertEqual(multi.json(), dict(result=[['original-fallback'], ['original-fallback']]))
        self.assertEqual(other_topk.json(), dict(result=[['original-fallback']]))
        self.assertEqual(original.json(), dict(result=[['original-fallback']]))
        self.assertEqual(fallback, [(['2', '3'], 3, True, threading.get_ident()),
                                    (['4'], 5, False, threading.get_ident()),
                                    (['5'], 3, False, threading.get_ident())])
        self.assertEqual(encoder.calls, [])
        self.assertEqual(index.batch_sizes, [])


if __name__ == '__main__':
    unittest.main()
