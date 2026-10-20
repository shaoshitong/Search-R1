"""CPU-only regression for cross-request batching and backend ownership."""
import asyncio
import threading
import unittest

from reproduction.operations.qwen7b_resample.shared_vram_retriever import (
    AsyncBatchedRequests, OwnerThreadRetriever, shard_counts)


class SharedRetrieverMicrobatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_owner_fifo_scores_failure_and_shutdown(self):
        self.assertEqual(shard_counts([17, 22, 22, 33, 34]), [17, 22, 22, 33, 34])
        self.assertEqual(shard_counts([8, 11, 11, 17, 17], 64), [8, 11, 11, 17, 17])
        main_thread = threading.get_ident()
        first_started, release_first = threading.Event(), threading.Event()
        events, calls = [], []

        class FakeRetriever:
            def __init__(self):
                events.append(('initialize', threading.get_ident()))

            def batch_search(self, *, query_list, num, return_score):
                events.append(('search', threading.get_ident()))
                calls.append(list(query_list))
                if len(calls) == 1:
                    first_started.set()
                    if not release_first.wait(timeout=5):
                        raise TimeoutError('Test did not release its first backend call')
                if 'boom' in query_list:
                    raise RuntimeError('fake backend failure')
                assert num == 3 and return_score is True
                docs = [[{'query': query, 'rank': rank} for rank in range(3)] for query in query_list]
                scores = [[float(len(query)) + rank / 10 for rank in range(3)] for query in query_list]
                return docs, scores

            def close(self):
                events.append(('close', threading.get_ident()))

        def factory():
            backend = FakeRetriever()
            return backend, {'ready': True}, backend.close

        owner = OwnerThreadRetriever(factory)
        batching = AsyncBatchedRequests(owner, batch_queries=4, delay_seconds=.002)
        try:
            first = asyncio.create_task(batching.submit(['first'], True))

            async def wait_started():
                while not first_started.is_set():
                    await asyncio.sleep(.001)

            await asyncio.wait_for(wait_started(), timeout=2)
            second = asyncio.create_task(batching.submit(['second', 'third'], False))
            third = asyncio.create_task(batching.submit(['fourth'], True))
            await asyncio.sleep(.01)
            # HTTP-loop work and queue admission continue while the sole
            # backend thread is deliberately blocked in its first search.
            self.assertEqual(len(batching.pending), 2)
            self.assertFalse(second.done())
            release_first.set()
            results = await asyncio.wait_for(asyncio.gather(first, second, third), timeout=2)
            self.assertEqual(calls[:2], [['first'], ['second', 'third', 'fourth']])
            self.assertEqual(results[0]['result'][0][2],
                             {'document': {'query': 'first', 'rank': 2}, 'score': 5.2})
            self.assertEqual(results[1]['result'],
                             [[{'query': query, 'rank': rank} for rank in range(3)]
                              for query in ('second', 'third')])
            self.assertEqual(results[2]['result'][0][1],
                             {'document': {'query': 'fourth', 'rank': 1}, 'score': 6.1})
            self.assertEqual((batching.requests, batching.queries, batching.batches), (3, 4, 2))

            failed = await asyncio.wait_for(asyncio.gather(
                batching.submit(['boom'], True), batching.submit(['also-fails'], False),
                return_exceptions=True), timeout=2)
            self.assertEqual(calls[2], ['boom', 'also-fails'])
            self.assertTrue(all(isinstance(error, RuntimeError) and str(error) == 'fake backend failure'
                                for error in failed))
            recovered = await asyncio.wait_for(batching.submit(['recovered'], True), timeout=2)
            self.assertEqual(recovered['result'][0][0]['document']['query'], 'recovered')
            self.assertEqual((batching.requests, batching.queries, batching.batches), (4, 5, 3))
            self.assertGreaterEqual(batching.retrieval_seconds, 0)
        finally:
            release_first.set()
            try:
                await batching.aclose()
            finally:
                await owner.aclose()
        self.assertEqual([kind for kind, _ in events].count('initialize'), 1)
        self.assertEqual([kind for kind, _ in events].count('close'), 1)
        self.assertEqual({ident for _, ident in events}, {owner.thread_id})
        self.assertNotEqual(owner.thread_id, main_thread)
        self.assertTrue(owner._released)

        # Cancelling shutdown while cleanup is queued behind an active search
        # must leave the loop responsive and still release on the owner.
        search_started, finish_search = threading.Event(), threading.Event()
        close_events = []

        class ClosingRetriever:
            def batch_search(self, **_):
                search_started.set()
                if not finish_search.wait(timeout=5):
                    raise TimeoutError('Test did not release shutdown search')
                return [], []

            def close(self):
                close_events.append(threading.get_ident())

        def closing_factory():
            backend = ClosingRetriever()
            return backend, {}, backend.close

        closing_owner = OwnerThreadRetriever(closing_factory)
        search = asyncio.create_task(closing_owner.search(['closing']))
        closing = None
        try:
            async def wait_closing_search():
                while not search_started.is_set():
                    await asyncio.sleep(.001)

            await asyncio.wait_for(wait_closing_search(), timeout=2)
            closing = asyncio.create_task(closing_owner.aclose())
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(.005)
            self.assertFalse(closing.done())
            finish_search.set()
            await asyncio.wait_for(search, timeout=2)
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(closing, timeout=2)
            self.assertEqual(close_events, [closing_owner.thread_id])
            self.assertTrue(closing_owner._released)
        finally:
            finish_search.set()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            closing_owner.close()


if __name__ == '__main__':
    unittest.main()
