import asyncio
import threading
import unittest

import numpy as np

from search_g1.singleton_faiss_batcher import SingletonFaissBatcher


class FakeEncoder:
    def __init__(self):
        self.calls, self.threads = [], []

    def encode(self, queries):
        self.calls.append(list(queries)); self.threads.append(threading.get_ident())
        return np.array([[float(query), 1] for query in queries], dtype=np.float32)


class FakeIndex:
    def __init__(self, fail_negative=False):
        self.batch_sizes, self.threads = [], []
        self.fail_negative = fail_negative

    def search(self, vectors, k):
        self.batch_sizes.append(len(vectors)); self.threads.append(threading.get_ident())
        if self.fail_negative and (vectors[:, 0] < 0).any():
            raise ValueError('fake batch failure')
        assert k == 3 and vectors.dtype == np.float32
        scores = vectors[:, :1] + np.array([[.3, .2, .1]], dtype=np.float32)
        ids = (vectors[:, :1] * 10).astype(np.int64) + np.array([[0, 1, 2]])
        return scores, ids


class SingletonFaissBatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_singleton_encoder_order_batch_mapping_and_owner_thread(self):
        encoder, index = FakeEncoder(), FakeIndex()
        batcher = SingletonFaissBatcher(encoder, index, max_batch_size=3, max_wait_ms=2)
        results = await asyncio.gather(*(batcher.submit(str(i)) for i in range(5)))
        await batcher.close()
        self.assertEqual(encoder.calls, [[str(i)] for i in range(5)])
        self.assertEqual(index.batch_sizes, [3, 2])
        self.assertEqual(set(encoder.threads + index.threads), {threading.get_ident()})
        for value, (scores, ids) in enumerate(results):
            np.testing.assert_array_equal(ids, np.array([value * 10 + i for i in range(3)]))
            np.testing.assert_array_equal(scores, np.array([value], dtype=np.float32) + np.array([.3, .2, .1], dtype=np.float32))

    async def test_batch_error_delivery_close_drains_and_rejects_new_requests(self):
        encoder, index = FakeEncoder(), FakeIndex(fail_negative=True)
        batcher = SingletonFaissBatcher(encoder, index, max_batch_size=2, max_wait_ms=100)
        tasks = [asyncio.create_task(batcher.submit(query)) for query in ('-1', '2', '3')]
        await asyncio.sleep(0)
        await batcher.close()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.assertTrue(all(isinstance(result, ValueError) for result in results[:2]))
        np.testing.assert_array_equal(results[2][1], np.array([30, 31, 32]))
        self.assertEqual(encoder.calls, [['-1'], ['2'], ['3']])
        self.assertEqual(index.batch_sizes, [2, 1])
        await batcher.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            await batcher.submit('4')


if __name__ == '__main__':
    unittest.main()
