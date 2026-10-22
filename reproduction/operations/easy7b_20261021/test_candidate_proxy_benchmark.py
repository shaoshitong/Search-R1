import asyncio
from copy import deepcopy
import threading
import time
import unittest

from reproduction.operations.easy7b_20261021.candidate_proxy_benchmark import (
    PrivateHTTPBackend, UpstreamHTTPError, dispatch, make_forwarder,
)


READY = dict(encoder_manifest_sha256='encoder', index_sha256='index', corpus_sha256='corpus')


class FakeBackend(PrivateHTTPBackend):
    def __init__(self, pause=0):
        super().__init__('http://private/retrieve-original')
        self.calls, self.threads, self.pause = [], [], pause

    def request(self, payload=None, *, health=False):
        self.calls.append(deepcopy(payload)); self.threads.append(threading.get_ident())
        if self.pause:
            time.sleep(self.pause)
        if payload.get('fail'):
            raise UpstreamHTTPError(503, dict(detail='fake failure'))
        return dict(result=[dict(original_payload=deepcopy(payload))])


class CandidateProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_modes_match_and_cache_the_complete_ordered_payload(self):
        payloads = [dict(queries=['a', 'b'], topk=3, return_scores=True),
            dict(queries=['b', 'a'], topk=3, return_scores=True),
            dict(queries=['a', 'b'], topk=4, return_scores=True),
            dict(queries=['a', 'b'], topk=3, return_scores=False)]
        observed = []
        for mode in ('legacy', 'async'):
            backend = FakeBackend(); forwarder = make_forwarder(mode, backend, READY)
            self.assertEqual(forwarder.statistics()['cache']['entries'], 0)
            results = [await dispatch(mode, forwarder, payload) for payload in payloads]
            results[0]['result'][0]['original_payload']['queries'].append('caller mutation')
            cached = await dispatch(mode, forwarder, payloads[0])
            self.assertEqual(cached['result'][0]['original_payload'], payloads[0])
            self.assertEqual(backend.calls, payloads)
            self.assertEqual(forwarder.statistics()['cache']['served_hit_queries'], 1)
            self.assertEqual(forwarder.statistics()['cache']['capacity'], 4096)
            if mode == 'legacy':
                self.assertEqual(set(backend.threads), {threading.get_ident()})
            else:
                self.assertNotIn(threading.get_ident(), backend.threads)
            observed.append(cached)
        self.assertEqual(*observed)

    async def test_async_does_not_merge_inflight_or_cache_failed_requests(self):
        backend = FakeBackend(pause=.01); forwarder = make_forwarder('async', backend, READY)
        payload = dict(queries=['same'], topk=3)
        first, second = await asyncio.gather(forwarder.retrieve(payload), forwarder.retrieve(payload))
        self.assertEqual(first, second)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(await forwarder.retrieve(payload), first)
        for _ in range(2):
            with self.assertRaises(UpstreamHTTPError):
                await forwarder.retrieve(dict(queries=['failed'], fail=True))
        stats = forwarder.statistics()['cache']
        self.assertEqual((stats['backend_calls'], stats['served_hit_queries'], stats['backend_exceptions']), (4, 1, 2))
        self.assertEqual(stats['entries'], 1)


if __name__ == '__main__':
    unittest.main()
