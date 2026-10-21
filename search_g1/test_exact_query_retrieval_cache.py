"""CPU coverage of original-batch/value preservation and bounded exact reuse."""
from copy import deepcopy
import threading
import unittest

from search_g1.exact_query_retrieval_cache import (
    ExactQueryRetrievalCache, RetrievalAssetVersions,
)


VERSIONS = RetrievalAssetVersions('encoder-sha', 'index-sha', 'corpus-sha')


class Backend:
    def __init__(self):
        self.calls = []
        self.failure = None

    def batch_search(self, query_list, num=None, return_score=False):
        self.calls.append((query_list, num, return_score, threading.get_ident()))
        if self.failure is not None:
            raise self.failure
        queries = [query_list] if isinstance(query_list, str) else query_list
        topk = 3 if num is None else num
        # Include batch shape/order and position in the score, so incorrectly
        # serving a row from another batch cannot silently pass this fixture.
        shape = sum((i + 1) * sum(map(ord, query)) for i, query in enumerate(queries))
        documents = [[dict(contents=query, nested=dict(tags=[query, j]))
                      for j in range(topk)] for query in queries]
        scores = [[shape + i / 8 + j / 16 for j in range(topk)]
                  for i in range(len(queries))]
        return (documents, scores) if return_score else documents


class ExactQueryRetrievalCacheTests(unittest.TestCase):
    def make(self, capacity=16, enabled=True, versions=VERSIONS):
        backend = Backend()
        wrapper = ExactQueryRetrievalCache(backend.batch_search, versions=versions,
            default_topk=3, capacity=capacity, enabled=enabled)
        return backend, wrapper

    def test_default_disabled_keeps_original_calls_and_result_shape(self):
        backend = Backend()
        wrapper = ExactQueryRetrievalCache(backend.batch_search,
            versions=VERSIONS, default_topk=3)
        queries = ['Exact query?']
        first = wrapper.batch_search(queries, return_score=True)
        second = wrapper.batch_search(queries, return_score=True)
        self.assertEqual(first, second)
        self.assertIs(type(first), tuple)
        self.assertEqual(len(backend.calls), 2)
        self.assertTrue(all(call[0] is queries and call[1] is None for call in backend.calls))
        self.assertEqual(wrapper.statistics()['entries'], 0)

    def test_exact_one_query_hit_preserves_scores_and_nested_copy(self):
        backend, wrapper = self.make()
        original = wrapper.batch_search(['Who?'], return_score=True)
        expected = deepcopy(original)
        original[0][0][0]['nested']['tags'].append('caller mutation')
        original[1][0][0] = -1
        second = wrapper.batch_search(['Who?'], return_score=True)
        self.assertEqual(second, expected)
        second[0][0].clear()
        self.assertEqual(wrapper.batch_search(['Who?'], return_score=True), expected)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(wrapper.statistics()['served_hit_queries'], 2)

    def test_exact_query_no_case_space_or_punctuation_normalization(self):
        backend, wrapper = self.make()
        for query in ['Q?', 'q?', 'Q? ', ' Q?', 'Q!']:
            wrapper.batch_search([query])
        self.assertEqual(len(backend.calls), 5)
        wrapper.batch_search(['Q?'])
        self.assertEqual(len(backend.calls), 5)

    def test_topk_and_score_mode_are_separate_keys_and_none_uses_default(self):
        backend, wrapper = self.make()
        expected = wrapper.batch_search(['Q'])
        self.assertEqual(wrapper.batch_search(['Q'], num=3), expected)
        wrapper.batch_search(['Q'], num=2)
        scored = wrapper.batch_search(['Q'], num=3, return_score=True)
        self.assertIs(type(scored), tuple)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual([call[1] for call in backend.calls], [None, 2, 3])

    def test_mixed_batch_passes_original_batch_and_does_not_subselect_or_dedup(self):
        backend, wrapper = self.make()
        wrapper.batch_search(['A'], return_score=True)
        queries = ['A', 'B', 'A']
        actual = wrapper.batch_search(queries, num=2, return_score=True)
        self.assertIs(backend.calls[-1][0], queries)
        self.assertEqual(backend.calls[-1][1:3], (2, True))
        self.assertNotEqual(actual[1][0], actual[1][2])
        self.assertEqual(wrapper.batch_search(queries, num=2, return_score=True), actual)
        self.assertEqual(len(backend.calls), 2)
        # Permuting a known batch must recompute with that original permutation.
        reverse = ['B', 'A', 'A']
        wrapper.batch_search(reverse, num=2, return_score=True)
        self.assertIs(backend.calls[-1][0], reverse)
        self.assertEqual(len(backend.calls), 3)

    def test_partial_eviction_still_calls_whole_original_batch(self):
        backend, wrapper = self.make(capacity=2)
        queries = ['A', 'B', 'C']
        expected = wrapper.batch_search(queries, return_score=True)
        self.assertEqual(wrapper.statistics()['entries'], 2)
        self.assertEqual(wrapper.batch_search(queries, return_score=True), expected)
        self.assertEqual(len(backend.calls), 2)
        self.assertTrue(all(call[0] is queries for call in backend.calls))
        stats = wrapper.statistics()
        self.assertEqual(stats['served_hit_queries'], 0)
        self.assertEqual(stats['lookup_hit_queries'], 2)

    def test_lru_evicts_least_recent_and_cache_never_exceeds_capacity(self):
        backend, wrapper = self.make(capacity=2)
        wrapper.batch_search(['A']); wrapper.batch_search(['B'])
        wrapper.batch_search(['A']); wrapper.batch_search(['C'])
        wrapper.batch_search(['A'])
        self.assertEqual(len(backend.calls), 3)
        wrapper.batch_search(['B'])
        self.assertEqual(len(backend.calls), 4)
        self.assertEqual(wrapper.statistics()['entries'], 2)
        self.assertEqual(wrapper.statistics()['evictions'], 2)

    def test_backend_exception_is_propagated_not_cached(self):
        backend, wrapper = self.make()
        error = RuntimeError('actual lookup failed')
        backend.failure = error
        for _ in range(2):
            with self.assertRaises(RuntimeError) as caught:
                wrapper.batch_search(['A'], return_score=True)
            self.assertIs(caught.exception, error)
        self.assertEqual(wrapper.statistics()['entries'], 0)
        backend.failure = None
        expected = wrapper.batch_search(['A'], return_score=True)
        self.assertEqual(wrapper.batch_search(['A'], return_score=True), expected)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(wrapper.statistics()['backend_exceptions'], 2)

    def test_versions_are_explicit_immutable_and_part_of_every_key(self):
        backend, wrapper = self.make()
        wrapper.batch_search(['A'])
        for key in wrapper._entries:
            self.assertEqual(key[0], VERSIONS)
        for values in [('', 'i', 'c'), ('e', '', 'c'), ('e', 'i', '')]:
            with self.assertRaises(ValueError):
                RetrievalAssetVersions(*values)
        with self.assertRaises(AttributeError):
            wrapper.versions = RetrievalAssetVersions('new', 'i', 'c')
        other = ExactQueryRetrievalCache(backend.batch_search,
            versions=RetrievalAssetVersions('new', 'i', 'c'), default_topk=3, enabled=True)
        other.batch_search(['A'])
        self.assertEqual(len(backend.calls), 2)

    def test_string_empty_batch_and_detached_statistics(self):
        backend, wrapper = self.make()
        expected = wrapper.batch_search('A', return_score=True)
        self.assertEqual(wrapper.batch_search(['A'], return_score=True), expected)
        self.assertEqual(wrapper.batch_search([], return_score=True), ([], []))
        self.assertEqual(wrapper.batch_search([]), [])
        snapshot = wrapper.statistics()
        self.assertEqual(snapshot['logical_calls'], 4)
        self.assertEqual(snapshot['logical_queries'], 2)
        self.assertEqual(snapshot['served_hit_rate'], .5)
        snapshot['versions']['encoder'] = 'corrupted'
        self.assertEqual(wrapper.statistics()['versions']['encoder'], VERSIONS.encoder)
        wrapper.clear()
        self.assertEqual(wrapper.statistics()['entries'], 0)
        self.assertEqual(wrapper.statistics()['logical_calls'], 4)

    def test_no_backend_dispatch_on_a_different_thread_even_if_cache_hits(self):
        backend, wrapper = self.make()
        wrapper.batch_search(['A'])
        errors = []
        def worker():
            try:
                wrapper.batch_search(['A'])
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=worker)
        thread.start(); thread.join()
        self.assertEqual(errors, ['Retrieval cache must stay on the initialization thread'])
        self.assertEqual(len(backend.calls), 1)


if __name__ == '__main__':
    unittest.main()
