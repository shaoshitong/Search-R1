"""Opt-in memoization of completed original retrieval batches.

This wrapper has the upstream ``batch_search(query_list, num=None,
return_score=False)`` interface.  It never changes a backend batch, combines
requests, starts a thread, or caches an in-flight/failed lookup.  A miss in any
row sends the *entire original call* to the bound backend.

Keys include the original ordered batch and row position as well as the exact
query, effective top-k, score mode, and explicit asset versions.  Reusing a query
from a different batch could change FP16 encoder/FAISS numerics, so even an
otherwise identical query is deliberately a miss in that case.  The native
asynchronous rollout's one-query requests still benefit from exact reuse.

Create on, and call from, the retriever's original initialization thread.  The
default is disabled.  Statistics are separate from API results and logical
search counts remain the caller's responsibility.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class RetrievalAssetVersions:
    encoder: str
    index: str
    corpus: str

    def __post_init__(self):
        if any(type(value) is not str or not value.strip()
               for value in (self.encoder, self.index, self.corpus)):
            raise ValueError('Explicit encoder, index, and corpus versions are required')


class ExactQueryRetrievalCache:
    """Bounded, synchronous row LRU around an existing bound batch_search."""

    def __init__(self, batch_search, *, versions, default_topk, capacity=4096,
                 enabled=False):
        if not callable(batch_search):
            raise TypeError('batch_search must be callable')
        if not isinstance(versions, RetrievalAssetVersions):
            raise TypeError('versions must be RetrievalAssetVersions')
        if type(default_topk) is not int or default_topk <= 0:
            raise ValueError('default_topk must be a positive integer')
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('capacity must be a positive integer')
        if type(enabled) is not bool:
            raise TypeError('enabled must be a boolean')
        self._backend = batch_search
        self._versions = versions
        self._default_topk = default_topk
        self._capacity = capacity
        self._enabled = enabled
        self._owner_thread = threading.get_ident()
        self._entries = OrderedDict()
        self._counts = dict(logical_calls=0, logical_queries=0, backend_calls=0,
                            backend_queries=0, served_hit_queries=0,
                            lookup_hit_queries=0, lookup_miss_queries=0,
                            backend_exceptions=0, uncacheable_calls=0, evictions=0)

    @property
    def versions(self):
        return self._versions

    def clear(self):
        """Drop completed values without resetting the cumulative counters."""
        self._assert_owner()
        self._entries.clear()

    def statistics(self):
        """Return a detached JSON-compatible snapshot; no logging side effects."""
        value = dict(self._counts, enabled=self._enabled, entries=len(self._entries),
                     capacity=self._capacity,
                     versions=dict(encoder=self._versions.encoder,
                                   index=self._versions.index, corpus=self._versions.corpus))
        total = self._counts['logical_queries']
        value['served_hit_rate'] = self._counts['served_hit_queries'] / total if total else 0.0
        return value

    def _assert_owner(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError('Retrieval cache must stay on the initialization thread')

    @staticmethod
    def _queries(query_list):
        # DenseRetriever supports a single string as well as List[str].
        if type(query_list) is str:
            return (query_list,)
        if type(query_list) is list and all(type(query) is str for query in query_list):
            return tuple(query_list)
        return None

    @staticmethod
    def _rows(result, count, return_score):
        if return_score:
            if type(result) is not tuple or len(result) != 2:
                return None
            documents, scores = result
            if (type(documents) is not list or type(scores) is not list or
                    len(documents) != count or len(scores) != count):
                return None
            return list(zip(documents, scores))
        if type(result) is not list or len(result) != count:
            return None
        return result

    def batch_search(self, query_list, num=None, return_score=False):
        self._assert_owner()
        queries = self._queries(query_list)
        count = len(queries) if queries is not None else 0
        self._counts['logical_calls'] += 1
        self._counts['logical_queries'] += count
        topk = self._default_topk if num is None else num
        cacheable = (self._enabled and queries is not None and bool(queries) and
                     type(topk) is int and topk > 0 and type(return_score) is bool)
        keys = []
        if cacheable:
            keys = [(self._versions, query, topk, return_score, queries, position)
                    for position, query in enumerate(queries)]
            present = [key in self._entries for key in keys]
            self._counts['lookup_hit_queries'] += sum(present)
            self._counts['lookup_miss_queries'] += count - sum(present)
            if all(present):
                rows = [deepcopy(self._entries[key]) for key in keys]
                for key in keys:
                    self._entries.move_to_end(key)
                self._counts['served_hit_queries'] += count
                if return_score:
                    return ([row[0] for row in rows], [row[1] for row in rows])
                return rows
        elif self._enabled:
            self._counts['uncacheable_calls'] += 1

        self._counts['backend_calls'] += 1
        self._counts['backend_queries'] += count
        try:
            # Preserve object/order, original num (including None), score mode,
            # and the backend's original internal encoder batch partitions.
            result = self._backend(query_list, num=num, return_score=return_score)
        except BaseException:
            self._counts['backend_exceptions'] += 1
            raise
        if cacheable:
            rows = self._rows(result, count, return_score)
            if rows is None:
                self._counts['uncacheable_calls'] += 1
            else:
                # Stage every copy first: an incomplete response is never a hit.
                staged = [(key, deepcopy(row)) for key, row in zip(keys, rows)]
                for key, row in staged:
                    self._entries[key] = row
                    self._entries.move_to_end(key)
                    while len(self._entries) > self._capacity:
                        self._entries.popitem(last=False)
                        self._counts['evictions'] += 1
        return deepcopy(result)
