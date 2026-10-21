import threading
from types import SimpleNamespace
import unittest

from reproduction.easy2560_joint4.retrieval_service import (
    make_four_shard_cloner, original_config, track_batch_search,
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


if __name__ == '__main__':
    unittest.main()
