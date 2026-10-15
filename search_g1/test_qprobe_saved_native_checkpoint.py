"""Small CPU fixtures for the added native saved-CP5 qualification branch."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search_g1 import qprobe_saved_checkpoint as saved
from search_g1.qprobe_runtime import _token_cache_binding
from search_g1.test_qprobe_token_runtime import MemoryTokenCache


class NativeSavedCheckpointTests(unittest.TestCase):
    def fixture(self):
        cache = MemoryTokenCache()
        labels = [dict(qid='q'+str(i), question_hash=cache.index['q'+str(i)]['question_hash'], source_outer=outer)
                  for outer in range(1, 6) for i in range(512)]
        head = SimpleNamespace(completed_outer=5, update_count=5, version='qprobe-qwen:5:5:fixture',
            replay_labels=labels, replay_features=[row['qid'] for row in labels],
            initialization_provenance=dict(warmup_source_commit=saved.NATIVE_WARMUP_SOURCE,
                warmup_ready=saved.NATIVE_WARMUP_READY, teacher_replay_excluded=True),
            verify_native_provenance=lambda: dict(accepted=True, teacher_replay_excluded=True,
                initial_block_proof_sha256='a'*64, native_implementation_sha256='b'*64),
            _adam_steps=lambda: 260)
        qstate = dict(config=dict(head_kind=saved.NATIVE_HEAD_KIND, head_device='cuda:0'),
            head_kind=saved.NATIVE_HEAD_KIND, cache_binding=_token_cache_binding(cache),
            head_sha256='c'*64, head_version=head.version)
        return cache, head, qstate

    def test_native_feature_set_prefix_and_cpu_provenance_keep_actor_inventory_unchanged(self):
        config = dict(head_kind=saved.NATIVE_HEAD_KIND)
        self.assertEqual(set(saved._feature_cache_files(config)),
            {'manifest.json', 'index.jsonl', 'offsets.npy', 'lengths.npy', 'packed_features.npy'})
        self.assertEqual(saved._head5_prefix(config), 'qprobe-qwen:5:5:')
        self.assertEqual(len(saved.INVENTORY_FILES), 21)
        self.assertEqual(len(saved.DECLARED_FILES), 13)
        self.assertEqual(saved._feature_cache_files({}), ('manifest.json', 'index.jsonl', 'features.npy'))
        self.assertEqual(saved._head5_prefix({}), 'qprobe:5:5:')
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_HEAD_KIND'):
            saved._feature_cache_files(dict(head_kind='unreviewed'))
        cache, head, qstate = self.fixture()
        with patch('search_g1.qprobe_runtime.load_head_checkpoint', return_value=head) as load:
            proof = saved._native_head_proof(qstate, '/data/fixture/head.pt', 'run:5', cache)
        self.assertTrue(proof['teacher_replay_excluded'])
        self.assertEqual(proof['actual_head_adam_updates'], 260)
        self.assertEqual(proof['formal_replay_groups'], 2560)
        self.assertFalse(proof['actual_cuda_restore_accepted'])
        self.assertIsNone(load.call_args.kwargs['feature_cache'])
        self.assertEqual(load.call_args.kwargs['expected_actor_snapshot_id'], 'run:5')

    def test_native_saved_proof_rejects_teacher_rows_old_versions_wrong_adam_and_cache(self):
        for fault in ('teacher', 'version', 'adam', 'source', 'feature_hash', 'head_kind', 'cache_binding'):
            cache, head, qstate = self.fixture()
            if fault == 'teacher': head.replay_labels[0]['source_outer'] = 0
            elif fault == 'version': head.version = 'qprobe-token:5:5:old'
            elif fault == 'adam': head._adam_steps = lambda: 200
            elif fault == 'source': head.initialization_provenance['warmup_source_commit'] = '9'*40
            elif fault == 'feature_hash': head.replay_labels[0]['question_hash'] = '0'*64
            elif fault == 'head_kind': qstate['head_kind'] = 'token_binary_v1'
            elif fault == 'cache_binding': qstate['cache_binding'] = {}
            with self.subTest(fault=fault), patch('search_g1.qprobe_runtime.load_head_checkpoint', return_value=head):
                with self.assertRaises(ValueError):
                    saved._native_head_proof(qstate, '/data/fixture/head.pt', 'run:5', cache)


if __name__ == '__main__': unittest.main()
