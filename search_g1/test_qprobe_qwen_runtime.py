"""Native Qwen routing and complete-outer transactions with tiny CPU providers."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search_g1 import qprobe_runtime as native
from search_g1.test_qprobe_token_runtime import MemoryTokenCache, RecordingHead, questions, runtime, stage


class NativeRecordingHead(RecordingHead):
    def __init__(self):
        super().__init__()
        self.initialization_provenance.update(teacher_replay_excluded=True, formal_replay_groups_at_adoption=0)
        self.replay_labels = []
        self.replay_features = []
        self.protocol = None
        self.formal_update_count = 0
        self.native_verifications = 0

    def verify_native_provenance(self):
        self.native_verifications += 1
        return dict(teacher_replay_excluded=True, native_class='Qwen2DecoderLayer')


def native_runtime(base, **kwargs):
    kwargs.setdefault('head', NativeRecordingHead())
    return runtime(base, head_kind=native.NATIVE_QWEN_HEAD, bootstrap_head_epochs=10,
        head_epochs_per_outer=2, replay_window_outer=8, teacher_warmup_in_replay=False,
        normalize_replay_weights=False, prediction_batch_size=32, **kwargs)


def initial_ready(value, base, **ready_changes):
    checkpoint = base/'initial-native.pt'
    value.head.save(checkpoint, actor_snapshot_id='run:0')
    digest = native._sha(checkpoint)
    ready = dict(stage='QWEN_NATIVE_FORMAL_INITIAL_HEAD_ADOPTED', head_receipt=dict(sha256=digest),
        feature_spec_sha=value.feature_spec_sha, actual_head_adam_updates=200, completed_outer=0,
        warmup_questions=5120, trajectories_regenerated=False, no_head_refit=True,
        raw_warmup_checkpoint_sha256='b' * 64, teacher_replay_excluded=True,
        native_block_proof=dict(native_class='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer',
            official_block_copy_verified=True, block_index=35, transformers_version='fixture-native'))
    ready.update(ready_changes)
    path = base/'native-ready.json'
    path.write_text(json.dumps(ready))
    value.config.update(initial_head_checkpoint=str(checkpoint), initial_head_checkpoint_sha256=digest,
        initial_head_ready=str(path), initial_head_ready_sha256=native._sha(path))
    return ready


class NativeQwenRuntimeTests(unittest.TestCase):
    def test_native_recipe_rejects_old_epochs_replay_teacher_inclusion_and_normalization(self):
        config = dict(head_kind=native.NATIVE_QWEN_HEAD)
        native.validate_config(config)
        for changed in (dict(bootstrap_head_epochs=20), dict(head_epochs_per_outer=1), dict(replay_window_outer=16),
                        dict(teacher_warmup_in_replay=True), dict(normalize_replay_weights=True),
                        dict(ranking='expected_sample_std')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                native.validate_config(dict(config, **changed))

    def test_native_cache_factory_selects_token_representation_and_no_head_loader(self):
        with patch('search_g1.qprobe_token_features.TokenFeatureCache', return_value='packed-native') as load:
            value = native.cache_for_config(dict(head_kind=native.NATIVE_QWEN_HEAD, cache_dir='/data/packed'))
        self.assertEqual(value, 'packed-native')
        load.assert_called_once_with('/data/packed')

    def test_typed_head_factory_uses_native_class_and_checks_bytes_before_loading(self):
        calls = []
        class Loader:
            @classmethod
            def load(cls, path, **kwargs):
                calls.append((path, kwargs)); return 'native-head'
        module = SimpleNamespace(OnlineQwenBlockQProbe=Loader)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'head.pt'; path.write_bytes(b'fixture')
            cache = MemoryTokenCache()
            with patch.dict(sys.modules, {'search_g1.qprobe_qwen_block_online': module}):
                value = native.load_head_checkpoint(path, dict(head_kind=native.NATIVE_QWEN_HEAD, head_device='cpu'),
                    feature_cache=cache, expected_actor_snapshot_id='run:1',
                    expected_feature_spec_sha=cache.spec_sha, expected_checkpoint_sha=native._sha(path))
                with self.assertRaisesRegex(ValueError, 'checkpoint hash differs'):
                    native.load_head_checkpoint(path, dict(head_kind=native.NATIVE_QWEN_HEAD), expected_checkpoint_sha='0'*64)
            self.assertEqual(value, 'native-head')
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][1]['feature_provider'], cache)
            self.assertEqual(calls[0][1]['expected_actor_snapshot_id'], 'run:1')
            with patch.dict(sys.modules, {'search_g1.qprobe_qwen_block_online': module}):
                native.load_head_checkpoint(path, dict(head_kind=native.NATIVE_QWEN_HEAD, head_device='cuda:0'))
            self.assertEqual(calls[-1][1]['device'], 'cpu')
            self.assertIsNone(calls[-1][1]['feature_provider'])

    def test_native_all_candidate_prediction_is_qid_batched_and_no_count_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            value = native_runtime(temp)
            qids, predictions, version = value.score(questions())
            self.assertTrue(value.native_qwen)
            self.assertEqual(value.head_kind, native.NATIVE_QWEN_HEAD)
            self.assertEqual(len(qids), 5120)
            self.assertLessEqual(max(map(len, value.head.predict_calls)), 32)
            self.assertEqual(set(predictions), {'mixed_probability', 'predicted_sample_std'})
            self.assertEqual(version, value.version)

    def test_native_initial_adoption_requires_actual_copy_and_empty_formal_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            value = native_runtime(base)
            initial_ready(value, base)
            adoption = value.bootstrap_initial_head()
            self.assertTrue(adoption['teacher_replay_excluded'])
            self.assertEqual(value.head.native_verifications, 1)
            self.assertEqual(value.head.update_count, 0)
            value.head.replay_labels = [{'teacher': True}]
            with self.assertRaisesRegex(ValueError, 'empty formal replay proof'):
                value.bootstrap_initial_head()
            value.head.replay_labels = []
            value.head.initialization_provenance['teacher_replay_excluded'] = False
            with self.assertRaisesRegex(ValueError, 'empty formal replay proof'):
                value.bootstrap_initial_head()

    def test_native_aborted_actor_does_not_fit_and_completed_actor_commits_compact_qids_once(self):
        with tempfile.TemporaryDirectory() as temp:
            value = native_runtime(temp)
            stage(value, 'failed-native')
            value.abort_outer(1, {'attempt_id': 'failed-native'})
            self.assertEqual(value.head.update_count, 0)
            stage(value, 'successful-native')
            self.assertNotIn('features', value.pending)
            event = value.complete_outer(1, 'run:1', {'attempt_id': 'successful-native'})
            self.assertEqual(len(value.head.updated_qids), 512)
            self.assertTrue(all(isinstance(qid, str) for qid in value.head.updated_qids))
            self.assertEqual(event['prequential']['all']['questions'], 512)
            self.assertIn('binary_accuracy', event['prequential']['all'])
            self.assertNotIn('count_accuracy', event['prequential']['all'])
            with self.assertRaises(RuntimeError): value.complete_outer(1, 'run:1')

    def test_native_snapshot_binds_head_kind_and_rejects_token_head_state(self):
        with tempfile.TemporaryDirectory() as temp:
            value = native_runtime(temp)
            stage(value)
            value.complete_outer(1, 'run:1')
            receipt = value.save(Path(temp)/'paired.pt', 'run:1')
            import torch
            state = torch.load(receipt['path'], map_location='cpu', weights_only=False)
            self.assertEqual(state['head_kind'], native.NATIVE_QWEN_HEAD)
            self.assertEqual(state['cache_binding'], value.cache_binding)
            legacy = runtime(temp)
            with self.assertRaisesRegex(ValueError, 'head/cache representation differs'):
                legacy.restore(receipt['path'], 'run:1')


if __name__ == '__main__': unittest.main()
