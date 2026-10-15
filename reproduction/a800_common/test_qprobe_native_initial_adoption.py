"""Owned native logical0 adoption fixtures; no real Head/GPU/reward calibration."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from reproduction.a800_common import qprobe_adoption_source_delta as sources
from search_g1.qprobe_qwen_block_online import QwenBlockOnlineConfig


class NativeInitialAdoptionTests(unittest.TestCase):
    def setUp(self):
        sources._NATIVE_INITIAL_CACHE.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.base = self.root/'formal'
        self.warmup = self.root/'native-warmup'
        self.adoption = self.base/'artifacts/question-probe/native-qwen-warmup-adoption'
        self.adoption.mkdir(parents=True)
        (self.warmup/'warmup').mkdir(parents=True)
        self.head_path = self.adoption/'initial.pt'
        self.head_path.write_bytes(b'tiny synthetic Head bytes; no real model state')
        self.run = dict(run_id='e'*32, source_commit='9'*40,
            checkpoint_dir=str(self.base/'checkpoints/search-g1-common8'))
        proof = dict(native_class='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer',
            official_block_copy_verified=True, block_index=35, official_revision=sources.MODEL_REVISION,
            transformers_version='CPU-fixture')
        self.raw = dict(stage='QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE', source_commit=sources.NATIVE_WARMUP_SOURCE,
            source_map_sha256=sources.NATIVE_WARMUP_SOURCE_MAP, head_sha256='b'*64, prepared_sha256='c'*64,
            head_optimizer_updates=200, head_adam_steps=200, epoch_records=11, minibatch_records=200, questions=5120,
            trajectories_generated=0, formal_training_started=False, old_head_inherited=False,
            token_feature_spec_sha256='a'*64, native_block_proof=proof, initial_block_proof={'source_layer': 35},
            initial_readout_proof={'fresh': True, 'seed': 1729}, native_implementation={'native': True})
        self.raw_path = self.warmup/'warmup/ready.json'
        self.raw_path.write_text(json.dumps(self.raw))
        provenance = dict(warmup_source_commit=sources.NATIVE_WARMUP_SOURCE, warmup_ready=str(self.raw_path),
            warmup_ready_sha256=sources._file_sha(self.raw_path), warmup_checkpoint_sha256=self.raw['head_sha256'],
            warmup_head_adam_steps=200, no_refit=True, teacher_replay_excluded=True, formal_replay_groups_at_adoption=0)
        self.ready = dict(stage='QWEN_NATIVE_FORMAL_INITIAL_HEAD_ADOPTED', run_id=self.run['run_id'],
            source_commit=self.run['source_commit'], head_receipt=dict(path=str(self.head_path),
                actor_snapshot_id=self.run['run_id']+':0', sha256=sources._file_sha(self.head_path)),
            actual_head_adam_updates=200, completed_outer=0, warmup_questions=5120, teacher_trajectories_reused=25600,
            no_head_refit=True, teacher_replay_excluded=True, trajectories_regenerated=False,
            actor_optimizer_inherited=False, actor_checkpoint_inherited=False, actual_saved_head_cpu_readback=True,
            formal_training_started=False, native_block_proof=proof, feature_spec_sha='a'*64,
            raw_warmup_checkpoint_sha256=self.raw['head_sha256'], initialization_provenance=provenance,
            actual_warmup=dict(ready_path=str(self.raw_path), ready_sha256=sources._file_sha(self.raw_path),
                source_commit=sources.NATIVE_WARMUP_SOURCE, source_map_sha256=sources.NATIVE_WARMUP_SOURCE_MAP,
                head_sha256=self.raw['head_sha256'], prepared_sha256=self.raw['prepared_sha256']))
        self.settings = dict(head_kind=sources.NATIVE_HEAD_KIND, head_device='cuda:0', seed=20261013, head_seed=1729,
            candidate_questions=5120, selected_questions=512, responses_per_question=5, exploration_questions=0,
            bootstrap_max_groups_per_outer=5120, teacher_warmup_cache='/data/fixed/teacher',
            output_dir=str(self.base/'artifacts/question-probe'), initial_head_checkpoint=str(self.head_path),
            initial_head_checkpoint_sha256=sources._file_sha(self.head_path), initial_head_ready=str(self.adoption/'ready.json'))
        self.head = SimpleNamespace(completed_outer=0, update_count=0, feature_spec_sha='a'*64,
            config=QwenBlockOnlineConfig.from_mapping(self.settings), _adam_steps=lambda: 200,
            replay_labels=[], replay_features=[], protocol=None, initialization_provenance=copy.deepcopy(provenance),
            initial_block_proof=copy.deepcopy(self.raw['initial_block_proof']),
            initial_readout_proof=copy.deepcopy(self.raw['initial_readout_proof']),
            native_implementation=copy.deepcopy(self.raw['native_implementation']),
            verify_native_provenance=lambda: dict(accepted=True, teacher_replay_excluded=True))
        self.rebind_ready()
        self.patches = [patch.object(sources, '_native_storage_path', side_effect=lambda value: Path(value).resolve()),
                        patch.object(sources, 'NATIVE_WARMUP_BASE', self.warmup)]
        for item in self.patches: item.start()

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def rebind_ready(self):
        path = self.adoption/'ready.json'
        path.write_text(json.dumps(self.ready))
        self.settings['initial_head_ready_sha256'] = sources._file_sha(path)

    def load(self, path, config, **kwargs):
        self.assertNotIn('feature_cache', kwargs)  # Detached factory enforces CPU regardless manifest CUDA.
        self.assertEqual(kwargs['expected_actor_snapshot_id'], self.run['run_id']+':0')
        if sources._file_sha(path) != kwargs['expected_checkpoint_sha']:
            raise ValueError('fixture checkpoint SHA changed')
        return self.head

    def validate(self):
        return sources.validate_native_initial_head(self.run, self.settings)

    def test_exact_current_experiment_native_adoption_is_verified_once_for_many_source_pairs(self):
        with patch('search_g1.qprobe_runtime.load_head_checkpoint', side_effect=self.load) as load:
            self.assertTrue(self.validate()['accepted'])
            self.assertTrue(self.validate()['teacher_replay_excluded'])
        self.assertEqual(load.call_count, 1)

    def test_old_warmup_route_outside_artifacts_actor_inheritance_and_missing_hash_are_rejected(self):
        cases = ('old_warmup', 'outside', 'actor', 'head_hash', 'refit')
        for fault in cases:
            settings, ready = copy.deepcopy(self.settings), copy.deepcopy(self.ready)
            if fault == 'old_warmup': self.ready['actual_warmup']['ready_path'] = '/data/old494/warmup/ready.json'
            elif fault == 'outside': self.settings['initial_head_checkpoint'] = '/data/old/head.pt'
            elif fault == 'actor': self.ready['actor_checkpoint_inherited'] = True
            elif fault == 'head_hash': self.settings['initial_head_checkpoint_sha256'] = None
            elif fault == 'refit': self.ready['no_head_refit'] = False
            self.rebind_ready()
            with self.subTest(fault=fault), patch('search_g1.qprobe_runtime.load_head_checkpoint', side_effect=self.load):
                with self.assertRaises(ValueError): self.validate()
            self.settings, self.ready = settings, ready
            self.rebind_ready()

    def test_teacher_replay_wrong_adam_or_different_initial_copy_cannot_pass_typed_readback(self):
        for fault in ('teacher', 'adam', 'copy', 'native_implementation'):
            head = copy.deepcopy(self.head)
            if fault == 'teacher': self.head.replay_labels = [{'source_outer': 0}]
            elif fault == 'adam': self.head._adam_steps = lambda: 0
            elif fault == 'copy': self.head.initial_block_proof = {'source_layer': 34}
            elif fault == 'native_implementation': self.head.native_implementation = {'manual': True}
            sources._NATIVE_INITIAL_CACHE.clear()
            with self.subTest(fault=fault), patch('search_g1.qprobe_runtime.load_head_checkpoint', side_effect=self.load):
                with self.assertRaisesRegex(ValueError, 'STATE_OR_LINEAGE_CHANGED'): self.validate()
            self.head = head

    def test_cached_validation_is_invalidated_by_changed_head_file(self):
        with patch('search_g1.qprobe_runtime.load_head_checkpoint', side_effect=self.load) as load:
            self.validate()
            self.head_path.write_bytes(self.head_path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA changed'): self.validate()
        self.assertEqual(load.call_count, 2)

    def test_native_question_head_does_not_relax_official_fresh_actor_or_reward_identity(self):
        scientific = dict(initial_model='/data/search-g1/assets/qwen2.5-3b-instruct',
            reference_model='/data/search-g1/assets/qwen2.5-3b-instruct', initial_model_sha256={'weights': 'a'*64},
            model_revision=sources.MODEL_REVISION, input_sha256={'train.parquet': 'b'*64},
            calibration_question_ids=['reserved'], retrieval_asset_paths={'index': '/data/index'}, retrieval_encoder='/data/e5')
        common = dict(profile='common8', execution_backend='ssh_local', training_world_size=4,
            completed_updates=0, target_updates=200)
        old = dict(scientific, **common, run_id=sources.ORIGINAL_RUN_ID, execution_id='d'*32,
            source_commit=sources.ORIGINAL_SOURCE_COMMIT)
        run = dict(scientific, **common, **self.run, execution_id='f'*32, precision_profile=sources.PRECISION_PROFILE,
            initial_probe_adoption_source_base=sources.ORIGINAL_BASE, mixed_first_selection=False,
            retriever_version='index', question_probe=self.settings)
        with patch('search_g1.qprobe_runtime.load_head_checkpoint', side_effect=self.load):
            sources.validate_fresh_run(run, old)
            changed = dict(run, full_state_restore={'checkpoint': 'inherited actor'})
            with self.assertRaisesRegex(ValueError, 'FRESH_FOUR_RANK_ACTOR'):
                sources.validate_fresh_run(changed, old)
            changed = dict(run, initial_model_sha256={'weights': '9'*64})
            with self.assertRaisesRegex(ValueError, 'SCIENTIFIC_IDENTITY_CHANGED'):
                sources.validate_fresh_run(changed, old)

    def test_frozen_reward_calibration_paths_are_not_added_to_integration_allowlist(self):
        self.assertFalse(set(sources.FROZEN_G1_SOURCES) & sources.INTEGRATION_EDITS)
        self.assertIn('reproduction/common8/qprobe_qwen_bind.py', sources.INTEGRATION_EDITS)
        self.assertIn('search_g1/qprobe_qwen_block_model.py', sources.INTEGRATION_EDITS)
        self.assertNotIn('search_g1/unreviewed_native_head.py', sources.INTEGRATION_EDITS)


if __name__ == '__main__': unittest.main()
