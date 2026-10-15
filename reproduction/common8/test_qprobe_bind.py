"""Fail closed before changing a prospective run; no remote or GPU work."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.common8.qprobe_bind import bind
from search_g1.qprobe_model import OnlineQProbe, QProbeConfig


class QProbeBindingTests(unittest.TestCase):
    def fixture(self, root):
        base=Path(root)/'new-run';(base/'logs').mkdir(parents=True)
        repo=base/'repo'
        for name in ('verl/version/version','verl/single_controller/version/version'):
            path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
        run=dict(profile='common8',completed_updates=0,run_id='pristine',input_sha256={'train.parquet':'actual-train-sha'},
                 source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in repo.rglob('*') if p.is_file()})
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        (base/'remote-features').mkdir()
        (base/'remote-features/manifest.json').write_text(json.dumps({'schema_version':1}))
        cache=SimpleNamespace(manifest={'count':169028},spec={'train_parquet_sha256':'actual-train-sha'},
                              spec_sha='f'*64,verify_all=lambda:None)
        return base,run,cache

    def test_fresh_binding_records_all_effective_head_parameters_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as root:
            base,old,cache=self.fixture(root)
            with patch('search_g1.qprobe_features.FeatureCache',return_value=cache):
                run=bind(base,base/'remote-features',exploration_questions=128,retriever_version='immutable-index')
            self.assertEqual(run['question_probe']['exploration_questions'],128)
            self.assertEqual(run['question_probe']['seed'],20261013)
            self.assertEqual(run['question_probe']['head_learning_rate'],3e-4)
            self.assertEqual(json.loads((base/'preparation/question-probe-pre-bind.json').read_text()),old)
            with self.assertRaises(ValueError):
                bind(base,base/'remote-features',retriever_version='immutable-index')

    def test_sequence_binding_records_architecture_micros_and_uniform64(self):
        with tempfile.TemporaryDirectory() as root:
            base,old,cache=self.fixture(root)
            (base/'remote-features/manifest.json').write_text(json.dumps({'schema_version':2}))
            with patch('search_g1.qprobe_sequence.SequenceCache',return_value=cache):
                run=bind(base,base/'remote-features',retriever_version='index',
                    sequence_device='cpu',sequence_train_micro=4,sequence_predict_micro=32)
            q=run['question_probe']
            self.assertEqual(q['exploration_questions'],64)
            self.assertEqual(q['selected_questions']-q['exploration_questions'],448)
            self.assertEqual(q['probe_architecture'],'qwen2_decoder3_binary_mix_v1')
            self.assertEqual(q['sequence_train_micro'],4)
            self.assertEqual(q['sequence_predict_micro'],32)
            self.assertFalse(run.get('full_state_restore'))

    def test_wrong_train_artifact_or_incomplete_cache_rejected_without_manifest_write(self):
        for bad in ('count','train'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as root:
                base,old,cache=self.fixture(root)
                if bad=='count':cache.manifest['count']=10
                else:cache.spec['train_parquet_sha256']='different-split'
                with patch('search_g1.qprobe_features.FeatureCache',return_value=cache),self.assertRaises(ValueError):
                    bind(base,base/'remote-features',retriever_version='index')
                self.assertEqual(json.loads((base/'logs/run-manifest.json').read_text()),old)
                self.assertFalse((base/'preparation/question-probe-pre-bind.json').exists())

    def test_later_head_or_wrong_head_seed_cannot_initialize_fresh_actor(self):
        for bad in ('outer','seed'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as root:
                base,old,cache=self.fixture(root)
                head=OnlineQProbe(4,QProbeConfig(seed=0 if bad=='seed' else 20261013),feature_spec_sha=cache.spec_sha)
                if bad=='outer':head.completed_outer=50
                path=base/'head.pt';head.save(path,actor_snapshot_id='source:50' if bad=='outer' else 'source:0')
                with patch('search_g1.qprobe_features.FeatureCache',return_value=cache),self.assertRaises(ValueError):
                    bind(base,base/'remote-features',retriever_version='index',initial_head_checkpoint=path)
                self.assertEqual(json.loads((base/'logs/run-manifest.json').read_text()),old)

    def test_staged_source_change_requires_new_preparation_without_backdating_method(self):
        with tempfile.TemporaryDirectory() as root:
            base,old,cache=self.fixture(root)
            (base/'repo/new-runtime.py').write_text('new code after preparation')
            with patch('search_g1.qprobe_features.FeatureCache',return_value=cache),self.assertRaisesRegex(ValueError,'prepare a new run'):
                bind(base,base/'remote-features',retriever_version='index')
            self.assertEqual(json.loads((base/'logs/run-manifest.json').read_text()),old)
            self.assertFalse((base/'preparation/question-probe-pre-bind.json').exists())

    def test_claimed_task_is_never_modified(self):
        with tempfile.TemporaryDirectory() as root:
            base,old,_=self.fixture(root)
            (base/'logs/training-launch-claim.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'already trained'):
                bind(base,base/'remote-features',retriever_version='index')
            self.assertEqual(json.loads((base/'logs/run-manifest.json').read_text()),old)


if __name__=='__main__':unittest.main()
