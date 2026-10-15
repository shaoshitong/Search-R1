"""Small CPU fixtures for formal token acquisition and actor/head transactions."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from search_g1.qprobe_runtime import QProbeRuntime, validate_config
from search_g1.qprobe_selection import (
    PredictQuestion, binary_predictions, binary_prequential_metrics, select_binary_questions,
)


def questions(size=5120):
    return [PredictQuestion(i, 'q' + str(i), 'question ' + str(i)) for i in range(size)]


class MemoryTokenCache:
    feature_dim = 4
    spec_sha = 'a' * 64
    spec = dict(global_max_tokens=244)

    def __init__(self, size=5120):
        self.ids = [q.qid for q in questions(size)]
        self.index = {q.qid: dict(qid=q.qid, question_hash=q.question_hash, length=1)
                      for q in questions(size)}
        self.manifest = dict(representation='all_nonpadding_question_tokens_packed_v1', count=size)

    def get_batch(self, qids, device='cpu'):
        return dict(features=torch.tensor([[[int(qid[1:]) / 5120, 1., 0., 0.]] for qid in qids], device=device),
                    attention_mask=torch.ones((len(qids), 1), dtype=torch.bool, device=device))

    def get_features(self, *_args):
        raise AssertionError('A token runtime must never materialize the pooled/all-candidate tensor API')


class RecordingHead:
    feature_dim = 4
    feature_spec_sha = MemoryTokenCache.spec_sha

    def __init__(self):
        self.completed_outer = 0
        self.update_count = 0
        self.predict_calls = []
        self.updated_qids = []
        self.mutate_prediction = False
        self.initialization_provenance = dict(warmup_checkpoint_sha256='b' * 64,
            warmup_head_adam_steps=200, adopted_at_completed_outer=0, no_refit=True)

    @property
    def version(self):
        return 'head-' + str(self.update_count)

    def _adam_steps(self):
        return 200 + self.update_count

    def predict(self, qids):
        self.predict_calls.append(list(qids))
        if self.mutate_prediction:
            self.update_count += 1
        return dict(mixed_probability=torch.tensor([int(qid[1:]) / 5120 for qid in qids]),
                    predicted_sample_std=torch.tensor([.5 - .4 * int(qid[1:]) / 5120 for qid in qids]))

    def update_completed_outer(self, outer, labels, qids, **contracts):
        if any(not isinstance(qid, str) for qid in qids):
            raise AssertionError('Only compact qids may enter token replay')
        self.updated_qids = list(qids)
        self.completed_outer = outer
        self.update_count += 1
        return dict(groups=len(labels), **contracts)

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__ = copy.deepcopy(state)

    def save(self, path, *, actor_snapshot_id):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(path=str(path), sha256=sha, bytes=path.stat().st_size, actor_snapshot_id=actor_snapshot_id)


def runtime(base, cache=None, head=None, **overrides):
    config = dict(head_kind='token_binary_v1', ranking='binary_mixed_probability', candidate_questions=5120,
        selected_questions=512, responses_per_question=5, exploration_questions=0, max_tokens=244,
        target_statistic='sample_std', std_loss_weight=.3, prediction_batch_size=127,
        seed=20261013, head_seed=1729, feature_dim=4)
    config.update(overrides)
    return QProbeRuntime(config, base, feature_cache=cache or MemoryTokenCache(), head=head or RecordingHead())


def stage(value, attempt='first'):
    selected = questions(512)
    qids, prediction, version = value.score(selected)
    labels = [dict(complete=True, qid=q.qid, question_hash=q.question_hash,
                   strict_em=[int(j < q.ordinal % 6) for j in range(5)],
                   selection_source='predicted_binary_mixed') for q in selected]
    value.begin_outer(outer=1, attempt_id=attempt, selected=list(range(512)), features=qids,
        probabilities=prediction, prediction_version=version, policy_snapshot_id='sampling-actor:0',
        contracts=dict(run_id='run', sampling_contract_sha='sampling', strict_scoring_contract_sha='strict',
                       retriever_version='retriever'))
    value.observe_labels(labels, [row['strict_em'] for row in labels])
    return labels


class BinaryAcquisitionTests(unittest.TestCase):
    def test_top512_is_probability_only_and_hash_ties_ignore_std_and_source_order(self):
        candidates = questions()
        probability = np.ones(5120) * .7
        std = np.linspace(0, np.sqrt(.3), 5120)
        expected = {q.qid for q in sorted(candidates, key=lambda q: hashlib.sha256(q.qid.encode()).hexdigest())[:512]}
        first = select_binary_questions(candidates, probability, std)
        second = select_binary_questions(candidates[::-1], probability[::-1], std[::-1])
        third = select_binary_questions(candidates, probability, std[::-1])
        self.assertEqual({row.qid for row in first}, expected)
        self.assertEqual([row.qid for row in first], [row.qid for row in second])
        self.assertEqual([row.qid for row in first], [row.qid for row in third])
        self.assertEqual([row.ordinal for row in first], sorted(row.ordinal for row in first))
        self.assertTrue(all(row.selection_source == 'predicted_binary_mixed' for row in first))
        with self.assertRaises(ValueError):
            select_binary_questions(candidates, probability, std, exploration_questions=64)

    def test_binary_accuracy_and_raw_confusion_use_frozen_predictions_without_count_outputs(self):
        labels = [[0] * 5, [1] * 5, [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]]
        result = binary_prequential_metrics([.1, .8, .9, .4], [0, 0, np.sqrt(.2), np.sqrt(.3)],
                                            labels, ['predicted_binary_mixed'] * 4)
        self.assertEqual(result['all']['binary_accuracy'], .5)
        self.assertEqual(result['all']['binary_confusion'], dict(tp=1, tn=1, fp=1, fn=1))
        self.assertEqual(result['all']['std_mse_raw'], 0)
        self.assertFalse({'count_accuracy', 'sigma3_accuracy', 'count_nll', 'mean_mae'} & set(result['all']))
        with self.assertRaises(ValueError): binary_predictions([1.1], [0])
        with self.assertRaises(ValueError): binary_predictions([.5], [1])

    def test_token_config_rejects_legacy_ranking_exploration_or_other_auxiliary_target(self):
        token = dict(head_kind='token_binary_v1')
        validate_config(token)
        for changes in (dict(ranking='expected_sample_std'), dict(exploration_questions=1),
                        dict(target_statistic='sample_variance'), dict(std_loss_weight=.1),
                        dict(candidate_questions=600), dict(max_tokens=245), dict(prediction_batch_size=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(dict(token, **changes))


class TokenRuntimeTests(unittest.TestCase):
    def test_all5120_scored_in_bounded_qid_batches_with_frozen_version(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            before_rng = copy.deepcopy(value.selection_rng.bit_generator.state)
            qids, prediction, version = value.score(questions())
            self.assertIsInstance(qids, tuple)
            self.assertEqual(qids, tuple(q.qid for q in questions()))
            self.assertEqual(sum(map(len, value.head.predict_calls)), 5120)
            self.assertLessEqual(max(map(len, value.head.predict_calls)), 127)
            self.assertEqual(set(prediction), {'mixed_probability', 'predicted_sample_std'})
            self.assertEqual(prediction['mixed_probability'].shape, (5120,))
            self.assertEqual(version, value.version)
            self.assertEqual(value.selection_rng.bit_generator.state, before_rng)
            value.head.mutate_prediction = True
            with self.assertRaisesRegex(RuntimeError, 'frozen question prediction'):
                value.score(questions(1))

    def test_exact_question_hash_is_checked_before_head_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            with self.assertRaisesRegex(ValueError, 'text/hash differs'):
                value.score([PredictQuestion(0, 'q0', 'changed question')])
            with self.assertRaisesRegex(ValueError, 'qid is absent'):
                value.score([PredictQuestion(0, 'missing', 'question 0')])
            self.assertEqual(value.head.predict_calls, [])

    def test_pending_stores_compact_selected_qids_and_abort_never_fits(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            labels = stage(value)
            self.assertNotIn('features', value.pending)
            self.assertNotIn('probabilities', value.pending)
            self.assertEqual(value.pending['qids'], [row['qid'] for row in labels])
            self.assertEqual(value.pending['prequential']['all']['questions'], 512)
            self.assertEqual(value.head.update_count, 0)
            value.abort_outer(1, {'attempt_id': 'first'})
            self.assertEqual(value.head.update_count, 0)
            stage(value, 'success')
            event = value.complete_outer(1, 'run:1', {'attempt_id': 'success'})
            self.assertEqual(value.head.update_count, 1)
            self.assertEqual(value.head.updated_qids, [row['qid'] for row in labels])
            self.assertEqual(event['prequential']['all']['binary_confusion'], dict(tp=0, tn=171, fp=0, fn=341))
            with self.assertRaises(RuntimeError): value.complete_outer(1, 'run:1')

    def test_commit_publication_failure_restores_head_and_retains_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            stage(value)
            original = value.version
            with patch('search_g1.qprobe_runtime._write_json', side_effect=OSError('failed publication')):
                with self.assertRaises(OSError): value.complete_outer(1, 'run:1')
            self.assertEqual(value.version, original)
            self.assertEqual(value.head.updated_qids, [])
            self.assertIsNotNone(value.pending)

    def test_selected_labels_must_match_handles_hash_and_exact_strict_vector(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            labels = stage(value)
            value.pending['labels'] = None
            for key, bad in (('qid', 'q1'), ('question_hash', '0' * 64), ('complete', False), ('strict_em', [1] * 5)):
                changed = copy.deepcopy(labels)
                changed[0][key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    value.observe_labels(changed, [row['strict_em'] for row in labels])
            with self.assertRaises(ValueError): value.observe_labels(labels[:-1], [row['strict_em'] for row in labels[:-1]])
            self.assertEqual(value.head.update_count, 0)

    def test_snapshot_restores_binary_head_and_rejects_changed_cache_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            stage(value)
            value.complete_outer(1, 'run:1')
            receipt = value.save(Path(temp) / 'paired.pt', 'run:1')
            other = runtime(temp)
            def load_fake(path, config, **kwargs):
                self.assertEqual(kwargs['expected_actor_snapshot_id'], 'run:1')
                self.assertIs(kwargs['feature_cache'], other.cache)
                head = RecordingHead()
                head.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
                return head
            with patch('search_g1.qprobe_runtime.load_head_checkpoint', side_effect=load_fake):
                state = other.restore(receipt['path'], 'run:1')
            self.assertEqual(other.version, value.version)
            self.assertEqual(state['head_kind'], 'token_binary_v1')
            self.assertEqual(other.head.updated_qids, value.head.updated_qids)
            changed_cache = MemoryTokenCache()
            changed_cache.index['q0']['question_hash'] = '0' * 64
            changed = runtime(temp, cache=changed_cache)
            with self.assertRaisesRegex(ValueError, 'head/cache representation differs'):
                changed.restore(receipt['path'], 'run:1')

    def test_ready_adoption_is_verified_without_fit_or_teacher_rollout(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            value = runtime(base)
            checkpoint = base / 'initial.pt'
            value.head.save(checkpoint, actor_snapshot_id='run:0')
            head_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ready = dict(stage='TOKEN_BINARY_FORMAL_INITIAL_HEAD_ADOPTED', head_receipt=dict(sha256=head_sha),
                feature_spec_sha=value.feature_spec_sha, actual_head_adam_updates=200, completed_outer=0,
                warmup_questions=5120, trajectories_regenerated=False, no_head_refit=True,
                raw_warmup_checkpoint_sha256='b' * 64)
            ready_path = base / 'ready.json'
            ready_path.write_text(json.dumps(ready))
            value.config.update(initial_head_checkpoint=str(checkpoint), initial_head_checkpoint_sha256=head_sha,
                initial_head_ready=str(ready_path), initial_head_ready_sha256=hashlib.sha256(ready_path.read_bytes()).hexdigest())
            before = value.head.state_dict()
            first = value.bootstrap_initial_head()
            self.assertEqual(value.bootstrap_initial_head(), first)
            self.assertEqual(value.head.state_dict(), before)
            self.assertEqual(first['logical_outer'], 0)
            self.assertEqual(value.head.predict_calls, [])
            value.head.initialization_provenance['no_refit'] = False
            with self.assertRaisesRegex(ValueError, 'exact warmup10'):
                value.bootstrap_initial_head()


if __name__ == '__main__':
    unittest.main()
