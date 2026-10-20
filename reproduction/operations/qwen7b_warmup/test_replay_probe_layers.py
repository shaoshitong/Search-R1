"""Synthetic replay alignment tests only; no local model or actual feature bank."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import torch

from .replay_probe_layers import (LAYERS, QMAX, REVISION, Comparison, ReplayLayerWriter,
                                   canonical, fingerprints, normalize_prefix, replay_features)


def row():
    result = dict(trajectory_id='q0-r0', qid='q0', actor_version='official7b-'+REVISION,
        initial_prompt_token_ids=[11, 12, 13], policy_token_ids=[21, 22, 31, 32],
        policy_position_ids=[3, 4, 8, 9], retained_policy_length=4,
        prefix_segments=[dict(turn_index=0, policy_start=0, policy_length=2,
            request_length=3, position_start=3, context_tokens=[11, 12, 13, 21, 22]),
            dict(turn_index=1, policy_start=2, policy_length=2, request_length=8,
                 position_start=8, context_tokens=[11, 12, 13, 21, 22, 90, 91, 92, 31, 32])],
        terminal=True, future_information_included=False)
    result['prefix_sha256'] = hashlib.sha256(canonical(result)).hexdigest()
    return result


class ReplayAlignmentTests(unittest.TestCase):
    def test_token_position_alignment_one_pass_per_segment_and_no_future(self):
        normalized = normalize_prefix(row(), 'q0-r0', 'q0', require_digest=True)
        calls = []
        def capture(context, positions):
            calls.append((list(context), list(positions)))
            # Distinguish every token/position and layer.  A later tool context
            # would visibly corrupt question features if used to overwrite them.
            turn = len(calls)*1000
            return {layer:torch.tensor([[turn+layer+token, pos] for token, pos in zip(context, positions)],
                                      dtype=torch.float32) for layer in LAYERS}
        replay = replay_features(normalized, capture, hidden_size=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ([11, 12, 13, 21, 22], list(range(5))))
        self.assertEqual(calls[1][0], [11, 12, 13, 21, 22, 90, 91, 92, 31, 32])
        for layer in LAYERS:
            features = replay[layer]['features'].float()
            self.assertEqual(features[0, 0], torch.tensor(1000+layer+11).bfloat16().float())
            torch.testing.assert_close(features[QMAX:QMAX+4, 1], torch.tensor([3., 4., 8., 9.]))
            self.assertEqual(replay[layer]['position_ids'][QMAX:QMAX+4].tolist(), [3, 4, 8, 9])
            self.assertEqual(int(replay[layer]['attention_mask'].sum()), 7)
            self.assertTrue(torch.equal(features[3:QMAX], torch.zeros_like(features[3:QMAX])))
            self.assertTrue(torch.equal(features[QMAX+4:], torch.zeros_like(features[QMAX+4:])))

    def test_rejects_wrong_ids_overlap_positions_or_future_context(self):
        for mutate in (
            lambda r:r['prefix_segments'][1].update(policy_start=1),
            lambda r:r['prefix_segments'][1].update(position_start=9),
            lambda r:r['prefix_segments'][1]['context_tokens'].append(999),
            lambda r:r['prefix_segments'][1]['context_tokens'].__setitem__(-1, 999),
            lambda r:r['prefix_segments'][1]['context_tokens'].__setitem__(0, 999),
            lambda r:r['policy_position_ids'].__setitem__(2, 7),
        ):
            bad = row()
            bad.pop('prefix_sha256')
            mutate(bad)
            with self.assertRaises(ValueError):
                normalize_prefix(bad)
        with self.assertRaises(ValueError):
            normalize_prefix(row(), expected_tid='another-trajectory')
        with self.assertRaises(ValueError):
            normalize_prefix(row(), expected_qid='another-question')

    def test_native_feature_locator_append_does_not_change_sealed_token_digest(self):
        original = row()
        original.update(feature_layer=26, feature_file='/remote/synthetic/complete.json',
                        feature_boundary='post_block_hidden_plus_residual')
        normalized = normalize_prefix(original, require_digest=True)
        self.assertEqual(normalized['source_prefix_sha256'], row()['prefix_sha256'])
        original['policy_token_ids'][0] = 99
        with self.assertRaises(ValueError):
            normalize_prefix(original, require_digest=True)

    def test_short_and_zero_prefix_require_real_terminal_and_capture_initial_prompt(self):
        zero = row()
        zero.pop('prefix_sha256')
        zero.update(policy_token_ids=[], policy_position_ids=[], retained_policy_length=0, prefix_segments=[])
        normalized = normalize_prefix(zero)
        calls = []
        def capture(context, positions):
            calls.append(list(context))
            return {layer:torch.ones(len(context), 2) for layer in LAYERS}
        result = replay_features(normalized, capture, hidden_size=2)
        self.assertEqual(calls, [[11, 12, 13]])
        self.assertEqual(int(result[26]['attention_mask'].sum()), 3)
        self.assertEqual(result[26]['policy_length'], 0)
        zero['terminal'] = False
        with self.assertRaises(ValueError):
            normalize_prefix(zero)

    def test_alignment_rejects_grad_captures_or_missing_layers(self):
        normalized = normalize_prefix(row())
        with self.assertRaises(ValueError):
            replay_features(normalized, lambda c,p:{26:torch.ones(len(c), 2)}, hidden_size=2)
        with self.assertRaises(ValueError):
            replay_features(normalized, lambda c,p:{layer:torch.ones(len(c), 2, requires_grad=True)
                            for layer in LAYERS}, hidden_size=2)

    def test_comparison_reports_known_error_and_handles_empty_policy(self):
        reference = torch.tensor([[1., 2.], [2., 4.]])
        metric = Comparison()
        metric.add(2*reference, reference)
        metric.add(torch.empty(0, 2), torch.empty(0, 2))
        summary = metric.result()
        self.assertEqual(summary['elements'], 4)
        self.assertEqual(summary['tokens'], 2)
        self.assertAlmostEqual(summary['relative_l2'], 1.)
        self.assertAlmostEqual(summary['global_cosine'], 1.)
        self.assertEqual(summary['max_absolute_error'], 4.)
        self.assertEqual(summary['mean_absolute_error'], 2.25)

    def test_arbitrary_fixture_counts_pack_losslessly_without_old_g5_rule(self):
        import numpy as np
        selected = []
        for tid, qid in (('q0-r0', 'q0'), ('q0-r1', 'q0'), ('q1-r0', 'q1')):
            source = row()
            source.pop('prefix_sha256')
            source.update(trajectory_id=tid, qid=qid)
            selected.append(normalize_prefix(source))
        expected = {}
        with tempfile.TemporaryDirectory() as directory:
            writer = ReplayLayerWriter(Path(directory)/'layer26', 26, selected, dimension=2)
            for ordinal, source in enumerate(selected):
                def capture(context, positions):
                    return {layer:torch.tensor([[token+.125*ordinal, position] for token, position in zip(context, positions)],
                                              dtype=torch.float32) for layer in LAYERS}
                data = replay_features(source, capture, hidden_size=2)[26]
                expected[source['trajectory_id']] = data['features'].view(torch.uint16).numpy().copy()
                writer.write(source, data)
            descriptor = writer.finish()
            manifest = json.loads(Path(descriptor['path']).read_text())
            self.assertEqual(manifest['count'], 3)
            self.assertEqual(manifest['status'], 'LOSSLESS_HF_REPLAY_FEATURES_COMPLETE')
            for entry in manifest['files'].values():
                self.assertEqual(entry['sha256'], fingerprints(entry['path'])['sha256'])
            root = Path(descriptor['path']).parent
            prototype = np.frombuffer((root/'prototype.bin').read_bytes(), dtype=np.uint16).reshape(QMAX, 2)
            question_bytes, delta_bytes, policy_bytes = [(root/name).read_bytes() for name in
                                                        ('question.bin', 'delta.bin', 'policy.bin')]
            for line in (root/'index.jsonl').read_text().splitlines():
                record = json.loads(line)
                qlen, plen = record['question_length'], record['policy_length']
                offset, count = record['question_offset'], record['question_bytes']
                question = np.bitwise_xor(prototype[:qlen],
                    np.frombuffer(zlib.decompress(question_bytes[offset:offset+count]), dtype=np.uint16).reshape(qlen, 2))
                offset, count = record['delta_offset'], record['delta_bytes']
                if count:
                    question = np.bitwise_xor(question,
                        np.frombuffer(zlib.decompress(delta_bytes[offset:offset+count]), dtype=np.uint16).reshape(qlen, 2))
                offset = record['policy_offset']
                policy = np.frombuffer(policy_bytes[offset:offset+plen*2*2], dtype=np.uint16).reshape(plen, 2)
                original = expected[record['trajectory_id']]
                self.assertTrue(np.array_equal(question, original[:qlen]))
                self.assertTrue(np.array_equal(policy, original[QMAX:QMAX+plen]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
