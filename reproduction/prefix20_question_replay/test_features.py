"""Small CPU contract tests; never load an encoder or touch remote data."""
import hashlib
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from .features import (
    FEATURE_DIM, FEATURE_ROLE, assemble_batch, initial_question_segment,
    initial_request_length_from_rollout, question_max_from_rows, retained_twenty,
    scan_question_max,
)


def source_row(first=8, second=22, question=3, later_request=40):
    return dict(length=first + second, segments=[
        dict(turn_index=0, request_length=question, policy_start=0,
             policy_length=first, position_start=question, context_offset=0,
             context_length=question + first),
        dict(turn_index=1, request_length=later_request, policy_start=first,
             policy_length=second, position_start=later_request,
             context_offset=question + first, context_length=later_request + second),
    ])


class QuestionPolicyFeatureTests(unittest.TestCase):
    def test_compact_qmax_reads_initial_actual_request_not_later_context(self):
        row = dict(trajectory_id='t', prefix_segments=[
            dict(turn_index=0, request_length=3, position_start=3, context_tokens=[11, 12, 13, 21, 22]),
            dict(turn_index=1, request_length=100, position_start=100, context_tokens=list(range(120))),
        ])
        self.assertEqual(initial_request_length_from_rollout(row), 3)
        row['prefix_segments'] = row['prefix_segments'][1:]
        with self.assertRaisesRegex(ValueError, 'turn0'):
            initial_request_length_from_rollout(row)

    def test_qmax_inventory_accepts_compact_epochs_without_rebuilding_scans(self):
        train = dict(source_row(question=7), split='train')
        test = dict(source_row(question=11), split='test')
        source_scan = (Path('/mnt/test/original'), {'files': {'index.jsonl': {'sha256': 'index'}}},
                       [train] * 25600 + [test] * 2560, Path('/mnt/test/original'))
        inventory = {'result': dict(replay_sources=[
            dict(source_data_epoch=1, kind='cached_inputs', inputs='/mnt/test/original', split='train'),
            dict(source_data_epoch=2, kind='rollout_complete', rollout_complete='/mnt/test/e2.json',
                 sha256='source-sha', split='train')],
            test_source=dict(kind='cached_inputs', inputs='/mnt/test/original', split='test'))}
        with mock.patch('reproduction.prefix20_question_replay.features._source_scan', return_value=source_scan), \
             mock.patch('reproduction.prefix20_question_replay.features._rollout_question_lengths',
                        return_value=({'epoch': 2, 'shards': []}, 75, 3, 25600)) as compact, \
             mock.patch('reproduction.prefix20_question_replay.features.sha', return_value='source-sha'), \
             mock.patch('reproduction.prefix20_question_replay.features.once') as publish:
            result = scan_question_max(inventory, '/mnt/test/qmax.json', expected_train_epochs=2)
        compact.assert_called_once_with(Path('/mnt/test/e2.json'))
        self.assertEqual(result['question_max_tokens'], 75)
        self.assertEqual(result['head_max_tokens'], 95)
        self.assertEqual(result['train_epochs'], [1, 2])
        self.assertEqual(len(result['sources']), 3)
        publish.assert_called_once()

    def test_initial_request_is_not_later_tool_context(self):
        row = source_row()
        actual = initial_question_segment(row)
        self.assertEqual(actual, dict(context_offset=0, context_length=3,
                                     question_length=3, turn_index=0, position_start=0))
        self.assertEqual(question_max_from_rows([row, source_row(question=7)]), (7, 3, 2))
        missing = source_row()
        missing['segments'] = missing['segments'][1:]
        missing['segments'][0]['policy_start'] = 0
        with self.assertRaisesRegex(ValueError, 'turn0'):
            initial_question_segment(missing)

    def test_first_twenty_keep_actual_multiturn_positions(self):
        row = source_row(first=8, second=72)
        segments, bounds, n = retained_twenty(row)
        self.assertEqual(n, 20)
        self.assertEqual([s['policy_length'] for s in segments], [8, 12])
        self.assertEqual([s['context_length'] for s in segments], [11, 52])
        self.assertEqual([s['position_start'] for s in bounds], [3, 40])
        # Source width80 and all original metadata remain unchanged.
        self.assertEqual(row['segments'][1]['policy_length'], 72)
        self.assertEqual(row['segments'][1]['context_length'], 112)

    def test_question_padding_and_policy_fixed_tail_preserve_identity(self):
        qmax, qlen, plen = 7, 3, 5
        # A cross-turn policy has non-contiguous original absolute positions.
        bounds = [dict(turn_index=0, policy_start=0, policy_length=2, position_start=3),
                  dict(turn_index=1, policy_start=2, policy_length=3, position_start=40)]
        h = torch.arange((qlen + plen) * FEATURE_DIM, dtype=torch.float32).reshape(qlen + plen, FEATURE_DIM).to(torch.bfloat16)
        raw = h.view(torch.int16).numpy().view(np.uint16)
        row = dict(question_length=qlen, policy_length=plen, feature_offset=0,
                   feature_length=qlen + plen, feature_shard=0, policy_offset=0,
                   feature_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                   segment_bounds=bounds)
        actual = assemble_batch([row], {0: raw}, np.asarray([3, 4, 40, 41, 42], dtype=np.int64), qmax)
        self.assertEqual(actual['features'].shape, (1, 27, FEATURE_DIM))
        self.assertTrue(torch.equal(actual['features'][0, :qlen], h[:qlen]))
        self.assertTrue(torch.equal(actual['features'][0, qmax:qmax + plen], h[qlen:]))
        self.assertEqual(actual['attention_mask'][0].tolist(), [True] * 3 + [False] * 4 + [True] * 5 + [False] * 15)
        self.assertEqual(actual['question_mask'].sum().item(), qlen)
        self.assertEqual(actual['policy_mask'].sum().item(), plen)
        self.assertEqual(actual['position_ids'][0, :qlen].tolist(), [0, 1, 2])
        self.assertEqual(actual['position_ids'][0, qmax:qmax + plen].tolist(), [3, 4, 40, 41, 42])
        self.assertEqual(actual['question_lengths'].tolist(), [3])
        self.assertEqual(actual['feature_role'], FEATURE_ROLE)
        invalid = dict(row, feature_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            assemble_batch([invalid], {0: raw}, np.asarray([3, 4, 40, 41, 42], dtype=np.int64), qmax)


if __name__ == '__main__':
    unittest.main()
