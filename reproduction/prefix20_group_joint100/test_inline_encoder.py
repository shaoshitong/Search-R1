"""CPU causal mocks for authentic batch-inline question/policy assembly."""
import unittest

import numpy as np
import torch

from .inline_encoder import encode_rows


class Capture:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def __call__(self, tokens, mask, positions):
        self.calls.append((tokens.detach().clone(), mask.detach().clone(), positions.detach().clone()))
        hidden = torch.zeros((*tokens.shape, 2048), dtype=torch.float32)
        hidden[..., 0] = tokens * self.weight
        hidden[..., 1] = positions
        # A causal mock: retained policy representations see their real prior
        # request tokens, while initial question positions cannot see the future.
        hidden[..., 2] = (tokens * mask).cumsum(dim=1)
        return hidden


def fixture():
    contexts = np.asarray([11, 12, 21, 22,
                           11, 12, 21, 22, 99, 100, 23, 24,
                           41, 42, 43, 51], dtype=np.int64)
    rows = [dict(trajectory_id='q0-t0', qid='q0', strict_em=1, length=4, policy_offset=0,
                 segments=[dict(turn_index=0, request_length=2, policy_start=0, policy_length=2,
                                position_start=2, context_offset=0, context_length=4),
                           dict(turn_index=1, request_length=6, policy_start=2, policy_length=2,
                                position_start=6, context_offset=4, context_length=8)]),
            dict(trajectory_id='q1-t0', qid='q1', strict_em=0, length=1, policy_offset=4,
                 segments=[dict(turn_index=0, request_length=3, policy_start=0, policy_length=1,
                                position_start=3, context_offset=12, context_length=4)])]
    return rows, contexts, np.asarray([21, 22, 23, 24, 51]), np.asarray([2, 3, 6, 7, 3])


class InlineEncoderTests(unittest.TestCase):
    def test_real_context_question_identity_and_original_multiturn_positions(self):
        capture = Capture()
        batch = encode_rows(*fixture(), 5, capture, micro_batch_size=2)
        self.assertEqual(batch['trajectory_ids'], ['q0-t0', 'q1-t0'])
        self.assertEqual(batch['labels'].tolist(), [1, 0])
        self.assertEqual(batch['features'].shape, (2, 25, 2048))
        self.assertEqual(batch['features'][0, :2, 0].tolist(), [11, 12])
        self.assertEqual(batch['features'][0, :2, 2].tolist(), [11, 23])
        self.assertEqual(batch['features'][0, 5:9, 0].tolist(), [21, 22, 23, 24])
        self.assertEqual(batch['features'][0, 5:9, 2].tolist(), [44, 66, 288, 312])
        self.assertEqual(batch['position_ids'][0, 5:9].tolist(), [2, 3, 6, 7])
        self.assertEqual(batch['attention_mask'][0].tolist(), [True, True, False, False, False] + [True] * 4 + [False] * 16)
        self.assertEqual(batch['question_lengths'].tolist(), [2, 3])
        self.assertEqual(batch['question_mask'].sum(dim=1).tolist(), [2, 3])
        self.assertEqual(batch['policy_mask'].sum(dim=1).tolist(), [4, 1])
        self.assertEqual(len(capture.calls), 2)

    def test_forward_only_first_twenty_and_never_future_policy_tail(self):
        capture = Capture()
        ids = np.arange(100, 180, dtype=np.int64)
        row = dict(trajectory_id='q-t', qid='q', strict_em=0, length=80, policy_offset=0,
                   segments=[dict(turn_index=0, request_length=2, policy_start=0, policy_length=80,
                                  position_start=2, context_offset=0, context_length=82)])
        batch = encode_rows([row], np.concatenate((np.asarray([1, 2]), ids)), ids,
                            np.arange(2, 82), 5, capture)
        self.assertEqual(capture.calls[0][0].shape, (1, 22))
        self.assertEqual(capture.calls[0][0][0, -1].item(), 119)
        self.assertEqual(batch['features'][0, 5:25, 0].tolist(), list(range(100, 120)))
        self.assertEqual(batch['policy_lengths'].tolist(), [20])

    def test_normal_detached_features_allow_head_backward_without_encoder_gradient(self):
        capture = Capture()
        batch = encode_rows(*fixture(), 5, capture)
        features = batch['features']
        self.assertFalse(features.requires_grad)
        self.assertFalse(torch.is_inference(features))
        head = torch.nn.Linear(2048, 1)
        loss = head(features.float()).square().mean()
        loss.backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertIsNone(capture.weight.grad)

    def test_mismatched_own_token_identity_is_rejected_without_repairing_data(self):
        rows, contexts, ids, positions = fixture()
        ids[2] = 999
        with self.assertRaisesRegex(ValueError, 'exact causal request'):
            encode_rows(rows, contexts, ids, positions, 5, Capture())


if __name__ == '__main__':
    unittest.main()
