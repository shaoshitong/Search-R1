"""Small CPU fixtures for two-rank plans and prefix-only absolute positions."""
import random
import types
import unittest

import torch

from .model import validate_prefix_positions
from .features import retained_segments
from .train_ddp import batch_plan, validate_cache, EPOCHS, STEPS_PER_EPOCH, WORLD, LOCAL_BATCH


class PrefixOnlyContractTests(unittest.TestCase):
    def test_fifty_plus_thirty_uses_second_real_context_without_tool_positions(self):
        prompt = [1, 2, 3]
        first = list(range(100, 190))
        second = list(range(200, 260))
        observation = [900, 901, 902, 903]
        request2 = prompt + first[:50] + observation
        def turn(index, request, raw, kept):
            return dict(trajectory_id='same', turn_index=index,
                        actual_request_token_ids=request, raw_generated_token_ids=raw,
                        raw_generated_token_count=len(raw), kept_token_count=kept,
                        prefix_cut=dict(kept_token_ids=raw[:kept]))
        turns = [turn(0, prompt, first, 50), turn(1, request2, second, 60)]
        segments, tokens, positions = retained_segments(turns, 'same', 80)
        self.assertEqual(tokens, first[:50] + second[:30])
        self.assertEqual([s['policy_length'] for s in segments], [50, 30])
        self.assertEqual(segments[1]['context_tokens'], request2 + second[:30])
        self.assertEqual(positions, list(range(3, 53)) + list(range(len(request2), len(request2) + 30)))
        self.assertTrue(set(observation).isdisjoint(tokens))
        self.assertTrue(set(first[50:]).isdisjoint(tokens))
        self.assertEqual(len(retained_segments(turns, 'same', 40)[0]), 1)

    def test_same_balanced_plans_across_four_arms_and_two_ranks(self):
        labels = [1] * 5627 + [0] * (25600 - 5627)
        plans = [batch_plan(labels, 'balanced', random.Random(1729)) for _ in (10, 20, 40, 80)]
        self.assertTrue(all(plan == plans[0] for plan in plans))
        self.assertEqual((WORLD, LOCAL_BATCH, EPOCHS * STEPS_PER_EPOCH), (2, 128, 2000))
        for batch in plans[0]:
            self.assertEqual(len(batch), 256)
            self.assertEqual(sum(labels[i] for i in batch), 128)
            self.assertEqual(batch[:LOCAL_BATCH] + batch[LOCAL_BATCH:], batch)

    def test_natural_optional_mode_has_exact_epoch_coverage(self):
        labels = [1] * 5000 + [0] * 20600
        batches = batch_plan(labels, 'natural', random.Random(1729))
        self.assertEqual(len(batches), 100)
        self.assertEqual(sorted(sum(batches, [])), list(range(25600)))

    def test_absolute_positions_preserve_turn_gaps_resets_and_short_prefixes(self):
        mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
        positions = torch.tensor([[242, 243, 300, 301], [111, 0, 0, 0]], dtype=torch.long)
        segments = [[dict(policy_start=0, policy_length=2, position_start=242, turn_index=0),
                     dict(policy_start=2, policy_length=2, position_start=300, turn_index=1)],
                    [dict(policy_start=0, policy_length=1, position_start=111, turn_index=0)]]
        validate_prefix_positions(mask, positions, segments)
        # Actual subsequent request context cropping can reset absolute positions.
        segments[0][1]['position_start'] = 100
        positions[0, 2:] = torch.tensor([100, 101])
        validate_prefix_positions(mask, positions, segments)

    def test_within_segment_reset_and_false_segment_coverage_are_rejected(self):
        segments = [[dict(policy_start=0, policy_length=2, position_start=242, turn_index=0)]]
        mask = torch.tensor([[True, True, False]])
        for positions in (torch.tensor([[0, 1, 0]]), torch.tensor([[242, 244, 0]])):
            with self.assertRaises(ValueError):
                validate_prefix_positions(mask, positions, segments)
        with self.assertRaises(ValueError):
            validate_prefix_positions(torch.tensor([[True, False, True]]), torch.tensor([[242, 0, 243]]), segments)
        segments[0][0]['policy_start'] = 1
        with self.assertRaises(ValueError):
            validate_prefix_positions(mask, torch.tensor([[242, 243, 0]]), segments)

    def test_cache_only_prefix_slots_and_same_capacity80(self):
        train = ['train-%d' % i for i in range(25600)]
        test = ['test-%d' % i for i in range(2560)]
        index = {}
        for split, ids in [('train', train), ('test', test)]:
            for i, tid in enumerate(ids):
                index[tid] = dict(qid='%s-q%d' % (split, i // 5), question_hash='%s-h%d' % (split, i // 5),
                                  strict_em=i % 2, prefix_length=10, length=10,
                                  segment_bounds=[dict(policy_start=0, policy_length=10, position_start=242, turn_index=0)])
        cache = types.SimpleNamespace(ids_by_split=dict(train=train, test=test), index=index,
                                      max_tokens=80, feature_dim=2048,
                                      manifest=dict(representation='multi_turn_retained_policy_prefix_layer27_v1'),
                                      spec=dict(prefix_length=10))
        self.assertEqual(validate_cache(cache, 10), (train, test))
        cache.max_tokens = 262
        with self.assertRaises(ValueError):
            validate_cache(cache, 10)
        cache.max_tokens = 80
        cache.index[train[0]]['length'] = 252  # Prompt positions must never enter the head.
        with self.assertRaises(ValueError):
            validate_cache(cache, 10)


if __name__ == '__main__':
    unittest.main()
